import json
from itertools import combinations
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets import load_dataset

CLASSES = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "car",
    "chair",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "monitor",
    "night_stand",
    "piano",
    "plant",
    "radio",
    "sink",
    "sofa",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
]

HF_REPO_ID = "kopetri/human_viewpoint_preferences"


def add_noise(label, mean=0.0, std=0.2):
    """Add Gaussian noise to a label, reverting if it changes the class."""
    label_noise = label + torch.normal(mean=mean, std=std, size=(1, 1))
    label_noise = label_noise.clamp(0, 1).reshape(1)
    if label_noise.round() == label:
        return label_noise
    return label  # if the noise changed the label return original label


def anno2int(anno):
    """Convert annotation dict to numeric label (best=2, second=1, worst=0)."""
    label = anno["label"]
    if label == "best":
        return 2.0
    if label == "second":
        return 1.0
    if label == "worst":
        return 0.0
    raise ValueError("Invalid label", label)


def _hf_label(annotation_labels: list, annotation_failed: list, use_failed: bool) -> list:
    """Convert HF dataset annotation columns to per-annotator numeric labels."""
    _map = {"best": 2.0, "second": 1.0, "worst": 0.0}
    labels = [
        _map[lbl] for lbl, failed in zip(annotation_labels, annotation_failed, strict=False) if not failed or use_failed
    ]
    if not labels:  # fallback: include failed annotations
        labels = [_map[lbl] for lbl in annotation_labels]
    return labels


def load_rgb(img_path):
    """Load an image from disk and convert to RGB."""
    return Image.open(img_path).convert("RGB")


def convert2label(data, convert_func, use_failed):
    """Filter annotations and apply a label conversion function."""
    return [convert_func(anno) for anno in data if ((not anno["failed"]) or use_failed)]


def parse_label(anno_file, use_failed, convert_labels):
    """Load an annotation JSON file and return converted labels."""
    with open(anno_file) as json_file:
        data = json.load(json_file)
        if isinstance(data, dict):
            data = data["annotations"]
        labels = convert2label(data, convert_labels, use_failed)
        if len(labels) == 0:
            print("WARNING: Missing labels. using failed ones aswell! {}".format(anno_file))
            labels = convert2label(data, convert_labels, True)
        assert len(labels) > 0, "No labels!!"
        return labels


def data_augment(img_a, img_b, use_rot):
    """Apply random color jitter and optional rotation augmentation to an image pair."""
    if np.random.uniform() > 0.5:  # noqa: NPY002, PLR2004
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
        img_a = color_jitter(img_a)
        img_b = color_jitter(img_b)
    if np.random.uniform() > 0.5 and use_rot:  # noqa: NPY002, PLR2004
        rand_rot = transforms.RandomRotation(90)
        img_a = rand_rot(img_a)
        img_b = rand_rot(img_b)
    return img_a, img_b


class ViewDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        split,
        use_failed=False,
        classes=None,
        filter_agree=False,
        discard_rate=0.0,
        use_rot=False,
        use_triplets=True,
        n_instances=-1,
    ):
        super(ViewDataset, self).__init__()
        assert split in ["train", "valid", "test", "all"], "unknown data set split: {}".format(split)
        self.filter_agree = filter_agree
        self.use_failed = use_failed
        self.split = split
        self.use_triplets = use_triplets
        self.use_rot = use_rot
        self.images = [
            img
            for img in Path(path).glob("**/*")
            if img.suffix == ".jpg" and "mask" not in img.name and split in (img.parent.name, "all")
        ]
        self.models = sorted({model.as_posix().split(".off")[0] for model in self.images})
        self.models = [model for model in self.models if classes is None or any(c in model for c in classes)]
        self.models, self.images = self.filter_instances(n_instances)
        self.n_triplets = self.parse_triplet_size()
        self.pairs = self.make_pairs()
        print("Found {} triplets per model".format(self.n_triplets))
        print("Found {} pairs for split {}.".format(len(self.pairs), split))
        print("Found {} models for split {}.".format(len(self.models), split))
        print("Found {} images for split {}.".format(len(self.images), split))
        if not self.use_triplets:
            print("Not using triplets.")
        if discard_rate > 0:
            self.discard(discard_rate)
            print("{} pairs for split {} after discard with rate: {}".format(len(self.pairs), split, discard_rate))

    def __len__(self):
        return len(self.pairs)

    def parse_triplet_size(self):
        return (np.max([int(Path(i).stem.split("_")[-1]) for i in self.images]) // 3) + 1

    def filter_instances(self, n):
        if n < 1:
            return self.models, self.images
        classes = sorted({Path(m).parents[1].name for m in self.models})
        models = []
        for cls in classes:
            models += [m for m in self.models if Path(m).parents[1].name == cls][0:n]
        m_names = [Path(m).stem for m in models]
        images = [i for i in self.images if i.stem.split(".off")[0] in m_names]
        return models, images

    def discard(self, rate):
        np.random.seed(1337)  # noqa: NPY002
        count = len(self.pairs)
        n = int((1.0 - rate) * count)
        indices = np.arange(count)
        np.random.shuffle(indices)  # noqa: NPY002
        indices = indices[0:n]
        self.pairs = np.array(self.pairs)[indices].tolist()

    def get_raw(self, index):
        [x0, y0], [x1, y1] = self.pairs[index]
        image0 = load_rgb(x0)
        image1 = load_rgb(x1)

        if self.split == "train":
            image0, image1 = data_augment(image0, image1, self.use_rot)

        image0 = TF.to_tensor(image0)
        image1 = TF.to_tensor(image1)

        assert y0 != y1, "same label! at index {} and images: {} {}".format(index, x0, x1)

        label = bool(y0 > y1)

        if self.split == "train" and np.random.uniform() > 0.5:  # noqa: NPY002, PLR2004
            image0, image1, label = image1, image0, not label

        label = torch.tensor([label]).float()  # convert label to float
        if self.split == "train":
            label = add_noise(label, mean=0.0, std=0.2)
        inverse = 1.0 - label
        return image0, image1, label, inverse

    def __getitem__(self, index):
        return self.get_raw(index)

    def make_pairs(self):
        pairs = []
        for model in tqdm(self.models, "Making pairs"):
            for trip_id in range(self.n_triplets):
                triplet = []
                for i in range(3):
                    img_path = Path("{}.off_camera_{}.jpg".format(model, 3 * trip_id + i))
                    ano_path = Path("{}.off_camera_{}.json".format(model, 3 * trip_id + i))
                    assert img_path.exists(), "Missing image file " + img_path.as_posix()
                    assert ano_path.exists(), "Missing anotation file " + ano_path.as_posix()
                    triplet.append((img_path.as_posix(), ano_path.as_posix()))
                labels = [parse_label(f, self.use_failed, anno2int) for (_, f) in triplet]
                if self.filter_agree:
                    if len(labels[0]) != 2:  # noqa: PLR2004
                        continue
                    best_idx = 0 if labels[0][0] == 2.0 else 1 if labels[1][0] == 2.0 else 2  # noqa: PLR2004
                    if labels[best_idx][1] != 2.0:  # noqa: PLR2004
                        continue
                for participant in range(len(labels[0])):
                    img0 = triplet[0][0]
                    img1 = triplet[1][0]
                    img2 = triplet[2][0]
                    label0 = labels[0][participant]  # best
                    label1 = labels[1][participant]  # middle
                    label2 = labels[2][participant]  # worst
                    if self.use_triplets:
                        pairs.append([[img0, label0], [img1, label1]])  # (A=Best, B= Middle)
                        pairs.append([[img1, label1], [img2, label2]])  # (A=Middle, B=Worst)
                    pairs.append([[img0, label0], [img2, label2]])  # (A=Best, B=Worst)
        return pairs


class NeuralViewLabelDataset(torch.utils.data.Dataset):
    def __init__(self, path, classes=None) -> None:
        classes = classes or []
        super().__init__()
        self.classes = classes
        self.images = [
            img for img in Path(path).glob("**/*") if img.suffix == ".jpg" and any(c in img.name for c in classes)
        ]
        self.models = sorted({model.as_posix().split(".off")[0] for model in self.images})
        print("Labeling images using classes: {}".format(self.classes))
        print("Found {} images.".format(len(self.images)))
        self.pairs = self.make_pairs()

    def __len__(self):
        return len(self.pairs)

    def get_raw(self, index):
        [x0, x1] = self.pairs[index]
        image0 = load_rgb(x0)
        image1 = load_rgb(x1)

        image0 = TF.to_tensor(image0)
        image1 = TF.to_tensor(image1)

        return {"image0": image0, "image1": image1, "path0": x0.as_posix(), "path1": x1.as_posix()}

    def __getitem__(self, index):
        return self.get_raw(index)

    def make_pairs(self):
        pairs = []
        for c in self.classes:
            models = [m for m in self.models if c in m]
            images = [i for i in self.images if c in i.name]
            for model in tqdm(models, desc="Making combinations for class " + c):
                model_images = [img for img in images if Path(model).name in img.name]
                pairs += list(combinations(model_images, 2))
        return pairs


class PreferencesDataset(torch.utils.data.Dataset):
    """Dataset that pairs each rendered image with its human view-preference score.

    For every image in ``metric_path/{class}/{split}/``, the corresponding JSON
    file provides the camera's ``cartesian_coords``.  The nearest point on the
    shared Fibonacci sphere (loaded from ``prefs_path/fibonacci.npy``) is found
    by Euclidean distance, and the score at that index is read from the per-model
    ``.npy`` file in ``prefs_path/{class}/{split}/``.

    Images whose ``.npy`` counterpart does not exist are silently skipped.
    Scores are normalised to [0, 1] based on the total number of Fibonacci
    points (i.e. divided by ``len(fibonacci) - 1``).
    """

    def __init__(self, metric_path, prefs_path, split, classes=None):
        super().__init__()
        assert split in ["train", "test"], f"unknown split: {split}"

        metric_root = Path(metric_path)
        prefs_root = Path(prefs_path)

        self.fibonacci = np.load(prefs_root / "fibonacci.npy")  # (1000, 3)
        self.n_fib = len(self.fibonacci)

        # Cache loaded .npy score arrays keyed by their path to avoid redundant I/O
        self._score_cache: dict[Path, np.ndarray] = {}

        all_images = [
            img
            for img in metric_root.glob("**/*.jpg")
            if "mask" not in img.name and "depth" not in img.name and img.parent.name == split
        ]

        if classes is not None:
            all_images = [img for img in all_images if img.parents[1].name in classes]

        self.samples = []  # list of (img_path, json_path, npy_path)
        for img in sorted(all_images):
            cls = img.parents[1].name
            model_stem = img.stem.split(".off_camera_")[0]
            npy = prefs_root / cls / split / (model_stem + ".npy")
            if not npy.exists():
                continue
            json_path = img.with_suffix(".json")
            if not json_path.exists():
                continue
            self.samples.append((img, json_path, npy))

        print(f"PreferencesDataset [{split}]: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def _get_score(self, json_path: Path, npy_path: Path) -> float:
        with open(json_path) as f:
            coord = np.array(json.load(f)["cartesian_coords"])
        nearest_idx = int(np.argmin(np.linalg.norm(self.fibonacci - coord, axis=1)))

        if npy_path not in self._score_cache:
            self._score_cache[npy_path] = np.load(npy_path)
        scores = self._score_cache[npy_path]

        return float(scores[nearest_idx]) / (self.n_fib - 1)

    def __getitem__(self, index):
        img_path, json_path, npy_path = self.samples[index]
        image = TF.to_tensor(load_rgb(img_path))
        score = torch.tensor([self._get_score(json_path, npy_path)], dtype=torch.float32)
        return image, score


class HFPreferencesDataset(torch.utils.data.Dataset):
    """Viewpoint score dataset loaded automatically from a HuggingFace dataset repo."""

    def __init__(self, repo_id: str = HF_REPO_ID, split: str = "train", classes: list | None = None):
        super().__init__()
        assert split in ["train", "test"]
        ds = load_dataset(repo_id, split=split)
        if classes is not None:
            cat_feature = ds.features["category"]
            class_ids = {cat_feature.str2int(c) for c in classes}
            cat_col = ds["category"]
            ds = ds.select([i for i, c in enumerate(cat_col) if c in class_ids])
        # Keep only rows that have a preference score (.npy was present for this split)
        score_col = ds["viewpoint_score"]
        ds = ds.select([i for i, s in enumerate(score_col) if s is not None])
        self.ds = ds
        print(f"HFPreferencesDataset [{split}]: {len(self.ds)} images")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        image = TF.to_tensor(row["image"].convert("RGB"))
        score = torch.tensor([row["viewpoint_score"]], dtype=torch.float32)
        return image, score


class PreferencesDataModule(L.LightningDataModule):
    def __init__(
        self,
        metric_path: str | None = None,
        prefs_path: str | None = None,
        repo_id: str = HF_REPO_ID,
        batch_size: int = 16,
        num_workers: int = 8,
        classes: list | None = None,
    ):
        super().__init__()
        self.metric_path = metric_path
        self.prefs_path = prefs_path
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = classes
        self._use_hf = metric_path is None or prefs_path is None

    def _make_dataset(self, split: str):
        if self._use_hf:
            return HFPreferencesDataset(repo_id=self.repo_id, split=split, classes=self.classes)
        return PreferencesDataset(
            metric_path=self.metric_path,
            prefs_path=self.prefs_path,
            split=split,
            classes=self.classes,
        )

    def train_dataloader(self):
        dataset = self._make_dataset("train")
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        dataset = self._make_dataset("test")
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)


class HFViewDataset(torch.utils.data.Dataset):
    """Ranking-pair dataset loaded automatically from a HuggingFace dataset repo."""

    def __init__(
        self,
        repo_id: str = HF_REPO_ID,
        split: str = "train",
        use_failed: bool = False,
        classes: list | None = None,
        filter_agree: bool = False,
        discard_rate: float = 0.0,
        use_rot: bool = False,
        use_triplets: bool = True,
        n_instances: int = -1,
    ):
        super().__init__()
        assert split in ["train", "valid", "test"]
        self.split = split
        self.use_rot = use_rot
        self.use_triplets = use_triplets

        ds = load_dataset(repo_id, split=split)
        if classes is not None:
            cat_feature = ds.features["category"]
            class_ids = {cat_feature.str2int(c) for c in classes}
            cat_col = ds["category"]
            ds = ds.select([i for i, c in enumerate(cat_col) if c in class_ids])

        view_info, model_to_cat = self._build_index(ds, use_failed)
        all_models = self._filter_models(sorted(model_to_cat), model_to_cat, n_instances)
        n_triplets = (max((cam for (_, cam) in view_info), default=0) // 3) + 1 if view_info else 0
        self.pairs = self._make_pairs(all_models, view_info, n_triplets, filter_agree, use_triplets)

        if discard_rate > 0:
            np.random.seed(1337)  # noqa: NPY002
            n = int((1.0 - discard_rate) * len(self.pairs))
            indices = np.arange(len(self.pairs))
            np.random.shuffle(indices)  # noqa: NPY002
            self.pairs = [self.pairs[i] for i in indices[:n]]

        self.ds = ds
        print(f"Found {n_triplets} triplets per model")
        print(f"Found {len(self.pairs)} pairs for split {split}.")
        print(f"Found {len(all_models)} models for split {split}.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_index(ds, use_failed: bool) -> tuple[dict, dict]:
        """Return (view_info, model_to_cat) by fetching metadata columns in bulk."""
        cat_feature = ds.features["category"]
        view_info: dict = {}  # (model_id, camera_idx) -> (row_idx, labels)
        model_to_cat: dict = {}
        for i, (mid, cam, lbls, fails, cat_id) in enumerate(
            zip(
                ds["model_id"],
                ds["camera_idx"],
                ds["annotation_labels"],
                ds["annotation_failed"],
                ds["category"],
                strict=False,
            )
        ):
            view_info[(mid, cam)] = (i, _hf_label(lbls, fails, use_failed))
            if mid not in model_to_cat:
                model_to_cat[mid] = cat_feature.int2str(cat_id)
        return view_info, model_to_cat

    @staticmethod
    def _filter_models(all_models: list, model_to_cat: dict, n_instances: int) -> list:
        if n_instances < 1:
            return all_models
        by_class: dict = {}
        for mid in all_models:
            by_class.setdefault(model_to_cat[mid], []).append(mid)
        result = []
        for cat_name in sorted(by_class):
            result += sorted(by_class[cat_name])[:n_instances]
        return sorted(result)

    @staticmethod
    def _make_pairs(all_models, view_info, n_triplets, filter_agree, use_triplets) -> list:
        pairs = []
        for model in tqdm(all_models, "Making pairs"):
            for trip_id in range(n_triplets):
                keys = [(model, 3 * trip_id + i) for i in range(3)]
                if any(k not in view_info for k in keys):
                    continue
                triplet = [view_info[k] for k in keys]
                labels = [t[1] for t in triplet]
                if filter_agree:
                    if len(labels[0]) != 2:  # noqa: PLR2004
                        continue
                    best_idx = 0 if labels[0][0] == 2.0 else 1 if labels[1][0] == 2.0 else 2  # noqa: PLR2004
                    if labels[best_idx][1] != 2.0:  # noqa: PLR2004
                        continue
                n_participants = min(len(lab) for lab in labels)
                for participant in range(n_participants):
                    idx0, idx1, idx2 = [triplet[j][0] for j in range(3)]
                    l0, l1, l2 = [labels[j][participant] for j in range(3)]
                    if use_triplets:
                        pairs.append([[idx0, l0], [idx1, l1]])
                        pairs.append([[idx1, l1], [idx2, l2]])
                    pairs.append([[idx0, l0], [idx2, l2]])
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        [x0, y0], [x1, y1] = self.pairs[index]
        image0 = self.ds[x0]["image"].convert("RGB")
        image1 = self.ds[x1]["image"].convert("RGB")

        if self.split == "train":
            image0, image1 = data_augment(image0, image1, self.use_rot)

        image0 = TF.to_tensor(image0)
        image1 = TF.to_tensor(image1)

        assert y0 != y1, f"same label! at index {index}"

        label = bool(y0 > y1)
        if self.split == "train" and np.random.uniform() > 0.5:  # noqa: NPY002, PLR2004
            image0, image1, label = image1, image0, not label

        label = torch.tensor([label]).float()
        if self.split == "train":
            label = add_noise(label, mean=0.0, std=0.2)
        inverse = 1.0 - label
        return image0, image1, label, inverse


class ViewDataModule(L.LightningDataModule):
    def __init__(  # noqa: PLR0913
        self,
        dataset_path: str | None = None,
        valid_path: str = "",
        repo_id: str = HF_REPO_ID,
        batch_size: int = 16,
        num_workers: int = 8,
        classes: list | None = None,
        use_rot: bool = True,
        filter_agree: bool = False,
        discard_rate: float = 0.0,
        use_triplets: bool = True,
        n_models: int = -1,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.valid_path = valid_path or dataset_path
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = classes or CLASSES
        self.use_rot = use_rot
        self.filter_agree = filter_agree
        self.discard_rate = discard_rate
        self.use_triplets = use_triplets
        self.n_models = n_models
        self._use_hf = dataset_path is None

    def train_dataloader(self):
        if self._use_hf:
            dataset = HFViewDataset(
                repo_id=self.repo_id,
                split="train",
                use_rot=self.use_rot,
                use_failed=False,
                filter_agree=self.filter_agree,
                classes=self.classes,
                discard_rate=self.discard_rate,
                use_triplets=self.use_triplets,
                n_instances=self.n_models,
            )
        else:
            dataset = ViewDataset(
                path=self.dataset_path,
                split="train",
                use_rot=self.use_rot,
                use_failed=False,
                filter_agree=self.filter_agree,
                classes=self.classes,
                discard_rate=self.discard_rate,
                use_triplets=self.use_triplets,
                n_instances=self.n_models,
            )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        if self._use_hf:
            dataset = HFViewDataset(
                repo_id=self.repo_id,
                split="valid",
                use_rot=False,
                use_failed=False,
                filter_agree=self.filter_agree,
                classes=self.classes,
            )
        else:
            dataset = ViewDataset(
                path=self.valid_path,
                split="valid",
                use_rot=False,
                use_failed=False,
                filter_agree=self.filter_agree,
                classes=self.classes,
            )
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    dataset = ViewDataset(
        "G:/projects/3DAnnotation/additional_study",
        "train",
        classes=["airplane", "chair", "flower_pot", "sofa", "toilet"],
    )
