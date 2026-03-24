import lightning as L
import numpy as np
import torch
import torchvision.transforms.functional as TF
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


def add_noise(label, mean=0.0, std=0.2):
    """Add Gaussian noise to a label, reverting if it changes the class."""
    label_noise = label + torch.normal(mean=mean, std=std, size=(1, 1))
    label_noise = label_noise.clamp(0, 1).reshape(1)
    if label_noise.round() == label:
        return label_noise
    return label  # if the noise changed the label return original label


def _hf_label(annotation_labels: list, annotation_failed: list, use_failed: bool) -> list:
    """Convert HF dataset annotation columns to per-annotator numeric labels."""
    _map = {"best": 2.0, "second": 1.0, "worst": 0.0}
    labels = [
        _map[lbl] for lbl, failed in zip(annotation_labels, annotation_failed, strict=False) if not failed or use_failed
    ]
    if not labels:  # fallback: include failed annotations
        labels = [_map[lbl] for lbl in annotation_labels]
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


class PreferencesDataset(torch.utils.data.Dataset):
    """Viewpoint score dataset loaded automatically from a HuggingFace dataset repo."""

    def __init__(self, repo_id: str, split: str = "train", classes: list | None = None):
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
        print(f"PreferencesDataset [{split}]: {len(self.ds)} images")

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
        repo_id: str,
        batch_size: int,
        num_workers: int,
        classes: list | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = classes

    def train_dataloader(self):
        dataset = PreferencesDataset(repo_id=self.repo_id, split="train", classes=self.classes)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        dataset = PreferencesDataset(repo_id=self.repo_id, split="test", classes=self.classes)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)


class ViewDataset(torch.utils.data.Dataset):
    """Ranking-pair dataset loaded automatically from a HuggingFace dataset repo."""

    def __init__(
        self,
        repo_id: str,
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
    def __init__(
        self,
        repo_id: str,
        batch_size: int,
        num_workers: int,
        classes: list | None = None,
        use_rot: bool = True,
        filter_agree: bool = False,
        discard_rate: float = 0.0,
        use_triplets: bool = True,
        n_models: int = -1,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = classes or CLASSES
        self.use_rot = use_rot
        self.filter_agree = filter_agree
        self.discard_rate = discard_rate
        self.use_triplets = use_triplets
        self.n_models = n_models

    def train_dataloader(self):
        dataset = ViewDataset(
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
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        dataset = ViewDataset(
            repo_id=self.repo_id,
            split="valid",
            use_rot=False,
            use_failed=False,
            filter_agree=self.filter_agree,
            classes=self.classes,
        )
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
