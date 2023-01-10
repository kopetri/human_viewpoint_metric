import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json 
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
from itertools import combinations

CLASSES = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'car', 'chair', 'desk', 'door', 'dresser', 'flower_pot', 'guitar', 'keyboard', 'lamp', 'laptop', 'monitor', 'night_stand', 'piano', 'plant', 'radio', 'sink', 'sofa', 'table', 'tent', 'toilet', 'tv_stand', 'vase']

def add_noise(label, mean=0.0, std=0.2):
    label_noise = label + torch.normal(mean=mean, std=std, size=(1,1)) # add noise
    label_noise = label_noise.clamp(0, 1).reshape(1)
    if label_noise.round() == label:
        return label_noise
    else:
        return label # if the noise changed the label return original label

def anno2int(anno):
    label = anno['label']
    if label == 'best':
        return 2.0
    elif label == 'second':
        return 1.0
    elif label == 'worst':
        return 0.0
    raise ValueError("Invalid label", label)

def load_rgb(img_path):
    image = Image.open(img_path).convert('RGB')
    return image

def convert2label(data, convert_func, use_failed):
    return [convert_func(anno) for anno in data if ((not anno["failed"]) or use_failed)]

def parse_label(anno_file, use_failed, convert_labels):
    with open(anno_file, "r") as json_file:
        data = json.load(json_file)
        if isinstance(data, dict):
            data = data["annotations"]
        labels = convert2label(data, convert_labels, use_failed)
        if len(labels) == 0:
            print("WARNING: Missing labels. using failed ones aswell! {}".format(anno_file))
            labels = convert2label(data, convert_labels, True)
        assert len(labels) > 0, "No labels!!"
        return labels

def data_augment(imgA, imgB, use_rot):
    if np.random.uniform() > 0.5:
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
        imgA = color_jitter(imgA)
        imgB = color_jitter(imgB)
    if np.random.uniform() > 0.5 and use_rot:
        rand_rot = transforms.RandomRotation(90)
        imgA = rand_rot(imgA)
        imgB = rand_rot(imgB)
    return imgA, imgB


class ViewDataset(torch.utils.data.Dataset):
    def __init__(self, path, split, use_failed=False, classes=None, filter_agree=False, discard_rate=0.0, use_rot=False, use_triplets=True, n_instances=-1):
        super(ViewDataset, self).__init__()
        assert split in ["train", "valid", "test", "all"], "unknown data set split: {}".format(split)
        self.filter_agree = filter_agree
        self.use_failed = use_failed
        self.split = split
        self.use_triplets = use_triplets
        self.use_rot = use_rot
        self.images = [img for img in Path(path).glob("**/*") if img.suffix == ".jpg" and not "mask" in img.name and (img.parent.name == split or split == 'all')]
        self.models = sorted(list(set([model.as_posix().split(".off")[0] for model in self.images])))
        self.models = [model for model in self.models if classes is None or any([c in model for c in classes])]
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
        if n < 1: return self.models, self.images
        classes = list(set([Path(m).parents[1].name for m in self.models]))
        classes.sort()
        models = []
        for cls in classes:
            models += [m for m in self.models if Path(m).parents[1].name == cls][0:n]
        m_names = [Path(m).stem for m in models]
        images = [i for i in self.images if i.stem.split(".off")[0] in m_names]
        return models, images

    def discard(self, rate):
        np.random.seed(1337)
        N = len(self.pairs)
        n = int((1.0 - rate) * N)
        indices = np.arange(N)
        np.random.shuffle(indices)
        indices = indices[0:n]
        self.pairs = np.array(self.pairs)[indices].tolist()

    def get_raw(self, index):
        [x0, y0], [x1, y1] = self.pairs[index]
        image0 = load_rgb(x0)
        image1 = load_rgb(x1)

        if self.split == 'train':
            image0, image1 = data_augment(image0, image1, self.use_rot)

        image0 = TF.to_tensor(image0)
        image1 = TF.to_tensor(image1)

        assert not y0 == y1, "same label! at index {} and images: {} {}".format(index, x0, x1)

        label = bool(y0 > y1)

        if self.split == 'train' and np.random.uniform() > 0.5:
            image0, image1, label = image1, image0, not label

        label = torch.tensor([label]).float() # convert label to float
        if self.split == 'train':
            label = add_noise(label, mean=0.0, std=0.2)
        inverse = 1.0 - label
        return image0, image1, label, inverse

    def __getitem__(self, index):
        return self.get_raw(index)

    def make_pairs(self):
        pairs = []
        for model in tqdm(self.models, "Making pairs"):
            for tId in range(self.n_triplets):
                triplet = []
                for i in range(3):
                    img_path = Path("{}.off_camera_{}.jpg".format(model, 3 * tId + i))
                    ano_path = Path("{}.off_camera_{}.json".format(model, 3 * tId + i))
                    assert img_path.exists(), "Missing image file " + img_path.as_posix()
                    assert ano_path.exists(), "Missing anotation file " + ano_path.as_posix()
                    triplet.append((img_path.as_posix(), ano_path.as_posix()))
                labels = [parse_label(f, self.use_failed, anno2int) for (_, f) in triplet]
                if self.filter_agree:
                    if (not len(labels[0]) == 2):
                        continue
                    best_idx = 0 if labels[0][0] == 2.0 else 1 if labels[1][0] == 2.0 else 2
                    if not labels[best_idx][1] == 2.0:
                        continue
                for participant in range(len(labels[0])):
                    img0 = triplet[0][0]
                    img1 = triplet[1][0]
                    img2 = triplet[2][0]
                    label0 = labels[0][participant] # best
                    label1 = labels[1][participant] # middle
                    label2 = labels[2][participant] # worst
                    if self.use_triplets:
                        pairs.append([[img0, label0], [img1, label1]]) # (A=Best, B= Middle)                    
                        pairs.append([[img1, label1], [img2, label2]]) # (A=Middle, B=Worst)
                    pairs.append([[img0, label0], [img2, label2]]) # (A=Best, B=Worst)
        return pairs

class NeuralViewLabelDataset(torch.utils.data.Dataset):
    def __init__(self, path, classes=[]) -> None:
        super().__init__()
        self.classes = classes
        self.images = [img for img in Path(path).glob("**/*") if img.suffix == ".jpg" and any([c in img.name for c in classes])]
        self.models = sorted(list(set([model.as_posix().split(".off")[0] for model in self.images])))        
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

        return {'image0': image0, 'image1': image1, 'path0': x0.as_posix(), 'path1': x1.as_posix()}


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


if __name__ == '__main__':
    dataset = ViewDataset("G:/projects/3DAnnotation/additional_study", "train", classes=["airplane", "chair", "flower_pot", "sofa", "toilet"])