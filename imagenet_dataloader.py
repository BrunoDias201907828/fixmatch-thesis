import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import random

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

weak_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

strong_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class AugmentedLabeledDataset(Dataset):
    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        data, target = self._dataset[idx]
        return self._transform(data), target

class AugmentedUnlabeledDataset(Dataset):
    def __init__(self, dataset, weak_transform, strong_transform):
        self._dataset = dataset
        self._weak_transform = weak_transform
        self._strong_transform = strong_transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        data, _ = self._dataset[idx]
        return self._weak_transform(data), self._strong_transform(data)

class ValDataset(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        data, target = self._dataset[idx]
        return data, target

def split_labeled_training(dataset, n_labeled=500, n_classes=1000):
    labels = np.array(dataset.targets)
    labeled_idxs = []
    unlabeled_idxs = []

    for i in range(n_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)

        labeled_idxs.extend(idxs[:n_labeled//n_classes])
        unlabeled_idxs.extend(idxs[n_labeled//n_classes:])

    return labeled_idxs, unlabeled_idxs

def imagenet_get_dataset(root='/data/toys/ILSVRC2012', n_labeled=13000, n_classes=1000):
    """
    Create and return labeled and unlabeled datasets for ImageNet.
    """

    train_dataset = torchvision.datasets.ImageNet(
        root=root,
        split='train'
    )

    val_dataset = torchvision.datasets.ImageNet(
        root=root,
        split='val',
        transform=val_transform
    )

    labeled_idxs, unlabeled_idxs = split_labeled_training(train_dataset, n_labeled, n_classes)

    labeled_train_dataset = AugmentedLabeledDataset(
        Subset(train_dataset, labeled_idxs), weak_transform)
    unlabeled_train_dataset = AugmentedUnlabeledDataset(
        Subset(train_dataset, unlabeled_idxs), weak_transform, strong_transform)
    val_dataset = ValDataset(val_dataset)

    return labeled_train_dataset, unlabeled_train_dataset, val_dataset


if __name__ == '__main__':

    label, unlabel, val = imagenet_get_dataset()

    from IPython import embed; embed()
