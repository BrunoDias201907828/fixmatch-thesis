import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np

mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)

weak_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

strong_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandAugment(2, 10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


strong_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.RandAugment(num_ops=2, magnitude=10),
    # transforms.Cutout(num_holes=1, size=16),
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

class TestDataset(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        data, target = self._dataset[idx]
        return data, target


def split_labeled_training(dataset, n_labeled=500, n_classes=10):
    """
    Split the labeled training dataset into labeled and unlabeled datasets
    Input:    
    - param dataset: the training dataset
    - param n_labeled: the number of labeled samples
    - param n_classes: the number of classes
    Return:
    - labeled_idxs, unlabeled_idxs
    """

    labels = np.array(dataset.targets)
    labeled_idxs = []
    unlabeled_idxs = []

    for i in range(n_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)

        labeled_idxs.extend(idxs[:n_labeled//n_classes])
        unlabeled_idxs.extend(idxs[n_labeled//n_classes:])

    return labeled_idxs, unlabeled_idxs

def get_dataset(root='./data', n_labeled=500, n_classes=10):
    """
        Create and return labeled and unlabeled datasets.
    """

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )

    labeled_idxs, unlabeled_idxs = split_labeled_training(train_dataset, n_labeled, n_classes)

    labeled_train_dataset = AugmentedLabeledDataset(
        Subset(train_dataset, labeled_idxs), weak_transform)
    unlabeled_train_dataset = AugmentedUnlabeledDataset(
        Subset(train_dataset, unlabeled_idxs), weak_transform, strong_transform)
    test_dataset = TestDataset(test_dataset)

    # print(f"Labeled dataset size: {len(labeled_train_dataset)}")
    # print(f"Unlabeled dataset size: {len(unlabeled_train_dataset)}")

    return labeled_train_dataset, unlabeled_train_dataset, test_dataset

# from IPython import embed; embed()
