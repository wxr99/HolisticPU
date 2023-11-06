import logging
import math
import random
import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset
from torchvision import datasets
from torchvision import transforms
from .randaugment import RandAugmentMC
from .randaugment import RandAugmentSim

logger = logging.getLogger(__name__)

fashionmnist_mean = (0.1307,)
fashionmnist_std = (0.3081,)


def p_u_split(args, labels):
    print('available classes:', args.available_label_list, 'positive classes:', args.positive_label_list, 'number of class:', args.num_classes)
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    positive_idx = []
    # unlabeled data:
    unlabeled_idx = np.array(range(len(labels)))
    for i in args.available_label_list:
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        positive_idx.extend(idx)
    positive_idx = np.array(positive_idx)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, positive_idx)
    assert len(positive_idx) == args.num_labeled
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        positive_idx = np.hstack([positive_idx for _ in range(num_expand_x)])
    np.random.shuffle(positive_idx)
    return positive_idx, unlabeled_idx


def get_fmnist(args, root):
    transform_normal = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=4,
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=fashionmnist_mean, std=fashionmnist_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=fashionmnist_mean, std=fashionmnist_std)
    ])
    base_dataset = datasets.FashionMNIST(root, train=True, download=True)
    train_positive_idxs, train_unlabeled_idxs = p_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = FashionMnistSSL(
        root, train_positive_idxs, train=True,
        transform=TransformFixMatch(mean=fashionmnist_mean, std=fashionmnist_std),
        target_transform=TransformPTarget(available_label_list=args.available_label_list))

    train_unlabeled_dataset = FashionMnistSSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=fashionmnist_mean, std=fashionmnist_std),
        target_transform=TransformNTarget(available_label_list=args.available_label_list)
    )

    test_dataset = datasets.FashionMNIST(
        root, train=False, transform=transform_val, download=False,
        target_transform=TransformTestTarget(positive_label_list=args.positive_label_list))

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=4,
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=4,
                                  padding_mode='reflect'),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.ColorJitter()
            ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformPTarget(object):
    def __init__(self, available_label_list):
        self.p_transform = lambda x: 0 if x in available_label_list else 1

    def __call__(self, x):
        x = self.p_transform(x)
        return x


class TransformNTarget(object):
    def __init__(self, available_label_list):
        self.u_transform = lambda x: 1
        self.transform = lambda x: 0 if x in available_label_list else 1

    def __call__(self, x):
        x_unlabeled = self.u_transform(x)
        x_true = self.transform(x)
        return x_unlabeled, x_true


class TransformTestTarget(object):
    def __init__(self, positive_label_list):
        self.p_transform = lambda x: 0 if x in positive_label_list else 1

    def __call__(self, x):
        x = self.p_transform(x)
        return x


class FashionMnistSSL(datasets.FashionMNIST):
    def   __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.array(img)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


