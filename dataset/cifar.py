import logging
import math
import random
import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset
from torchvision import datasets
from torchvision import transforms
from .randaugment import RandAugmentMC



logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def make_lt_imb(gamma, class_num, max_num):
    mu = np.power(1/gamma, 1/(class_num-1))
    # setting1:
    class_list = [0, 1, 8, 9, 2, 3, 4, 5, 6, 7]
    # setting2:
    # class_list = [2, 3, 4, 5, 6, 7, 0, 1, 8, 9]
    class_num_list = np.ones(class_num)
    for i in range(class_num):
        class_num_list[class_list[i]] = (int(max_num * np.power(mu, i)))
    print(class_num_list)
    return list(class_num_list)


def get_cifar10(args, root):
    transform_normal = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    if args.imbalance:
        train_positive_idxs, train_unlabeled_idxs = lt_p_u_split(
            args, base_dataset.targets)
    else:
        train_positive_idxs, train_unlabeled_idxs = p_u_split(
            args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_positive_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
        target_transform=TransformPTarget(available_label_list=args.available_label_list))

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
        target_transform=TransformNTarget(available_label_list=args.available_label_list)
        )

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False,
        target_transform=TransformTestTarget(positive_label_list=args.positive_label_list))

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


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


def lt_p_u_split(args, labels):
    print('available classes:', args.available_label_list, 'positive classes:', args.positive_label_list, 'number of class:', args.num_classes)
    label_per_class = args.num_labeled // args.num_classes
    num_per_class = make_lt_imb(gamma=1000, class_num=10, max_num=4000)
    labels = np.array(labels)
    positive_idx = []
    lt_unlabeled_idx = []
    # unlabeled data:
    for i in args.available_label_list:
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        positive_idx.extend(idx)
    positive_idx = np.array(positive_idx)
    for i in range(10):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, int(num_per_class[i]), False)
        lt_unlabeled_idx.extend(idx)
    lt_unlabeled_idx = np.array(lt_unlabeled_idx)
    assert len(positive_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        positive_idx = np.hstack([positive_idx for _ in range(num_expand_x)])
    np.random.shuffle(positive_idx)

    return positive_idx, lt_unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
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


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
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
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
