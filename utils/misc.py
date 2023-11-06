'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import LambdaLR
import math

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    prob = torch.clamp(F.softmax(output.detach(), dim=1), 1e-10, 1-1e-10)
    prob = np.array(prob.cpu().detach())
    predicted = np.array(predicted.cpu().detach())
    target = np.array(target.cpu().detach())
    acc = accuracy_score(target, predicted)
    auc = roc_auc_score(target, prob[:, 1])
    # auc = 0
    f1 = f1_score(target, predicted)
    # prec = precision_score(target, predicted)
    # recall = recall_score(target, predicted)
    return acc, auc, f1

def accuracy1(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    prob = torch.clamp(F.softmax(output.detach(), dim=1), 1e-10, 1-1e-10)
    prob = np.array(prob.cpu().detach())
    predicted = np.array(predicted.cpu().detach())
    target = np.array(target.cpu().detach())
    acc = accuracy_score(target, predicted)
    auc = roc_auc_score(target, prob[:, 1])
    # auc = 0
    f1 = f1_score(target, predicted)
    prec = precision_score(target, predicted)
    recall = recall_score(target, predicted)
    return acc, auc, f1, prec, recall


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def three_sigma(x):

    # idx = np.where(x < 0.2/14)
    idx = np.where(x < 0.2 / 9)
    return x[idx]


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


