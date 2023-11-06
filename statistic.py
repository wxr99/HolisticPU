import argparse
import logging
import math
import os
import shutil
import time
from collections import OrderedDict
from model.resnet_zoo import resnet18, resnet34, resnet50
from model.ResNet_Zoo import ResNet18
from model.alexnet import alexnet
from model.cnn7 import cnn_cifar, cnn_stl
from model.lenet5 import lenet_fmnist

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import get_cifar10
from dataset.stl10 import get_stl10
from dataset.fahionmnist import get_fmnist
from utils.misc import AverageMeter, accuracy, set_seed, get_cosine_schedule_with_warmup, interleave, de_interleave
import pandas as pd


def test(args, logger, best_acc, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    aucs = AverageMeter()
    f1s = AverageMeter()
    precs = AverageMeter()
    recalls = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            # inputs = inputs.resize()
            outputs = model(inputs)
            # print(outputs)
            loss = F.cross_entropy(outputs, targets)

            acc, auc, f1 = accuracy(outputs, targets)

            losses.update(loss.item(), inputs.shape[0])
            accs.update(acc.item(), inputs.shape[0])
            aucs.update(auc.item(), inputs.shape[0])
            f1s.update(f1.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. acc: {acc:.2f}. auc: {auc:.2f}. f1_score: {f1:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    acc=accs.avg,
                    auc=aucs.avg,
                    f1=f1s.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("acc: {:.6f}".format(accs.avg))
    logger.info("auc: {:.6f}".format(aucs.avg))
    logger.info("f1: {:.6f}".format(f1s.avg))
    return losses.avg, accs.avg, aucs.avg, f1s.avg, logger, best_acc


def validation(args, logger, valid_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    end = time.time()
    if not args.no_progress:
        valid_loader = tqdm(valid_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, ((inputs, inputs_s), targets) in enumerate(valid_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            lam = np.random.beta(args.alpha, args.alpha)
            index = torch.randperm(inputs.size(0)).cuda()
            inputs = lam * inputs + (1 - lam) * inputs[index, :]

            targets = targets.to(args.device)
            # inputs = inputs.resize()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            predicted = np.array(predicted.cpu().detach())
            targets = np.array(targets.cpu().detach())
            acc = accuracy_score(targets, predicted)
            losses.update(loss.item(), inputs.shape[0])
            accs.update(acc.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                valid_loader.set_description("Valid Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. acc: {acc:.2f}.".format(
                    batch=batch_idx + 1,
                    iter=len(valid_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    acc=accs.avg,
                ))
        if not args.no_progress:
            valid_loader.close()

    logger.info("acc: {:.6f}".format(accs.avg))
    logger.info("loss: {:.6f}".format(losses.avg))
    return losses.avg, accs.avg, logger


def record(args, logger, valid_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    if not args.no_progress:
        valid_loader = tqdm(valid_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        preds = np.array(args.batch_size * 2)
        for batch_idx, ((inputs, inputs_s), (target_u, target)) in enumerate(valid_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            target = target.to(args.device)
            outputs = model(inputs)
            softmax = torch.nn.Softmax(dim=1)
            pred = softmax(outputs)
            pred = np.array((pred[:, 0]).cpu())
            target = target.cpu()
            if batch_idx == 0:
                preds = pred
                targets = target
            else:
                preds = np.concatenate([preds, pred], axis=0)
                targets = np.concatenate([targets, target], axis=0)
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                valid_loader.set_description("Record Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. ".format(
                    batch=batch_idx + 1,
                    iter=len(valid_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            valid_loader.close()
    return logger, preds, targets




