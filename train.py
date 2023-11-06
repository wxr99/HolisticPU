import argparse
import logging
import math
import os
import shutil
import time
from collections import OrderedDict
import jenkspy
import pandas as pd

from model.resnet_zoo import resnet18, resnet34, resnet50
from model.ResNet_Zoo import ResNet18
from model.alexnet import alexnet
from model.cnn7 import cnn_cifar, cnn_stl
from model.lenet5 import lenet_fmnist
from model.loss import loss_coteaching, loss_ft
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
from utils.misc import AverageMeter, three_sigma, get_cosine_schedule_with_warmup, interleave, de_interleave
from statistic import test, record, validation


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def train(args, logger, best_acc, labeled_trainloader, unlabeled_trainloader, test_loader, pseudo_targets,
          model1, optimizer1, scheduler1, model2, optimizer2, scheduler2, ema_model):
    if args.amp:
        from apex import amp
    end = time.time()
    test_accs = []
    unlabeled_num = len(pseudo_targets)
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    global preds_sequence
    model1.train()
    unlabeled_idx = 0
    for epoch in range(args.start_epoch, args.ft_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                (inputs_x_w, inputs_x_s), targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (inputs_x_w, inputs_x_s), targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), (targets_u, targets_t) = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), (targets_u, targets_t) = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x_w.shape[0]

            # lam = np.random.beta(args.alpha, args.alpha)
            # index = torch.randperm(batch_size).cuda()
            # inputs_mix = lam * inputs_u_w + (1-lam) * inputs_u_w[index]

            inputs = interleave(
                torch.cat((inputs_u_w, inputs_u_s, inputs_x_w, inputs_x_s)), 4*args.mu).to(args.device)

            targets_x = targets_x.to(args.device)
            targets_u = targets_u.to(args.device)
            targets_t = targets_t.to(args.device)
            targets_p = pseudo_targets[unlabeled_idx: unlabeled_idx+args.mu * batch_size]
            targets_p = torch.tensor(targets_p)
            targets_p = targets_p.to(torch.long).to(args.device)
            unlabeled_idx = (unlabeled_idx + args.mu * batch_size) % unlabeled_num

            logits1 = model1(inputs)
            logits1 = de_interleave(logits1, 4 * args.mu)
            logits1_u, logits1_u_s = logits1[: 2 * args.mu * batch_size].chunk(2)
            logits1_x_w, logits1_x_s = logits1[2 * args.mu * batch_size:].chunk(2)

            del logits1
            Lx1 = F.cross_entropy(logits1_x_w, targets_x, reduction='mean')
            # print(torch.softmax(logits1_x_w.detach() / args.T, dim=-1), targets_x)
            # Lu1 = loss_ft(args, logits1_u, logits1_u_s, targets_u, targets_p, epoch=epoch)
            Lu1 = F.cross_entropy(logits1_u, targets_t, reduction='mean')
            '''
             # loss function:
            '''
            loss1 = Lx1 + Lu1
            # loss1 = Lu1
            # loss2 = Lx2 + Lu2
            loss = loss1

            losses.update(loss.item())
            losses_x.update(Lx1.item())
            losses_u.update(Lu1.item())

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            scheduler1.step()
            # print(torch.softmax(logits1_u.detach() / args.T, dim=-1), targets_u, targets_t)
            # optimizer2.zero_grad()
            # loss2.backward()
            # optimizer2.step()
            # scheduler2.step()

            if args.use_ema:
                ema_model.update(model1)
            model1.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.ft_epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler1.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,

                    ))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            # test_model = ema_model.ema
            test_model = model1
        else:
            test_model = model1

        # Validation & Test
        if args.local_rank in [-1, 0]:
            # stat_loss, stat_acc, pred_u_p, pred_u_n, var_p, var_n, logger = statistic(args, logger, unlabeled_trainloader, test_model, epoch)
            # valid_loss, valid_acc, logger = validation(args, logger, labeled_trainloader, test_model, epoch)
            test_loss, test_acc, test_auc, test_f1, logger, best_acc = test(args, logger, best_acc, test_loader,
                                                                                test_model)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)


            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_auc', test_auc, epoch)
            args.writer.add_scalar('test/3.test_f1', test_f1, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model1.module if hasattr(model1, "module") else model1
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema


            test_accs.append(test_acc)
            logger.info('Best acc: {:.6f}'.format(best_acc))
            logger.info('Mean acc: {:.6f}\n'.format(
                np.mean(test_accs[-20:])))

        if args.local_rank in [-1, 0]:
            args.writer.close()


def train_phase1(args, logger, labeled_trainloader, unlabeled_trainloader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    global preds_sequence
    model.train()
    for epoch in range(args.start_epoch, args.warming_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_n = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                (inputs_x_w, inputs_x_s), targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (inputs_x_w, inputs_x_s), targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), (targets_u, targets_t) = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), (targets_u, targets_t) = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x_w.shape[0]
            inputs = interleave(
                torch.cat((inputs_u_w, inputs_u_s, inputs_x_w, inputs_x_s)), 4*args.mu).to(args.device)
            targets_x = targets_x.to(args.device)
            targets_u = targets_u.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 4*args.mu)
            logits_u, logits_u_w = logits[:2*args.mu*batch_size].chunk(2)
            logits_x_w, logits_x_s = logits[2*args.mu*batch_size:].chunk(2)
            del logits

            Lx = (F.cross_entropy(logits_x_w, targets_x, reduction='mean', label_smoothing=args.rho) +
                  F.cross_entropy(logits_x_s, targets_x, reduction='mean', label_smoothing=args.rho)) / 2
            # Lx = F.cross_entropy(logits_x_w, targets_x, reduction='mean')
            '''
             # Negative target by moving average 
            '''
            # label_one_hot = F.one_hot(targets_u, 2).float().to(args.device)
            # label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
            # targets_u = label_one_hot
            # Negative by CE
            Ln = F.cross_entropy(logits_u, targets_u, reduction='mean')
            # print(torch.softmax(logits_u.detach() / args.T, dim=-1), targets_u, targets_t)
            '''
             # loss function:
            '''
            # print('1\n', targets_x, '\n2\n', targets_u)
            # loss = Lx + Ln
            loss = Lx + Ln
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_n.update(Ln.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_n: {loss_n:.4f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.warming_epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_n=losses_n.avg,
                    ))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        # Validation & Test
        logger, preds, targets = record(args, logger, unlabeled_trainloader, test_model, epoch)
        if epoch == 0:
            preds_sequence = preds
        else:
            preds_sequence = np.vstack((preds_sequence, preds))
        args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
        args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
        args.writer.add_scalar('train/3.train_loss_n', losses_n.avg, epoch)
        if args.local_rank in [-1, 0]:
            args.writer.close()
    preds_sequence = preds_sequence.T
    trends = np.zeros(len(preds_sequence))
    for i, sequence in enumerate(preds_sequence):
        sequence = pd.Series(sequence)
        diff_1 = sequence.diff(periods=1)
        diff_1 = np.array(diff_1)
        diff_1 = diff_1[1:]
        diff_1 = np.log(1+diff_1+0.5*diff_1**2)
        trends[i] = diff_1.mean()
    intervals = jenkspy.jenks_breaks(trends, n_classes=2)
    break_point = intervals[1]
    if break_point > 0:
        trends_std = three_sigma(trends)
        intervals = jenkspy.jenks_breaks(trends_std, n_classes=2)
        break_point = intervals[1]
    logger.info(f"The interval is {intervals}; Break Point is {break_point}")
    pseudo_targets = np.where(trends > break_point, 0, 1)
    print(pseudo_targets.sum())
    estimated_prior = 1 - (pseudo_targets.sum() + args.num_labeled) / (len(pseudo_targets) + args.num_labeled)
    logger.info(f" Estimated positive prior is {estimated_prior}")
    if args.dataset != 'stl10_1' and args.dataset != 'stl10_2':
        acc = accuracy_score(targets, pseudo_targets)
        f1 = f1_score(targets, pseudo_targets)
        logger.info(f" After training phase 1: acc {acc}; f1 {f1}")
    return pseudo_targets
