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
from dataset.alzheimer import get_alzheimer
from utils.misc import AverageMeter, accuracy, set_seed, get_cosine_schedule_with_warmup, interleave, de_interleave
from train import train_phase1, train


def main():
    logger = logging.getLogger(__name__)
    best_acc = 0
    parser = argparse.ArgumentParser(description='Naive PU Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10_1', type=str,
                        choices=['cifar10_1', 'stl10_1', 'fmnist_1', 'cifar10_2', 'stl10_2', 'fmnist_2', 'alzheimer'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--class-missing', action='store_true', default=False,
                        help='not all positive classes in dataset is available in P')
    parser.add_argument('--imbalance', action='store_true', default=False,
                        help='an imbalanced data setting')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='ResNet18', type=str,
                        choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'AlexNet', 'CNN_CIFAR', 'CNN_STL', 'LeNet'],
                        help='model name')
    parser.add_argument('--total-steps', default=25 * 2 ** 9, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=512, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    # parser.add_argument('--use-ema', action='store_true', default=False,
    #                     help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--beta', default=0.9, type=float,
                        help='ensembling parameter')
    parser.add_argument('--lambda-u', default=5, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lambda-n', default=1, type=float,
                        help='coefficient of pseudo negative loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--warming_steps', default=15 * 2 ** 9, type=int,
                        help='number of epochs in training phase 1')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='mixup parameter on validation set')
    parser.add_argument('--rho', default=0.1, type=float,
                        help='smoothing parameter')
    args = parser.parse_args()
    # setting 1:
    if args.dataset == 'cifar10_1':
        args.positive_label_list = [0, 1, 8, 9]
        if args.class_missing:
            args.available_label_list = [0, 1]
            args.num_classes = 2
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_cifar10(args, './dataset/cifar10')
        else:
            args.available_label_list = [0, 1, 8, 9]
            args.num_classes = 4
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_cifar10(args, './dataset/cifar10')
    # setting 2:
    if args.dataset == 'cifar10_2':
        args.positive_label_list = [2, 3, 4, 5, 6, 7]
        if args.class_missing:
            args.available_label_list = [2, 3, 4]
            args.num_classes = 3
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_cifar10(args, './dataset/cifar10')
        else:
            args.available_label_list = [2, 3, 4, 5, 6, 7]
            args.num_classes = 6
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_cifar10(args, './dataset/cifar10')
    # # setting 1:
    if args.dataset == 'stl10_1':
        args.positive_label_list = [0, 2, 3, 8, 9]
        if args.class_missing:
            args.available_label_list = [0, 2, 8]
            args.num_classes = 3
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_stl10(args, './dataset/stl10')
        else:
            args.available_label_list = [0, 2, 3, 8, 9]
            args.num_classes = 5
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_stl10(args, './dataset/stl10')
    # setting 2:
    if args.dataset == 'stl10_2':
        args.positive_label_list = [1, 4, 5, 6, 7]
        if args.class_missing:
            args.available_label_list = [1, 4, 5]
            args.num_classes = 3
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_stl10(args, './dataset/stl10')
        else:
            args.available_label_list = [1, 4, 5, 6, 7]
            args.num_classes = 5
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_stl10(args, './dataset/stl10')
    # setting 1:
    if args.dataset == 'fmnist_1':
        args.positive_label_list = [0, 2, 4, 6]
        if args.class_missing:
            args.available_label_list = [0, 4]
            args.num_classes = 2
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_fmnist(args, './dataset')
        else:
            args.available_label_list = [0, 2, 4, 6]
            args.num_classes = 4
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_fmnist(args, './dataset')
    # setting 2:
    if args.dataset == 'fmnist_2':
        args.positive_label_list = [1, 3, 5, 7, 8, 9]
        if args.class_missing:
            args.available_label_list = [0, 2, 3, 5, 8]
            args.num_classes = 2
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_fmnist(args, './dataset')
        else:
            args.available_label_list = [1, 3, 5, 7, 8, 9]
            args.num_classes = 6
            train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_fmnist(args, './dataset')

    if args.dataset == 'alzheimer':
        args.num_classes = 2
        args.num_labeled = 769
        train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_alzheimer()
    print(len(train_labeled_dataset), len(train_unlabeled_dataset), len(test_dataset))
    args.rate = len(train_unlabeled_dataset) // len(train_labeled_dataset)

    def create_model(args):
        if args.arch == 'ResNet18':
            if args.dataset == 'cifar10':
                model = ResNet18()
            if args.dataset == 'stl10' or args.dataset=='alzheimer':
                model = resnet18()
        elif args.arch == 'ResNet34':
            model = resnet34()
        elif args.arch == 'ResNet50':
            model = resnet50()
        elif args.arch == 'AlexNet':
            model = alexnet()
        elif args.arch == 'CNN_CIFAR':
            model = cnn_cifar()
        elif args.arch == 'CNN_STL':
            model = cnn_stl()
        elif args.arch == 'LeNet':
            model = lenet_fmnist()
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device
    args.ft_epochs = args.total_steps // args.eval_step
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    test_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    valid_sampler = SequentialSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        train_labeled_dataset,
        sampler=train_sampler(train_labeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        train_unlabeled_dataset,
        sampler=valid_sampler(train_unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)


    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler(test_dataset),
        batch_size=256,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    args.warming_epochs = math.ceil(args.warming_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from model.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    pseudo_targets = train_phase1(args, logger, labeled_trainloader, unlabeled_trainloader,
          model, optimizer, ema_model, scheduler)
    del model, optimizer, scheduler

    print(len(pseudo_targets))
    '''
    create model 1
    '''
    model1 = create_model(args)
    model1.to(args.device)
    no_decay = ['bias', 'bn']
    grouped_parameters1 = [
        {'params': [p for n, p in model1.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model1.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer1 = optim.SGD(grouped_parameters1, lr=args.lr ,
                          momentum=0.9, nesterov=args.nesterov)
    scheduler1 = get_cosine_schedule_with_warmup(
        optimizer1, args.warmup, args.total_steps)
    '''
        create model 2
    '''

    model2 = create_model(args)
    model2.to(args.device)
    no_decay = ['bias', 'bn']
    grouped_parameters2 = [
        {'params': [p for n, p in model2.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model2.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer2 = optim.SGD(grouped_parameters2, lr=args.lr ,
                           momentum=0.9, nesterov=args.nesterov)
    scheduler2 = get_cosine_schedule_with_warmup(
        optimizer2, args.warmup, args.total_steps)

    model1.zero_grad()
    train(args, logger, best_acc, labeled_trainloader, unlabeled_trainloader, test_loader, pseudo_targets,
                 model1, optimizer1, scheduler1, model2, optimizer2, scheduler2, ema_model)


if __name__ == '__main__':
    main()


