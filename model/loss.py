import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1.detach(), t, reduce=False)
    ind_1_sorted = torch.argsort(loss_1)
    loss_1_sorted = loss_1[ind_1_sorted]
    loss_2 = F.cross_entropy(y_2.detach(), t, reduce=False)
    ind_2_sorted = torch.argsort(loss_2.data)
    loss_2_sorted = loss_2[ind_2_sorted]
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember


# def loss_ft(args, logits_u, logits_u_s, targets, forget_rate, epoch, threshold=0.9):
#     loss = F.cross_entropy(logits_u.detach(), targets, reduce=False)
#     index = torch.argsort(loss)
#     num_remember = int((1-forget_rate)*len(loss))
#     index_update = index[:num_remember]
#
#     # pred = F.softmax(logits_u, dim=1)
#     # label_one_hot = F.one_hot(targets, 2).float().to(args.device)
#     # label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
#     # lamda = (epoch / 2 * args.ft_epochs) ** 2
#     # targets_u = lamda * pred + (1 - lamda) * label_one_hot
#
#     loss1 = F.cross_entropy(logits_u[index_update], targets[index_update], reduction='mean', label_smoothing=0.1)
#
#     # pseudo_label = torch.softmax(logits_u.detach() / args.T, dim=-1)
#     # max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
#     # mask = max_probs.ge(threshold).float()
#     # loss2 = (F.cross_entropy(logits_u_s, pseudo_targets_u,
#     #                       reduction='none') * mask).mean()
#     return loss1

def loss_ft(args, logits1_u, logits1_u_s, targets_u, targets_p, epoch):
    label_u = F.one_hot(targets_u, 2).float().to(args.device)
    label_p = F.one_hot(targets_p, 2).float().to(args.device)
    lamda = (epoch / args.ft_epochs) ** 0.8
    label = lamda * label_p + (1-lamda) * label_u
    loss = F.cross_entropy(logits1_u, label, reduction='mean')

    pseudo_label = torch.softmax(logits1_u.detach() / args.T, dim=-1)
    max_probs, pseudo_targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(0.9).float()
    loss2 = (F.cross_entropy(logits1_u_s, pseudo_targets_u,
                              reduction='none') * mask).mean()
    return loss + loss2