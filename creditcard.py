import numpy as np
import pandas as pd
import torch
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model.mlp import CardMLP
from utils.misc import three_sigma, accuracy1, set_seed, get_cosine_schedule_with_warmup, interleave, de_interleave
import torch.nn.functional as F
import jenkspy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

logger = logging.getLogger(__name__)
data = pd.read_csv('/home/parnec/wxr/dataset/creditcard/creditcard.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = 1-y
sc = StandardScaler()
x = sc.fit_transform(x)
positive_index = np.where(y == 0)[0]

negative_index = np.where(y == 1)[0]
# print(len(positive_index))
labeled_train_index = np.random.choice(positive_index, 100, False)
positive_index = np.setdiff1d(positive_index, labeled_train_index)
positive_test_index = np.random.choice(positive_index, 100, False)
unlabeled_positive_index = np.setdiff1d(positive_index, positive_test_index)

unlabeled_negative_index = np.random.choice(negative_index, 8008, False)
negative_index = np.setdiff1d(negative_index, unlabeled_negative_index)
test_negative_index = np.random.choice(negative_index, 2000, False)

labeled_x = x[labeled_train_index]
labeled_y = y[labeled_train_index]

unlabeled_x = np.append(x[unlabeled_negative_index], x[unlabeled_positive_index],axis=0)
unlabeled_y_t = np.append(y[unlabeled_negative_index], y[unlabeled_positive_index])
print(unlabeled_y_t.sum())
unlabeled_y_u = np.ones(len(unlabeled_y_t))


x_test = np.append(x[test_negative_index], x[positive_test_index], axis=0)
y_test = np.append(y[test_negative_index], y[positive_test_index])
print(np.where(y_test==0))

p_data = torch.from_numpy(labeled_x)
p_label = torch.from_numpy(labeled_y).double()
u_data = torch.from_numpy(unlabeled_x)
u_label_u = torch.from_numpy(unlabeled_y_u).double()
u_label_t = torch.from_numpy(unlabeled_y_t).double()
t_data = torch.from_numpy(x_test)
t_label = torch.from_numpy(y_test).double()




postive_set = data_utils.TensorDataset(p_data, p_label)
unlabeled_set = data_utils.TensorDataset(u_data, u_label_u, u_label_t)
test_set = data_utils.TensorDataset(t_data, t_label)
print(len(postive_set), len(unlabeled_set), len(test_set))
labeled_trainloader = DataLoader(postive_set, sampler=SequentialSampler(postive_set), batch_size=50, shuffle=False)
unlabeled_trainloader = DataLoader(unlabeled_set, batch_size=50, sampler=SequentialSampler(unlabeled_set), shuffle=False, drop_last=True)
testloader = DataLoader(test_set, batch_size=2100, sampler=SequentialSampler(test_set), shuffle=False, drop_last=True)


def train_phase1(device, model, optimizer):
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    global preds_sequence, targets
    model.train()
    for epoch in range(0, 10):

        for batch_idx in range(500):
            try:
                inputs_p, targets_p = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_p, targets_p = labeled_iter.next()
            try:
                inputs_u, targets_u, targets_t = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                inputs_u, targets_u, targets_t = unlabeled_iter.next()
            batch_size = inputs_p.shape[0]
            inputs = interleave(torch.cat((inputs_p, inputs_u)), 2).to(device)
            targets_p = targets_p.to(device)
            targets_u = targets_u.to(device)
            targets_t = targets_t.to(device)
            logits = model(inputs)
            logits = de_interleave(logits, 2)
            logits_p, logits_u = logits.chunk(2)
            Lx = F.cross_entropy(logits_p, targets_p.long(), reduction='mean', label_smoothing=0)
            Lu = F.cross_entropy(logits_u, targets_u.long(), reduction='mean', label_smoothing=0)
            loss = Lx + Lu
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = np.array(batch_size)
            for batch_idx, (inputs, targets_u, targets_t) in enumerate(unlabeled_trainloader):
                model.eval()
                inputs = inputs.to(device)
                outputs = model(inputs)
                softmax = torch.nn.Softmax(dim=1)
                pred = softmax(outputs)
                pred = np.array((pred[:, 0]).cpu())
                if batch_idx == 0:
                    preds = pred
                    targets = targets_t
                else:
                    preds = np.concatenate([preds, pred], axis=0)
                    targets = np.concatenate([targets, targets_t], axis=0)
        if epoch == 0:
            preds_sequence = preds

        else:
            preds_sequence = np.vstack((preds_sequence, preds))

    preds_sequence = preds_sequence.T
    trends = np.zeros(len(preds_sequence))
    for i, sequence in enumerate(preds_sequence):
        sequence = pd.Series(sequence)
        diff_1 = sequence.diff(periods=1)
        diff_1 = np.array(diff_1)
        diff_1 = diff_1[1:]
        diff_1 = np.log(1 + diff_1 + 0.5 * diff_1 ** 2)
        trends[i] = diff_1.mean()
    # print(np.min(trends), np.max(trends), trends.mean())
    intervals = jenkspy.jenks_breaks(trends, n_classes=2)
    break_point = intervals[1]
    # if break_point > 0:
    #     trends_std = three_sigma(trends)
    #     intervals = jenkspy.jenks_breaks(trends_std, n_classes=2)
    #     break_point = intervals[1]
    print(f"The interval is {intervals}; Break Point is {break_point}")
    pseudo_targets = np.where(trends > break_point, 0, 1)
    acc = accuracy_score(targets, pseudo_targets)
    f1 = f1_score(targets, pseudo_targets)
    prec = precision_score(targets, pseudo_targets)
    recall = recall_score(targets, pseudo_targets)
    print('acc', acc, 'f1', f1, 'prec', prec, 'recall', recall, 'estimated prior', 1 - pseudo_targets.sum()/8400)
    return pseudo_targets


def train(device, model, optimizer, pseudo_targets):
    unlabeled_num = len(pseudo_targets)
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    global preds_sequence
    model.train()
    unlabeled_idx = 0
    for epoch in range(0, 30):

        for batch_idx in range(500):
            try:
                inputs_p, targets_p = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_p, targets_p = labeled_iter.next()
            try:
                inputs_u, targets_u, targets_t = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                inputs_u, targets_u, targets_t = unlabeled_iter.next()
            batch_size = inputs_p.shape[0]
            inputs = interleave(torch.cat((inputs_p, inputs_u)), 2).to(device)
            targets_p = targets_p.to(device)
            targets_u = targets_u.to(device)
            targets_t = targets_t.to(device)
            targets_pseudo = pseudo_targets[unlabeled_idx: unlabeled_idx + batch_size]
            unlabeled_idx = (unlabeled_idx + batch_size) % unlabeled_num
            targets_pseudo = torch.tensor(targets_pseudo)
            targets_pseudo = targets_pseudo.to(torch.long).to(device)
            logits = model(inputs)
            logits = de_interleave(logits, 2)
            logits_p, logits_u = logits.chunk(2)
            Lx = F.cross_entropy(logits_p, targets_p.long(), reduction='mean', label_smoothing=0)
            Lu = F.cross_entropy(logits_u, targets_pseudo.long(), reduction='mean', label_smoothing=0)
            loss = Lx + Lu
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # test
        test_model = model


        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                test_model.eval()
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = test_model(inputs)
                acc, auc, f1, prec, recall = accuracy1(outputs, targets)
                print(batch_idx)
                print(acc, auc, f1, prec, recall)


if __name__ == '__main__':
    device = torch.device('cuda', 0)
    model = CardMLP().double()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    model.to(device)
    print('Training phase 1:')
    pseudo_targets = train_phase1(device, model, optimizer)
    del model, optimizer
    print('Fine tuning phase:')
    model = CardMLP().double()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    model.to(device)
    train(device, model, optimizer, pseudo_targets)


