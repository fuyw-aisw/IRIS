import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch.nn as nn
from datasets import merged_tudatasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random



from copy import deepcopy



def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred, _, _, _ = model(data)
            pred = pred.max(1)[1]
        
        all_preds.append(pred.cpu())
        all_labels.append(data.y.view(-1).cpu())
     
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return acc, f1_macro


def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, _, _, _ = model(data)
        loss += F.cross_entropy(out, data.y.view(-1), reduction='sum').item()#F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

def eval_idx_aug(model, dataset, device, batch_size, threshold, with_eval_mode, score):
    dataset.aug = "none"
    loader = DataLoader(dataset, batch_size, shuffle=False)
    if with_eval_mode:
        model.eval()
     
    #loss = 0 
    lists = [[] for _ in range(dataset.num_classes)]
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred, _, _, _ = model(data)
            pred = torch.softmax(pred, dim=-1)
            prob = pred.max(1)[0]
            pred = pred.max(1)[1]
            for i in range(pred.shape[0]):
                lists[pred[i]].append(prob[i].detach().cpu().numpy())
    class_counts = np.array([len(lst) for lst in lists])
    avg_count = np.mean(class_counts)
    lists = np.array(lists, dtype=object)
    #print(class_counts)
    for i in range(lists.shape[0]):
        lists[i] = np.sort(np.array(lists[i]))[::-1]
    rho = 1    
    idx = np.argmax(class_counts)
    for i in range(lists[idx].shape[0]):
        if lists[idx][i] < threshold:
            break
        rho = (i + 1) / lists[idx].shape[0]

    for i in range(lists.shape[0]):
        if lists[i].shape[0] != 0:
            adj_rho = rho * (avg_count / lists[i].shape[0]) ** 0.5
            idx = max(0, np.round(lists[i].shape[0] * adj_rho - 1).astype(int))
            idx = min(idx, lists[i].shape[0]-1)
            score[i] = min(threshold, lists[i][idx])
    return score


def get_one_hot_encoding(data, n_class):
    y = data.y.view(-1)
    encoding = np.zeros([len(y), n_class])
    for i in range(len(y)):
        encoding[i, int(y[i])] = 1
    return torch.from_numpy(encoding).to(device)


def train(model, optimizer, dataset, device, batch_size, eta, tau, score):
    #dataset = aug_extend_wo_label(model, dataset, aug, aug_ratio, aug_num, batch_size, score)
    #print(len(dataset))
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset1.aug, dataset1.aug_ratio = "none", 0.0
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = "none", 0.0

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)

    model.train()

    total_loss = 0
    for data1, data2 in zip(loader1, loader2):
        # print(data1, data2)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        out1, x1, pred1, pred_gcn1 = model.forward_cl(data1, False, None, eta=eta)
        graph_grad = torch.autograd.grad(out1,x1,retain_graph=True, grad_outputs=torch.ones_like(out1))[0]
        out2, _, pred2, pred_gcn2 = model.forward_cl(data2, True, None, grads=graph_grad, eta=eta)

        eq = torch.argmax(pred1, axis=-1) - torch.argmax(pred2, axis=-1)
        indices = (eq == 0).nonzero().reshape(-1)
        #print(len(indices))
        loss = model.loss_cl(out1[indices], out2[indices], tau)

        prob1 = torch.softmax(pred1, dim=1) / torch.from_numpy(score).float().to(device)
        max_rp1, rp1_hat = torch.max(prob1, dim=1)
        mask1 = max_rp1.ge(1.0)
        
        pred, _, h, _ = model(data1)

        loss += (F.cross_entropy(pred, rp1_hat, reduction='none') * mask1).mean()
        #print(loss.item())
        if len(indices) > 1:
            loss.backward()
            total_loss += loss.item() * num_graphs(data1)
            optimizer.step()
                
    return total_loss / len(loader1.dataset)



def train_label(model, optimizer, dataset, device, batch_size, eta , tau, criterion_ib=None):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset1.aug, dataset1.aug_ratio = "none", 0.0
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = "none", 0.0

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)

    model.train()
    total_loss = 0
    correct = 0
    for data1, data2 in zip(loader1, loader2):
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)  

        out1, x1, pred1, pred_gcn1 = model.forward_cl(data1, False, get_one_hot_encoding(data1, model.n_class), eta=eta)
        graph_grad = torch.autograd.grad(out1,x1,retain_graph=True, grad_outputs=torch.ones_like(out1))[0]
        out2, x2, pred2, pred_gcn2 = model.forward_cl(data2, True, get_one_hot_encoding(data2, model.n_class), grads=graph_grad, eta=eta)
        
        eq = torch.argmax(pred1, axis=-1) - torch.argmax(pred2, axis=-1)
        indices = (eq == 0).nonzero().reshape(-1)
        loss = model.loss_cl(out1[indices], out2[indices], tau)

        #print(loss.item())
        out, _, hidden, pred_gcn = model(data1)
        if criterion_ib is None:
            loss += (F.cross_entropy(pred1, data1.y.view(-1))+ F.cross_entropy(pred2[indices], data2.y.view(-1)[indices])) #* 0.01
        else:
            loss += (criterion_ib(pred1, data1.y.view(-1), x1)+ criterion_ib(pred2[indices], data2.y.view(-1)[indices], x2[indices]))
                
        pred = out.max(1)[1]
        correct += pred.eq(data1.y.view(-1)).sum().item()

            
        if len(indices) > 1:
            loss.backward()
            total_loss += loss.item() * num_graphs(data1)
            optimizer.step()
    return total_loss / len(loader1.dataset), correct / len(loader1.dataset)

def get_class_num(imb_ratio, num_train, num_val):
    c_train_num = [int(imb_ratio * num_train), num_train -
                   int(imb_ratio * num_train)]

    c_val_num = [int(imb_ratio * num_val), num_val - int(imb_ratio * num_val)]

    return c_train_num, c_val_num

def shuffle_semi(dataset, c_train_num, c_val_num, n_percents=0.3):
   
    y = torch.tensor([data.y.item() for data in dataset])
    classes = torch.unique(y)
    counts = torch.bincount(y)
    if counts[0].item() > counts[1].item():
        c_train_num.reverse()
        c_val_num.reverse()
    indices = []

    for i in range(len(classes)):
        index = torch.nonzero(y == classes[i]).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index, train_index_unlabel, val_index, test_index = [], [], [], []

    for i in range(len(classes)):
        train_index.append(indices[classes[i]][:round(c_train_num[classes[i]] * n_percents)])
        train_index_unlabel.append(indices[classes[i]][round(c_train_num[classes[i]] * n_percents):c_train_num[classes[i]]])

        val_index.append(indices[classes[i]][c_train_num[classes[i]]:(
            c_train_num[classes[i]] + c_val_num[classes[i]])])

        test_index.append(indices[classes[i]][(
            c_train_num[classes[i]] + c_val_num[classes[i]]):])

    train_index = torch.cat(train_index, dim=0)
    train_index_unlabel = torch.cat(train_index_unlabel, dim=0)
    val_index = torch.cat(val_index, dim=0)
    test_index = torch.cat(test_index, dim=0)

    train_dataset = dataset[train_index]
    train_dataset_unlabel = dataset[train_index_unlabel]    
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]
    print(train_index, train_index_unlabel)
    return train_dataset, val_dataset, test_dataset, train_dataset_unlabel




def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()

class IBLoss(nn.Module):
    def __init__(self, weight=None, alpha=1000):
        super(IBLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight

    def forward(self, out, target, features): 
        grads = torch.sum(torch.abs(F.softmax(out, dim=1) - F.one_hot(target, 2)),1) # N * 1
        features = torch.sum(torch.abs(features), 1).reshape(-1, 1)
        ib = grads*features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.cross_entropy(out, target, reduction='none', weight=self.weight), ib)

def ib_focal_loss(input_values, ib, gamma):
    """Computes the ib focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values * ib
    return loss.mean()

class IB_FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000., gamma=0.):
        super(IB_FocalLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, 2)),1) # N * 1
        features = torch.sum(torch.abs(features), 1).reshape(-1, 1)
        ib = grads*(features.reshape(-1))
        ib = self.alpha / (ib + self.epsilon)
        return ib_focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)


def embed_smote(embed, num_training_graph, y, k):
    max_num_training_graph = max(num_training_graph)
    classes = torch.unique(y)

    embed_aug = []
    y_aug = []

    for i in range(len(classes)):
        train_idx = torch.where((y == classes[i]) == True)[0].tolist()

        c_embed = embed[train_idx]
        c_dist = torch.cdist(c_embed, c_embed, p=2)

        # different from original smote, we also include itself in case of no other nodes to use
        c_min_idx = torch.topk(c_dist, min(k, c_dist.size(0)), largest=False)[
            1][:, :].tolist()

        up_sample_ratio = max_num_training_graph / \
            num_training_graph[i]
        up_sample_num = int(
            num_training_graph[i] * up_sample_ratio - num_training_graph[i])

        tmp = 1
        head_list = list(np.arange(0, len(train_idx)))

        while(tmp <= up_sample_num):
            head_id = random.choice(head_list)
            tail_id = random.choice(c_min_idx[head_id])

            delta = torch.rand(1).to(c_embed.device)
            new_embed = torch.lerp(
                c_embed[head_id], c_embed[tail_id], delta)
            embed_aug.append(new_embed)
            y_aug.append(classes[i])

            tmp += 1

    if(embed_aug == []):
        return embed, y

    return torch.stack(embed_aug), torch.stack(y_aug).to(embed.device)

def cross_validation_with_label(dataset,
                                model_func,
                                epochs,
                                batch_size,
                                lr,
                                lr_decay_factor,
                                lr_decay_step_size,
                                weight_decay,
                                epoch_select,
                                num_train, 
                                num_val, 
                                imb_ratio=0.1,
                                n_percents=0.3,
                                with_eval_mode=True,
                                logger=None,
                                dataset_name=None,
                                eta=1.0, threshold=0.9, tau=0.5
                               ):
    assert epoch_select in ['val_max', 'test_max'], epoch_select
    val_losses, train_accs, test_accs, durations, test_f1s = [], [], [], [], []
    c_train_num, c_val_num = get_class_num(imb_ratio, num_train, num_val) 
    
    train_dataset, val_dataset, test_dataset, train_dataset_unlabel = shuffle_semi(dataset, c_train_num, c_val_num, n_percents)  
    #print(torch.bincount(torch.tensor([data.y.item() for data in train_dataset])))

    #train_dataset = aug_extend_with_label(train_dataset, aug, aug_ratio, aug_num)
    #val_dataset = aug_extend_with_label(val_dataset, aug, aug_ratio, aug_num)
    cls_num_list = torch.bincount(torch.tensor([data.y.item() for data in train_dataset]))
    #print(cls_num_list)
    per_cls_weights = 1.0 / np.array(cls_num_list)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    criterion_ib = IBLoss(weight=per_cls_weights, alpha=1000).to(device)
    #criterion_ib = IB_FocalLoss(weight=per_cls_weights, alpha=1000, gamma=1).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    train_loader_unlabel = DataLoader(train_dataset_unlabel, batch_size, shuffle=True)

    score = np.array([threshold] * dataset.num_classes)
    
    dataset.aug = "none"
    model = model_func(dataset).to(device)
    optimizer_label = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = Adam(model.parameters(), lr=lr/5, weight_decay=weight_decay)
    #breakpoint()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    thresholds_all, train_losses, train_label_losses, epochs_all = [], [], [], []
    #c_train_num = len(train_dataset)
    for epoch in range(1, epochs+1):
        score = eval_idx_aug(model, train_dataset_unlabel, device, batch_size, threshold, with_eval_mode, score)
        #print(score)
        train_loss = train(model, optimizer, train_dataset_unlabel, device, batch_size, eta, tau, score)
        if epoch < 10:
            train_label_loss, train_acc = train_label(model, optimizer_label, train_dataset, device, batch_size, eta, tau)
        else:
            train_label_loss, train_acc = train_label(model, optimizer_label, train_dataset, device, batch_size, eta, tau, criterion_ib)            

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        train_label_losses.append(train_label_loss)
        epochs_all.append(epoch)

        thresholds_all.append(score)
        val_losses.append(eval_loss(
            model, val_loader, device, with_eval_mode))
        test_accs.append(eval_acc(
            model, test_loader, device, with_eval_mode)[0])
        test_f1s.append(eval_acc(
            model, test_loader, device, with_eval_mode)[1])        
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
    eval_info = {
        'epoch': epochs_all,
        'train_loss': train_losses,
        'train_label_loss': train_label_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'test_acc': test_accs,
        'thresholds': thresholds_all
    }

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    durations.append(t_end - t_start)

    duration = tensor(durations)
    train_acc, test_acc, test_f1 = tensor(train_accs), tensor(test_accs), tensor(test_f1s)
    val_loss = tensor(val_losses)
    if epoch_select == 'test_max':
        _, selected_epoch = (test_acc+test_f1).max(dim=0)
    else:
        _, selected_epoch = val_loss.min(dim=0)
    print(selected_epoch)
    test_acc = test_acc[selected_epoch]
    test_f1 = test_f1[selected_epoch]
    sys.stdout.flush()

    return test_acc, test_f1, eval_info
