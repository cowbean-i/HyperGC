import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch import Tensor
from sklearn import metrics
from torchmetrics import AveragePrecision
from torchmetrics.classification import BinaryAccuracy
from HyperGC.logreg import MLP, MLP_HENN

def accuracy(logits: list, labels: list):
    average_precision = AveragePrecision(task="binary")
    auc_roc = metrics.roc_auc_score(labels, logits)
    labels=torch.from_numpy(labels)
    labels=labels.type(torch.int32)
    ap = average_precision(torch.from_numpy(logits), labels)
    binary_acc= BinaryAccuracy()
    acc=binary_acc(torch.from_numpy(logits), labels)
    return auc_roc,ap,acc

def HE_evaluator(z,classifier, vidx, eidx,label):
    with torch.no_grad():
        classifier.eval()
        pred=classifier(z,vidx,eidx).to('cpu').detach().squeeze(-1).numpy()
        auroc,ap,acc=accuracy(pred,label)
    return auroc,ap,acc


def HE_predictor(z,edge_buckets,name):
    if name=='cora_cite':
        lr=0.001
        max_epoch=1000
        h_dim=512
    elif name=='citeseer_cite':
        lr=0.001
        max_epoch=1000
        h_dim=512
    else:
        lr=0.001
        max_epoch=1000
        h_dim=512
    
    classifier=MLP_HENN(in_dim=z.shape[1],hidden_dim=h_dim).to(z.device)
    optimizer=torch.optim.AdamW(classifier.parameters(),lr=lr,weight_decay=0.0)
    criterion=nn.BCELoss()
    train_vidx = torch.tensor(edge_buckets[0][0]).to(z.device)
    train_eidx = torch.tensor(edge_buckets[0][1]).to(z.device)
    train_label = edge_buckets[0][2].float().to(z.device)
    
    valid_vidx = torch.tensor(edge_buckets[1][0]).to(z.device)
    valid_eidx = torch.tensor(edge_buckets[1][1]).to(z.device)
    valid_label = edge_buckets[1][2].float().cpu().detach().numpy()
    
    test_vidx = torch.tensor(edge_buckets[2][0]).to(z.device)
    test_eidx = torch.tensor(edge_buckets[2][1]).to(z.device)
    test_label = edge_buckets[2][2].float().cpu().detach().numpy()

    valid_score=0
    best_epoch=0

    
    z=z.detach()
    for epoch in tqdm(range(1, max_epoch + 1)):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        pred=classifier(z,train_vidx,train_eidx)
        
        loss = criterion(pred, train_label)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            cur_score=HE_evaluator(z=z,classifier=classifier, vidx=valid_vidx, eidx=valid_eidx,label=valid_label)
            
            path=f'./logs/log_he_{name}.txt'
            with open(path, 'a+') as write_obj:
                write_obj.write(f'{name}{epoch}=> auroc: {cur_score[0]}, ap: {cur_score[1]}, acc: {cur_score[2]}\n')
            if epoch<=10:
                cur_score=list(cur_score)
                cur_score[0]=0
            if cur_score[0]>valid_score:
                valid_score=cur_score[0]
                param=copy.deepcopy(classifier.state_dict())
                best_epoch=epoch
                valid_results=cur_score
    classifier.load_state_dict(param)
    test_results=HE_evaluator(z=z, classifier=classifier, vidx=test_vidx, eidx=test_eidx,label=test_label)

    return valid_results, test_results, best_epoch

def NC_evaluator_finetuning(GNN,X,hyperedge_index, label, idx):
    with torch.no_grad() :
        GNN.eval()
        n_node, n_edge = torch.max(hyperedge_index[0]) + 1, torch.max(hyperedge_index[1]) + 1
        pred = torch.argmax(GNN(X, hyperedge_index, n_node, n_edge), dim=1)
        acc=torch.sum((pred==label)[idx])/len(idx)
    return acc.item()

def NC_evaluator_linear(z,classifier, idx, label):
    with torch.no_grad():
        classifier.eval()
        pred=torch.argmax(classifier(z),dim=1)
        acc=torch.sum((pred==label)[idx])/len(idx)
    return acc.item()


def NC_predictor_linear(z,data_splits,data):
    if data.name=='cora_cite':
        lr=0.001
        max_epoch=500
    elif data.name=='citeseer_cite':
        lr=0.001
        max_epoch=1000
    else:
        lr=0.001
        max_epoch=1000
    valid_results, test_results, best_epochs=[],[],[]

    for train_idx, valid_idx, test_idx in data_splits:
        classifier = MLP(in_dim=z.shape[1],n_class = data.n_class).to(z.device)
        optimizer=torch.optim.AdamW(classifier.parameters(),lr=lr,weight_decay=0.0)
        criterion=nn.CrossEntropyLoss()
        valid_score=0
        best_epoch=0        
        z=z.detach()
        for epoch in range(1, max_epoch + 1):
            classifier.train()
            optimizer.zero_grad(set_to_none=True)
            pred=classifier(z)[train_idx,:]
            y=data.labels
            loss = criterion(pred, y[train_idx])
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1 == 0:
                cur_score=NC_evaluator_linear(z=z, classifier=classifier, idx=valid_idx, label=y)
                
                path=f'./logs/log_{data.name}.txt'
                with open(path, 'a+') as write_obj:
                    write_obj.write(f'{data.name}{epoch}=>  val_acc: {cur_score}\n')
                if epoch<=10:
                    cur_score==0
                if cur_score>valid_score:
                    valid_score=cur_score
                    param=copy.deepcopy(classifier.state_dict())
                    best_epoch=epoch
                    valid_result=cur_score
        classifier.load_state_dict(param)
        test_result=NC_evaluator_linear(z=z, classifier=classifier, idx=test_idx, label=y)


        valid_results.append(valid_result)
        test_results.append(test_result)
        best_epochs.append(best_epoch)

    return valid_results, test_results, best_epochs
