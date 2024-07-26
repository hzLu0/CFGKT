import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
import random
import os

class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, './best_model.pth')
        self.val_loss_min = val_loss

early_stopping = EarlyStopping()

def train_one_epoch(model, train_iterator, val_loader, optim, loss_function, epoch,device="cpu"):
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=4, gamma=0.9)
    for j in range(epoch):
        print('----------------epoch:' ,j)
        for i, (qa, qid, labels, mask, concept, dotime, interval) in enumerate(train_iterator):
            qa, qid, labels, mask, concept, dotime, interval = (qa.to(device), qid.to(device), labels.to(device),
                                                              mask.to(device), concept.to(device),
                                                              dotime.to(device), interval.to(device))
            optim.zero_grad()
            pred = model(concept,qa,labels,qid, dotime, interval)
            loss = loss_function(pred, labels, mask)
            if i % 5 == 0:
                print("Train--Loss: {:.5}".format(loss.item()))

            if i % (len(train_iterator)-1) == 0 and i != 0:
                stop_sign = eval_one_epoch(model, val_loader, device, loss_function)
                if stop_sign == True:
                    return
            loss.backward()
            optim.step()
            model.train()
        scheduler.step()

def eval_one_epoch(model, val_iterator, device,loss_function):
    stop_sign = False
    model.eval()
    with torch.no_grad():
        preds_list = []
        truths_list = []
        for i, (qa, qid, labels, mask, concept, dotime, interval) in enumerate(val_iterator):
            qa, qid, labels, mask, concept, dotime, interval = (qa.to(device), qid.to(device), labels.to(device),
                                                              mask.to(device), concept.to(device),
                                                              dotime.to(device), interval.to(device))

            pred = model(concept, qa, labels, qid, dotime, interval)
            loss = loss_function(pred, labels, mask)
            print('val-loss  :', loss.item())
            truth, pred = values_after_mask(pred, labels, mask)
            truths_list.append(truth)
            preds_list.append(pred)

        truths_list = np.concatenate(truths_list)
        preds_list = np.concatenate(preds_list)

        auc = roc_auc_score(truths_list, preds_list)
        acc = accuracy_score(truths_list, preds_list.round())
        f1 = f1_score(truths_list, preds_list.round())
        print("\nval-auc=%.4f acc=%.4f f1=%.4f" % (auc, acc, f1))
        early_stopping(auc*(-1), model)
        if early_stopping.early_stop:
            print("stop training!!!!!!!!")
            stop_sign = True
            return stop_sign
        return stop_sign

def test_one_epoch(model, test_iterator, device):
    model.eval()
    with torch.no_grad():
        preds_list = []
        truths_list = []

        for i, (qa, qid, labels, mask, concept, dotime, interval) in enumerate(test_iterator):
            qa, qid, labels, mask, concept, dotime, interval = (qa.to(device), qid.to(device), labels.to(device),
                                                               mask.to(device), concept.to(device),
                                                               dotime.to(device), interval.to(device))
            with torch.no_grad():
                pred = model(concept, qa, labels, qid, dotime, interval)
            truth, pred = values_after_mask(pred, labels, mask)
            truths_list.append(truth)
            preds_list.append(pred)

        truths_list = np.concatenate(truths_list)
        preds_list = np.concatenate(preds_list)
        mse = mean_squared_error(truths_list,preds_list)
        auc = roc_auc_score(truths_list, preds_list)
        acc = accuracy_score(truths_list, preds_list.round())
        f1 = f1_score(truths_list, preds_list.round())
        print("\nauc=%.4f acc=%.4f f1=%.4f mse=%.4f" % (auc, acc, f1, mse))
        print('MSE',mse)

def values_after_mask(pred, labels, mask):
    mask_except_first = mask.gt(0)
    mask_except_first = mask_except_first.view(-1)
    pred_except_first = torch.masked_select(pred.view(-1), mask_except_first).detach().cpu().numpy()
    truth_except_first = torch.masked_select(labels.view(-1), mask_except_first).detach().cpu().numpy()
    return truth_except_first, pred_except_first
