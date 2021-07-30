from models.macro_f1_loss import macro_double_soft_f1, custom_loss

from progress.bar import ShadyBar
from utils import params as p
from sklearn import metrics
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch


pos_weight = torch.tensor(p.pos_weight)
# loss_function = nn.BCEWithLogitsLoss(reduction='mean')
bce_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
loss_function = custom_loss(bce_loss, macro_double_soft_f1)

def train(model, train_loader, optimizer, epoch, writer=None):
    model.train()
    # pos_weight a weight of positive examples. Must be a vector with length equal to the number of classes.
    f1_c = 0
    bACC_c = 0 #balanced accuracy
    loss = 0
    bar = ShadyBar(f"Training epoch {epoch}...", max=len(train_loader))
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch
        data, target = data.to(p.device).float(), target.to(p.device).float()
        optimizer.zero_grad()
        output = model(data)
        loss_comp = loss_function(output, target)
        loss_comp.backward()
        loss += loss_comp.item()
        optimizer.step()
        output = torch.sigmoid(output)
        output_round = torch.round(output)
        #train_acc = torch.sum(output_round == target).float()/(target.shape[0]*target.shape[1])
        f1 = 0
        bACC = 0
        for i_batch in range(output_round.shape[0]):
            y_true = target.detach().cpu().numpy().astype(int)[i_batch]
            y_pred = output_round.detach().cpu().numpy().astype(int)[i_batch]
            f1 += metrics.f1_score(y_true, y_pred)
            bACC += metrics.balanced_accuracy_score(y_true, y_pred)
        f1 /= output_round.shape[0]
        bACC /= output_round.shape[0]

        f1_c += f1
        bACC_c += bACC

        bar.next()

    train_loss = loss/len(train_loader)
    train_bACC = bACC_c/len(train_loader)
    train_F1 = f1_c/len(train_loader)
    print("\nf1-score: {:.2f}, balanced accuracy: {:.2f}, loss: {:.2f}, epoch: {}".format(f1_c/len(train_loader), bACC_c/len(train_loader), train_loss, epoch))

    if writer:
        writer.add_scalar("Train/f1-score", train_F1, epoch)
        writer.add_scalar("Train/bAcc", train_bACC, epoch)
        writer.add_scalar("Train/Loss", loss, epoch)

    return train_loss, train_bACC, train_F1


def test(model, test_loader, scheduler, epoch, writer=None):
    model.eval()
    test_loss = 0
    f1_c = 0
    bACC_c = 0
    bar = ShadyBar(f"Validation...", max=len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(p.device).float(), target.to(p.device).float()
            output = model(data)
            temp = loss_function(output, target)
            test_loss += temp.item()
            output = torch.sigmoid(output)
            output_round = torch.round(output)
            #test_acc = torch.sum(output_round == target).float()/(target.shape[0]*target.shape[1])
            f1 = 0
            bACC = 0
            for i_batch in range(output_round.shape[0]):
                y_true = target.detach().cpu().numpy().astype(int)[i_batch]
                y_pred = output_round.detach().cpu().numpy().astype(int)[i_batch]
                f1 += metrics.f1_score(y_true, y_pred)
                bACC += metrics.balanced_accuracy_score(y_true, y_pred)
            f1 /= output_round.shape[0]
            bACC /= output_round.shape[0]

            f1_c += f1
            bACC_c += bACC
            
            bar.next()
    test_loss /= len(test_loader.dataset)
    test_bACC = bACC_c/len(test_loader)
    test_F1 = f1_c/len(test_loader)
    print("\nf1-score: {:.2f}, balanced accuracy: {:.2f}, loss: {:.2f}".format(f1_c/len(test_loader), bACC_c/len(test_loader), test_loss))
    
    if scheduler:
        scheduler.step(test_loss)
    
    if writer:
        writer.add_scalar("Val/f1-score", test_F1, epoch)
        writer.add_scalar("Val/bAcc", test_bACC, epoch)
        writer.add_scalar("Val/Loss", test_loss, epoch)

    return test_loss, test_bACC, test_F1
