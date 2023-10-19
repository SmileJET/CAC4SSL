import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def Binary_dice_loss(predictive, target, ep=1e-8, mask=None):
    if mask is not None:
        predictive = torch.masked_select(predictive, mask)
        target = torch.masked_select(target, mask)
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, mask=None, ep=1e-8, reduction='mean'):
    kl_loss=nn.KLDivLoss(reduction=reduction)
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    if mask is not None:
        consist_loss = torch.mean(torch.masked_select(consist_loss, mask.unsqueeze(dim=1).bool()))
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2, mask=None):
    if mask is None:
        return torch.mean((input1 - input2)**2)
    else:
        mse = (input1 - input2)**2
        return torch.mean(torch.masked_select(mse, mask.bool().unsqueeze(1)))

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, mask=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            if mask is not None:
                dice = self._dice_loss(torch.masked_select(inputs[:, i], mask), torch.masked_select(target[:, i], mask))
            else:
                dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    