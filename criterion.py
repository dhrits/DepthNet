import torch
import torch.nn as nn
import kornia


def edge_loss(pred, target, valid_mask):
    depth_edges = kornia.filters.sobel(target)
    pred_edges = kornia.filters.sobel(pred)
    return depth_edges[valid_mask] - pred_edges[valid_mask]


# Defined custom loss for the purpose of this project
class SmoothedL1Loss(nn.Module):
    def __init__(self, preserve_edges=False):
        super(SmoothedL1Loss, self).__init__()
        self.preserve_edges = preserve_edges
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='none')
        self.loss = 0.

    def forward(self, preds, targets):
        valid_mask = (targets > 0).detach()
        self.loss = self.loss_fn(preds, targets)
        self.loss = self.loss[valid_mask].mean()
        if self.preserve_edges:
            depth_edges = kornia.filters.sobel(targets)
            pred_edges = kornia.filters.sobel(preds)
            edge_loss = self.loss_fn(pred_edges, depth_edges)
            edge_loss = edge_loss[valid_mask].mean()
            self.loss += 12 * edge_loss
            
        return self.loss


# The masking and loss function structure inspired from https://github.com/tau-adl/FastDepth
class L1Loss(nn.Module):
    def __init__(self, preserve_edges=False):
        super(L1Loss, self).__init__()
        self.preserve_edges = preserve_edges
        self.loss_fn = torch.nn.L1Loss(reduction='none')
        self.loss = 0.

    def forward(self, preds, targets):
        valid_mask = (targets > 0).detach()
        self.loss = self.loss_fn(preds, targets)
        self.loss = self.loss[valid_mask].mean()
        if self.preserve_edges:
            # 10x difference
            self.loss += 4 * (edge_loss(preds, targets, valid_mask).abs()).mean()
        return self.loss


class L2Loss(nn.Module):
    def __init__(self, preserve_edges=False):
        super(L2Loss, self).__init__()
        self.preserve_edges = preserve_edges
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.loss = 0.

    def forward(self, preds, targets):
        valid_mask = (targets > 0).detach()
        self.loss = self.loss_fn(preds, targets)
        self.loss = self.loss[valid_mask].mean()
        if self.preserve_edges:
            # 50x difference
            self.loss += 7 * (edge_loss(preds, targets, valid_mask) ** 2).mean()
        return self.loss

