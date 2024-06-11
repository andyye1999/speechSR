import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import numpy as np

class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

class LabelSmoothDirichletCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, alpha=None):
        with torch.no_grad():            
            if alpha is not None:
                # oh = torch.empty(size=(targets.size(0), n_classes),
                #                     device=targets.device) \
                #     .fill_(0).scatter_(1, targets.data.unsqueeze(1), 1.)
                # a_mu = alpha / alpha.sum()
                # aa = torch.gather(a_mu, 0, targets)
                # a = a_mu.view(1, -1).expand(targets.shape[0], -1)
                # aa = a[targets]
                # label_noise = torch.from_numpy(np.random.dirichlet(a_mu.cpu(), size=targets.shape[0])).to(targets.device)
                # idx = targets==1
                # targets[idx] == 1-label_noise[idx].float()
                # targets[torch.logical_not(idx)] == label_noise[torch.logical_not(idx)].float()
                targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(0.1 / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - 0.1)
            else:
                targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(0.1 / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - 0.1)
                
        return targets

    def forward(self, inputs, targets, alpha):        
        targets = LabelSmoothDirichletCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              alpha)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

if __name__ == '__main__':
    pass