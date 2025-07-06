import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, loss_type='l2', eps=1e-6):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps

    def forward(self, x, target):
        diff = x - target
        if self.loss_type == 'l2':
            return torch.mean(torch.sqrt(self.eps + torch.sum(diff ** 2, (1, 2, 3))))
        elif self.loss_type == 'l1':
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        else:
            assert False, f"loss_type '{self.loss_type}' is not supported.)"
