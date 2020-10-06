import torch
from torch import nn
from torch.nn import functional as F


class RademacherDropout(nn.Module):
    def forward(self, input):
        S = torch.norm(input, dim=0)
        max_Sj = torch.max(S)
        q = 1 - torch.bernoulli(S / max_Sj)
        
        return input * q
