from torch import nn, tanh
from torch.nn import functional as F


class Mish(nn.Module):
    def forward(self, input):
        return input * tanh(F.softplus(input))
