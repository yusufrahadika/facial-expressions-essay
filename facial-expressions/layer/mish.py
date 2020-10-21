import torch
from torch import nn, tanh
from torch.nn import functional as F


class Mish(nn.Module):
    def forward(self, input):
        return input * tanh(F.softplus(input))


class MemoryEfficientMishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.tanh(F.softplus(x))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MemoryEfficientMishFunction.apply(x)
