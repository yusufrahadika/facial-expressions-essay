import math
from torch.optim.lr_scheduler import _LRScheduler


class FlatCosineAnnealing(_LRScheduler):
    def __init__(self, optimizer, max_iter, step_size=0.75, eta_min=0, last_epoch=-1):
        self.flat_range = int(max_iter * step_size)
        self.T_max = max_iter - self.flat_range
        self.eta_min = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.flat_range:
            return [base_lr for base_lr in self.base_lrs]
        else:
            cr_epoch = self.last_epoch - self.flat_range
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (cr_epoch / self.T_max)))
                / 2
                for base_lr in self.base_lrs
            ]
