import math
import warnings
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        warmup_steps: int = 0,
        decay_steps: int = 0,
        eta_min=1e-7,
        last_step=-1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.stable_steps = max_steps - warmup_steps - decay_steps
        self.decay_steps = decay_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        base_lrs = torch.tensor(self.base_lrs)

        if self.last_epoch < self.warmup_steps:
            lr = 0.5 * (
                (self.eta_min - base_lrs)
                * math.cos(math.pi * self.last_epoch / self.warmup_steps)
                + base_lrs
                + self.eta_min
            )
            return lr
        elif self.last_epoch < self.warmup_steps + self.stable_steps:
            return base_lrs
        else:
            lr = 0.5 * (
                (base_lrs - self.eta_min)
                * math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_steps - self.stable_steps)
                    / self.decay_steps
                )
                + base_lrs
                + self.eta_min
            )
            return lr
