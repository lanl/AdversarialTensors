import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    """
    A PyTorch learning rate scheduler that combines linear warmup and cosine annealing schedules.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rates are to be scheduled.
    warmup_epochs : int
        The maximum number of epochs for the linear warmup phase.
    max_epochs : int
        The maximum number of epochs for the cosine annealing phase.
    warmup_start_lr : float, optional
        The initial learning rate for the linear warmup phase. Default is 1e-8.
    eta_min : float, optional
        The minimum learning rate during the cosine annealing phase. Default is 1e-8.
    last_epoch : int, optional
        The index of the last epoch. Default is -1.

    Attributes
    ----------
    warmup_epochs : int
        Stores the maximum number of epochs for the warmup phase.
    max_epochs : int
        Stores the maximum number of epochs for the cosine annealing phase.
    warmup_start_lr : float
        Stores the initial learning rate for the warmup phase.
    eta_min : float
        Stores the minimum learning rate during the cosine annealing phase.

    Warnings
    --------
    1. Calling `.step()`: It's recommended to call `.step()` after each iteration. If called only after each epoch, the starting learning rate will remain at `warmup_start_lr` for the first epoch, which is often 0.
    2. Passing `epoch` to `.step()`: This is deprecated and triggers an EPOCH_DEPRECATION_WARNING. Make sure to call `.step()` before your training and validation routines when using this approach.

    Usage Example
    --------------
    .. code-block:: python

        # Initialize optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=0.01)
        scheduler = WarmupCosineLR(optimizer, warmup_epochs=5, max_epochs=50)

        # Training loop
        for epoch in range(50):
            # train(...)
            # validate(...)
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        last_epoch: int = -1,
    ) -> None:

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]

