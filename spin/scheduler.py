import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineSchedulerWithRestarts(LambdaLR):

    def __init__(self, optimizer: Optimizer,
                 num_warmup_steps: int,
                 num_training_steps: int,
                 min_factor: float = 0.1,
                 linear_decay: float = 0.67,
                 num_cycles: int = 1,
                 last_epoch: int = -1):
        """From https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/optimization.py#L138

        Create a schedule with a learning rate that decreases following the values
        of the cosine function between the initial lr set in the optimizer to 0,
        with several hard restarts, after a warmup period during which it increases
        linearly between 0 and the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            num_cycles (`int`, *optional*, defaults to 1):
                The number of hard restarts to use.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.
        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                factor = float(current_step) / float(max(1, num_warmup_steps))
                return max(min_factor, factor)
            progress = float(current_step - num_warmup_steps)
            progress /= float(max(1, num_training_steps - num_warmup_steps))
            if progress >= 1.0:
                return 0.0
            factor = (float(num_cycles) * progress) % 1.0
            cos = 0.5 * (1.0 + math.cos(math.pi * factor))
            lin = 1.0 - (progress * linear_decay)
            return max(min_factor, cos * lin)

        super(CosineSchedulerWithRestarts, self).__init__(optimizer, lr_lambda,
                                                          last_epoch)
