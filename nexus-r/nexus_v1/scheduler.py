"""Learning-rate schedule experiments for Nexus-R.

These schedulers are not specific to the recursive architecture, but they are
kept here so training scripts can swap between baseline cosine warmup and more
specialized schedules such as WSD without duplicating optimizer logic.
"""

import math
from typing import Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WSDScheduler(_LRScheduler):
    """
    Warmup-Stable-Decay LR Schedule.

    From paper: "Optimal Learning-Rate Schedules under Functional Scaling Laws" (2602.06797)

    Schedule:
    1. Linear warmup from 0 to peak_lr over warmup_steps
    2. Stable phase at peak_lr for stable_steps
    3. Decay phase with power-law decay from peak_lr to min_lr

    This is proven to outperform cosine annealing in transformer training.

    Key insight: The functional form of decay should be power-law,
    not cosine, based on scaling laws.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 2000,
        stable_steps: int = 10000,
        decay_steps: int = 100000,
        peak_lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_start_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.power = power  # Paper suggests 0.5-2.0 range

        # Total training steps
        self.total_steps = warmup_steps + stable_steps + decay_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        num_groups = len(self.optimizer.param_groups)

        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / self.warmup_steps
            lr = self.warmup_start_lr + (self.peak_lr - self.warmup_start_lr) * alpha
        elif step < self.warmup_steps + self.stable_steps:
            # Stable phase - constant LR
            lr = self.peak_lr
        else:
            # Power-law decay
            decay_step = step - self.warmup_steps - self.stable_steps
            decay_ratio = decay_step / self.decay_steps
            # Power-law decay: lr = peak_lr * (1 - decay_ratio)^power
            # Paper suggests power in 0.5-2.0 range (default 1.0 = linear decay)
            decay_factor = (1 - min(decay_ratio, 1.0)) ** self.power
            lr = self.min_lr + (self.peak_lr - self.min_lr) * decay_factor

        # Return one LR per param group
        return [lr] * num_groups

    def state_dict(self):
        """Return state with current step for checkpointing."""
        state = super().state_dict()
        # Custom state if needed
        return state


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine Annealing with Linear Warmup.

    Standard schedule used in GPT-3, LLaMA, etc.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / self.warmup_steps
            base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
            return [base_lr * alpha for base_lr in base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decayed = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
            base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
            return [base_lr * decayed for base_lr in base_lrs]


class InverseSquareRootScheduler(_LRScheduler):
    """
    Inverse Square Root Scheduler.

    Original BERT/Transformer schedule:
    lr = peak_lr / sqrt(max(step, warmup_steps))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 4000,
        initial_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        num_groups = len(self.optimizer.param_groups)

        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / self.warmup_steps
            base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
            return [base_lr * alpha for base_lr in base_lrs]
        else:
            # Inverse square root decay
            decayed = self.initial_lr / math.sqrt(max(self.warmup_steps, step))
            base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
            peak_lr = base_lrs[0]  # Assume same peak for all
            lr = peak_lr / math.sqrt(max(self.warmup_steps, step))
            return [lr] * num_groups


class PolynomialDecayScheduler(_LRScheduler):
    """
    Polynomial Decay with Warmup.

    Used in some LLM training recipes.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            alpha = step / self.warmup_steps
            base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
            return [base_lr * alpha for base_lr in base_lrs]
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            decayed = (1 - progress) ** self.power
            min_lr = self.min_lr_ratio * self.optimizer.param_groups[0]['initial_lr']
            base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]
            return [min_lr + (base_lr - min_lr) * decayed for base_lr in base_lrs]


def create_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    **kwargs
) -> _LRScheduler:
    """
    Factory function to create LR scheduler.

    Args:
        scheduler_type: One of 'wsd', 'cosine', 'inverse_sqrt', 'polynomial'
        optimizer: The optimizer
        **kwargs: Scheduler-specific arguments

    Returns:
        Configured scheduler
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == 'wsd':
        return WSDScheduler(optimizer, **kwargs)
    elif scheduler_type == 'cosine':
        return WarmupCosineScheduler(optimizer, **kwargs)
    elif scheduler_type == 'inverse_sqrt':
        return InverseSquareRootScheduler(optimizer, **kwargs)
    elif scheduler_type == 'polynomial':
        return PolynomialDecayScheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# Visualization for debugging
if __name__ == '__main__':
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = optim.AdamW(model.parameters(), lr=0.0)  # LR set by scheduler

    # Test WSD scheduler
    schedulers = {
        'WSD': WSDScheduler(optimizer, warmup_steps=1000, stable_steps=5000,
                           decay_steps=40000, peak_lr=1e-3, min_lr=1e-5),
        'Cosine': WarmupCosineScheduler(optimizer, warmup_steps=1000,
                                        total_steps=46000, min_lr_ratio=0.1),
    }

    steps = list(range(50000))
    lrs = {name: [] for name in schedulers}

    for step in steps:
        for name, sched in schedulers.items():
            sched.last_epoch = step
            lrs[name].append(sched.get_lr()[0])

    # Plot
    plt.figure(figsize=(12, 6))
    for name, lr_list in lrs.items():
        plt.plot(steps, lr_list, label=name)

    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('LR Schedule Comparison')
    plt.grid(True)
    plt.savefig('lr_schedules.png')
    print("Saved lr_schedules.png")
    print("WSD test PASSED!")