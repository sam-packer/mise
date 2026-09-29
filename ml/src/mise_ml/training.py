"""Shared training schedule."""

import math

import torch


def cosine_schedule(optimizer, total: int, warmup_ratio: float):
    warmup = max(1, int(total * warmup_ratio))

    def scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        fraction = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1 + math.cos(math.pi * fraction))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
