"""Share prediction heads and loss functions between the teacher and student models."""

import torch
import torch.nn.functional as F
from torch import nn


class Mlp(nn.Sequential):
    def __init__(self, d_in: int, hidden: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )


def decode_palette(raw: torch.Tensor) -> torch.Tensor:
    """Convert (B, 15) values to (B, 5, 3) OKLab. Limit L to [0, 1] and a/b to [-0.4, 0.4]."""
    x = raw.view(-1, 5, 3)
    lightness = torch.sigmoid(x[..., :1])
    ab = 0.4 * torch.tanh(x[..., 1:])
    return torch.cat([lightness, ab], dim=-1)


def palette_loss(pred: torch.Tensor, target: torch.Tensor, lightness_weight: float) -> torch.Tensor:
    """Mean squared OKLab error plus sorted lightness error, both per channel."""
    slot = (pred - target).pow(2).sum(-1).mean()
    sorted_l = F.mse_loss(pred[..., 0].sort(-1).values, target[..., 0].sort(-1).values)
    return (slot + lightness_weight * sorted_l) / 3


def kl_logits(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL(teacher || student) over the last axis, scaled by T^2."""
    log_p = F.log_softmax(teacher / temperature, dim=-1)
    log_q = F.log_softmax(student / temperature, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
    return kl * temperature**2


class ChoiceHeads(nn.Module):
    def __init__(self, d_in: int, hidden: int, sizes: tuple[int, int, int], dropout: float) -> None:
        super().__init__()
        self.palette = Mlp(d_in, hidden, 15, dropout)
        self.light = Mlp(d_in, hidden, sizes[0], dropout)
        self.typeface = Mlp(d_in, hidden, sizes[1], dropout)
        self.scent = Mlp(d_in, hidden, sizes[2], dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return decode_palette(self.palette(x)), self.light(x), self.typeface(x), self.scent(x)
