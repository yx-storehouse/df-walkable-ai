"""损失函数：交叉熵 + 可走类的 soft Dice，二者都忽略 IGNORE_INDEX 像素。"""

from __future__ import annotations

import torch
from torch import nn

from . import IGNORE_INDEX, WALKABLE


class SegLoss(nn.Module):
    """loss = ce_weight * CE + dice_weight * (1 - Dice_walkable)

    - CE：F.cross_entropy(ignore_index=ignore_index)，按有效像素求均值。
    - Dice：p = softmax(logits)[:, WALKABLE]，g = (target == WALKABLE)，只在有效像素（target != ignore）上计算
      整个 batch 的 soft Dice = (2 * sum(p*g) + smooth) / (sum(p) + sum(g) + smooth)。
    - batch 中所有像素都被忽略时：返回 0（且可反传，梯度为 0），不能返回 NaN。
    - logits: (N, C, H, W) float；target: (N, H, W) int64。
    """

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 ignore_index: int = IGNORE_INDEX, smooth: float = 1.0) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
