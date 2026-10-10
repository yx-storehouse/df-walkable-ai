"""损失函数：交叉熵 + 可走类的 soft Dice，二者都忽略 IGNORE_INDEX 像素。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import IGNORE_INDEX, WALKABLE


class SegLoss(nn.Module):
    """loss = ce_weight * CE + dice_weight * (1 - Dice_walkable)

    - CE：F.cross_entropy(ignore_index=ignore_index)，按有效像素求均值。
    - Dice：p = softmax(logits)[:, WALKABLE]，g = (target == WALKABLE)，只在有效像素（target != ignore）上计算
      整个 batch 的 soft Dice = (2 * sum(p*g) + smooth) / (sum(p) + sum(g) + smooth)。
    - batch 中所有像素都被忽略时：返回 0（且可反传，梯度为 0），不能返回 NaN。
    - logits: (N, C, H, W) float；target: (N, H, W) int64。

    实现细节：
    - 不做数据相关的分支（没有 .item() / if valid.any()），CUDA 上不会每步强制同步：
      CE = sum(有效像素的 CE) / max(有效像素数, 1)，全部忽略时分子为 0 → 结果恰为 0；
      Dice 项在全部忽略时乘以 0 屏蔽（smooth > 0 时它本身也恰为 1 - 1 = 0）。
    - 计算统一转成 float32（AMP 下 logits 可能是 float16，大 batch 的 sum(p) 会超出 float16 范围）。
    - 被忽略像素的 logits 对 loss 没有任何影响，梯度也恰为 0（即使这些位置是 inf/NaN）。
    - target 允许任意整数 dtype（如直接由 uint8 掩码得到），内部转 int64；浮点 target 抛 TypeError。
    - 权重为 0 的项直接跳过计算；ce_weight、dice_weight、smooth 不能为负（ValueError）。
    """

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 ignore_index: int = IGNORE_INDEX, smooth: float = 1.0) -> None:
        super().__init__()
        for name, value in (("ce_weight", ce_weight), ("dice_weight", dice_weight), ("smooth", smooth)):
            if not float(value) >= 0.0:  # 同时拦住 NaN
                raise ValueError(f"{name} 必须 >= 0，得到 {value!r}")
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)

    def extra_repr(self) -> str:
        return (f"ce_weight={self.ce_weight}, dice_weight={self.dice_weight}, "
                f"ignore_index={self.ignore_index}, smooth={self.smooth}")

    def _check_inputs(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"logits 必须是 (N, C, H, W)，得到 shape={tuple(logits.shape)}")
        if not logits.is_floating_point():
            raise TypeError(f"logits 必须是浮点张量，得到 dtype={logits.dtype}")
        if logits.shape[1] <= WALKABLE:
            raise ValueError(f"logits 的类别维 C={logits.shape[1]} 必须大于 WALKABLE={WALKABLE}")
        if target.ndim != 3:
            raise ValueError(f"target 必须是 (N, H, W)，得到 shape={tuple(target.shape)}")
        n, _, h, w = logits.shape
        if tuple(target.shape) != (n, h, w):
            raise ValueError(f"target shape {tuple(target.shape)} 与 logits shape {tuple(logits.shape)} 不匹配")
        if target.is_floating_point() or target.is_complex():
            raise TypeError(f"target 必须是整数张量（类别索引），得到 dtype={target.dtype}")
        if target.dtype != torch.long:
            target = target.long()
        return target

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = self._check_inputs(logits, target)
        valid = target != self.ignore_index
        n_valid = valid.sum()
        # 先把被忽略像素的 logits 置 0：where 的反向只做选择不做乘法，
        # 因此即使这些位置是 inf/NaN，它们的梯度也恰为 0，不会把 NaN 传回网络
        logits = torch.where(valid.unsqueeze(1), logits.float(), 0.0)
        # 空切片求和：值恰为 0.0 且挂在计算图上（backward 得到全 0 梯度）。
        # 两个权重都为 0 时直接返回它，保证 loss.backward() 依然可用
        loss = logits.reshape(-1)[:0].sum()

        if self.ce_weight > 0.0:
            ce_sum = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="sum")
            ce = ce_sum / n_valid.clamp(min=1).to(ce_sum.dtype)
            loss = loss + self.ce_weight * ce

        if self.dice_weight > 0.0:
            prob = torch.where(valid, torch.softmax(logits, dim=1)[:, WALKABLE], 0.0)
            gt = ((target == WALKABLE) & valid).to(logits.dtype)
            inter = (prob * gt).sum()
            denom = prob.sum() + gt.sum() + self.smooth
            # smooth=0 且全部忽略时 denom=0：钳到极小正数避免 0/0，再由 has_valid 屏蔽
            dice = (2.0 * inter + self.smooth) / denom.clamp(min=torch.finfo(denom.dtype).tiny)
            has_valid = (n_valid > 0).to(logits.dtype)
            loss = loss + self.dice_weight * (1.0 - dice) * has_valid

        return loss
