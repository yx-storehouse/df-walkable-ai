"""分割评估指标：基于混淆矩阵累计，忽略 IGNORE_INDEX 像素。"""

from __future__ import annotations

import numpy as np

from . import IGNORE_INDEX, NUM_CLASSES


class ConfusionMatrix:
    """matrix[t, p] = 真实类别 t 被预测为 p 的像素数（int64）。

    update(pred, target)：pred/target 为相同形状的整数 torch.Tensor 或 np.ndarray（任意维度）；
    target == ignore_index 的像素跳过；pred 或 target 出现 [0, num_classes) 以外的值（ignore 除外）时抛 ValueError。
    """

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, pred, target) -> None:
        raise NotImplementedError

    @property
    def matrix(self) -> np.ndarray:
        raise NotImplementedError

    def compute(self) -> dict[str, float]:
        """返回：
        iou_non_walkable, iou_walkable：IoU = TP / (TP + FP + FN)，分母为 0 时为 NaN
        miou：上述两项的 nanmean（全为 NaN 时为 NaN）
        pixel_acc：对角线和 / 总数，总数为 0 时为 NaN
        precision_walkable, recall_walkable, f1_walkable：分母为 0 时为 NaN
        所有值为 Python float。
        """
        raise NotImplementedError
