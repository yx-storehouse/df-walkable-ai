"""分割评估指标：基于混淆矩阵累计，忽略 IGNORE_INDEX 像素。

只依赖 numpy；torch.Tensor 通过鸭子类型（.detach().cpu().numpy()）转换，本模块不导入 torch。
"""

from __future__ import annotations

import math

import numpy as np

from . import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, WALKABLE


def _to_numpy(x, name: str) -> np.ndarray:
    """torch.Tensor（任意设备）/ np.ndarray / 嵌套列表 → np.ndarray。"""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch.Tensor
        return x.detach().cpu().numpy()
    try:
        return np.asarray(x)
    except Exception as e:  # pragma: no cover - 极少见的不可转换对象
        raise TypeError(f"{name} 无法转换为数组: {type(x).__name__}") from e


def _as_int64(a: np.ndarray, name: str) -> np.ndarray:
    if a.dtype == np.bool_:
        return a.astype(np.int64)
    if not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"{name} 必须是整数（类别索引）数组/张量，得到 dtype={a.dtype}")
    return a.astype(np.int64, copy=False)


def _safe_div(num: float, den: float) -> float:
    """den == 0 时返回 NaN，不触发 numpy 的除零警告。"""
    return float(num) / float(den) if den != 0 else math.nan


class ConfusionMatrix:
    """matrix[t, p] = 真实类别 t 被预测为 p 的像素数（int64）。

    update(pred, target)：pred/target 为相同形状的整数 torch.Tensor 或 np.ndarray（任意维度）；
    target == ignore_index 的像素跳过；pred 或 target 出现 [0, num_classes) 以外的值（ignore 除外）时抛 ValueError。

    补充约定：
    - 形状不同抛 ValueError；浮点 dtype 抛 TypeError；bool 视为 0/1；空数组什么都不做。
    - "ignore 除外" 的含义：值等于 ignore_index 不算越界。target 中的 ignore 像素整体跳过，此处的 pred 值
      只要在 [0, num_classes) 或等于 ignore_index 即可；但在 target 有效的像素上 pred == ignore_index 无法计数，
      同样抛 ValueError（预测结果不应含忽略值）。
    - 校验失败时混淆矩阵保持不变（先校验后累计）。
    - num_classes >= 2（WALKABLE=1 必须是合法类别），ignore_index 不能落在 [0, num_classes) 内。
    """

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> None:
        num_classes = int(num_classes)
        ignore_index = int(ignore_index)
        if num_classes <= WALKABLE:
            raise ValueError(f"num_classes 必须 > {WALKABLE}（WALKABLE 必须是合法类别），得到 {num_classes}")
        if 0 <= ignore_index < num_classes:
            raise ValueError(f"ignore_index={ignore_index} 不能是合法类别 [0, {num_classes})")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = tuple(CLASS_NAMES[:num_classes]) + tuple(
            f"class{i}" for i in range(len(CLASS_NAMES), num_classes)
        )
        self._matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def __repr__(self) -> str:
        return (f"ConfusionMatrix(num_classes={self.num_classes}, ignore_index={self.ignore_index}, "
                f"total={int(self._matrix.sum())})")

    def reset(self) -> None:
        self._matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target) -> None:
        p = _to_numpy(pred, "pred")
        t = _to_numpy(target, "target")
        if p.shape != t.shape:
            raise ValueError(f"pred 与 target 形状必须相同，得到 {p.shape} 与 {t.shape}")
        p = _as_int64(p, "pred").reshape(-1)
        t = _as_int64(t, "target").reshape(-1)
        if t.size == 0:
            return
        c = self.num_classes

        valid = t != self.ignore_index
        tv = t[valid]
        pv = p[valid]
        if tv.size:
            if tv.min() < 0 or tv.max() >= c:
                bad = np.unique(tv[(tv < 0) | (tv >= c)])
                raise ValueError(f"target 含越界值 {bad[:10].tolist()}，合法范围 [0, {c}) 或 {self.ignore_index}")
            if pv.min() < 0 or pv.max() >= c:
                bad = np.unique(pv[(pv < 0) | (pv >= c)])
                raise ValueError(f"pred 在有效像素上含越界值 {bad[:10].tolist()}，合法范围 [0, {c})")
        if tv.size < t.size:
            pi = p[~valid]
            bad_mask = ((pi < 0) | (pi >= c)) & (pi != self.ignore_index)
            if bad_mask.any():
                bad = np.unique(pi[bad_mask])
                raise ValueError(f"pred 含越界值 {bad[:10].tolist()}，合法范围 [0, {c}) 或 {self.ignore_index}")
        if tv.size == 0:
            return

        counts = np.bincount(tv * c + pv, minlength=c * c)
        self._matrix += counts.reshape(c, c).astype(np.int64, copy=False)

    @property
    def matrix(self) -> np.ndarray:
        """混淆矩阵的副本（修改返回值不会影响内部累计）。"""
        return self._matrix.copy()

    def compute(self) -> dict[str, float]:
        """返回：
        iou_non_walkable, iou_walkable：IoU = TP / (TP + FP + FN)，分母为 0 时为 NaN
        miou：上述两项的 nanmean（全为 NaN 时为 NaN）
        pixel_acc：对角线和 / 总数，总数为 0 时为 NaN
        precision_walkable, recall_walkable, f1_walkable：分母为 0 时为 NaN
        所有值为 Python float。

        细节：
        - precision = TP / (TP + FP)，recall = TP / (TP + FN)（均针对 WALKABLE 类）。
        - f1 = 2·TP / (2·TP + FP + FN)：TP > 0 时与 2PR/(P+R) 完全相等；它只在 TP = FP = FN = 0
          （真值和预测里都没有可走像素）时为 NaN。模型一个可走像素都没预测对（TP = 0 而 FP + FN > 0）时
          f1 = 0 而不是 NaN，便于比较模型优劣。
        - num_classes > 2 时额外类别的 IoU 键名为 iou_class{i}，miou 为所有类别 IoU 的 nanmean。
        - 计算过程不产生 numpy 警告。
        """
        m = self._matrix
        tp = np.diag(m)
        fp = m.sum(axis=0) - tp
        fn = m.sum(axis=1) - tp

        out: dict[str, float] = {}
        ious = []
        for i, name in enumerate(self.class_names):
            iou = _safe_div(tp[i], tp[i] + fp[i] + fn[i])
            out[f"iou_{name}"] = iou
            ious.append(iou)
        finite = [v for v in ious if not math.isnan(v)]
        out["miou"] = float(sum(finite) / len(finite)) if finite else math.nan
        out["pixel_acc"] = _safe_div(tp.sum(), m.sum())

        w_tp, w_fp, w_fn = int(tp[WALKABLE]), int(fp[WALKABLE]), int(fn[WALKABLE])
        out["precision_walkable"] = _safe_div(w_tp, w_tp + w_fp)
        out["recall_walkable"] = _safe_div(w_tp, w_tp + w_fn)
        out["f1_walkable"] = _safe_div(2 * w_tp, 2 * w_tp + w_fp + w_fn)
        return out
