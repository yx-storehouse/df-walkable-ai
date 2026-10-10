"""训练/评估的数据变换（图像与掩码同步），只依赖 numpy/cv2/torch。

随机数一律使用 Python 标准库 random 模块：PyTorch DataLoader 会为每个 worker 重新设置 random 的种子，
而 numpy 的全局 RNG 在各 worker 中会重复，导致不同 worker 产生相同的增强。

输出约定：
    image: torch.float32 (3, H, W)，已按 mean/std 归一化
    mask:  torch.int64 (H, W)，值域 {0, 1, 255}；输入 mask 为 None 时输出 None
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from . import IMAGENET_MEAN, IMAGENET_STD

Transform = Callable[[np.ndarray, "np.ndarray | None"], tuple]


def to_tensor_normalized(image_rgb: np.ndarray, mean: Sequence[float] = IMAGENET_MEAN,
                         std: Sequence[float] = IMAGENET_STD):
    """HxWx3 uint8 RGB → float32 张量 (3, H, W)：/255 后按 mean/std 归一化。"""
    raise NotImplementedError


class EvalTransform:
    """把图像缩放到 image_size=(H, W)（cv2.INTER_LINEAR；缩小时 INTER_AREA），掩码用 INTER_NEAREST。"""

    def __init__(self, image_size: Sequence[int], mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD) -> None:
        raise NotImplementedError

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple:
        raise NotImplementedError


class TrainTransform:
    """训练增强，按顺序：
    1. 随机缩放：s ~ U(scale)，把图像整体缩放到 (round(H*s), round(W*s))（与原图宽高比无关，直接拉伸到目标比例），
       图像 INTER_LINEAR，掩码 INTER_NEAREST。
    2. 若小于 image_size 则在右/下方填充（图像填 0，掩码填 IGNORE_INDEX），再随机裁剪到 image_size。
    3. 以概率 hflip_p 水平翻转（图像与掩码同步）。
    4. 仅对图像做颜色抖动：亮度、对比度、饱和度各乘以 U(1-jitter, 1+jitter)，色相偏移 U(-hue, hue)（单位：圈，
       hue=0.03 约等于 ±11°），结果裁剪到 [0, 255]。
    5. 以概率 blur_p 做 3x3 高斯模糊（模拟运动模糊/低画质）。
    6. to_tensor_normalized。
    """

    def __init__(
        self,
        image_size: Sequence[int],
        scale: tuple[float, float] = (0.75, 1.25),
        hflip_p: float = 0.5,
        jitter: float = 0.3,
        hue: float = 0.03,
        blur_p: float = 0.2,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        raise NotImplementedError

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple:
        raise NotImplementedError
