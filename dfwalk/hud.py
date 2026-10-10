"""HUD 区域处理：HUD 像素在标注、训练和评估中一律视为忽略 (IGNORE_INDEX=255)。

矩形为归一化坐标 [x0, y0, x1, y1]（0~1，左上角原点，x 为列方向，y 为行方向）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rect_to_pixels(rect: Sequence[float], height: int, width: int) -> tuple[int, int, int, int]:
    """归一化矩形 → 像素范围 (c0, r0, c1, r1)，半开区间 [c0, c1) x [r0, r1)。
    起点向下取整 (floor)、终点向上取整 (ceil)，保证 HUD 被完整覆盖；结果裁剪到图像范围内。"""
    raise NotImplementedError


def hud_mask(height: int, width: int, rects: Sequence[Sequence[float]]) -> np.ndarray:
    """返回 HxW bool 数组，HUD 矩形内为 True。rects 为空时全 False。"""
    raise NotImplementedError


def apply_hud_ignore(mask: np.ndarray, rects: Sequence[Sequence[float]]) -> np.ndarray:
    """返回 mask 的副本，HUD 矩形内的像素置为 IGNORE_INDEX。不修改输入。"""
    raise NotImplementedError


def draw_hud_preview(
    image_rgb: np.ndarray,
    rects: Sequence[Sequence[float]],
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
    thickness: int = 2,
) -> np.ndarray:
    """在图像副本上画出 HUD 矩形：半透明填充 + 边框 + 序号文字（从 0 开始，对应配置中的顺序），
    用于核对 hud.rects 配置。返回 HxWx3 uint8 RGB。"""
    raise NotImplementedError
