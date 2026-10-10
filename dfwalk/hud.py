"""HUD 区域处理：HUD 像素在标注、训练和评估中一律视为忽略 (IGNORE_INDEX=255)。

矩形为归一化坐标 [x0, y0, x1, y1]（0~1，左上角原点，x 为列方向，y 为行方向）。
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from . import IGNORE_INDEX

# 归一化坐标乘以像素尺寸时的浮点误差容差（单位：像素）。
# 例如 0.3 * 100 = 30.000000000000004，若直接 ceil 会多覆盖 1 列；先减去容差再 ceil 得到 30。
_PIXEL_EPS = 1e-9


def rect_to_pixels(rect: Sequence[float], height: int, width: int) -> tuple[int, int, int, int]:
    """归一化矩形 → 像素范围 (c0, r0, c1, r1)，半开区间 [c0, c1) x [r0, r1)。
    起点向下取整 (floor)、终点向上取整 (ceil)，保证 HUD 被完整覆盖；结果裁剪到图像范围内。

    - 取整前加/减一个极小容差，消除 0.3 * 100 = 30.000000000000004 这类浮点误差带来的多余一行/列。
    - 超出 0~1 的坐标会被裁剪；完全在图像外或 x0 >= x1 的矩形得到空区间（c0 == c1 或 r0 == r1）。
    - rect 不是 4 个数、含 NaN/inf、或 height/width 为负时抛 ValueError。
    """
    if len(rect) != 4:
        raise ValueError(f"HUD 矩形必须是 [x0, y0, x1, y1]，得到 {list(rect)}")
    height, width = int(height), int(width)
    if height < 0 or width < 0:
        raise ValueError(f"图像尺寸不能为负: height={height}, width={width}")
    x0, y0, x1, y1 = (float(v) for v in rect)
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        raise ValueError(f"HUD 矩形坐标必须是有限数，得到 {list(rect)}")

    c0 = math.floor(x0 * width + _PIXEL_EPS)
    r0 = math.floor(y0 * height + _PIXEL_EPS)
    c1 = math.ceil(x1 * width - _PIXEL_EPS)
    r1 = math.ceil(y1 * height - _PIXEL_EPS)

    # 裁剪到图像范围，并保证终点不小于起点（空矩形表示为零宽/零高）
    c0 = min(max(c0, 0), width)
    r0 = min(max(r0, 0), height)
    c1 = min(max(c1, c0), width)
    r1 = min(max(r1, r0), height)
    return c0, r0, c1, r1


def hud_mask(height: int, width: int, rects: Sequence[Sequence[float]]) -> np.ndarray:
    """返回 HxW bool 数组，HUD 矩形内为 True。rects 为空时全 False。多个矩形取并集。"""
    mask = np.zeros((int(height), int(width)), dtype=bool)
    for rect in _iter_rects(rects):
        c0, r0, c1, r1 = rect_to_pixels(rect, height, width)
        mask[r0:r1, c0:c1] = True
    return mask


def apply_hud_ignore(mask: np.ndarray, rects: Sequence[Sequence[float]]) -> np.ndarray:
    """返回 mask 的副本，HUD 矩形内的像素置为 IGNORE_INDEX。不修改输入。

    mask 一般为 HxW uint8（值域 {0, 1, 255}）；输出保持输入的 dtype 和形状。
    """
    out = np.array(mask, copy=True)
    if out.ndim < 2:
        raise ValueError(f"掩码至少是二维 HxW，得到 shape={out.shape}")
    height, width = out.shape[:2]
    for rect in _iter_rects(rects):
        c0, r0, c1, r1 = rect_to_pixels(rect, height, width)
        out[r0:r1, c0:c1] = IGNORE_INDEX
    return out


def draw_hud_preview(
    image_rgb: np.ndarray,
    rects: Sequence[Sequence[float]],
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
    thickness: int = 2,
) -> np.ndarray:
    """在图像副本上画出 HUD 矩形：半透明填充 + 边框 + 序号文字（从 0 开始，对应配置中的顺序），
    用于核对 hud.rects 配置。返回 HxWx3 uint8 RGB。

    - 不修改输入；即使 rects 为空也返回新数组。
    - 半透明填充按所有矩形的并集只混合一次，重叠区域不会叠加变深。
    - alpha 会被裁剪到 [0, 1]；thickness <= 0 时不画边框。
    - 输入也可以是 HxW 灰度图或 HxWx4（丢弃 alpha 通道），统一转为 HxWx3 输出。
    """
    out = _to_rgb_uint8(image_rgb)
    height, width = out.shape[:2]
    rect_list = list(_iter_rects(rects))
    if not rect_list or height == 0 or width == 0:
        return out

    if len(color) != 3:
        raise ValueError(f"color 必须是 (R, G, B)，得到 {color}")
    color_px = tuple(int(np.clip(c, 0, 255)) for c in color)
    a = float(np.clip(alpha, 0.0, 1.0))

    # 1) 半透明填充（并集只混合一次）
    region = hud_mask(height, width, rect_list)
    if region.any():
        blended = out[region].astype(np.float32) * (1.0 - a) + np.asarray(color_px, np.float32) * a
        out[region] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    # 2) 边框与序号
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(height, width) / 600.0)
    text_thick = max(1, int(round(font_scale * 1.5)))
    border = max(int(thickness), 0)
    for idx, rect in enumerate(rect_list):
        c0, r0, c1, r1 = rect_to_pixels(rect, height, width)
        if c1 <= c0 or r1 <= r0:
            continue  # 空矩形（完全在画面外等）不画
        if border > 0:
            # cv2.rectangle 的终点是闭区间，所以用 c1-1 / r1-1
            cv2.rectangle(out, (c0, r0), (c1 - 1, r1 - 1), color_px, border, lineType=cv2.LINE_8)

        text = str(idx)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thick)
        # 文字放在矩形左上角内侧，并保证落在画面内
        x = min(max(c0 + border + 2, 0), max(width - tw - 1, 0))
        y = min(max(r0 + border + 2 + th, th), max(height - 1 - baseline, th))
        # 黑色描边 + 白色文字，任何背景色下都清晰
        cv2.putText(out, text, (x, y), font, font_scale, (0, 0, 0), text_thick + 2, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), font, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA)
    return out


def _iter_rects(rects: Sequence[Sequence[float]] | None):
    """rects 为 None 时视为空；兼容 list / tuple / numpy 数组。"""
    if rects is None:
        return ()
    return rects


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """复制并规整为 C 连续的 HxWx3 uint8（cv2 绘图函数需要）。"""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3:
        raise ValueError(f"图像必须是 HxW 或 HxWxC，得到 shape={img.shape}")
    channels = img.shape[2]
    if channels == 1:
        img = np.repeat(img, 3, axis=2)
    elif channels == 4:
        img = img[:, :, :3]
    elif channels != 3:
        raise ValueError(f"图像通道数必须是 1/3/4，得到 {channels}")
    return np.array(img, dtype=np.uint8, copy=True, order="C")
