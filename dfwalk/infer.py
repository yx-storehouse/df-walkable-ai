"""推理：图片 / 图片目录 / 录像 → 可走区域掩码与叠加可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from torch import nn

from . import IMAGENET_MEAN, IMAGENET_STD


class Predictor:
    """封装模型推理。输入任意分辨率 RGB 图像，内部缩放到 image_size，输出原分辨率结果。"""

    def __init__(self, model: nn.Module, image_size: Sequence[int], mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD, device="cpu") -> None:
        """model 移到 device 并 eval()。"""
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Predictor":
        """dfwalk.checkpoint.load_checkpoint + model_from_checkpoint；image_size/mean/std 取自 checkpoint。"""
        raise NotImplementedError

    def predict_prob(self, image_rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB → HxW float32 可走概率（softmax 第 WALKABLE 通道），原图分辨率。
        预处理与 transforms.EvalTransform 完全一致（复用 EvalTransform）；概率图双线性插值回原尺寸。"""
        raise NotImplementedError

    def predict_mask(self, image_rgb: np.ndarray, threshold: float = 0.5,
                     hud_rects: Sequence[Sequence[float]] = ()) -> np.ndarray:
        """HxW uint8：prob >= threshold → 1，否则 0；hud_rects 区域置为 255。"""
        raise NotImplementedError


def overlay(image_rgb: np.ndarray, mask: np.ndarray, color: Sequence[int] = (0, 255, 0),
            alpha: float = 0.45) -> np.ndarray:
    """返回叠加图（HxWx3 uint8 RGB，不修改输入）：mask == 1 的像素与 color 按 alpha 混合，
    并用同色 2px 描出可走区域轮廓；mask == 255 的像素整体压暗 50%（标示 HUD/忽略区）。"""
    raise NotImplementedError


def infer_images(
    predictor: Predictor,
    inputs: Sequence[str | Path],
    out_dir: str | Path,
    threshold: float = 0.5,
    alpha: float = 0.45,
    color: Sequence[int] = (0, 255, 0),
    hud_rects: Sequence[Sequence[float]] = (),
    save_mask: bool = True,
) -> list[Path]:
    """对每张图片写 out_dir/{stem}_overlay.jpg，save_mask 时再写 out_dir/{stem}_mask.png（0/1/255）。
    返回写出的叠加图路径列表。显示 tqdm 进度条。"""
    raise NotImplementedError


def infer_video(
    predictor: Predictor,
    video_path: str | Path,
    out_path: str | Path,
    threshold: float = 0.5,
    alpha: float = 0.45,
    color: Sequence[int] = (0, 255, 0),
    hud_rects: Sequence[Sequence[float]] = (),
    every_n: int = 1,
    max_frames: int | None = None,
) -> Path:
    """逐帧推理并写出叠加视频（cv2.VideoWriter，fourcc "mp4v"，与源视频相同 fps（读不到按 30）和分辨率）。
    every_n > 1 时每 every_n 帧推理一次，中间帧复用上一次的掩码。max_frames 限制处理的帧数。
    视频打不开或 writer 打不开时抛 ValueError。返回 out_path。"""
    raise NotImplementedError
