"""公共小工具：设备选择、随机种子、图像/掩码读写。

图像读写统一走 np.fromfile + cv2.imdecode / cv2.imencode + tofile，
避免 Windows 下中文路径导致 cv2.imread 返回 None。
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from . import IMAGE_EXTS


def resolve_device(name: str = "auto"):
    """"auto" → cuda > mps > cpu；其余字符串原样交给 torch.device。"""
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_images(folder: str | Path) -> list[Path]:
    """列出目录下（不递归）的图片文件，按文件名排序。"""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_image_rgb(path: str | Path) -> np.ndarray:
    """读取图片为 HxWx3 uint8 RGB。读取失败抛 FileNotFoundError / ValueError。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图片: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image_rgb(path: str | Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    """写 HxWx3 uint8 RGB 图片，格式由后缀决定；自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)] if path.suffix.lower() in (".jpg", ".jpeg") else []
    ok, enc = cv2.imencode(path.suffix, bgr, params)
    if not ok:
        raise ValueError(f"无法编码图片: {path}")
    enc.tofile(str(path))


def read_mask(path: str | Path) -> np.ndarray:
    """读取单通道 uint8 掩码 PNG（值域 {0, 1, 255}）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    buf = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"无法解码掩码: {path}")
    if mask.ndim != 2:
        raise ValueError(f"掩码必须是单通道 PNG: {path}, shape={mask.shape}")
    return mask.astype(np.uint8, copy=False)


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    """写单通道 uint8 掩码为无损 PNG；自动创建父目录。"""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"掩码必须保存为 .png: {path}")
    if mask.ndim != 2:
        raise ValueError(f"掩码必须是 HxW，得到 {mask.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", mask.astype(np.uint8, copy=False))
    if not ok:
        raise ValueError(f"无法编码掩码: {path}")
    enc.tofile(str(path))
