"""公共小工具：设备选择、随机种子、图像/掩码读写、打开视频。

图像读写统一走 np.fromfile + cv2.imdecode / cv2.imencode + tofile，
避免 Windows 下中文路径导致 cv2.imread 返回 None。
视频只能交给 cv2.VideoCapture 按路径打开：Windows 下打不开中文路径时改用 8.3 短路径重试（open_video）。
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np

from . import IMAGE_EXTS


def resolve_device(name: str = "auto"):
    """"auto" → cuda > mps > cpu；其余字符串交给 torch.device。

    显式指定 cuda（cuda:0 等）/ mps 但当前 PyTorch 用不了时抛 ValueError 并给出解决办法——最常见的原因是
    装了 CPU 版 PyTorch（Windows 上直接 pip install 得到的就是 CPU 版，此时要先卸载再装 CUDA 版）；否则要等到
    model.to(device) 才抛出没有说明的 AssertionError（“Torch not compiled with CUDA enabled”）。
    """
    import torch

    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            if torch.version.cuda is None:
                # 必须先卸载：已装 CPU 版时第 3 步的命令只会提示 “Requirement already satisfied”，不会替换成 CUDA 版
                reason = ("当前安装的是 CPU 版 PyTorch。请先运行 pip uninstall -y torch torchvision 卸载 CPU 版，"
                          "再按 README 安装步骤第 3 步安装 CUDA 版 PyTorch")
            else:
                reason = "PyTorch 支持 CUDA，但检测不到可用的 NVIDIA 显卡或驱动。请检查显卡驱动"
            raise ValueError(f"指定了设备 {name}，但当前 PyTorch 无法使用 CUDA：{reason}，或改用 --device auto / cpu")
        if device.type == "mps" and not (
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        ):
            raise ValueError(f"指定了设备 {name}，但当前 PyTorch 无法使用 MPS（仅支持 Apple Silicon 的 macOS），"
                             "请改用 --device auto / cpu")
        return device
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
    """读取单通道 uint8 掩码 PNG（值域 {0, 1, 255}）。

    不是单通道、或不是 8 位（例如 16 位 PNG：直接转 uint8 会把 256 变成 0、257 变成 1，
    后续的值域检查也发现不了）时抛 ValueError。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    buf = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"无法解码掩码: {path}")
    if mask.ndim != 2:
        raise ValueError(f"掩码必须是单通道 PNG: {path}, shape={mask.shape}")
    if mask.dtype != np.uint8:
        raise ValueError(f"掩码必须是 8 位单通道 PNG（uint8），得到 dtype={mask.dtype}: {path}")
    return mask


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


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """cv2.VideoCapture 读出的帧（BGR，个别后端给出灰度或 BGRA）→ HxWx3 uint8 RGB。"""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _windows_short_path(path: Path) -> str | None:  # pragma: no cover - 仅 Windows
    """Windows 下取 8.3 短路径（纯 ASCII），给不支持中文路径的视频后端兜底。取不到时返回 None。"""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        fn = ctypes.windll.kernel32.GetShortPathNameW
        fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        fn.restype = wintypes.DWORD
        need = fn(str(path), None, 0)
        if need == 0:
            return None
        buf = ctypes.create_unicode_buffer(need)
        if fn(str(path), buf, need) == 0:
            return None
        short = buf.value
        return short if short and short != str(path) and short.isascii() else None
    except Exception:
        return None


def open_video(path: str | Path) -> cv2.VideoCapture:
    """用 cv2.VideoCapture 打开本地视频文件，失败时抛 ValueError（信息中包含路径）。

    文件不存在也抛 ValueError（与“格式不支持/文件损坏”统一处理，调用方只需捕获一种异常）。
    Windows 下第一次打开失败时用 8.3 短路径重试一次（部分视频后端不支持中文路径）。
    调用方负责 release()。
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"无法打开视频（文件不存在）: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        short = _windows_short_path(path)
        if short is not None:  # pragma: no cover - 仅 Windows
            cap = cv2.VideoCapture(short)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"无法打开视频（格式不支持或文件损坏）: {path}")
    return cap
