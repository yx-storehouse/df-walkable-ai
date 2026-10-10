"""checkpoint 保存/加载。只存张量和基础类型，加载时使用 weights_only=True，不执行任意反序列化代码。"""

from __future__ import annotations

import pickle
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from . import IMAGENET_MEAN, IMAGENET_STD, __version__
from .model import build_model

CKPT_FORMAT = "dfwalk-ckpt-v1"

# torch.load(weights_only=True) 读到非 torch 文件 / 其他工具的 pickle / 损坏文件时抛出的异常类型（实测：ONNX、YAML、
# 图片 → UnpicklingError；空文件 → EOFError；CSV → IndexError；纯文本 → KeyError；截断的 .pt、普通 zip → RuntimeError）
_NOT_A_CHECKPOINT_ERRORS: tuple[type[BaseException], ...] = (
    pickle.UnpicklingError,
    EOFError,
    IndexError,
    KeyError,
    ValueError,
    TypeError,
    AttributeError,
    struct.error,
    RuntimeError,
)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    model_cfg: Mapping[str, Any],
    image_size: Sequence[int],
    epoch: int,
    metrics: Mapping[str, float],
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": CKPT_FORMAT,
        "version": __version__,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_cfg": dict(model_cfg),
        "image_size": [int(x) for x in image_size],
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "epoch": int(epoch),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """读取 save_checkpoint 写出的 checkpoint（weights_only=True）。

    文件不存在抛 FileNotFoundError；不是 dfwalk checkpoint 时一律抛 ValueError——包括根本不是 torch 文件的情况
    （例如误把导出的 model.onnx、config.yaml、图片传给 --ckpt）和已损坏/不完整的文件。torch.load 对这些文件抛出的
    UnpicklingError / EOFError / IndexError 等原始异常保留在 __cause__ 中（其中“改用 weights_only=False”的建议
    并不适用）。
    """
    path = Path(path)
    try:
        state = torch.load(path, map_location=map_location, weights_only=True)
    except _NOT_A_CHECKPOINT_ERRORS as e:
        if path.suffix.lower() == ".onnx":
            hint = "这是导出的 ONNX 模型，请用 onnxruntime 加载（见 README“部署 ONNX 模型”）；"
        else:
            hint = ""
        raise ValueError(
            f"不是 dfwalk checkpoint，或文件已损坏，无法读取: {path}。{hint}--ckpt 需要 dfwalk train 输出的 best.pt / last.pt"
        ) from e
    if not isinstance(state, dict) or state.get("format") != CKPT_FORMAT:
        raise ValueError(f"不是 dfwalk checkpoint（缺少 format={CKPT_FORMAT}）: {path}")
    return state


def model_from_checkpoint(state: Mapping[str, Any]) -> nn.Module:
    """按 checkpoint 里的结构描述重建模型并加载权重（不下载预训练编码器）。返回 eval 模式的模型（CPU 上）。"""
    cfg = state["model_cfg"]
    model = build_model(cfg["arch"], cfg["encoder"], encoder_weights=None, num_classes=cfg["num_classes"])
    model.load_state_dict(state["model_state"])
    model.eval()
    return model
