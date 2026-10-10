"""checkpoint 保存/加载。只存张量和基础类型，加载时使用 weights_only=True，不执行任意反序列化代码。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from . import IMAGENET_MEAN, IMAGENET_STD, __version__
from .model import build_model

CKPT_FORMAT = "dfwalk-ckpt-v1"


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
    state = torch.load(Path(path), map_location=map_location, weights_only=True)
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
