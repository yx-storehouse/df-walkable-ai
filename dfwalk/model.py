"""分割模型构建（segmentation_models_pytorch）。"""

from __future__ import annotations

from torch import nn

from . import NUM_CLASSES


def build_model(
    arch: str = "Unet",
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """smp.create_model(arch, encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3,
    classes=num_classes)。输出 logits (N, num_classes, H, W)，H、W 与输入相同（输入需为 32 的倍数）。
    arch 名不合法时抛 ValueError，信息中列出可用架构名。"""
    raise NotImplementedError


def model_config(arch: str, encoder: str, num_classes: int = NUM_CLASSES) -> dict:
    """写入 checkpoint 的模型结构描述：{"arch": arch, "encoder": encoder, "num_classes": num_classes}。"""
    raise NotImplementedError
