"""分割模型构建（segmentation_models_pytorch）。

segmentation_models_pytorch 在函数内部延迟导入：dfwalk.checkpoint 等模块会 import 本模块，
只用到 model_config 时不必为导入 smp（及其依赖的 timm / huggingface_hub）付出启动开销。
"""

from __future__ import annotations

import pickle

from torch import nn

from . import NUM_CLASSES

# encoder_weights 写成这些字符串时等同于 None（随机初始化）。
# YAML 里只有 null / ~ 会被解析成 None，命令行上 --set train.encoder_weights=none 很常见，这里一并兼容。
_NO_WEIGHTS_STRINGS = ("", "none", "null")

# 下载/读取预训练权重时可能出现的异常：网络错误（OSError 及其子类，包括 huggingface_hub 的
# 连接/离线错误）、下载不完整（EOFError）、缓存文件损坏或格式不兼容（UnpicklingError / RuntimeError）
_WEIGHT_LOAD_ERRORS: tuple[type[BaseException], ...] = (OSError, EOFError, RuntimeError, pickle.UnpicklingError)


def available_archs() -> list[str]:
    """当前安装的 smp 支持的架构名（类名写法，如 "Unet"、"DeepLabV3Plus"），按 smp 注册顺序。"""
    import segmentation_models_pytorch as smp

    return [cls.__name__ for cls in smp.MODEL_ARCHITECTURES_MAPPING.values()]


def _normalize_weights(encoder_weights: str | None) -> str | None:
    if encoder_weights is None:
        return None
    if not isinstance(encoder_weights, str):
        raise TypeError(f"encoder_weights 必须是字符串或 None，得到 {type(encoder_weights).__name__}")
    if encoder_weights.strip().lower() in _NO_WEIGHTS_STRINGS:
        return None
    return encoder_weights


def build_model(
    arch: str = "Unet",
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """smp.create_model(arch, encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3,
    classes=num_classes)。输出 logits (N, num_classes, H, W)，H、W 与输入相同（输入需为 32 的倍数）。
    arch 名不合法时抛 ValueError，信息中列出可用架构名。

    补充约定：
    - arch 大小写不敏感（与 smp 一致），"unet" 与 "Unet" 等价。
    - encoder_weights 为 None 或字符串 "none"/"null"/""（大小写不敏感）时随机初始化，不会联网下载；
      其余值（如 "imagenet"）交给 smp，首次使用会下载预训练权重。
    - encoder 名或 encoder_weights 不被 smp 支持时同样抛 ValueError（smp 原本抛 KeyError），信息中保留
      smp 给出的可选项列表。
    - num_classes 必须是正整数，否则抛 ValueError。
    - 预训练权重下载/读取失败（网络不通、缓存文件损坏等）时抛 RuntimeError，信息中给出 HF 镜像、
      清理缓存和 encoder_weights=null 三种解决办法，原始异常保留在 __cause__ 中。
    - 返回的模型处于 train 模式（nn.Module 默认），位于 CPU。
    """
    import segmentation_models_pytorch as smp

    if not isinstance(arch, str) or arch.lower() not in smp.MODEL_ARCHITECTURES_MAPPING:
        raise ValueError(f"未知的模型架构 {arch!r}，可用架构: {', '.join(available_archs())}")
    if not isinstance(encoder, str) or not encoder:
        raise ValueError(f"encoder 必须是非空字符串，得到 {encoder!r}")
    # bool 是 int 的子类，单独排除
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"num_classes 必须是正整数，得到 {num_classes!r}")
    weights = _normalize_weights(encoder_weights)

    try:
        return smp.create_model(
            arch,
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )
    except KeyError as e:
        # smp 对未知编码器 / 未知预训练权重抛 KeyError，统一转成 ValueError，保留原始的可选项说明
        detail = e.args[0] if e.args else str(e)
        raise ValueError(
            f"无法创建模型 arch={arch!r}, encoder={encoder!r}, encoder_weights={weights!r}: {detail}"
        ) from e
    except _WEIGHT_LOAD_ERRORS as e:
        if weights is None:
            raise
        # smp 0.5 先从 Hugging Face Hub（smp-hub/...）下载，失败后改从原始 URL（torch.hub）下载；
        # 网络不通、下载中断、或缓存里留下的损坏/旧格式文件都会落到这里，原始异常保留在 __cause__ 中
        raise RuntimeError(
            f"无法获取预训练编码器权重 encoder={encoder!r}, encoder_weights={weights!r}"
            f"（{type(e).__name__}: {str(e).strip() or '无详细信息'}）。可以："
            "1) 国内网络设置环境变量 HF_ENDPOINT=https://hf-mirror.com 后重试；"
            "2) 删除 ~/.cache/torch/hub/checkpoints 中可能损坏的权重文件后重试；"
            "3) 离线环境用 --set train.encoder_weights=null 随机初始化（需要更多数据和轮数）"
        ) from e


def model_config(arch: str, encoder: str, num_classes: int = NUM_CLASSES) -> dict:
    """写入 checkpoint 的模型结构描述：{"arch": arch, "encoder": encoder, "num_classes": num_classes}。

    只含基础类型（str / int），可被 torch.load(weights_only=True) 安全读回；
    dfwalk.checkpoint.model_from_checkpoint 用它调用 build_model 重建同结构模型。
    """
    return {"arch": str(arch), "encoder": str(encoder), "num_classes": int(num_classes)}
