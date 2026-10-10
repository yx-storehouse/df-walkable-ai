"""配置：dataclass 定义默认值，YAML 文件和命令行 --set 逐层覆盖。

dataclass 里的默认值是唯一真实来源；configs/default.yaml 与之保持一致（有测试校验）。
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass
class DataConfig:
    root: str = "data"
    image_size: list[int] = field(default_factory=lambda: [384, 672])
    val_ratio: float = 0.2
    seed: int = 42


@dataclass
class HudConfig:
    rects: list[list[float]] = field(default_factory=list)


@dataclass
class FramesConfig:
    fps: float = 2.0
    max_side: int = 1280
    dedup_threshold: float = 6.0
    jpeg_quality: int = 95


DEFAULT_WALKABLE_LABELS = [
    "floor", "road", "grass", "sidewalk", "earth", "rug", "field", "sand", "path",
    "stairs", "runway", "stairway", "bridge", "dirt track", "land", "step",
]


@dataclass
class AutolabelConfig:
    model: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    short_side: int = 512
    hi_threshold: float = 0.7
    lo_threshold: float = 0.3
    device: str = "auto"
    walkable_labels: list[str] = field(default_factory=lambda: list(DEFAULT_WALKABLE_LABELS))


@dataclass
class TrainConfig:
    arch: str = "Unet"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    epochs: int = 50
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    amp: bool = True
    out_dir: str = "runs/exp"
    device: str = "auto"
    seed: int = 42


@dataclass
class InferConfig:
    threshold: float = 0.5
    alpha: float = 0.45
    color: list[int] = field(default_factory=lambda: [0, 255, 0])
    device: str = "auto"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    autolabel: AutolabelConfig = field(default_factory=AutolabelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(value: Any, type_name: str) -> Any:
    """float 类型的配置项收到数字字符串时转成 float。

    YAML 1.1（PyYAML）只把带小数点的科学计数法当作浮点数："1e-3" 会被解析成字符串，
    而 train.lr=1e-3 这种写法很常见。其余情况原样返回，类型不对由 validate 报错。
    """
    if type_name == "float" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _merge_section(section: Any, values: dict[str, Any], where: str) -> None:
    types = {f.name: str(f.type) for f in dataclasses.fields(section)}
    for key, value in values.items():
        if key not in types:
            raise KeyError(f"未知配置项: {where}.{key}")
        setattr(section, key, _coerce(value, types[key]))


def apply_dict(cfg: Config, data: dict[str, Any]) -> Config:
    """把嵌套字典 {section: {key: value}} 合并进 cfg（原地修改并返回）。"""
    for section_name, values in (data or {}).items():
        if not hasattr(cfg, section_name) or not dataclasses.is_dataclass(getattr(cfg, section_name)):
            raise KeyError(f"未知配置段: {section_name}")
        if values is None:
            continue
        if not isinstance(values, dict):
            raise TypeError(f"配置段 {section_name} 必须是映射，得到 {type(values).__name__}")
        _merge_section(getattr(cfg, section_name), values, section_name)
    return cfg


def parse_override(text: str) -> tuple[str, str, Any]:
    """解析 "section.key=value"，value 按 YAML 语法解析（数字、列表、null、布尔都可以）。"""
    if "=" not in text:
        raise ValueError(f"覆盖项格式应为 section.key=value: {text!r}")
    dotted, raw = text.split("=", 1)
    parts = dotted.strip().split(".")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"覆盖项键应为 section.key: {dotted!r}")
    return parts[0], parts[1], yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Config:
    """加载配置：dataclass 默认值 ← YAML 文件（可选）← 覆盖项列表。"""
    cfg = Config()
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            apply_dict(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        section, key, value = parse_override(item)
        apply_dict(cfg, {section: {key: value}})
    validate(cfg)
    return cfg


def _is_number(value: Any) -> bool:
    """int / float（含 numpy 标量），bool 不算数字。"""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _check_number(value: Any, name: str, lo: float | None = None, hi: float | None = None,
                  lo_open: bool = False, allow_inf: bool = False) -> None:
    """数字（不是 bool、不是 NaN；allow_inf=False 时不能是 ±inf），可选范围检查。"""
    if not _is_number(value) or math.isnan(value) or (not allow_inf and math.isinf(value)):
        raise ValueError(f"{name} 必须是数字，得到 {value!r}")
    if lo is not None and (value <= lo if lo_open else value < lo):
        raise ValueError(f"{name} 必须{'>' if lo_open else '>='} {lo}，得到 {value!r}")
    if hi is not None and value > hi:
        raise ValueError(f"{name} 必须 <= {hi}，得到 {value!r}")


def _check_int(value: Any, name: str, lo: int | None = None, hi: int | None = None) -> None:
    if not _is_int(value):
        raise ValueError(f"{name} 必须是整数，得到 {value!r}")
    if lo is not None and value < lo:
        raise ValueError(f"{name} 必须 >= {lo}，得到 {value!r}")
    if hi is not None and value > hi:
        raise ValueError(f"{name} 必须 <= {hi}，得到 {value!r}")


def _check_str(value: Any, name: str, allow_path: bool = False, allow_none: bool = False) -> None:
    """非空字符串；allow_path 时也接受 pathlib.Path 等 os.PathLike。"""
    if value is None and allow_none:
        return
    text = os.fspath(value) if allow_path and isinstance(value, os.PathLike) else value
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{name} 必须是非空字符串，得到 {value!r}")


def _is_seq(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def validate(cfg: Config) -> None:
    """检查配置的类型和取值范围，不合法时抛 ValueError（信息以 "section.key" 开头，便于定位）。

    YAML / --set 写错类型（例如 data.image_size=5、train.lr=abc、train.amp=yes 之外的字符串）同样是 ValueError。
    各模块在使用参数时还会做自己的检查，这里负责尽早、统一地报出配置文件中的错误。
    """
    d = cfg.data
    _check_str(d.root, "data.root", allow_path=True)
    size = d.image_size
    if (
        not _is_seq(size)
        or len(size) != 2
        or not all(_is_int(v) for v in size)
        or any(v <= 0 or v % 32 for v in size)
    ):
        raise ValueError(f"data.image_size 必须是 [高, 宽]，两个正的 32 的倍数，得到 {size!r}")
    if not _is_number(d.val_ratio) or not 0.0 < d.val_ratio < 1.0:
        raise ValueError(f"data.val_ratio 必须在 (0, 1) 之间，得到 {d.val_ratio!r}")
    _check_int(d.seed, "data.seed")

    if not _is_seq(cfg.hud.rects):
        raise ValueError(f"hud.rects 必须是矩形列表 [[x0, y0, x1, y1], ...]，得到 {cfg.hud.rects!r}")
    for rect in cfg.hud.rects:
        if not _is_seq(rect) or len(rect) != 4 or not all(_is_number(v) for v in rect):
            raise ValueError(f"hud.rects 的每一项必须是 [x0, y0, x1, y1] 四个数，得到 {rect!r}")
        x0, y0, x1, y1 = rect
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"hud.rects 坐标必须归一化到 0~1 且 x0<x1, y0<y1，得到 {rect}")

    f = cfg.frames
    _check_number(f.fps, "frames.fps", allow_inf=True)  # <= 0 或 >= 源帧率表示每帧都取
    _check_int(f.max_side, "frames.max_side", lo=0)
    _check_number(f.dedup_threshold, "frames.dedup_threshold", lo=0.0)
    _check_int(f.jpeg_quality, "frames.jpeg_quality", lo=0, hi=100)

    a = cfg.autolabel
    _check_str(a.model, "autolabel.model", allow_path=True)
    _check_int(a.short_side, "autolabel.short_side", lo=1)
    if (
        not _is_number(a.lo_threshold)
        or not _is_number(a.hi_threshold)
        or not 0.0 <= a.lo_threshold < a.hi_threshold <= 1.0
    ):
        raise ValueError(
            "需要满足 0 <= autolabel.lo_threshold < autolabel.hi_threshold <= 1，"
            f"得到 lo={a.lo_threshold!r}, hi={a.hi_threshold!r}"
        )
    _check_str(a.device, "autolabel.device")
    if not _is_seq(a.walkable_labels) or not all(isinstance(x, str) for x in a.walkable_labels):
        raise ValueError(f"autolabel.walkable_labels 必须是类别名（字符串）列表，得到 {a.walkable_labels!r}")

    t = cfg.train
    _check_str(t.arch, "train.arch")
    _check_str(t.encoder, "train.encoder")
    _check_str(t.encoder_weights, "train.encoder_weights", allow_none=True)
    _check_int(t.epochs, "train.epochs", lo=1)
    _check_int(t.batch_size, "train.batch_size", lo=1)
    _check_number(t.lr, "train.lr", lo=0.0, lo_open=True)
    _check_number(t.weight_decay, "train.weight_decay", lo=0.0)
    _check_int(t.num_workers, "train.num_workers", lo=0)
    _check_number(t.ce_weight, "train.ce_weight", lo=0.0)
    _check_number(t.dice_weight, "train.dice_weight", lo=0.0)
    if t.ce_weight == 0 and t.dice_weight == 0:
        raise ValueError("train.ce_weight 与 train.dice_weight 不能同时为 0（loss 恒为 0，模型学不到东西）")
    if not isinstance(t.amp, bool):
        raise ValueError(f"train.amp 必须是 true / false，得到 {t.amp!r}")
    _check_str(t.out_dir, "train.out_dir", allow_path=True)
    _check_str(t.device, "train.device")
    _check_int(t.seed, "train.seed")

    i = cfg.infer
    if not _is_number(i.threshold) or not 0.0 <= i.threshold <= 1.0:
        raise ValueError(f"infer.threshold 必须在 [0, 1] 之间，得到 {i.threshold!r}")
    _check_number(i.alpha, "infer.alpha", lo=0.0, hi=1.0)
    if not _is_seq(i.color) or len(i.color) != 3 or not all(_is_int(c) and 0 <= c <= 255 for c in i.color):
        raise ValueError(f"infer.color 必须是 [R, G, B] 三个 0~255 的整数，得到 {i.color!r}")
    _check_str(i.device, "infer.device")
