"""配置：dataclass 定义默认值，YAML 文件和命令行 --set 逐层覆盖。

dataclass 里的默认值是唯一真实来源；configs/default.yaml 与之保持一致（有测试校验）。
"""

from __future__ import annotations

import dataclasses
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


def _merge_section(section: Any, values: dict[str, Any], where: str) -> None:
    known = {f.name for f in dataclasses.fields(section)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"未知配置项: {where}.{key}")
        setattr(section, key, value)


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


def validate(cfg: Config) -> None:
    h, w = cfg.data.image_size
    if h % 32 or w % 32 or h <= 0 or w <= 0:
        raise ValueError(f"data.image_size 必须是正的 32 的倍数，得到 {cfg.data.image_size}")
    if not 0.0 < cfg.data.val_ratio < 1.0:
        raise ValueError("data.val_ratio 必须在 (0, 1) 之间")
    a = cfg.autolabel
    if not 0.0 <= a.lo_threshold < a.hi_threshold <= 1.0:
        raise ValueError("需要满足 0 <= autolabel.lo_threshold < autolabel.hi_threshold <= 1")
    for rect in cfg.hud.rects:
        if len(rect) != 4:
            raise ValueError(f"hud.rects 的每一项必须是 [x0, y0, x1, y1]，得到 {rect}")
        x0, y0, x1, y1 = rect
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"hud.rects 坐标必须归一化到 0~1 且 x0<x1, y0<y1，得到 {rect}")
    if not 0.0 <= cfg.infer.threshold <= 1.0:
        raise ValueError("infer.threshold 必须在 [0, 1] 之间")
