"""数据集与训练/验证划分。

目录约定（相对于 data.root）：
    images/        抽帧得到的图片 {video}_{frame:06d}.jpg
    pseudo_masks/  autolabel 生成的伪标签（未经人工修正）
    labelme/       LabelMe JSON（人工修正）
    masks/         最终训练用掩码 {stem}.png，值域 {0, 1, 255}
    splits/        train.txt / val.txt，每行一个 stem
"""

from __future__ import annotations

import math
import random
import re
import warnings
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from . import IGNORE_INDEX, NON_WALKABLE, WALKABLE
from .utils import list_images, read_image_rgb, read_mask

IMAGES_SUBDIR = "images"
MASKS_SUBDIR = "masks"
SPLITS_SUBDIR = "splits"

# "{prefix}_{digits}"：prefix 至少 1 个字符，digits 只认 ASCII 数字（frames.py 用 :06d 生成）
_FRAME_STEM = re.compile(r"^(.+)_([0-9]+)$")
# 浮点比较容差：0.1 * 30 == 3.0000000000000004，不应向上取整成 4
_EPS = 1e-9
# 错误信息中最多列出的条目数
_MAX_LISTED = 5
_VALID_LABELS = (NON_WALKABLE, WALKABLE, IGNORE_INDEX)


def group_of(stem: str) -> str:
    """帧所属视频分组：stem 形如 "{prefix}_{digits}" 时返回 prefix，否则返回 stem 本身。
    例："dam_run1_000120" → "dam_run1"；"screenshot" → "screenshot"；"a_b" → "a_b"。

    digits 只匹配 ASCII 数字，按最后一个 '_' 切分；prefix 不能为空（"_123" → "_123"）。
    """
    m = _FRAME_STEM.match(stem)
    return m.group(1) if m else stem


def _frame_index(stem: str) -> int:
    """stem 末尾的帧序号（整数比较，"v_99" < "v_100"）；没有序号时返回 -1。"""
    m = _FRAME_STEM.match(stem)
    return int(m.group(2)) if m else -1


def _preview(items: Sequence[str]) -> str:
    shown = ", ".join(str(x) for x in items[:_MAX_LISTED])
    return shown + (f" …（共 {len(items)} 个）" if len(items) > _MAX_LISTED else "")


def index_images(images_dir: str | Path) -> dict[str, Path]:
    """stem → 图片路径（dfwalk.utils.list_images）。同一 stem 存在多个后缀时抛 ValueError。

    只看目录第一层、只认 IMAGE_EXTS 中的后缀（大小写不敏感）。目录不存在时抛 FileNotFoundError。
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    index: dict[str, Path] = {}
    dups: dict[str, list[str]] = {}
    for path in list_images(images_dir):
        stem = path.stem
        if stem in index:
            dups.setdefault(stem, [index[stem].name]).append(path.name)
        else:
            index[stem] = path
    if dups:
        detail = "; ".join(f"{stem}: {', '.join(names)}" for stem, names in list(dups.items())[:_MAX_LISTED])
        raise ValueError(f"{images_dir} 中有 {len(dups)} 个 stem 对应多个图片文件（只能保留一个）: {detail}")
    return index


def _mask_path(masks_dir: Path, stem: str) -> Path:
    return masks_dir / f"{stem}.png"


def find_pairs(root: str | Path, images_subdir: str = IMAGES_SUBDIR, masks_subdir: str = MASKS_SUBDIR) -> list[str]:
    """返回同时拥有图片和 masks/{stem}.png 的 stem 列表（排序）。

    图片目录不存在时抛 FileNotFoundError；掩码目录不存在时视为没有掩码，返回空列表。
    """
    root = Path(root)
    index = index_images(root / images_subdir)
    masks_dir = root / masks_subdir
    if not masks_dir.is_dir():
        return []
    return sorted(stem for stem in index if _mask_path(masks_dir, stem).is_file())


def make_splits(stems: Sequence[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """按视频分组（group_of）划分训练/验证集，同一视频的帧不会同时出现在两边，避免数据泄漏。

    - 分组数 >= 2：用 random.Random(seed) 打乱分组顺序，依次把分组放入验证集，直到验证集帧数
      >= val_ratio * 总帧数（至少 1 组，且至少留 1 组给训练集）。
    - 分组数 == 1（只有一段录像）：按帧序号排序后把最后 ceil(val_ratio * N) 帧作为验证集（时间切分，
      减少相邻帧泄漏），并用 warnings.warn 提醒最好录制多段视频。N < 2 时抛 ValueError。
    - 输入为空时抛 ValueError。返回的两个列表各自排序，结果对相同输入和 seed 确定。

    细节：
    - 输入中重复的 stem 只算一次；val_ratio 不在 (0, 1) 之间时抛 ValueError。
    - 多分组时，按打乱后的顺序逐组考察：若放入该组后验证集帧数比不放更接近目标（平局时放入）才放入，
      否则跳过、继续看下一组；达到目标即停止。这样某段特别长的录像不会因为排在前面就把验证集撑爆。
      若全部分组都被跳过，则把最接近目标大小的一组（同样接近时取打乱顺序靠前者）放入验证集。
    - 单分组时验证集帧数取 ceil(val_ratio * N)，并限制在 [1, N-1]，保证两边都非空；
      帧序号按整数比较（"v_99" 在 "v_100" 之前），没有序号的 stem 排在最前。
    """
    ratio = float(val_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"val_ratio 必须在 (0, 1) 之间，得到 {val_ratio}")
    unique = sorted(set(stems))
    if not unique:
        raise ValueError("没有可划分的样本（stems 为空）：请先抽帧并生成 masks/")
    n = len(unique)

    groups: dict[str, list[str]] = {}
    for stem in unique:
        groups.setdefault(group_of(stem), []).append(stem)

    if len(groups) == 1:
        (name,) = groups
        if n < 2:
            raise ValueError(f"只有 1 个样本（{unique[0]}），无法划分训练集和验证集")
        k = min(max(math.ceil(ratio * n - _EPS), 1), n - 1)
        ordered = sorted(unique, key=lambda s: (_frame_index(s), s))
        warnings.warn(
            f"所有 {n} 帧都来自同一个视频分组 {name!r}，改为按帧序号做时间切分：最后 {k} 帧作为验证集。"
            "相邻帧高度相似，验证指标会偏乐观，建议录制多段视频（不同地图/时段）后再划分。",
            UserWarning,
            stacklevel=2,
        )
        return sorted(ordered[:-k]), sorted(ordered[-k:])

    order = sorted(groups)
    random.Random(seed).shuffle(order)
    target = ratio * n
    val_groups: list[str] = []
    val_count = 0
    for name in order:
        if val_count >= target - _EPS or len(val_groups) >= len(order) - 1:
            break
        size = len(groups[name])
        # 放入后的超出量不大于放入前的缺口 → 放入（更接近目标或同样接近）
        if (val_count + size - target) <= (target - val_count) + _EPS:
            val_groups.append(name)
            val_count += size
    if not val_groups:
        best = min(order, key=lambda g: abs(len(groups[g]) - target))  # min 取第一个最小值，保持确定
        val_groups.append(best)

    chosen = set(val_groups)
    val = sorted(s for g in val_groups for s in groups[g])
    train = sorted(s for g in order if g not in chosen for s in groups[g])
    return train, val


def _split_path(root: str | Path, name: str) -> Path:
    return Path(root) / SPLITS_SUBDIR / f"{name}.txt"


def write_split(root: str | Path, name: str, stems: Sequence[str]) -> Path:
    """写 root/splits/{name}.txt（每行一个 stem，UTF-8），返回路径。

    自动创建 splits/ 目录；换行统一为 "\\n"。stem 含换行符时抛 ValueError。
    """
    stems = [str(s) for s in stems]
    bad = [s for s in stems if "\n" in s or "\r" in s]
    if bad:
        raise ValueError(f"stem 不能包含换行符: {bad[:_MAX_LISTED]!r}")
    path = _split_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{s}\n" for s in stems)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def read_split(root: str | Path, name: str) -> list[str]:
    """读 root/splits/{name}.txt，忽略空行和首尾空白。文件不存在时抛 FileNotFoundError。

    兼容带 BOM 的 UTF-8（Windows 记事本保存）和 CRLF 换行。
    """
    path = _split_path(root, name)
    if not path.is_file():
        raise FileNotFoundError(f"划分文件不存在: {path}（可先运行 `dfwalk split`）")
    with open(path, "r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def _check_mask_values(mask: np.ndarray, path: Path) -> None:
    """掩码只能包含 {0, 1, 255}，否则抛 ValueError（信息含路径和非法值）。"""
    counts = np.bincount(mask.ravel(), minlength=256)
    counts[list(_VALID_LABELS)] = 0
    bad = np.flatnonzero(counts)
    if bad.size:
        values = ", ".join(str(int(v)) for v in bad[:10]) + (" …" if bad.size > 10 else "")
        raise ValueError(
            f"掩码包含非法值 [{values}]（只允许 {NON_WALKABLE}/{WALKABLE}/{IGNORE_INDEX}），"
            f"共 {int(counts.sum())} 个像素: {path}"
        )


class WalkableDataset(Dataset):
    """返回 (image, mask)：image float32 (3, H, W)，mask int64 (H, W)。

    - 初始化时用 index_images 建立 stem → 图片路径索引；stems 中任何 stem 缺图片或缺掩码都抛
      FileNotFoundError（信息中列出前几个缺失项）。
    - __getitem__ 用 dfwalk.utils.read_image_rgb / read_mask 读取；图片与掩码尺寸不一致、或掩码出现
      {0, 1, 255} 以外的值时抛 ValueError（信息包含文件路径和非法值）。
    - transform(image, mask) → (image_tensor, mask_tensor)，通常为 TrainTransform / EvalTransform。

    属性 stems / image_paths / mask_paths 与索引一一对应。对象只保存路径，可安全地被 DataLoader
    的多个 worker pickle。
    """

    def __init__(
        self,
        root: str | Path,
        stems: Sequence[str],
        transform: Callable[[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]],
        images_subdir: str = IMAGES_SUBDIR,
        masks_subdir: str = MASKS_SUBDIR,
    ) -> None:
        if not callable(transform):
            raise TypeError(f"transform 必须可调用，得到 {type(transform).__name__}")
        self.root = Path(root)
        self.stems = [str(s) for s in stems]
        self.transform = transform
        images_dir = self.root / images_subdir
        masks_dir = self.root / masks_subdir

        index = index_images(images_dir)
        missing_images = [s for s in self.stems if s not in index]
        missing_masks = [s for s in self.stems if not _mask_path(masks_dir, s).is_file()]
        if missing_images or missing_masks:
            parts = []
            if missing_images:
                parts.append(f"{len(missing_images)} 个缺图片（{images_dir}）: {_preview(missing_images)}")
            if missing_masks:
                parts.append(f"{len(missing_masks)} 个缺掩码（{masks_dir}/{{stem}}.png）: {_preview(missing_masks)}")
            raise FileNotFoundError("数据集中有样本文件缺失：" + "；".join(parts))

        self.image_paths = [index[s] for s in self.stems]
        self.mask_paths = [_mask_path(masks_dir, s) for s in self.stems]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.image_paths[idx], self.mask_paths[idx]
        image = read_image_rgb(image_path)
        mask = read_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"图片与掩码尺寸不一致：{image_path} 为 {image.shape[1]}x{image.shape[0]}，"
                f"{mask_path} 为 {mask.shape[1]}x{mask.shape[0]}（宽x高）"
            )
        _check_mask_values(mask, mask_path)
        return self.transform(image, mask)

    def __repr__(self) -> str:
        return f"WalkableDataset(root={str(self.root)!r}, n={len(self)}, transform={self.transform!r})"


def label_stats(root: str | Path, stems: Sequence[str], masks_subdir: str = MASKS_SUBDIR) -> dict[str, float]:
    """统计掩码像素分布：{"frames", "walkable_ratio", "non_walkable_ratio", "ignore_ratio"}（比例相对总像素）。

    frames 为整数帧数；三个比例之和为 1。stems 为空（总像素为 0）时比例均为 0.0。
    掩码缺失抛 FileNotFoundError，含 {0, 1, 255} 以外的值抛 ValueError（与 WalkableDataset 一致）。
    """
    masks_dir = Path(root) / masks_subdir
    counts = np.zeros(256, dtype=np.int64)
    stems = list(stems)
    for stem in stems:
        path = _mask_path(masks_dir, stem)
        mask = read_mask(path)
        _check_mask_values(mask, path)
        counts += np.bincount(mask.ravel(), minlength=256)
    total = int(counts.sum())

    def ratio(label: int) -> float:
        return float(counts[label]) / total if total else 0.0

    return {
        "frames": len(stems),
        "walkable_ratio": ratio(WALKABLE),
        "non_walkable_ratio": ratio(NON_WALKABLE),
        "ignore_ratio": ratio(IGNORE_INDEX),
    }
