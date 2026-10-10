"""数据集与训练/验证划分。

目录约定（相对于 data.root）：
    images/        抽帧得到的图片 {video}_{frame:06d}.jpg
    pseudo_masks/  autolabel 生成的伪标签（未经人工修正）
    labelme/       LabelMe JSON（人工修正）
    masks/         最终训练用掩码 {stem}.png，值域 {0, 1, 255}
    splits/        train.txt / val.txt，每行一个 stem
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGES_SUBDIR = "images"
MASKS_SUBDIR = "masks"
SPLITS_SUBDIR = "splits"


def group_of(stem: str) -> str:
    """帧所属视频分组：stem 形如 "{prefix}_{digits}" 时返回 prefix，否则返回 stem 本身。
    例："dam_run1_000120" → "dam_run1"；"screenshot" → "screenshot"；"a_b" → "a_b"。"""
    raise NotImplementedError


def index_images(images_dir: str | Path) -> dict[str, Path]:
    """stem → 图片路径（dfwalk.utils.list_images）。同一 stem 存在多个后缀时抛 ValueError。"""
    raise NotImplementedError


def find_pairs(root: str | Path, images_subdir: str = IMAGES_SUBDIR, masks_subdir: str = MASKS_SUBDIR) -> list[str]:
    """返回同时拥有图片和 masks/{stem}.png 的 stem 列表（排序）。"""
    raise NotImplementedError


def make_splits(stems: Sequence[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """按视频分组（group_of）划分训练/验证集，同一视频的帧不会同时出现在两边，避免数据泄漏。

    - 分组数 >= 2：用 random.Random(seed) 打乱分组顺序，依次把分组放入验证集，直到验证集帧数
      >= val_ratio * 总帧数（至少 1 组，且至少留 1 组给训练集）。
    - 分组数 == 1（只有一段录像）：按帧序号排序后把最后 ceil(val_ratio * N) 帧作为验证集（时间切分，
      减少相邻帧泄漏），并用 warnings.warn 提醒最好录制多段视频。N < 2 时抛 ValueError。
    - 输入为空时抛 ValueError。返回的两个列表各自排序，结果对相同输入和 seed 确定。
    """
    raise NotImplementedError


def write_split(root: str | Path, name: str, stems: Sequence[str]) -> Path:
    """写 root/splits/{name}.txt（每行一个 stem，UTF-8），返回路径。"""
    raise NotImplementedError


def read_split(root: str | Path, name: str) -> list[str]:
    """读 root/splits/{name}.txt，忽略空行和首尾空白。文件不存在时抛 FileNotFoundError。"""
    raise NotImplementedError


class WalkableDataset(Dataset):
    """返回 (image, mask)：image float32 (3, H, W)，mask int64 (H, W)。

    - 初始化时用 index_images 建立 stem → 图片路径索引；stems 中任何 stem 缺图片或缺掩码都抛
      FileNotFoundError（信息中列出前几个缺失项）。
    - __getitem__ 用 dfwalk.utils.read_image_rgb / read_mask 读取；图片与掩码尺寸不一致、或掩码出现
      {0, 1, 255} 以外的值时抛 ValueError（信息包含文件路径和非法值）。
    - transform(image, mask) → (image_tensor, mask_tensor)，通常为 TrainTransform / EvalTransform。
    """

    def __init__(
        self,
        root: str | Path,
        stems: Sequence[str],
        transform: Callable[[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]],
        images_subdir: str = IMAGES_SUBDIR,
        masks_subdir: str = MASKS_SUBDIR,
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def label_stats(root: str | Path, stems: Sequence[str], masks_subdir: str = MASKS_SUBDIR) -> dict[str, float]:
    """统计掩码像素分布：{"frames", "walkable_ratio", "non_walkable_ratio", "ignore_ratio"}（比例相对总像素）。"""
    raise NotImplementedError
