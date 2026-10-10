"""dfwalk.dataset 的单元测试：分组、图片索引、配对、训练/验证划分、划分文件读写、数据集读取与统计。

所有图片/掩码都在 tmp_path 下现场合成（含中文路径），不依赖网络和外部文件，只用 CPU。
"""

from __future__ import annotations

import math
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dfwalk import IGNORE_INDEX, NON_WALKABLE, WALKABLE
from dfwalk.dataset import (
    IMAGES_SUBDIR,
    MASKS_SUBDIR,
    SPLITS_SUBDIR,
    WalkableDataset,
    find_pairs,
    group_of,
    index_images,
    label_stats,
    make_splits,
    read_split,
    write_split,
)
from dfwalk.transforms import EvalTransform, TrainTransform
from dfwalk.utils import write_image_rgb, write_mask

SIZE = (32, 64)  # EvalTransform 输出 (H, W)


# ---------------------------------------------------------------- 合成数据工具


def make_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def make_mask(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    return rng.choice(np.array([NON_WALKABLE, WALKABLE, IGNORE_INDEX], np.uint8), size=(h, w))


def add_sample(root: Path, stem: str, hw: tuple[int, int] = (24, 40), ext: str = ".png",
               mask: np.ndarray | None = None, with_mask: bool = True, seed: int = 0) -> None:
    h, w = hw
    write_image_rgb(root / IMAGES_SUBDIR / f"{stem}{ext}", make_image(h, w, seed))
    if with_mask:
        write_mask(root / MASKS_SUBDIR / f"{stem}.png", make_mask(h, w, seed) if mask is None else mask)


def video_stems(groups: dict[str, int]) -> list[str]:
    """{分组名: 帧数} → frames.py 风格的 stem 列表。"""
    return [f"{g}_{i * 15:06d}" for g, n in groups.items() for i in range(n)]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "数据 根目录"  # 中文 + 空格路径
    r.mkdir()
    return r


# ---------------------------------------------------------------- group_of


@pytest.mark.parametrize(
    "stem,group",
    [
        ("dam_run1_000120", "dam_run1"),
        ("screenshot", "screenshot"),
        ("a_b", "a_b"),
        ("abc_123", "abc"),
        ("v_7", "v"),
        ("零号大坝_第1局_000003", "零号大坝_第1局"),
        ("abc_", "abc_"),
        ("_123", "_123"),
        ("123", "123"),
        ("a__5", "a_"),
        ("run_12a", "run_12a"),
        ("x_１２", "x_１２"),  # 全角数字不算帧序号
    ],
)
def test_group_of(stem, group):
    assert group_of(stem) == group


# ---------------------------------------------------------------- index_images / find_pairs


def test_index_images_maps_stems(root):
    d = root / IMAGES_SUBDIR
    write_image_rgb(d / "a.jpg", make_image(4, 4))
    write_image_rgb(d / "b.png", make_image(4, 4))
    write_image_rgb(d / "大坝_000001.PNG", make_image(4, 4))
    (d / "notes.txt").write_text("x", encoding="utf-8")
    (d / "sub").mkdir()
    write_image_rgb(d / "sub" / "c.jpg", make_image(4, 4))
    index = index_images(d)
    assert set(index) == {"a", "b", "大坝_000001"}
    assert index["a"] == d / "a.jpg"
    assert all(isinstance(p, Path) for p in index.values())


def test_index_images_duplicate_stem_raises(root):
    d = root / IMAGES_SUBDIR
    write_image_rgb(d / "a.jpg", make_image(4, 4))
    write_image_rgb(d / "a.png", make_image(4, 4))
    write_image_rgb(d / "b.png", make_image(4, 4))
    with pytest.raises(ValueError, match="a"):
        index_images(d)


def test_index_images_missing_dir(root):
    with pytest.raises(FileNotFoundError):
        index_images(root / "nope")


def test_index_images_empty_dir(root):
    (root / IMAGES_SUBDIR).mkdir()
    assert index_images(root / IMAGES_SUBDIR) == {}


def test_find_pairs(root):
    for stem in ["c_000002", "a_000000", "b_000001"]:
        add_sample(root, stem, with_mask=False)
    write_mask(root / MASKS_SUBDIR / "a_000000.png", make_mask(24, 40))
    write_mask(root / MASKS_SUBDIR / "c_000002.png", make_mask(24, 40))
    write_mask(root / MASKS_SUBDIR / "z_orphan.png", make_mask(24, 40))  # 没有图片
    write_image_rgb(root / MASKS_SUBDIR / "b_000001.jpg", make_image(24, 40))  # 后缀不对，不算掩码
    assert find_pairs(root) == ["a_000000", "c_000002"]


def test_find_pairs_custom_subdirs_and_missing_masks_dir(root):
    write_image_rgb(root / "imgs" / "x.jpg", make_image(4, 4))
    assert find_pairs(root, images_subdir="imgs", masks_subdir="labels") == []
    write_mask(root / "labels" / "x.png", make_mask(4, 4))
    assert find_pairs(root, images_subdir="imgs", masks_subdir="labels") == ["x"]
    with pytest.raises(FileNotFoundError):
        find_pairs(root)  # 默认 images/ 不存在


def test_find_pairs_duplicate_stem_raises(root):
    add_sample(root, "a", ext=".jpg")
    write_image_rgb(root / IMAGES_SUBDIR / "a.png", make_image(4, 4))
    with pytest.raises(ValueError):
        find_pairs(root)


# ---------------------------------------------------------------- make_splits


def assert_valid_split(stems, train, val):
    assert train == sorted(train) and val == sorted(val)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(stems)
    assert len(train) + len(val) == len(set(stems))
    assert train and val


def test_make_splits_by_group_no_leak():
    groups = {f"vid{i}": 5 + i for i in range(10)}  # 5..14 帧，共 95 帧
    stems = video_stems(groups)
    for seed in range(10):
        train, val = make_splits(stems, 0.2, seed)
        assert_valid_split(stems, train, val)
        train_groups = {group_of(s) for s in train}
        val_groups = {group_of(s) for s in val}
        assert train_groups.isdisjoint(val_groups)
        # 同组帧全部在同一边
        for g in val_groups:
            assert all(s in val for s in stems if group_of(s) == g)
        # 比例大致满足：误差不超过最大分组
        assert abs(len(val) / len(stems) - 0.2) <= max(groups.values()) / len(stems)


def test_make_splits_equal_groups_exact_ratio():
    stems = video_stems({f"g{i}": 10 for i in range(10)})
    for ratio, n_val_groups in [(0.2, 2), (0.25, 3), (0.3, 3), (0.5, 5), (0.1, 1)]:
        train, val = make_splits(stems, ratio, seed=1)
        assert len({group_of(s) for s in val}) == n_val_groups
        assert_valid_split(stems, train, val)


def test_make_splits_deterministic_and_order_independent():
    stems = video_stems({f"run{i}": 3 + (i % 4) for i in range(12)})
    a = make_splits(stems, 0.25, 42)
    b = make_splits(list(reversed(stems)), 0.25, 42)
    shuffled = stems[:]
    random.Random(0).shuffle(shuffled)
    c = make_splits(shuffled, 0.25, 42)
    assert a == b == c
    # 不同 seed 应能得到不同划分
    assert len({tuple(make_splits(stems, 0.25, s)[1]) for s in range(10)}) > 1


def test_make_splits_not_affected_by_global_random_state():
    stems = video_stems({f"g{i}": 4 for i in range(8)})
    random.seed(1)
    a = make_splits(stems, 0.3, 7)
    random.seed(999)
    random.random()
    b = make_splits(stems, 0.3, 7)
    assert a == b


@pytest.mark.parametrize("ratio", [0.01, 0.5, 0.99])
def test_make_splits_two_groups_one_each_side(ratio):
    stems = video_stems({"left": 7, "right": 3})
    for seed in range(5):
        train, val = make_splits(stems, ratio, seed)
        assert_valid_split(stems, train, val)
        assert len({group_of(s) for s in train}) == 1
        assert len({group_of(s) for s in val}) == 1


def test_make_splits_keeps_at_least_one_train_group():
    stems = video_stems({f"g{i}": 2 for i in range(5)})
    train, val = make_splits(stems, 0.95, 0)
    assert len({group_of(s) for s in train}) == 1
    assert len({group_of(s) for s in val}) == 4


def test_make_splits_skips_oversized_group():
    """一段特别长的录像不应因为打乱后排在前面就整段进入验证集。"""
    stems = video_stems({"long": 300, "short1": 50, "short2": 40})
    for seed in range(20):
        train, val = make_splits(stems, 0.2, seed)
        assert {group_of(s) for s in val} == {"short1", "short2"}
        assert {group_of(s) for s in train} == {"long"}


def test_make_splits_all_groups_too_big_still_one_val_group():
    stems = video_stems({"a": 100, "b": 100, "c": 100})
    for seed in range(5):
        train, val = make_splits(stems, 0.01, seed)
        assert_valid_split(stems, train, val)
        assert len({group_of(s) for s in val}) == 1


def test_make_splits_screenshots_without_frame_numbers():
    stems = ["shotA", "shotB", "shotC", "shotD", "shotE"]
    train, val = make_splits(stems, 0.2, 0)
    assert_valid_split(stems, train, val)
    assert len(val) == 1


def test_make_splits_single_group_temporal_split():
    stems = [f"run_{i}" for i in range(1, 13)]  # 不补零：字典序 run_9 > run_12，必须按数字排序
    with pytest.warns(UserWarning):
        train, val = make_splits(stems, 0.25, 0)
    assert val == ["run_10", "run_11", "run_12"]
    assert train == sorted(f"run_{i}" for i in range(1, 10))
    # 与 seed 无关
    with pytest.warns(UserWarning):
        assert make_splits(stems, 0.25, 123) == (train, val)


def test_make_splits_single_group_ceil_and_float_edge():
    stems = video_stems({"solo": 30})
    with pytest.warns(UserWarning):
        train, val = make_splits(stems, 0.1, 0)  # 0.1 * 30 = 3.0000000000000004 → 3 帧
    assert len(val) == 3
    assert val == stems[-3:]
    with pytest.warns(UserWarning):
        _, val = make_splits(stems, 0.21, 0)  # ceil(6.3) = 7
    assert len(val) == 7 == math.ceil(0.21 * 30)


def test_make_splits_single_group_both_sides_nonempty():
    stems = ["clip_000001", "clip_000002"]
    with pytest.warns(UserWarning):
        train, val = make_splits(stems, 0.99, 0)
    assert train == ["clip_000001"] and val == ["clip_000002"]
    with pytest.warns(UserWarning):
        train, val = make_splits(stems, 0.01, 0)
    assert train == ["clip_000001"] and val == ["clip_000002"]


def test_make_splits_single_group_with_unnumbered_stem():
    stems = ["x", "x_000002", "x_000001"]  # "x" 与 "x_..." 同属分组 "x"，无序号的排最前
    with pytest.warns(UserWarning):
        train, val = make_splits(stems, 0.34, 0)
    assert train == ["x"] and val == ["x_000001", "x_000002"]


def test_make_splits_multi_group_does_not_warn():
    stems = video_stems({"a": 3, "b": 3})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_splits(stems, 0.5, 0)


def test_make_splits_errors():
    with pytest.raises(ValueError):
        make_splits([], 0.2, 0)
    with pytest.raises(ValueError):
        make_splits(["only_000001"], 0.2, 0)
    with pytest.raises(ValueError):
        make_splits(["solo", "solo"], 0.2, 0)  # 去重后只剩 1 个
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            make_splits(video_stems({"a": 3, "b": 3}), bad, 0)


def test_make_splits_deduplicates():
    stems = video_stems({"a": 4, "b": 4, "c": 4})
    train, val = make_splits(stems + stems[:5], 0.3, 3)
    assert_valid_split(stems, train, val)
    assert (train, val) == make_splits(stems, 0.3, 3)


# ---------------------------------------------------------------- write_split / read_split


def test_split_roundtrip(root):
    stems = ["零号大坝_000001", "dam_run1_000120", "screenshot"]
    path = write_split(root, "train", stems)
    assert path == root / SPLITS_SUBDIR / "train.txt"
    assert path.read_bytes() == "零号大坝_000001\ndam_run1_000120\nscreenshot\n".encode("utf-8")
    assert read_split(root, "train") == stems


def test_split_write_overwrites_and_empty(root):
    write_split(root, "val", ["a", "b"])
    write_split(root, "val", [])
    assert read_split(root, "val") == []


def test_read_split_ignores_blank_lines_whitespace_bom_crlf(root):
    p = root / SPLITS_SUBDIR / "val.txt"
    p.parent.mkdir(parents=True)
    p.write_bytes("﻿  a_000001 \r\n\r\n\tb_000002\r\n   \nc\n".encode("utf-8"))
    assert read_split(root, "val") == ["a_000001", "b_000002", "c"]


def test_read_split_missing(root):
    with pytest.raises(FileNotFoundError):
        read_split(root, "train")


def test_write_split_rejects_newline_in_stem(root):
    with pytest.raises(ValueError):
        write_split(root, "train", ["ok", "bad\nstem"])


# ---------------------------------------------------------------- WalkableDataset


def test_dataset_items(root):
    stems = ["a_000000", "a_000015", "b_000000"]
    sizes = [(24, 40), (50, 80), (32, 64)]
    masks = {}
    for i, (stem, hw) in enumerate(zip(stems, sizes)):
        masks[stem] = make_mask(*hw, seed=i)
        add_sample(root, stem, hw=hw, ext=".jpg" if i % 2 else ".png", mask=masks[stem], seed=i)
    ds = WalkableDataset(root, stems, EvalTransform(SIZE))
    assert len(ds) == 3
    assert ds.stems == stems
    for i in range(len(ds)):
        img, mask = ds[i]
        assert img.dtype == torch.float32 and tuple(img.shape) == (3, *SIZE)
        assert mask.dtype == torch.int64 and tuple(mask.shape) == SIZE
        assert set(torch.unique(mask).tolist()) <= {NON_WALKABLE, WALKABLE, IGNORE_INDEX}
    # 与目标尺寸相同的样本：掩码原样保留
    np.testing.assert_array_equal(ds[2][1].numpy(), masks["b_000000"])
    assert ds[-1][1].shape == ds[2][1].shape


def test_dataset_passes_raw_arrays_to_transform(root):
    mask = make_mask(10, 12, seed=3)
    add_sample(root, "s", hw=(10, 12), mask=mask, seed=3)
    seen = {}

    def transform(image, m):
        seen["image"], seen["mask"] = image, m
        return torch.zeros(1), torch.ones(1)

    ds = WalkableDataset(root, ["s"], transform)
    out = ds[0]
    assert torch.equal(out[0], torch.zeros(1)) and torch.equal(out[1], torch.ones(1))
    assert seen["image"].shape == (10, 12, 3) and seen["image"].dtype == np.uint8
    np.testing.assert_array_equal(seen["image"], make_image(10, 12, 3))  # PNG 无损
    assert seen["mask"].dtype == np.uint8
    np.testing.assert_array_equal(seen["mask"], mask)


def test_dataset_dataloader_collate(root):
    stems = [f"vid_{i:06d}" for i in range(5)]
    for i, stem in enumerate(stems):
        add_sample(root, stem, hw=(20 + 4 * i, 30 + 6 * i), seed=i)
    ds = WalkableDataset(root, stems, EvalTransform(SIZE))
    loader = DataLoader(ds, batch_size=2, num_workers=0, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    images, masks = batches[0]
    assert images.dtype == torch.float32 and tuple(images.shape) == (2, 3, *SIZE)
    assert masks.dtype == torch.int64 and tuple(masks.shape) == (2, *SIZE)
    assert tuple(batches[-1][0].shape) == (1, 3, *SIZE)


def test_dataset_with_train_transform_is_seeded(root):
    stems = ["v_000001", "v_000002"]
    for i, stem in enumerate(stems):
        add_sample(root, stem, hw=(40, 70), seed=i)
    ds = WalkableDataset(root, stems, TrainTransform(SIZE))
    random.seed(5)
    a = [ds[i] for i in range(2)]
    random.seed(5)
    b = [ds[i] for i in range(2)]
    for (ai, am), (bi, bm) in zip(a, b):
        assert torch.equal(ai, bi) and torch.equal(am, bm)
        assert tuple(ai.shape) == (3, *SIZE) and tuple(am.shape) == SIZE


def test_dataset_is_picklable_for_workers(root):
    """DataLoader 多 worker 需要 pickle 数据集（连同 transform）。"""
    add_sample(root, "p_000001", hw=(40, 70))
    for transform in (EvalTransform(SIZE), TrainTransform(SIZE)):
        ds = WalkableDataset(root, ["p_000001"], transform)
        clone = pickle.loads(pickle.dumps(ds))
        assert clone.stems == ds.stems and clone.image_paths == ds.image_paths
        random.seed(0)
        a = ds[0]
        random.seed(0)
        b = clone[0]
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_dataset_empty_stems(root):
    (root / IMAGES_SUBDIR).mkdir()
    ds = WalkableDataset(root, [], EvalTransform(SIZE))
    assert len(ds) == 0


def test_dataset_missing_image(root):
    add_sample(root, "ok")
    write_mask(root / MASKS_SUBDIR / "noimg.png", make_mask(24, 40))
    with pytest.raises(FileNotFoundError, match="noimg"):
        WalkableDataset(root, ["ok", "noimg"], EvalTransform(SIZE))


def test_dataset_missing_mask(root):
    add_sample(root, "ok")
    add_sample(root, "nomask", with_mask=False)
    with pytest.raises(FileNotFoundError, match="nomask"):
        WalkableDataset(root, ["ok", "nomask"], EvalTransform(SIZE))


def test_dataset_missing_lists_first_few(root):
    add_sample(root, "ok")
    missing = [f"gone_{i:06d}" for i in range(8)]
    with pytest.raises(FileNotFoundError) as exc:
        WalkableDataset(root, ["ok", *missing], EvalTransform(SIZE))
    msg = str(exc.value)
    assert "gone_000000" in msg and "共 8 个" in msg
    assert "gone_000007" not in msg  # 只列出前几个


def test_dataset_missing_images_dir(root):
    with pytest.raises(FileNotFoundError):
        WalkableDataset(root, ["a"], EvalTransform(SIZE))


def test_dataset_size_mismatch(root):
    add_sample(root, "bad", hw=(24, 40), mask=make_mask(24, 41))
    ds = WalkableDataset(root, ["bad"], EvalTransform(SIZE))
    with pytest.raises(ValueError, match="bad"):
        ds[0]


def test_dataset_invalid_mask_values(root):
    mask = make_mask(24, 40)
    mask[0, 0] = 7
    mask[1, 1] = 128
    add_sample(root, "weird", mask=mask)
    ds = WalkableDataset(root, ["weird"], EvalTransform(SIZE))
    with pytest.raises(ValueError) as exc:
        ds[0]
    msg = str(exc.value)
    assert "[7, 128]" in msg and "weird.png" in msg


def test_dataset_rejects_non_callable_transform(root):
    add_sample(root, "a")
    with pytest.raises(TypeError):
        WalkableDataset(root, ["a"], None)  # type: ignore[arg-type]


def test_dataset_custom_subdirs(root):
    write_image_rgb(root / "imgs" / "a.png", make_image(24, 40))
    write_mask(root / "pseudo_masks" / "a.png", make_mask(24, 40))
    ds = WalkableDataset(root, ["a"], EvalTransform(SIZE), images_subdir="imgs", masks_subdir="pseudo_masks")
    assert ds.mask_paths == [root / "pseudo_masks" / "a.png"]
    assert tuple(ds[0][0].shape) == (3, *SIZE)


# ---------------------------------------------------------------- label_stats


def test_label_stats(root):
    m1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [255, 255, 255, 255]], np.uint8)
    m2 = np.ones((2, 2), np.uint8)
    write_mask(root / MASKS_SUBDIR / "a.png", m1)
    write_mask(root / MASKS_SUBDIR / "b.png", m2)
    stats = label_stats(root, ["a", "b"])
    assert stats["frames"] == 2
    assert stats["walkable_ratio"] == pytest.approx(12 / 20)
    assert stats["non_walkable_ratio"] == pytest.approx(4 / 20)
    assert stats["ignore_ratio"] == pytest.approx(4 / 20)
    assert set(stats) == {"frames", "walkable_ratio", "non_walkable_ratio", "ignore_ratio"}


def test_label_stats_empty(root):
    stats = label_stats(root, [])
    assert stats == {"frames": 0, "walkable_ratio": 0.0, "non_walkable_ratio": 0.0, "ignore_ratio": 0.0}


def test_label_stats_custom_subdir_and_errors(root):
    write_mask(root / "pseudo_masks" / "a.png", np.zeros((3, 3), np.uint8))
    stats = label_stats(root, ["a"], masks_subdir="pseudo_masks")
    assert stats["non_walkable_ratio"] == 1.0 and stats["walkable_ratio"] == 0.0
    with pytest.raises(FileNotFoundError):
        label_stats(root, ["missing"])
    write_mask(root / MASKS_SUBDIR / "bad.png", np.full((3, 3), 2, np.uint8))
    with pytest.raises(ValueError, match=r"\[2\]"):
        label_stats(root, ["bad"])
