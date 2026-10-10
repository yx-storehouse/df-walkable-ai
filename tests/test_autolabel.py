"""dfwalk.autolabel 的单元测试。

全部离线、CPU、使用假模型；唯一需要联网的测试在 DFWALK_NETWORK_TESTS=1 时才运行。
"""

from __future__ import annotations

import math
import os
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dfwalk import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD, NON_WALKABLE, WALKABLE
from dfwalk import autolabel as al
from dfwalk.autolabel import (
    AutoLabeler,
    autolabel_folder,
    build_walkable_lookup,
    compute_infer_size,
    normalize_label,
    walkable_prob_to_label,
)
from dfwalk.utils import read_mask, write_image_rgb, write_mask

# ---------------------------------------------------------------------------
# 假模型与辅助函数
# ---------------------------------------------------------------------------

# 模拟 HF config 反序列化后的 id2label：字符串键，且带一个尾随空格的 "bed "
FAKE_ID2LABEL = {"0": "wall", "1": "sky", "2": "tree", "3": "floor", "4": "bed ", "5": "road"}
SKY, FLOOR = 1, 3
WALKABLE_NAMES = [" Floor ", "BED", "road"]  # 大小写/空白不同，测试规范化


class FakeSegModel(torch.nn.Module):
    """输出 1/4 分辨率 logits：上半部分 "sky" 占优，下半部分 "floor" 占优。

    记录每次调用收到的 pixel_values，便于检查预处理。
    """

    def __init__(self, num_labels: int = 6, strength: float = 10.0, output: str = "obj") -> None:
        super().__init__()
        self.num_labels = num_labels
        self.strength = strength
        self.output = output
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # 让 .to()/.eval() 有意义
        self.calls: list[dict] = []

    def forward(self, pixel_values: torch.Tensor):
        self.calls.append(
            {
                "pixel_values": pixel_values.detach().clone(),
                "grad_enabled": torch.is_grad_enabled(),
                "training": self.training,
            }
        )
        n, _, h, w = pixel_values.shape
        lh, lw = max(1, h // 4), max(1, w // 4)
        logits = torch.zeros(n, self.num_labels, lh, lw, device=pixel_values.device)
        top = lh // 2
        logits[:, SKY, :top] = self.strength
        logits[:, FLOOR, top:] = self.strength
        if self.output == "obj":
            return SimpleNamespace(logits=logits)
        if self.output == "dict":
            return {"logits": logits}
        if self.output == "tuple":
            return (logits,)
        if self.output == "tensor":
            return logits
        raise AssertionError(self.output)


def make_labeler(model=None, **kwargs) -> AutoLabeler:
    params = dict(
        id2label=FAKE_ID2LABEL,
        walkable_labels=WALKABLE_NAMES,
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
        short_side=64,
        hi=0.7,
        lo=0.3,
        device="cpu",
    )
    params.update(kwargs)
    return AutoLabeler(model if model is not None else FakeSegModel(), **params)


def random_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _reference_apply_hud_ignore(mask: np.ndarray, rects) -> np.ndarray:
    """按 dfwalk.hud 文档约定的参考实现（起点 floor、终点 ceil、裁剪到图像内）。"""
    out = mask.copy()
    h, w = mask.shape[:2]
    for x0, y0, x1, y1 in rects:
        c0, r0 = max(0, math.floor(x0 * w)), max(0, math.floor(y0 * h))
        c1, r1 = min(w, math.ceil(x1 * w)), min(h, math.ceil(y1 * h))
        out[r0:r1, c0:c1] = IGNORE_INDEX
    return out


@pytest.fixture
def working_hud(monkeypatch):
    """优先使用真实的 dfwalk.hud.apply_hud_ignore；该模块尚未实现时退回文档约定的参考实现。"""
    from dfwalk import hud

    try:
        hud.apply_hud_ignore(np.zeros((2, 2), np.uint8), [[0.0, 0.0, 0.5, 0.5]])
    except NotImplementedError:
        monkeypatch.setattr(hud, "apply_hud_ignore", _reference_apply_hud_ignore)
    return hud


# ---------------------------------------------------------------------------
# normalize_label / build_walkable_lookup
# ---------------------------------------------------------------------------


def test_normalize_label():
    assert normalize_label("bed ") == "bed"
    assert normalize_label("  Dirt Track\t") == "dirt track"
    assert normalize_label("FLOOR") == "floor"


def test_lookup_int_keys():
    lookup = build_walkable_lookup({0: "wall", 1: "floor", 2: "road"}, ["floor"])
    assert lookup.dtype == bool
    assert lookup.tolist() == [False, True, False]


def test_lookup_str_keys_and_trailing_space():
    lookup = build_walkable_lookup(FAKE_ID2LABEL, WALKABLE_NAMES)
    assert len(lookup) == 6
    assert lookup.tolist() == [False, False, False, True, True, True]


def test_lookup_mixed_keys_and_unordered():
    lookup = build_walkable_lookup({"2": "grass", 0: "sky", "1": "wall"}, ["grass"])
    assert lookup.tolist() == [False, False, True]


def test_lookup_noncontiguous_ids_length_is_max_plus_one():
    lookup = build_walkable_lookup({0: "wall", 5: "floor"}, ["floor"])
    assert len(lookup) == 6
    assert np.flatnonzero(lookup).tolist() == [5]


def test_lookup_duplicate_names_all_marked():
    lookup = build_walkable_lookup({0: "floor", 1: "wall", 2: "Floor "}, ["floor"])
    assert lookup.tolist() == [True, False, True]


def test_lookup_unknown_names_listed():
    with pytest.raises(ValueError) as exc:
        build_walkable_lookup(FAKE_ID2LABEL, ["floor", "flor", "lava pit"])
    msg = str(exc.value)
    assert "flor" in msg and "lava pit" in msg
    assert "'floor'" in msg  # 拼写建议
    assert "2" in msg  # 找不到的个数


def test_lookup_single_string_is_one_name():
    lookup = build_walkable_lookup({0: "wall", 1: "floor"}, "floor")
    assert lookup.tolist() == [False, True]


def test_lookup_empty_walkable_list_all_false():
    lookup = build_walkable_lookup({0: "wall", 1: "floor"}, [])
    assert lookup.tolist() == [False, False]


@pytest.mark.parametrize("id2label", [{}, {"x": "floor"}, {-1: "floor"}])
def test_lookup_bad_id2label(id2label):
    with pytest.raises(ValueError):
        build_walkable_lookup(id2label, ["floor"])


def test_default_config_labels_are_ade_names():
    """默认配置中的可走类别名都应是 ADE20K 的类别名（这里只检查拼写规范化后无空白异常）。"""
    from dfwalk.config import DEFAULT_WALKABLE_LABELS

    assert all(normalize_label(n) == n for n in DEFAULT_WALKABLE_LABELS)


# ---------------------------------------------------------------------------
# walkable_prob_to_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prob_to_label_boundaries(dtype):
    prob = np.array([[0.0, 0.3, 0.3001, 0.5, 0.6999, 0.7, 1.0]], dtype=dtype)
    before = prob.copy()
    out = walkable_prob_to_label(prob, hi=0.7, lo=0.3)
    assert out.dtype == np.uint8
    assert out.shape == prob.shape
    assert out.tolist() == [[0, 0, 255, 255, 255, 1, 1]]
    np.testing.assert_array_equal(prob, before)  # 不修改输入


def test_prob_to_label_numpy_scalar_thresholds_on_float32():
    prob = np.array([0.7, 0.3], dtype=np.float32)
    out = walkable_prob_to_label(prob, hi=np.float64(0.7), lo=np.float64(0.3))
    assert out.tolist() == [WALKABLE, NON_WALKABLE]


def test_prob_to_label_extreme_thresholds_and_nan():
    prob = np.array([0.0, 0.5, 1.0, np.nan], dtype=np.float32)
    assert walkable_prob_to_label(prob, hi=1.0, lo=0.0).tolist() == [0, 255, 1, 255]


@pytest.mark.parametrize("hi,lo", [(0.5, 0.5), (0.3, 0.7), (1.1, 0.3), (0.7, -0.1)])
def test_prob_to_label_invalid_thresholds(hi, lo):
    with pytest.raises(ValueError):
        walkable_prob_to_label(np.zeros((2, 2), np.float32), hi=hi, lo=lo)


def test_prob_to_label_2d_values():
    prob = np.array([[0.1, 0.9], [0.5, 0.71]], dtype=np.float32)
    assert walkable_prob_to_label(prob, 0.7, 0.3).tolist() == [[0, 1], [255, 1]]


# ---------------------------------------------------------------------------
# compute_infer_size
# ---------------------------------------------------------------------------


def test_infer_size_1080p():
    assert compute_infer_size(1080, 1920, 512) == (512, 896)
    assert compute_infer_size(1920, 1080, 512) == (896, 512)
    assert compute_infer_size(512, 512, 512) == (512, 512)


def test_infer_size_half_rounds_up():
    # 80/32 = 2.5 → 3（Python 的 round 会给 2）
    assert compute_infer_size(100, 100, 80) == (96, 96)


def test_infer_size_minimum_multiple():
    assert compute_infer_size(10, 1000, 8) == (32, 800)
    assert compute_infer_size(1, 1, 1) == (32, 32)


def test_infer_size_custom_multiple():
    h, w = compute_infer_size(720, 1280, 360, multiple=16)
    assert (h, w) == (368, 640)
    assert h % 16 == 0 and w % 16 == 0


@pytest.mark.parametrize(
    "h,w,short",
    [(101, 157, 64), (720, 1280, 512), (1440, 2560, 512), (1080, 1920, 384), (333, 777, 256), (600, 400, 512), (37, 3000, 96)],
)
def test_infer_size_properties(h, w, short):
    nh, nw = compute_infer_size(h, w, short)
    assert nh % 32 == 0 and nw % 32 == 0
    assert nh >= 32 and nw >= 32
    scale = short / min(h, w)
    # 每个维度与精确缩放结果相差不超过半个倍数（或被抬到最小值 32）
    for exact, got in ((h * scale, nh), (w * scale, nw)):
        assert abs(got - exact) <= 16 or (got == 32 and exact < 32)
    # 宽高比基本保持
    assert abs((nw / nh) - (w / h)) / (w / h) < 0.1


@pytest.mark.parametrize("args", [(0, 10, 512), (10, -1, 512), (10, 10, 0), (10, 10, 32, 0)])
def test_infer_size_invalid(args):
    with pytest.raises(ValueError):
        compute_infer_size(*args)


# ---------------------------------------------------------------------------
# AutoLabeler 构造
# ---------------------------------------------------------------------------


def test_labeler_init_attributes_and_eval_mode():
    model = FakeSegModel().train()
    lab = make_labeler(model)
    assert model.training is False
    assert lab.model is model
    assert lab.device == torch.device("cpu")
    assert lab.id2label[4] == "bed "
    assert all(isinstance(k, int) for k in lab.id2label)
    assert lab.walkable_lookup.tolist() == [False, False, False, True, True, True]
    assert lab.image_mean == tuple(IMAGENET_MEAN)
    assert lab.image_std == tuple(IMAGENET_STD)
    assert (lab.short_side, lab.hi, lab.lo) == (64, 0.7, 0.3)


def test_labeler_device_auto_and_torch_device():
    assert make_labeler(device="auto").device.type in {"cpu", "cuda", "mps"}
    assert make_labeler(device=torch.device("cpu")).device == torch.device("cpu")


def test_labeler_accepts_plain_callable():
    calls = []

    def fn(pixel_values):
        calls.append(pixel_values.shape)
        return FakeSegModel()(pixel_values=pixel_values)

    lab = make_labeler(fn)
    prob = lab.walkable_prob(random_image(40, 60))
    assert prob.shape == (40, 60)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(hi=0.3, lo=0.7),
        dict(hi=0.5, lo=0.5),
        dict(short_side=0),
        dict(image_mean=(0.5, 0.5)),
        dict(image_std=(0.5, 0.0, 0.5)),
        dict(walkable_labels=["floor", "lava"]),
    ],
)
def test_labeler_init_validation(kwargs):
    with pytest.raises(ValueError):
        make_labeler(**kwargs)


# ---------------------------------------------------------------------------
# walkable_prob
# ---------------------------------------------------------------------------


def test_walkable_prob_odd_size_shape_and_range():
    model = FakeSegModel()
    lab = make_labeler(model)
    prob = lab.walkable_prob(random_image(101, 157))
    assert prob.shape == (101, 157)
    assert prob.dtype == np.float32
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0
    # 上半部分是 sky，下半部分是 floor
    assert float(prob[:30].max()) < 0.01
    assert float(prob[-30:].min()) > 0.99
    # 输入给模型的尺寸来自 compute_infer_size
    pv = model.calls[0]["pixel_values"]
    assert tuple(pv.shape) == (1, 3, *compute_infer_size(101, 157, 64))
    assert pv.dtype == torch.float32


def test_walkable_prob_runs_without_grad_in_eval_mode():
    model = FakeSegModel()
    lab = make_labeler(model)
    lab.walkable_prob(random_image(50, 70))
    assert model.calls[0]["grad_enabled"] is False
    assert model.calls[0]["training"] is False


def test_walkable_prob_normalization_without_resize():
    model = FakeSegModel()
    mean, std = (0.5, 0.4, 0.3), (0.2, 0.25, 0.5)
    lab = make_labeler(model, image_mean=mean, image_std=std)
    img = np.zeros((64, 96, 3), np.uint8)  # 已是推理尺寸 → 不缩放
    img[..., 0], img[..., 1], img[..., 2] = 10, 128, 250
    lab.walkable_prob(img)
    pv = model.calls[0]["pixel_values"]
    assert tuple(pv.shape) == (1, 3, 64, 96)
    for c, v in enumerate((10, 128, 250)):
        expected = (v / 255.0 - mean[c]) / std[c]
        assert torch.allclose(pv[0, c], torch.full_like(pv[0, c], expected), atol=1e-5)


def test_walkable_prob_channel_order_is_rgb():
    model = FakeSegModel()
    lab = make_labeler(model, image_mean=(0.0, 0.0, 0.0), image_std=(1.0, 1.0, 1.0))
    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0] = 255  # 纯红
    lab.walkable_prob(img)
    pv = model.calls[0]["pixel_values"][0]
    assert torch.allclose(pv[0], torch.ones_like(pv[0]))
    assert torch.allclose(pv[1:], torch.zeros_like(pv[1:]))


def test_walkable_prob_upscales_small_image():
    model = FakeSegModel()
    lab = make_labeler(model)
    prob = lab.walkable_prob(random_image(20, 30))
    assert prob.shape == (20, 30)
    assert tuple(model.calls[0]["pixel_values"].shape) == (1, 3, 64, 96)


def test_walkable_prob_downscales_large_image():
    model = FakeSegModel()
    lab = make_labeler(model, short_side=128)
    prob = lab.walkable_prob(random_image(540, 960))
    assert prob.shape == (540, 960)
    assert tuple(model.calls[0]["pixel_values"].shape) == (1, 3, 128, 224)


def test_walkable_prob_only_upsamples_one_channel(monkeypatch):
    """150 通道 logits 不能被上采样到原图分辨率：F.interpolate 只能收到 1 通道输入。"""
    seen: list[tuple[int, ...]] = []
    orig = torch.nn.functional.interpolate

    def spy(input, *args, **kwargs):
        seen.append(tuple(input.shape))
        assert kwargs.get("mode", "nearest") == "bilinear"
        assert kwargs.get("align_corners") is False
        return orig(input, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", spy)
    id2label = {i: f"class{i}" for i in range(150)}
    id2label[3] = "floor"
    lab = make_labeler(FakeSegModel(num_labels=150), id2label=id2label, walkable_labels=["floor"])
    prob = lab.walkable_prob(random_image(101, 157))
    assert prob.shape == (101, 157)
    assert seen and all(s[1] == 1 for s in seen)


def test_walkable_prob_sums_walkable_softmax():
    """均匀 logits 时，可走概率 = 可走类别数 / 总类别数。"""

    def uniform(pixel_values):
        _, _, h, w = pixel_values.shape
        return SimpleNamespace(logits=torch.zeros(1, 6, h // 4, w // 4))

    lab = make_labeler(uniform)
    prob = lab.walkable_prob(random_image(33, 47))
    np.testing.assert_allclose(prob, 3 / 6, atol=1e-6)


@pytest.mark.parametrize("output", ["dict", "tuple", "tensor"])
def test_walkable_prob_output_variants(output):
    img = random_image(70, 90)
    ref = make_labeler(FakeSegModel(output="obj")).walkable_prob(img)
    got = make_labeler(FakeSegModel(output=output)).walkable_prob(img)
    np.testing.assert_allclose(got, ref)


def test_walkable_prob_no_walkable_classes_is_zero():
    lab = make_labeler(walkable_labels=[])
    prob = lab.walkable_prob(random_image(40, 50))
    assert prob.shape == (40, 50)
    assert float(prob.max()) == 0.0
    assert set(np.unique(lab.label(random_image(40, 50))).tolist()) == {NON_WALKABLE}


def test_walkable_prob_channel_mismatch_raises():
    lab = make_labeler(FakeSegModel(num_labels=8))
    with pytest.raises(ValueError, match="类别"):
        lab.walkable_prob(random_image(40, 50))


def test_walkable_prob_bad_logits_rank_raises():
    lab = make_labeler(lambda pixel_values: SimpleNamespace(logits=torch.zeros(6, 4, 4)))
    with pytest.raises(ValueError):
        lab.walkable_prob(random_image(40, 50))


def test_walkable_prob_unusable_output_raises():
    lab = make_labeler(lambda pixel_values: "nope")
    with pytest.raises(TypeError):
        lab.walkable_prob(random_image(40, 50))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((32, 32), np.uint8),
        np.zeros((32, 32, 4), np.uint8),
        np.zeros((32, 32, 3), np.float32),
        np.zeros((0, 32, 3), np.uint8),
    ],
)
def test_walkable_prob_rejects_bad_images(image):
    with pytest.raises(ValueError):
        make_labeler().walkable_prob(image)


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def test_label_values_and_uncertain_band():
    lab = make_labeler()
    mask = lab.label(random_image(101, 157))
    assert mask.shape == (101, 157) and mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1, 255}
    assert (mask[:30] == NON_WALKABLE).all()
    assert (mask[-30:] == WALKABLE).all()
    # 天空/地面交界处概率介于 lo 和 hi 之间 → 忽略
    assert (mask[40:60] == IGNORE_INDEX).any()


def test_label_matches_prob_to_label():
    lab = make_labeler(hi=0.9, lo=0.1)
    img = random_image(61, 83)
    expected = walkable_prob_to_label(lab.walkable_prob(img), 0.9, 0.1)
    np.testing.assert_array_equal(lab.label(img), expected)


def test_label_with_hud_rects(working_hud):
    lab = make_labeler()
    img = random_image(101, 157)
    plain = lab.label(img)
    rects = [[0.0, 0.0, 0.25, 0.2], [0.6, 0.8, 1.0, 1.0]]
    mask = lab.label(img, hud_rects=rects)
    assert mask.shape == (101, 157)
    # 矩形内部全部为 255（右下角那块原本是可走区域）
    assert (mask[0:20, 0:39] == IGNORE_INDEX).all()
    assert (mask[81:101, 95:157] == IGNORE_INDEX).all()
    assert (plain[81:101, 95:157] == WALKABLE).all()
    # 矩形以外（留出 2 像素边距）保持不变
    outside = np.ones_like(mask, dtype=bool)
    outside[0:23, 0:42] = False
    outside[78:101, 92:157] = False
    np.testing.assert_array_equal(mask[outside], plain[outside])


def test_label_passes_rects_to_hud(monkeypatch):
    from dfwalk import hud

    got = {}

    def fake_apply(mask, rects):
        got["shape"], got["rects"] = mask.shape, rects
        out = mask.copy()
        out[0, 0] = 77
        return out

    monkeypatch.setattr(hud, "apply_hud_ignore", fake_apply)
    rects = [[0.1, 0.1, 0.2, 0.2]]
    mask = make_labeler().label(random_image(30, 40), rects)
    assert got == {"shape": (30, 40), "rects": rects}
    assert mask[0, 0] == 77


# ---------------------------------------------------------------------------
# autolabel_folder
# ---------------------------------------------------------------------------


def _make_images(folder: Path, specs: dict[str, tuple[int, int]]) -> None:
    for i, (name, (h, w)) in enumerate(specs.items()):
        write_image_rgb(folder / name, random_image(h, w, seed=i))


def test_autolabel_folder_writes_skips_and_overwrites(tmp_path):
    images = tmp_path / "images"
    _make_images(images, {"a_000001.png": (60, 80), "a_000002.jpg": (101, 157), "b_000001.png": (48, 64)})
    (images / "notes.txt").write_text("not an image", encoding="utf-8")
    out = tmp_path / "out" / "pseudo_masks"  # 嵌套目录自动创建
    model = FakeSegModel()
    lab = make_labeler(model)

    written = autolabel_folder(lab, images, out)
    assert sorted(p.name for p in written) == ["a_000001.png", "a_000002.png", "b_000001.png"]
    assert len(model.calls) == 3
    for p in written:
        assert p.parent == out and p.is_file()
        m = read_mask(p)
        assert set(np.unique(m).tolist()) <= {0, 1, 255}
    assert read_mask(out / "a_000002.png").shape == (101, 157)
    assert not list(out.glob(".*"))  # 没有残留的临时文件

    # 篡改一个输出：overwrite=False 时跳过，不重新推理
    sentinel = np.full((60, 80), 7, np.uint8)
    write_mask(out / "a_000001.png", sentinel)
    assert autolabel_folder(lab, images, out) == []
    assert len(model.calls) == 3
    np.testing.assert_array_equal(read_mask(out / "a_000001.png"), sentinel)

    # 只补缺失的那一个
    (out / "b_000001.png").unlink()
    again = autolabel_folder(lab, images, out)
    assert [p.name for p in again] == ["b_000001.png"]
    assert len(model.calls) == 4

    # overwrite=True 全部重写
    rewritten = autolabel_folder(lab, images, out, overwrite=True)
    assert len(rewritten) == 3
    assert len(model.calls) == 7
    m = read_mask(out / "a_000001.png")
    assert m.shape == (60, 80) and not (m == 7).any()


def test_autolabel_folder_applies_hud(tmp_path, working_hud):
    images = tmp_path / "images"
    _make_images(images, {"x.png": (100, 160)})
    out = tmp_path / "masks"
    autolabel_folder(make_labeler(), images, out, hud_rects=[[0.5, 0.5, 1.0, 1.0]])
    m = read_mask(out / "x.png")
    assert (m[50:, 80:] == IGNORE_INDEX).all()
    assert (m[:30, :70] == NON_WALKABLE).all()
    assert (m[-20:, :70] == WALKABLE).all()


def test_autolabel_folder_chinese_paths(tmp_path):
    images = tmp_path / "截图"
    _make_images(images, {"零号大坝_000001.png": (50, 70)})
    out = tmp_path / "伪标签"
    written = autolabel_folder(make_labeler(), images, out)
    assert [p.name for p in written] == ["零号大坝_000001.png"]
    assert read_mask(written[0]).shape == (50, 70)


def test_autolabel_folder_empty_dir(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    assert autolabel_folder(make_labeler(), images, tmp_path / "out") == []


def test_autolabel_folder_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        autolabel_folder(make_labeler(), tmp_path / "nope", tmp_path / "out")


def test_autolabel_folder_duplicate_stems_raise_before_writing(tmp_path):
    images = tmp_path / "images"
    _make_images(images, {"a.jpg": (40, 40), "a.png": (40, 40), "b.png": (40, 40)})
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="a"):
        autolabel_folder(make_labeler(), images, out)
    assert not out.exists() or not any(out.iterdir())


def test_autolabel_folder_refuses_same_dir(tmp_path):
    images = tmp_path / "images"
    _make_images(images, {"a.png": (40, 40)})
    before = (images / "a.png").read_bytes()
    with pytest.raises(ValueError):
        autolabel_folder(make_labeler(), images, images / ".." / "images", overwrite=True)
    assert (images / "a.png").read_bytes() == before


def test_autolabel_folder_skips_corrupt_image_with_warning(tmp_path):
    images = tmp_path / "images"
    _make_images(images, {"good.png": (40, 60)})
    (images / "bad.png").write_bytes(b"definitely not a png")
    out = tmp_path / "out"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        written = autolabel_folder(make_labeler(), images, out)
    assert [p.name for p in written] == ["good.png"]
    assert any("bad.png" in str(w.message) for w in rec)
    assert not (out / "bad.png").exists()


# ---------------------------------------------------------------------------
# from_pretrained
# ---------------------------------------------------------------------------


def test_from_pretrained_without_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # import transformers → ImportError
    with pytest.raises(ImportError, match=r"dfwalk\[autolabel\]"):
        AutoLabeler.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", ["floor"])


def test_from_pretrained_wiring_with_fake_transformers(monkeypatch):
    calls = {}

    class FakeProcessor:
        image_mean = np.array([0.5, 0.5, 0.5])  # numpy 数组也要能处理
        image_std = 0.25  # 标量 → 展开成 3 通道

        @classmethod
        def from_pretrained(cls, name):
            calls["processor"] = name
            return cls()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, name):
            calls["model"] = name
            m = FakeSegModel().train()
            m.config = SimpleNamespace(id2label={int(k): v for k, v in FAKE_ID2LABEL.items()})
            return m

    fake = types.ModuleType("transformers")
    fake.AutoImageProcessor = FakeProcessor
    fake.AutoModelForSemanticSegmentation = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake)

    lab = AutoLabeler.from_pretrained("some/model", ["floor", "road"], short_side=96, hi=0.8, lo=0.2, device="auto")
    assert calls == {"processor": "some/model", "model": "some/model"}
    assert lab.image_mean == (0.5, 0.5, 0.5) and lab.image_std == (0.25, 0.25, 0.25)
    assert (lab.short_side, lab.hi, lab.lo) == (96, 0.8, 0.2)
    assert lab.model.training is False
    assert lab.walkable_lookup.tolist() == [False, False, False, True, False, True]
    assert lab.walkable_prob(random_image(50, 50)).shape == (50, 50)


def test_from_pretrained_falls_back_to_imagenet_stats(monkeypatch):
    class NoStatsProcessor:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, name):
            m = FakeSegModel()
            m.config = SimpleNamespace(id2label=dict(FAKE_ID2LABEL))
            return m

    fake = types.ModuleType("transformers")
    fake.AutoImageProcessor = NoStatsProcessor
    fake.AutoModelForSemanticSegmentation = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake)
    lab = AutoLabeler.from_pretrained("x", ["floor"], device="cpu")
    assert lab.image_mean == tuple(IMAGENET_MEAN) and lab.image_std == tuple(IMAGENET_STD)


def test_from_pretrained_casts_model_to_float32(monkeypatch):
    class FakeProcessor:
        image_mean, image_std = IMAGENET_MEAN, IMAGENET_STD

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, name):
            m = FakeSegModel().half()  # 模拟按 fp16 权重加载
            m.config = SimpleNamespace(id2label=dict(FAKE_ID2LABEL))
            return m

    fake = types.ModuleType("transformers")
    fake.AutoImageProcessor = FakeProcessor
    fake.AutoModelForSemanticSegmentation = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake)
    lab = AutoLabeler.from_pretrained("x", ["floor"], device="cpu")
    assert lab.model.dummy.dtype == torch.float32


def test_from_pretrained_real_transformers_tiny_local_model(tmp_path):
    """用 transformers 构建随机初始化的极小 SegFormer 保存到本地目录，再走真实的 from_pretrained（离线）。"""
    transformers = pytest.importorskip("transformers")

    id2label = {0: "wall", 1: "sky", 2: "floor", 3: "bed ", 4: "road"}
    config = transformers.SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[1, 1, 1, 1],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[8, 16, 16, 16],
        num_attention_heads=[1, 1, 1, 1],
        decoder_hidden_size=16,
        num_labels=5,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    torch.manual_seed(0)
    transformers.SegformerForSemanticSegmentation(config).save_pretrained(tmp_path)
    transformers.SegformerImageProcessor(image_mean=[0.4, 0.5, 0.6], image_std=[0.2, 0.3, 0.4]).save_pretrained(tmp_path)

    lab = AutoLabeler.from_pretrained(str(tmp_path), ["floor", "BED", "road"], short_side=64, device="cpu")
    assert isinstance(lab.model, transformers.SegformerForSemanticSegmentation)
    assert lab.model.training is False
    np.testing.assert_allclose(lab.image_mean, (0.4, 0.5, 0.6))
    np.testing.assert_allclose(lab.image_std, (0.2, 0.3, 0.4))
    assert lab.walkable_lookup.tolist() == [False, False, True, True, True]
    img = random_image(101, 157)
    prob = lab.walkable_prob(img)
    assert prob.shape == (101, 157) and prob.dtype == np.float32
    assert 0.0 <= float(prob.min()) and float(prob.max()) <= 1.0
    mask = lab.label(img)
    assert set(np.unique(mask).tolist()) <= {0, 1, 255}


@pytest.mark.skipif(os.environ.get("DFWALK_NETWORK_TESTS") != "1", reason="需要联网：设置 DFWALK_NETWORK_TESTS=1 启用")
def test_from_pretrained_real_hub_model():
    pytest.importorskip("transformers")
    from dfwalk.config import DEFAULT_WALKABLE_LABELS

    lab = AutoLabeler.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512", DEFAULT_WALKABLE_LABELS, short_side=256, device="cpu"
    )
    assert len(lab.walkable_lookup) == 150
    assert int(lab.walkable_lookup.sum()) == len(DEFAULT_WALKABLE_LABELS)
    assert lab.id2label[7] == "bed "  # ADE20K 原始名字带尾随空格

    # 合成画面：上方蓝天、下方灰色地面
    img = np.zeros((216, 384, 3), np.uint8)
    img[:108] = (120, 170, 235)
    img[108:] = (110, 105, 100)
    prob = lab.walkable_prob(img)
    assert prob.shape == (216, 384) and prob.dtype == np.float32
    assert 0.0 <= float(prob.min()) and float(prob.max()) <= 1.0
    mask = lab.label(img, hud_rects=[])
    assert mask.shape == (216, 384)
    assert set(np.unique(mask).tolist()) <= {0, 1, 255}


def test_module_does_not_import_transformers_eagerly():
    """除 from_pretrained 外不依赖 transformers：模块源码顶层不应导入它。"""
    src = Path(al.__file__).read_text(encoding="utf-8")
    top_level = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("transformers" in ln for ln in top_level)
