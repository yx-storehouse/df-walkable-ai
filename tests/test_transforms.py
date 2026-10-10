"""dfwalk.transforms 的单元测试：归一化、评估缩放、训练增强（几何对齐、填充、翻转、颜色抖动、确定性）。

所有图像/掩码都在测试内合成，不依赖网络和外部文件，只用 CPU。
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest
import torch

from dfwalk import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD, NON_WALKABLE, WALKABLE
from dfwalk.transforms import (
    MASK_INTERP,
    EvalTransform,
    TrainTransform,
    _color_jitter,
    _shift_hue,
    to_tensor_normalized,
)

MEAN = np.asarray(IMAGENET_MEAN, np.float32)
STD = np.asarray(IMAGENET_STD, np.float32)
TARGET = (64, 96)  # (H, W)，测试用的小尺寸
VALID_LABELS = {NON_WALKABLE, WALKABLE, IGNORE_INDEX}


# ---------------------------------------------------------------- 合成数据工具


def denorm(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) 归一化张量 → HxWx3 float32，取值约 0~255。"""
    x = t.numpy().transpose(1, 2, 0)
    return (x * STD + MEAN) * 255.0


def random_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def random_mask(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 1000)
    return rng.choice(np.array([NON_WALKABLE, WALKABLE, IGNORE_INDEX], np.uint8), size=(h, w))


def position_pattern(h: int, w: int, block: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """位置编码图像 + 由同一图案导出的掩码。

    - R 通道：可走块为 255，不可走块为 0（块状、左右不对称的图案）
    - G 通道：x 坐标线性编码（0 → 0，w-1 → 255），用来判断是否翻转、裁剪位置
    - B 通道：y 坐标线性编码
    - 掩码：R 高的地方为 WALKABLE，其余为 NON_WALKABLE
    """
    ys, xs = np.mgrid[0:h, 0:w]
    by, bx = ys // block, xs // block
    walk = ((bx * 3 + by * 5) % 7 < 3) | (xs < w // 4)  # 左侧一条全可走，保证左右不对称
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.where(walk, 255, 0)
    img[..., 1] = np.round(xs * 255.0 / (w - 1)).astype(np.uint8)
    img[..., 2] = np.round(ys * 255.0 / (h - 1)).astype(np.uint8)
    mask = np.where(walk, WALKABLE, NON_WALKABLE).astype(np.uint8)
    return img, mask


def geometric_only(**kw) -> TrainTransform:
    """关闭所有颜色/模糊增强的 TrainTransform，只保留几何变换。"""
    params = dict(jitter=0.0, hue=0.0, blur_p=0.0)
    params.update(kw)
    return TrainTransform(TARGET, **params)


# ---------------------------------------------------------------- to_tensor_normalized


def test_to_tensor_normalized_matches_formula():
    img = random_image(5, 7)
    t = to_tensor_normalized(img)
    assert t.dtype == torch.float32
    assert tuple(t.shape) == (3, 5, 7)
    assert t.is_contiguous()
    expected = ((img.astype(np.float64) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    np.testing.assert_allclose(t.numpy(), expected, atol=1e-5)


def test_to_tensor_normalized_custom_mean_std_and_no_aliasing():
    img = random_image(4, 4, seed=3)
    t = to_tensor_normalized(img, mean=(0, 0, 0), std=(1, 1, 1))
    np.testing.assert_allclose(t.numpy(), img.transpose(2, 0, 1) / 255.0, atol=1e-6)
    t2 = to_tensor_normalized(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert float(t2.min()) >= -1.0 - 1e-6 and float(t2.max()) <= 1.0 + 1e-6
    # 修改张量不影响输入
    before = img.copy()
    t.zero_()
    np.testing.assert_array_equal(img, before)


def test_to_tensor_normalized_accepts_float_and_noncontiguous():
    img = random_image(6, 8, seed=4)
    a = to_tensor_normalized(img.astype(np.float32))
    b = to_tensor_normalized(img)
    torch.testing.assert_close(a, b)
    flipped = img[:, ::-1]
    c = to_tensor_normalized(flipped)
    torch.testing.assert_close(c, to_tensor_normalized(np.ascontiguousarray(flipped)))


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 4), (4, 4, 1), (3, 4, 4, 1)])
def test_to_tensor_normalized_rejects_bad_shape(shape):
    with pytest.raises(ValueError):
        to_tensor_normalized(np.zeros(shape, np.uint8))


@pytest.mark.parametrize("mean,std", [((0.5, 0.5), (0.2, 0.2, 0.2)), ((0.5, 0.5, 0.5), (0.2, 0.0, 0.2))])
def test_to_tensor_normalized_rejects_bad_mean_std(mean, std):
    with pytest.raises(ValueError):
        to_tensor_normalized(np.zeros((2, 2, 3), np.uint8), mean, std)


# ---------------------------------------------------------------- EvalTransform


@pytest.mark.parametrize("src", [(128, 192), (32, 48), (20, 300), (64, 96), (1, 1), (720, 1280)])
def test_eval_transform_exact_size_and_dtypes(src):
    h, w = src
    img, mask = random_image(h, w), random_mask(h, w)
    t_img, t_mask = EvalTransform(TARGET)(img, mask)
    assert t_img.dtype == torch.float32 and tuple(t_img.shape) == (3, *TARGET)
    assert t_mask.dtype == torch.int64 and tuple(t_mask.shape) == TARGET
    assert set(torch.unique(t_mask).tolist()) <= VALID_LABELS


def test_eval_transform_mask_none_passthrough():
    out = EvalTransform(TARGET)(random_image(30, 40))
    assert len(out) == 2
    assert out[1] is None
    assert tuple(out[0].shape) == (3, *TARGET)
    out2 = EvalTransform(TARGET)(random_image(30, 40), None)
    assert out2[1] is None


def test_eval_transform_same_size_is_plain_normalization():
    img, mask = random_image(*TARGET), random_mask(*TARGET)
    t_img, t_mask = EvalTransform(TARGET)(img, mask)
    torch.testing.assert_close(t_img, to_tensor_normalized(img))
    np.testing.assert_array_equal(t_mask.numpy(), mask)


def test_eval_transform_interpolation_rules():
    """缩小用 INTER_AREA，放大（或任一方向放大）用 INTER_LINEAR。"""
    big = random_image(128, 192, seed=5)
    t_img, _ = EvalTransform(TARGET)(big)
    ref = cv2.resize(big, (TARGET[1], TARGET[0]), interpolation=cv2.INTER_AREA)
    torch.testing.assert_close(t_img, to_tensor_normalized(ref))

    small = random_image(40, 50, seed=6)
    t_img, _ = EvalTransform(TARGET)(small)
    ref = cv2.resize(small, (TARGET[1], TARGET[0]), interpolation=cv2.INTER_LINEAR)
    torch.testing.assert_close(t_img, to_tensor_normalized(ref))

    mixed = random_image(200, 50, seed=7)  # 高缩小、宽放大
    t_img, _ = EvalTransform(TARGET)(mixed)
    ref = cv2.resize(mixed, (TARGET[1], TARGET[0]), interpolation=cv2.INTER_LINEAR)
    torch.testing.assert_close(t_img, to_tensor_normalized(ref))


def test_eval_transform_area_for_non_integer_downscale():
    """真实缩放比例不是整数（1080p → 384x672 约 2.8 倍）。恰好 2 倍时 INTER_LINEAR 与 INTER_AREA 结果相同，
    区分不了两者，所以这里用 1.5625 倍：必须是 INTER_AREA（抗锯齿）的结果。"""
    img = random_image(100, 150, seed=8)
    area = cv2.resize(img, (TARGET[1], TARGET[0]), interpolation=cv2.INTER_AREA)
    linear = cv2.resize(img, (TARGET[1], TARGET[0]), interpolation=cv2.INTER_LINEAR)
    assert np.abs(area.astype(int) - linear.astype(int)).max() > 20  # 本用例确实能区分两种插值
    t_img, _ = EvalTransform(TARGET)(img)
    torch.testing.assert_close(t_img, to_tensor_normalized(area))
    t_img, _ = TrainTransform(TARGET, scale=(1.0, 1.0), hflip_p=0.0, jitter=0.0, hue=0.0, blur_p=0.0)(img)
    torch.testing.assert_close(t_img, to_tensor_normalized(area))


def test_eval_transform_mask_nearest_upscale_and_downscale():
    mask = random_mask(32, 48, seed=2)
    _, up = EvalTransform(TARGET)(random_image(32, 48), mask)
    np.testing.assert_array_equal(up.numpy(), mask.repeat(2, 0).repeat(2, 1))

    blocky = mask.repeat(2, 0).repeat(2, 1)  # 每个 2x2 块同值
    _, down = EvalTransform((32, 48))(random_image(64, 96), blocky)
    np.testing.assert_array_equal(down.numpy(), mask)


def test_mask_resize_samples_pixel_centers():
    """缩小 3 倍时每个输出像素应取对应 3x3 区域中心的标签（与 INTER_AREA/INTER_LINEAR 的图像对齐）。"""
    h, w = 12, 18
    cols = np.arange(w)[None, :].repeat(h, 0)
    rows = np.arange(h)[:, None].repeat(w, 1)
    mask = np.where((cols % 3 == 1) & (rows % 3 == 1), WALKABLE, NON_WALKABLE).astype(np.uint8)
    _, out = EvalTransform((h // 3, w // 3))(random_image(h, w), mask)
    assert out.numpy().tolist() == np.full((h // 3, w // 3), WALKABLE).tolist()
    small = cv2.resize(mask, (w // 3, h // 3), interpolation=MASK_INTERP)
    assert (small == WALKABLE).all()


def test_eval_transform_does_not_modify_inputs():
    img, mask = random_image(50, 70), random_mask(50, 70)
    img0, mask0 = img.copy(), mask.copy()
    EvalTransform(TARGET)(img, mask)
    np.testing.assert_array_equal(img, img0)
    np.testing.assert_array_equal(mask, mask0)


def test_eval_transform_accepts_other_mask_dtypes():
    mask = random_mask(32, 48)
    _, a = EvalTransform(TARGET)(random_image(32, 48), mask)
    _, b = EvalTransform(TARGET)(random_image(32, 48), mask.astype(np.int64))
    torch.testing.assert_close(a, b)
    _, c = EvalTransform(TARGET)(random_image(32, 48), mask == WALKABLE)
    assert set(torch.unique(c).tolist()) <= {0, 1}


@pytest.mark.parametrize("size", [(0, 10), (10, -1), (10,), (1, 2, 3), "ab", None])
def test_eval_transform_rejects_bad_size(size):
    with pytest.raises(ValueError):
        EvalTransform(size)


@pytest.mark.parametrize("transform_factory", [lambda: EvalTransform(TARGET), lambda: TrainTransform(TARGET)])
def test_transforms_reject_bad_inputs(transform_factory):
    t = transform_factory()
    img = random_image(20, 30)
    with pytest.raises(ValueError):
        t(np.zeros((20, 30), np.uint8))  # 灰度图
    with pytest.raises(ValueError):
        t(np.zeros((20, 30, 4), np.uint8))  # RGBA
    with pytest.raises(ValueError):
        t(np.zeros((0, 30, 3), np.uint8))
    with pytest.raises(ValueError):
        t(img, np.zeros((20, 31), np.uint8))  # 尺寸不一致
    with pytest.raises(ValueError):
        t(img, np.zeros((20, 30, 1), np.uint8))  # 掩码不是 HxW
    with pytest.raises(ValueError):
        t(img, np.full((20, 30), 300, np.int64))  # 超出 uint8 范围
    with pytest.raises(ValueError):
        t(img, np.full((20, 30), -1, np.int64))


# ---------------------------------------------------------------- TrainTransform：形状与类型


@pytest.mark.parametrize("src", [(30, 50), (64, 96), (200, 120), (720, 1280), (1, 1)])
def test_train_transform_shapes_and_dtypes(src):
    h, w = src
    t = TrainTransform(TARGET, blur_p=0.5)
    random.seed(1)
    for _ in range(5):
        t_img, t_mask = t(random_image(h, w), random_mask(h, w))
        assert t_img.dtype == torch.float32 and tuple(t_img.shape) == (3, *TARGET)
        assert t_mask.dtype == torch.int64 and tuple(t_mask.shape) == TARGET
        assert t_img.is_contiguous() and t_mask.is_contiguous()
        assert torch.isfinite(t_img).all()
        assert set(torch.unique(t_mask).tolist()) <= VALID_LABELS


def test_train_transform_mask_none_passthrough():
    t = TrainTransform(TARGET)
    random.seed(0)
    for _ in range(5):
        t_img, t_mask = t(random_image(50, 80))
        assert t_mask is None
        assert tuple(t_img.shape) == (3, *TARGET)


def test_train_transform_mask_values_stay_valid():
    t = TrainTransform(TARGET, scale=(0.4, 1.8), hflip_p=0.5)
    random.seed(7)
    seen: set[int] = set()
    for i in range(20):
        _, t_mask = t(random_image(90, 140, seed=i), random_mask(90, 140, seed=i))
        values = set(torch.unique(t_mask).tolist())
        assert values <= VALID_LABELS
        seen |= values
    assert seen == VALID_LABELS


def test_train_transform_deterministic_under_random_seed():
    t = TrainTransform(TARGET, blur_p=0.5)  # 默认颜色抖动全部开启
    img, mask = random_image(100, 150), random_mask(100, 150)

    def run(seed: int):
        random.seed(seed)
        return [t(img, mask) for _ in range(4)]

    a, b, c = run(123), run(123), run(456)
    for (ai, am), (bi, bm) in zip(a, b):
        assert torch.equal(ai, bi) and torch.equal(am, bm)
    assert any(not torch.equal(ai, ci) for (ai, _), (ci, _) in zip(a, c))


def test_train_transform_does_not_modify_inputs():
    img, mask = random_image(70, 100), random_mask(70, 100)
    img0, mask0 = img.copy(), mask.copy()
    random.seed(3)
    t = TrainTransform(TARGET, hflip_p=1.0, blur_p=1.0)
    for _ in range(3):
        t(img, mask)
    np.testing.assert_array_equal(img, img0)
    np.testing.assert_array_equal(mask, mask0)


# ---------------------------------------------------------------- TrainTransform：几何对齐


def test_train_transform_geometric_alignment():
    """随机缩放/裁剪/翻转后，掩码可走区域必须仍与图像 R 通道的高值区域重合。"""
    img, mask = position_pattern(150, 260)
    t = geometric_only(scale=(0.5, 1.6), hflip_p=0.5)

    flips = set()
    left_g = set()
    for seed in range(30):
        random.seed(seed)
        t_img, t_mask = t(img, mask)
        out = denorm(t_img)
        m = t_mask.numpy()
        assert set(np.unique(m).tolist()) <= VALID_LABELS

        # 填充区域（mask==255）的图像必须是 0
        pad = m == IGNORE_INDEX
        if pad.any():
            assert np.abs(out[pad]).max() < 0.5

        valid = ~pad
        red_high = out[..., 0] > 127.5
        agree = (red_high == (m == WALKABLE))[valid].mean()
        assert agree > 0.98, f"seed={seed}: 图像与掩码不对齐，一致率 {agree:.4f}"
        # 反例：若掩码没有和图像一起翻转，一致率会明显下降（图案左右不对称）
        agree_flipped = (red_high == (m[:, ::-1] == WALKABLE))[valid & valid[:, ::-1]].mean()
        assert agree_flipped < 0.9

        # 用 G 通道（x 坐标编码）判断翻转方向
        g = out[..., 1]
        rows = valid.all(axis=1)
        if rows.any():
            dx = np.diff(g[rows], axis=1)
            flips.add(bool(np.median(dx) < 0))
            left_g.add(int(round(float(g[rows][0, 0]))))
    assert flips == {True, False}, "hflip_p=0.5 时 30 次中应同时出现翻转和不翻转"
    assert len(left_g) > 5, "随机裁剪位置应当变化"


def test_train_transform_exact_alignment_on_integer_upscale():
    """2 倍放大（INTER_LINEAR）+ 翻转时，掩码与图像 R 通道应逐像素完全一致。"""
    img, mask = position_pattern(32, 48, block=8)
    t = geometric_only(scale=(1.0, 1.0), hflip_p=1.0)
    random.seed(0)
    t_img, t_mask = t(img, mask)
    out = denorm(t_img)
    m = t_mask.numpy()
    np.testing.assert_array_equal(out[..., 0] > 127.5, m == WALKABLE)
    np.testing.assert_array_equal(m, mask.repeat(2, 0).repeat(2, 1)[:, ::-1])
    # G 通道（x 编码）在翻转后应单调不增
    assert (np.diff(out[..., 1], axis=1) <= 0.5).all()


def test_train_transform_hflip_exact():
    img, mask = random_image(*TARGET, seed=9), random_mask(*TARGET, seed=9)
    random.seed(0)
    t_img, t_mask = geometric_only(scale=(1.0, 1.0), hflip_p=1.0)(img, mask)
    torch.testing.assert_close(t_img, to_tensor_normalized(img[:, ::-1]))
    np.testing.assert_array_equal(t_mask.numpy(), mask[:, ::-1])

    random.seed(0)
    t_img, t_mask = geometric_only(scale=(1.0, 1.0), hflip_p=0.0)(img, mask)
    torch.testing.assert_close(t_img, to_tensor_normalized(img))
    np.testing.assert_array_equal(t_mask.numpy(), mask)


@pytest.mark.parametrize("hflip_p", [0.0, 1.0])
def test_train_transform_padding_when_scale_below_one(hflip_p):
    h, w = TARGET
    img = np.full((h, w, 3), 200, np.uint8)
    mask = random_mask(h, w, seed=4)
    mask[mask == IGNORE_INDEX] = WALKABLE  # 输入不含 255，输出中的 255 只能来自填充
    random.seed(0)
    t_img, t_mask = geometric_only(scale=(0.5, 0.5), hflip_p=hflip_p)(img, mask)
    out, m = denorm(t_img), t_mask.numpy()
    sh, sw = h // 2, w // 2
    content_cols = slice(0, sw) if hflip_p == 0.0 else slice(w - sw, w)
    pad_cols = slice(sw, w) if hflip_p == 0.0 else slice(0, w - sw)

    assert (m[sh:, :] == IGNORE_INDEX).all()
    assert (m[:, pad_cols] == IGNORE_INDEX).all()
    assert set(np.unique(m[:sh, content_cols]).tolist()) <= {NON_WALKABLE, WALKABLE}
    # 图像填 0（归一化后为 -mean/std）
    assert np.abs(out[sh:, :]).max() < 0.5
    assert np.abs(out[:, pad_cols]).max() < 0.5
    np.testing.assert_allclose(out[:sh, content_cols], 200.0, atol=0.5)
    # 内容区域的掩码就是原掩码的 2 倍缩小（中心对齐最近邻）
    expected = cv2.resize(mask, (sw, sh), interpolation=MASK_INTERP)
    if hflip_p:
        expected = expected[:, ::-1]
    np.testing.assert_array_equal(m[:sh, content_cols], expected)


def test_train_transform_scale_stretches_to_target_aspect():
    """缩放目标只取决于 image_size 和 s，与原图宽高比无关：scale=(1,1) 时等价于 EvalTransform。"""
    for src in [(120, 200), (300, 100), (64, 96), (20, 30)]:
        img, mask = random_image(*src, seed=src[0]), random_mask(*src, seed=src[1])
        random.seed(0)
        a_img, a_mask = geometric_only(scale=(1.0, 1.0), hflip_p=0.0)(img, mask)
        b_img, b_mask = EvalTransform(TARGET)(img, mask)
        torch.testing.assert_close(a_img, b_img)
        assert torch.equal(a_mask, b_mask)


def test_train_transform_large_scale_crops_inside_image():
    """scale > 1 时没有填充，全是原图内容（掩码中不出现 255）。"""
    img, mask = position_pattern(64, 96, block=8)
    t = geometric_only(scale=(1.3, 2.0), hflip_p=0.0)
    for seed in range(10):
        random.seed(seed)
        t_img, t_mask = t(img, mask)
        assert IGNORE_INDEX not in torch.unique(t_mask).tolist()
        assert np.abs(denorm(t_img)[..., 1]).max() > 0  # 有内容


# ---------------------------------------------------------------- TrainTransform：颜色增强


def test_color_jitter_only_changes_image():
    img, mask = random_image(*TARGET, seed=11), random_mask(*TARGET, seed=11)
    t = TrainTransform(TARGET, scale=(1.0, 1.0), hflip_p=0.0, jitter=0.3, hue=0.05, blur_p=1.0)
    changed = 0
    for seed in range(5):
        random.seed(seed)
        t_img, t_mask = t(img, mask)
        np.testing.assert_array_equal(t_mask.numpy(), mask)
        changed += int(not torch.allclose(t_img, to_tensor_normalized(img)))
        out = denorm(t_img)
        assert out.min() > -0.5 and out.max() < 255.5  # 裁剪到 [0, 255]
    assert changed == 5


def test_blur_applies_3x3_gaussian():
    img = random_image(*TARGET, seed=12)
    random.seed(0)
    t_img, _ = geometric_only(scale=(1.0, 1.0), hflip_p=0.0, blur_p=1.0)(img)
    torch.testing.assert_close(t_img, to_tensor_normalized(cv2.GaussianBlur(img, (3, 3), 0)))


def test_color_jitter_identity_and_basic_factors():
    img = random_image(8, 8, seed=13)
    assert _color_jitter(img, 1.0, 1.0, 1.0, 0.0) is img

    flat = np.full((4, 4, 3), 100, np.uint8)
    out = _color_jitter(flat, 2.0, 1.0, 1.0, 0.0)
    assert out.dtype == np.uint8 and (out == 200).all()
    out = _color_jitter(np.full((4, 4, 3), 200, np.uint8), 2.0, 1.0, 1.0, 0.0)
    assert (out == 255).all()  # 裁剪

    # 对比度 0 → 整图变成平均亮度
    out = _color_jitter(img, 1.0, 0.0, 1.0, 0.0)
    assert out.max() - out.min() <= 1
    # 饱和度 0 → 灰度图（R=G=B）
    out = _color_jitter(img, 1.0, 1.0, 0.0, 0.0)
    assert np.abs(out.astype(int) - out[..., :1].astype(int)).max() <= 1


def _hsv_color(h_deg: float) -> np.ndarray:
    hsv = np.array([[[h_deg, 1.0, 1.0]]], np.float32)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255.0


def test_shift_hue_rotations():
    red = np.array([[[255, 0, 0]]], np.float32)
    np.testing.assert_allclose(_shift_hue(red, 1 / 3), [[[0, 255, 0]]], atol=0.5)
    np.testing.assert_allclose(_shift_hue(red, -1 / 3), [[[0, 0, 255]]], atol=0.5)
    np.testing.assert_allclose(_shift_hue(red, 0.5), [[[0, 255, 255]]], atol=0.5)
    np.testing.assert_allclose(_shift_hue(red, 1.0), red, atol=0.5)
    assert _shift_hue(red, 0.0) is red


def test_shift_hue_wraparound():
    # 350° + 20° → 10°；10° - 20° → 350°
    np.testing.assert_allclose(_shift_hue(_hsv_color(350), 20 / 360), _hsv_color(10), atol=0.5)
    np.testing.assert_allclose(_shift_hue(_hsv_color(10), -20 / 360), _hsv_color(350), atol=0.5)
    # 小幅偏移不会因为量化而丢失：0.03 圈 ≈ 10.8°
    np.testing.assert_allclose(_shift_hue(_hsv_color(100), 0.03), _hsv_color(110.8), atol=0.5)


def test_shift_hue_keeps_gray_and_range():
    gray = np.full((3, 3, 3), 128, np.float32)
    np.testing.assert_allclose(_shift_hue(gray, 0.2), gray, atol=0.5)
    img = random_image(16, 16, seed=14).astype(np.float32)
    out = _shift_hue(img, -0.4)
    assert out.dtype == np.float32 and out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 255.0
    # 亮度（HSV 的 V = max(R,G,B)）不变
    np.testing.assert_allclose(out.max(axis=2), img.max(axis=2), atol=0.5)


# ---------------------------------------------------------------- 参数检查


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scale=(0.0, 1.0)),
        dict(scale=(1.2, 0.8)),
        dict(scale=(-1.0, 1.0)),
        dict(scale=(1.0,)),
        dict(hflip_p=1.5),
        dict(hflip_p=-0.1),
        dict(blur_p=2.0),
        dict(jitter=1.0),
        dict(jitter=-0.1),
        dict(hue=0.6),
        dict(hue=-0.01),
        dict(mean=(0.5, 0.5)),
        dict(std=(0.1, 0.0, 0.1)),
    ],
)
def test_train_transform_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        TrainTransform(TARGET, **kwargs)


def test_train_transform_rejects_bad_size():
    with pytest.raises(ValueError):
        TrainTransform((0, 96))


def test_transforms_repr():
    assert "64" in repr(EvalTransform(TARGET))
    assert "TrainTransform" in repr(TrainTransform(TARGET))
