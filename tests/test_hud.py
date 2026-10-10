"""dfwalk.hud 的单元测试：归一化矩形 → 像素、HUD 掩码、忽略标记、预览图。"""

from __future__ import annotations

import numpy as np
import pytest

from dfwalk import IGNORE_INDEX, NON_WALKABLE, WALKABLE
from dfwalk.hud import apply_hud_ignore, draw_hud_preview, hud_mask, rect_to_pixels

# ---------------------------------------------------------------- rect_to_pixels


def test_rect_full_image():
    assert rect_to_pixels([0.0, 0.0, 1.0, 1.0], 100, 200) == (0, 0, 200, 100)


def test_rect_exact_boundaries_x_is_columns_y_is_rows():
    # height=100, width=200：x 乘宽度、y 乘高度
    assert rect_to_pixels([0.25, 0.5, 0.75, 1.0], 100, 200) == (50, 50, 150, 100)


def test_rect_floor_start_ceil_end():
    # 0.101*10 = 1.01 → floor 1；0.199*10 = 1.99 → ceil 2
    assert rect_to_pixels([0.101, 0.101, 0.199, 0.199], 10, 10) == (1, 1, 2, 2)
    # 起点 0.15*10=1.5 → 1，终点 0.35*10=3.5 → 4：部分覆盖的像素也算进 HUD
    assert rect_to_pixels([0.15, 0.15, 0.35, 0.35], 10, 10) == (1, 1, 4, 4)


def test_rect_tiny_rect_still_covers_a_pixel():
    c0, r0, c1, r1 = rect_to_pixels([0.501, 0.501, 0.502, 0.502], 10, 10)
    assert (c0, r0, c1, r1) == (5, 5, 6, 6)
    assert c1 - c0 == 1 and r1 - r0 == 1


def test_rect_float_artifacts_do_not_add_extra_pixels():
    # 0.14*100 = 14.000000000000002，终点不应被 ceil 成 15
    assert 0.14 * 100 > 14
    assert rect_to_pixels([0.0, 0.0, 0.14, 0.14], 100, 100) == (0, 0, 14, 14)
    # 0.07*100 = 7.000000000000001，起点 floor 仍为 7
    assert rect_to_pixels([0.07, 0.07, 0.5, 0.5], 100, 100) == (7, 7, 50, 50)
    # 0.29*100 = 28.999999999999996，不应被 floor 成 28
    assert 0.29 * 100 < 29
    assert rect_to_pixels([0.29, 0.29, 0.5, 0.5], 100, 100) == (29, 29, 50, 50)
    # 1920x1080 下常见的配置值
    assert rect_to_pixels([0.2, 0.7, 0.8, 0.9], 1080, 1920) == (384, 756, 1536, 972)


def test_rect_is_clipped_to_image():
    assert rect_to_pixels([-0.1, -0.5, 1.2, 1.5], 50, 80) == (0, 0, 80, 50)
    assert rect_to_pixels([0.5, 0.5, 1.7, 1.7], 10, 10) == (5, 5, 10, 10)


def test_rect_completely_outside_is_empty():
    c0, r0, c1, r1 = rect_to_pixels([1.1, 1.1, 1.5, 1.5], 10, 20)
    assert (c0, r0, c1, r1) == (20, 10, 20, 10)
    c0, r0, c1, r1 = rect_to_pixels([-0.5, -0.5, -0.1, -0.1], 10, 20)
    assert c1 == c0 == 0 and r1 == r0 == 0


def test_rect_inverted_or_degenerate_is_empty_not_negative():
    c0, r0, c1, r1 = rect_to_pixels([0.8, 0.8, 0.2, 0.2], 10, 10)
    assert c1 >= c0 and r1 >= r0
    assert (c1 - c0) * (r1 - r0) == 0
    c0, r0, c1, r1 = rect_to_pixels([0.5, 0.5, 0.5, 0.5], 10, 10)
    assert c1 == c0 and r1 == r0


def test_rect_returns_python_ints_and_accepts_numpy_and_tuples():
    out = rect_to_pixels(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), np.int64(10), np.int64(10))
    assert all(type(v) is int for v in out)
    assert rect_to_pixels((0, 0, 1, 1), 4, 6) == (0, 0, 6, 4)


def test_rect_zero_size_image():
    assert rect_to_pixels([0.1, 0.1, 0.9, 0.9], 0, 0) == (0, 0, 0, 0)


@pytest.mark.parametrize("rect", [[0.1, 0.2, 0.3], [0, 0, 1, 1, 0], []])
def test_rect_wrong_length_raises(rect):
    with pytest.raises(ValueError):
        rect_to_pixels(rect, 10, 10)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_rect_non_finite_raises(bad):
    with pytest.raises(ValueError):
        rect_to_pixels([0.0, 0.0, bad, 1.0], 10, 10)


def test_rect_negative_size_raises():
    with pytest.raises(ValueError):
        rect_to_pixels([0, 0, 1, 1], -1, 10)


# ---------------------------------------------------------------- hud_mask


def test_hud_mask_empty_rects_all_false():
    m = hud_mask(12, 20, [])
    assert m.shape == (12, 20)
    assert m.dtype == bool
    assert not m.any()
    assert not hud_mask(5, 5, None).any()


def test_hud_mask_single_rect_matches_pixels():
    m = hud_mask(100, 200, [[0.25, 0.5, 0.75, 1.0]])
    expected = np.zeros((100, 200), bool)
    expected[50:100, 50:150] = True
    np.testing.assert_array_equal(m, expected)


def test_hud_mask_union_of_overlapping_rects():
    rects = [[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75], [0.9, 0.9, 1.0, 1.0]]
    m = hud_mask(8, 8, rects)
    expected = np.zeros((8, 8), bool)
    expected[0:4, 0:4] = True
    expected[2:6, 2:6] = True
    expected[7:8, 7:8] = True
    np.testing.assert_array_equal(m, expected)
    assert m.sum() == 16 + 16 - 4 + 1


def test_hud_mask_accepts_numpy_rect_array_and_out_of_range_rects():
    rects = np.array([[0.0, 0.0, 0.5, 1.0], [2.0, 2.0, 3.0, 3.0]])
    m = hud_mask(4, 4, rects)
    assert m[:, :2].all() and not m[:, 2:].any()


# ---------------------------------------------------------------- apply_hud_ignore


def _sample_mask(h: int = 10, w: int = 10) -> np.ndarray:
    m = np.full((h, w), NON_WALKABLE, np.uint8)
    m[h // 2 :, :] = WALKABLE
    return m


def test_apply_hud_ignore_sets_ignore_inside_and_keeps_outside():
    mask = _sample_mask()
    out = apply_hud_ignore(mask, [[0.0, 0.0, 0.3, 0.3], [0.8, 0.8, 1.0, 1.0]])
    assert out.dtype == np.uint8 and out.shape == mask.shape
    assert (out[0:3, 0:3] == IGNORE_INDEX).all()
    assert (out[8:10, 8:10] == IGNORE_INDEX).all()
    hud = hud_mask(10, 10, [[0.0, 0.0, 0.3, 0.3], [0.8, 0.8, 1.0, 1.0]])
    np.testing.assert_array_equal(out[~hud], mask[~hud])
    assert (out == IGNORE_INDEX).sum() == 9 + 4


def test_apply_hud_ignore_does_not_mutate_input():
    mask = _sample_mask()
    before = mask.copy()
    out = apply_hud_ignore(mask, [[0.0, 0.0, 1.0, 1.0]])
    np.testing.assert_array_equal(mask, before)
    assert (out == IGNORE_INDEX).all()
    assert not np.shares_memory(out, mask)


def test_apply_hud_ignore_empty_rects_returns_equal_copy():
    mask = _sample_mask()
    out = apply_hud_ignore(mask, [])
    np.testing.assert_array_equal(out, mask)
    assert out is not mask
    assert not np.shares_memory(out, mask)


def test_apply_hud_ignore_on_readonly_and_noncontiguous_input():
    base = _sample_mask(12, 12)
    view = base[::2, ::2]  # 6x6 非连续视图
    view.setflags(write=False)
    out = apply_hud_ignore(view, [[0.0, 0.0, 0.5, 0.5]])
    assert out.shape == (6, 6)
    assert (out[0:3, 0:3] == IGNORE_INDEX).all()
    np.testing.assert_array_equal(out[3:, 3:], view[3:, 3:])


def test_apply_hud_ignore_keeps_existing_ignore_pixels():
    mask = _sample_mask()
    mask[9, 0] = IGNORE_INDEX
    out = apply_hud_ignore(mask, [[0.0, 0.0, 0.2, 0.2]])
    assert out[9, 0] == IGNORE_INDEX


def test_apply_hud_ignore_rejects_1d():
    with pytest.raises(ValueError):
        apply_hud_ignore(np.zeros(5, np.uint8), [[0, 0, 1, 1]])


# ---------------------------------------------------------------- draw_hud_preview


def _sample_image(h: int = 120, w: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)


def test_preview_returns_new_uint8_rgb_same_shape_and_keeps_input():
    img = _sample_image()
    before = img.copy()
    out = draw_hud_preview(img, [[0.1, 0.1, 0.6, 0.8]])
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out is not img
    assert not np.shares_memory(out, img)
    np.testing.assert_array_equal(img, before)
    assert not np.array_equal(out, img)


def test_preview_blends_inside_and_leaves_outside_untouched():
    img = np.full((120, 200, 3), 100, np.uint8)
    rect = [0.1, 0.1, 0.6, 0.9]  # 像素 cols 20..120, rows 12..108
    out = draw_hud_preview(img, [rect], color=(255, 0, 0), alpha=0.5, thickness=2)
    # 矩形中心：远离边框和左上角的序号文字
    cy, cx = 80, 90
    expected = np.rint(np.array([100, 100, 100]) * 0.5 + np.array([255, 0, 0]) * 0.5)
    np.testing.assert_array_equal(out[cy, cx], expected.astype(np.uint8))
    # 矩形外（考虑边框线宽留出余量）完全不变
    outside = np.ones((120, 200), bool)
    outside[8:112, 16:124] = False
    np.testing.assert_array_equal(out[outside], img[outside])


def test_preview_draws_border_in_color():
    img = np.full((100, 100, 3), 50, np.uint8)
    out = draw_hud_preview(img, [[0.2, 0.2, 0.8, 0.8]], color=(0, 255, 0), alpha=0.0, thickness=1)
    # alpha=0：填充不改变颜色，只有边框和文字；边框在 [20, 80) 的边缘
    assert tuple(out[50, 20]) == (0, 255, 0)
    assert tuple(out[50, 79]) == (0, 255, 0)
    assert tuple(out[79, 50]) == (0, 255, 0)
    # 内部中心保持原色
    assert tuple(out[60, 60]) == (50, 50, 50)


def test_preview_alpha_one_fills_with_color_and_thickness_zero_no_border():
    img = np.full((100, 100, 3), 50, np.uint8)
    out = draw_hud_preview(img, [[0.2, 0.2, 0.8, 0.8]], color=(10, 20, 30), alpha=1.0, thickness=0)
    assert tuple(out[60, 60]) == (10, 20, 30)
    assert tuple(out[79, 79]) == (10, 20, 30)
    assert tuple(out[19, 19]) == (50, 50, 50)


def test_preview_alpha_is_clipped():
    img = np.full((60, 60, 3), 50, np.uint8)
    out_hi = draw_hud_preview(img, [[0, 0, 1, 1]], color=(200, 200, 200), alpha=5.0, thickness=0)
    out_lo = draw_hud_preview(img, [[0, 0, 1, 1]], color=(200, 200, 200), alpha=-1.0, thickness=0)
    assert tuple(out_hi[45, 45]) == (200, 200, 200)
    assert tuple(out_lo[45, 45]) == (50, 50, 50)


def test_preview_overlap_is_blended_once():
    img = np.full((100, 100, 3), 0, np.uint8)
    rects = [[0.0, 0.0, 0.7, 0.7], [0.3, 0.3, 1.0, 1.0]]
    out = draw_hud_preview(img, rects, color=(200, 0, 0), alpha=0.5, thickness=0)
    single = out[60, 25]  # 只在第一个矩形内
    overlap = out[55, 55]  # 两个矩形重叠
    np.testing.assert_array_equal(single, overlap)
    assert tuple(overlap) == (100, 0, 0)


def test_preview_draws_index_labels():
    img = np.full((200, 300, 3), 128, np.uint8)
    rects = [[0.05, 0.05, 0.45, 0.45], [0.55, 0.55, 0.95, 0.95]]
    out = draw_hud_preview(img, rects, alpha=0.0, thickness=0)
    # alpha=0 且无边框时，唯一的改动来自序号文字（白字黑边）
    changed = np.any(out != img, axis=2)
    assert changed[10:60, 15:80].any()  # 第 0 个矩形左上角
    assert changed[110:160, 165:230].any()  # 第 1 个矩形左上角
    assert (out[changed] == 255).all(axis=1).any()  # 有白色文字像素
    # 有明显比背景(128)暗的描边像素（小字号抗锯齿，达不到纯黑）
    assert (out[changed] < 80).all(axis=1).any()


def test_preview_empty_rects_returns_copy():
    img = _sample_image()
    out = draw_hud_preview(img, [])
    np.testing.assert_array_equal(out, img)
    assert out is not img and not np.shares_memory(out, img)


def test_preview_skips_rects_outside_image():
    img = _sample_image(50, 50)
    out = draw_hud_preview(img, [[1.2, 1.2, 1.5, 1.5]])
    np.testing.assert_array_equal(out, img)


def test_preview_handles_grayscale_rgba_and_noncontiguous_input():
    gray = np.full((40, 60), 90, np.uint8)
    out = draw_hud_preview(gray, [[0, 0, 0.5, 0.5]])
    assert out.shape == (40, 60, 3) and out.dtype == np.uint8

    rgba = np.full((40, 60, 4), 90, np.uint8)
    out = draw_hud_preview(rgba, [[0, 0, 0.5, 0.5]])
    assert out.shape == (40, 60, 3)

    base = _sample_image(80, 120)
    flipped = base[:, ::-1]  # 非 C 连续视图，cv2 绘图函数需要连续内存
    before = flipped.copy()
    out = draw_hud_preview(flipped, [[0.1, 0.1, 0.9, 0.9]], thickness=3)
    assert out.shape == flipped.shape
    np.testing.assert_array_equal(flipped, before)


def test_preview_tiny_image_and_rect_at_bottom_edge():
    img = np.full((8, 8, 3), 10, np.uint8)
    out = draw_hud_preview(img, [[0.9, 0.9, 1.0, 1.0]], thickness=5)
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


def test_preview_bad_color_raises():
    with pytest.raises(ValueError):
        draw_hud_preview(_sample_image(20, 20), [[0, 0, 1, 1]], color=(1, 2))
