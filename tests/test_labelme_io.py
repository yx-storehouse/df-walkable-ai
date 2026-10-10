"""dfwalk.labelme_io 的测试：掩码 ⇄ LabelMe JSON 互转。

全部离线、CPU、使用 tmp_path 和测试内合成的掩码/图片。
"""

from __future__ import annotations

import json
import time
import warnings

import cv2
import numpy as np
import pytest

from dfwalk import IGNORE_INDEX, NON_WALKABLE, WALKABLE
from dfwalk import labelme_io as lio
from dfwalk.labelme_io import (
    LABEL_TO_VALUE,
    LABELME_VERSION,
    VALUE_TO_LABEL,
    convert_labelme_dir,
    export_masks_to_labelme,
    label_to_value,
    labelme_to_mask,
    load_labelme,
    mask_to_labelme,
    mask_to_shapes,
    save_labelme,
)
from dfwalk.utils import read_mask, write_image_rgb, write_mask

N, W, I = NON_WALKABLE, WALKABLE, IGNORE_INDEX
SHAPE_KEYS = ["label", "points", "group_id", "description", "shape_type", "flags"]


# --------------------------------------------------------------------------- 辅助函数


def _poly(label: str, points, shape_type: str | None = "polygon") -> dict:
    shape = {"label": label, "points": [list(map(float, p)) for p in points], "group_id": None, "flags": {}}
    if shape_type is not None:
        shape["shape_type"] = shape_type
    return shape


def _doc(shapes, h: int = 10, w: int = 10) -> dict:
    return {"version": LABELME_VERSION, "shapes": shapes, "imageHeight": h, "imageWidth": w}


def _check_shapes_format(shapes) -> None:
    for s in shapes:
        assert list(s.keys()) == SHAPE_KEYS
        assert s["label"] in ("walkable", "non_walkable", "ignore")
        assert s["group_id"] is None
        assert s["description"] == ""
        assert s["shape_type"] == "polygon"
        assert s["flags"] == {}
        assert len(s["points"]) >= 3
        for p in s["points"]:
            assert isinstance(p, list) and len(p) == 2
            assert all(type(c) is float for c in p)
    json.dumps(shapes)  # 必须可直接序列化


def _roundtrip(mask: np.ndarray, epsilon: float = 0.0, min_area: float = 0.0) -> tuple[np.ndarray, list]:
    shapes = mask_to_shapes(mask, epsilon=epsilon, min_area=min_area)
    _check_shapes_format(shapes)
    h, w = mask.shape
    return labelme_to_mask({"shapes": shapes, "imageHeight": h, "imageWidth": w}), shapes


def _assert_exact(mask: np.ndarray) -> list:
    out, shapes = _roundtrip(mask)
    if not np.array_equal(out, mask):
        bad = np.argwhere(out != mask)
        pytest.fail(f"round-trip 不一致：{len(bad)} 个像素，例如 (y, x)={bad[:5].tolist()}\nmask=\n{mask}\nout=\n{out}")
    return shapes


def _areas(shapes) -> list[float]:
    return [cv2.contourArea(np.asarray(s["points"], np.float32).reshape(-1, 1, 2)) for s in shapes]


def _random_mask(rng: np.random.Generator, h: int, w: int, n_values: int = 3) -> np.ndarray:
    """模糊噪声阈值化得到的随机多值掩码（sigma=0 时为纯噪声，最容易出现对角交叉）。"""
    noise = rng.random((h, w)).astype(np.float32)
    sigma = float(rng.choice([0.0, 0.0, 0.7, 1.2, 2.0, 3.5]))
    if sigma > 0:
        noise = cv2.GaussianBlur(noise, (0, 0), sigma)
    lo, hi = np.sort(rng.uniform(0.1, 0.9, size=2))
    lo, hi = np.quantile(noise, [lo, hi])
    values = rng.permutation(np.array([N, W, I], np.uint8))
    if n_values == 2:
        values[2] = values[1]
    return np.where(noise < lo, values[0], np.where(noise < hi, values[1], values[2])).astype(np.uint8)


# --------------------------------------------------------------------------- 结构化掩码


def _m_rectangles() -> np.ndarray:
    m = np.zeros((20, 30), np.uint8)
    m[2:8, 3:12] = W
    m[10:18, 15:28] = I
    m[12:15, 5:9] = W
    return m


def _m_hole() -> np.ndarray:
    m = np.zeros((20, 20), np.uint8)
    m[2:18, 2:18] = W
    m[6:14, 6:14] = N  # 可走区域中的不可走孔洞
    return m


def _m_island_in_hole() -> np.ndarray:
    m = _m_hole()
    m[9:11, 9:11] = W  # 孔洞中的可走小岛
    return m


def _m_ignore_ring() -> np.ndarray:
    m = np.zeros((24, 24), np.uint8)
    m[3:21, 3:21] = I  # 忽略带
    m[6:18, 6:18] = W  # 被忽略带包围的可走区域
    m[10:13, 10:13] = N
    return m


def _m_hud_borders() -> np.ndarray:
    m = np.full((30, 40), W, np.uint8)
    m[18:, :] = N
    m[:6, :10] = I  # 左上角小地图
    m[26:, 32:] = I  # 右下角弹药
    m[:3, 15:25] = I  # 顶边
    m[14:16, 19:21] = I  # 准星
    m[10:20, 38:] = I  # 右边缘
    return m


def _m_thin_lines() -> np.ndarray:
    m = np.zeros((16, 20), np.uint8)
    m[2, 1:15] = W  # 水平线
    m[4:14, 3] = W  # 竖直线
    for k in range(8):  # 45° 斜线
        m[5 + k, 6 + k] = I
    m[8, 10:19] = W  # 与斜线交叉的水平线
    m[10:15, 17] = W  # 与上面那条组成 L 形
    m[0, :] = I  # 贴着上边界的整行
    m[:, 19] = N
    return m


def _m_single_pixels() -> np.ndarray:
    m = np.zeros((9, 9), np.uint8)
    m[0, 0] = W
    m[0, 8] = I
    m[8, 8] = W
    m[8, 0] = I
    m[4, 4] = W
    m[2, 6] = I
    m[6, 2] = W
    return m


def _m_diagonal_touch() -> np.ndarray:
    m = np.zeros((12, 12), np.uint8)
    # 2x2 棋盘格
    m[1, 1] = W
    m[2, 2] = W
    # 菱形：四个可走像素只靠对角相连，中心的不可走像素与外面的背景对角相连
    for r, c in [(4, 6), (5, 5), (5, 7), (6, 6)]:
        m[r, c] = W
    # 对角链 + 另一种值的反向对角链（交叉）
    for k in range(4):
        m[8 + k, 1 + k] = I
    m[8, 4] = W
    m[9, 3] = W
    m[10, 2] = W
    m[11, 1] = W
    return m


def _m_diamond() -> np.ndarray:
    m = np.zeros((5, 5), np.uint8)
    for r, c in [(1, 2), (2, 1), (2, 3), (3, 2)]:
        m[r, c] = W
    return m


def _m_checkerboard() -> np.ndarray:
    yy, xx = np.mgrid[0:9, 0:11]
    return np.where((yy + xx) % 2 == 0, W, N).astype(np.uint8)


def _m_checkerboard3() -> np.ndarray:
    yy, xx = np.mgrid[0:12, 0:12]
    return np.array([N, W, I], np.uint8)[(yy + 2 * xx) % 3]


def _m_border_touching() -> np.ndarray:
    m = np.zeros((15, 25), np.uint8)
    m[:, :5] = W
    m[:4, :] = I
    m[10:, 20:] = W
    m[7, 24] = I
    m[14, 10:13] = I
    return m


def _m_concentric() -> np.ndarray:
    m = np.zeros((31, 31), np.uint8)
    vals = [W, I, N, W, I, N, W, I]
    for k, v in enumerate(vals):
        m[2 * k : 31 - 2 * k, 2 * k : 31 - 2 * k] = v
    return m


def _m_diamond_rings() -> np.ndarray:
    """嵌套的“菱形环”（只靠对角相连的环），每层都会产生对角交叉。"""
    m = np.zeros((21, 21), np.uint8)
    c = 10
    for k, v in zip(range(9, 0, -2), [W, I, W, I, W]):
        for d in range(k + 1):
            for r, cc in [(c - k + d, c + d), (c + d, c + k - d), (c + k - d, c - d), (c - d, c - k + d)]:
                m[r, cc] = v
    return m


def _m_comb() -> np.ndarray:
    """梳状可走区域：多条 1 像素宽的齿，齿间不可走，外加忽略的齿根。"""
    m = np.zeros((12, 21), np.uint8)
    m[1:3, 1:20] = I
    m[3:11, 1:20:2] = W
    return m


STRUCTURED = {
    "rectangles": _m_rectangles,
    "hole": _m_hole,
    "island_in_hole": _m_island_in_hole,
    "ignore_ring": _m_ignore_ring,
    "hud_borders": _m_hud_borders,
    "thin_lines": _m_thin_lines,
    "single_pixels": _m_single_pixels,
    "diagonal_touch": _m_diagonal_touch,
    "diamond": _m_diamond,
    "checkerboard": _m_checkerboard,
    "checkerboard3": _m_checkerboard3,
    "border_touching": _m_border_touching,
    "concentric": _m_concentric,
    "diamond_rings": _m_diamond_rings,
    "comb": _m_comb,
    "all_non_walkable": lambda: np.zeros((7, 9), np.uint8),
    "all_walkable": lambda: np.full((7, 9), W, np.uint8),
    "all_ignore": lambda: np.full((7, 9), I, np.uint8),
    "one_pixel_image": lambda: np.array([[W]], np.uint8),
    "single_row": lambda: np.array([[N, W, W, I, W, N, N, I]], np.uint8),
    "single_column": lambda: np.array([[N], [W], [W], [I], [W], [N]], np.uint8),
}


# --------------------------------------------------------------------------- label_to_value


@pytest.mark.parametrize(
    "label, value",
    [
        ("walkable", 1),
        ("可走", 1),
        ("non_walkable", 0),
        ("不可走", 0),
        ("ignore", 255),
        ("忽略", 255),
        ("  Walkable ", 1),
        ("NON_WALKABLE", 0),
        ("IgNoRe\n", 255),
        ("　可走　", 1),  # 全角空格
    ],
)
def test_label_to_value(label, value):
    assert label_to_value(label) == value


@pytest.mark.parametrize("label", ["road", "non-walkable", "", "walk able", None, 1])
def test_label_to_value_unknown(label):
    with pytest.raises(ValueError) as ei:
        label_to_value(label)
    msg = str(ei.value)
    for name in LABEL_TO_VALUE:
        assert name in msg


def test_label_tables_consistent():
    for value, name in VALUE_TO_LABEL.items():
        assert LABEL_TO_VALUE[name] == value
    assert set(LABEL_TO_VALUE.values()) == {N, W, I}


# --------------------------------------------------------------------------- labelme_to_mask


def test_polygon_fill_includes_boundary():
    m = labelme_to_mask(_doc([_poly("walkable", [(2, 3), (6, 3), (6, 7), (2, 7)])]))
    expected = np.zeros((10, 10), np.uint8)
    expected[3:8, 2:7] = W  # 顶点在像素中心，边界像素也填充
    assert m.dtype == np.uint8 and m.shape == (10, 10)
    assert np.array_equal(m, expected)


def test_size_from_json_and_explicit_override():
    data = _doc([_poly("walkable", [(0, 0), (3, 0), (3, 3)])], h=6, w=8)
    assert labelme_to_mask(data).shape == (6, 8)
    assert labelme_to_mask(data, height=4, width=5).shape == (4, 5)
    assert labelme_to_mask(data, height=4).shape == (4, 8)


def test_missing_size_raises():
    with pytest.raises(ValueError):
        labelme_to_mask({"shapes": []})
    with pytest.raises(ValueError):
        labelme_to_mask({"shapes": [], "imageHeight": 5})
    with pytest.raises(ValueError):
        labelme_to_mask({"shapes": []}, width=5)
    with pytest.raises(ValueError):
        labelme_to_mask({"shapes": [], "imageHeight": 0, "imageWidth": 5})


def test_empty_shapes_default_value():
    assert np.array_equal(labelme_to_mask(_doc([]), 3, 4), np.zeros((3, 4), np.uint8))
    assert np.array_equal(labelme_to_mask({"imageHeight": 3, "imageWidth": 4}), np.zeros((3, 4), np.uint8))
    m = labelme_to_mask(_doc([_poly("walkable", [(0, 0), (1, 0), (1, 1), (0, 1)])], 4, 4), default_value=I)
    expected = np.full((4, 4), I, np.uint8)
    expected[:2, :2] = W
    assert np.array_equal(m, expected)


def test_rectangle_any_corner_order():
    expected = np.zeros((10, 10), np.uint8)
    expected[2:6, 3:9] = W  # 两个角都包含在内
    for pts in ([(3, 2), (8, 5)], [(8, 5), (3, 2)], [(3, 5), (8, 2)], [(8, 2), (3, 5)]):
        m = labelme_to_mask(_doc([_poly("walkable", pts, "rectangle")]))
        assert np.array_equal(m, expected), pts


def test_rectangle_too_few_points():
    with pytest.raises(ValueError, match=r"shapes\[0\]"):
        labelme_to_mask(_doc([_poly("walkable", [(3, 2)], "rectangle")]))


def test_chinese_labels():
    shapes = [
        _poly("可走", [(0, 0), (9, 0), (9, 9), (0, 9)]),
        _poly("不可走", [(2, 2), (4, 2), (4, 4), (2, 4)]),
        _poly("忽略", [(6, 6), (8, 6), (8, 8)]),
    ]
    m = labelme_to_mask(_doc(shapes))
    assert m[0, 0] == W and m[9, 9] == W
    assert (m[2:5, 2:5] == N).all()
    assert m[6, 6] == I and m[6, 8] == I and m[8, 8] == I
    assert m[8, 6] == W  # 三角形外


def test_shape_type_missing_or_none_is_polygon():
    square = [(0, 0), (4, 0), (4, 4), (0, 4)]
    a = labelme_to_mask(_doc([_poly("walkable", square, None)]))
    shape = _poly("walkable", square)
    shape["shape_type"] = None
    b = labelme_to_mask(_doc([shape]))
    c = labelme_to_mask(_doc([_poly("walkable", square)]))
    assert np.array_equal(a, c) and np.array_equal(b, c)
    assert c.sum() == 25


def test_unknown_label_raises_with_index():
    shapes = [_poly("walkable", [(0, 0), (3, 0), (3, 3)]), _poly("car", [(0, 0), (3, 0), (3, 3)])]
    with pytest.raises(ValueError, match=r"shapes\[1\]") as ei:
        labelme_to_mask(_doc(shapes))
    assert "car" in str(ei.value)
    assert "walkable" in str(ei.value)  # 列出允许的标签名


@pytest.mark.parametrize("shape_type", ["circle", "line", "point", "linestrip", "mask", "points"])
def test_unsupported_shape_type_raises_with_index(shape_type):
    shapes = [
        _poly("walkable", [(0, 0), (3, 0), (3, 3)]),
        _poly("ignore", [(0, 0), (3, 0), (3, 3)]),
        _poly("walkable", [(1, 1), (5, 5)], shape_type),
    ]
    with pytest.raises(ValueError, match=r"shapes\[2\]") as ei:
        labelme_to_mask(_doc(shapes))
    assert shape_type in str(ei.value)


@pytest.mark.parametrize("points", [[], [(1, 1)], [(1, 1), (5, 5)]])
def test_polygon_too_few_points(points):
    shapes = [_poly("walkable", [(0, 0), (3, 0), (3, 3)]), _poly("walkable", points)]
    with pytest.raises(ValueError, match=r"shapes\[1\]"):
        labelme_to_mask(_doc(shapes))


@pytest.mark.parametrize(
    "points", [[[0, 0], [1, float("nan")], [2, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]], "abc"]
)
def test_malformed_points_raise(points):
    shape = {"label": "walkable", "points": points, "shape_type": "polygon"}
    with pytest.raises(ValueError, match=r"shapes\[0\]"):
        labelme_to_mask(_doc([shape]))


def test_later_shapes_override_earlier():
    big = [(0, 0), (9, 0), (9, 9), (0, 9)]
    small = [(3, 3), (5, 3), (5, 5), (3, 5)]
    a = labelme_to_mask(_doc([_poly("walkable", big), _poly("ignore", small)]))
    assert (a[3:6, 3:6] == I).all() and a.sum() == (100 - 9) * W + 9 * I
    b = labelme_to_mask(_doc([_poly("ignore", small), _poly("walkable", big)]))
    assert (b == W).all()
    dot = [(4, 4)] * 3  # 退化多边形：3 个相同顶点，只画 1 个像素
    c = labelme_to_mask(_doc([_poly("walkable", big), _poly("non_walkable", small), _poly("walkable", dot)]))
    assert c[4, 4] == W and c[3, 3] == N


def test_out_of_range_coordinates_are_clipped():
    m = labelme_to_mask(_doc([_poly("walkable", [(-50, -50), (5, -50), (5, 4), (-50, 4)])]))
    expected = np.zeros((10, 10), np.uint8)
    expected[:5, :6] = W
    assert np.array_equal(m, expected)
    m2 = labelme_to_mask(_doc([_poly("walkable", [(20, 20), (30, 20), (30, 30)])]))
    assert not m2.any()
    m3 = labelme_to_mask(_doc([_poly("ignore", [(-1e12, -1e12), (1e12, -1e12), (0, 1e12)])]))
    assert (m3 == I).all()
    m4 = labelme_to_mask(_doc([_poly("walkable", [(12, -3), (25, 4), (12, 40)], "rectangle")]))
    assert not m4.any()


def test_far_above_coordinates_fast_and_consistent():
    """顶点在图像上方极远处时先做半平面裁剪（否则 fillPoly 要空扫上亿行），结果与直接填充一致。"""
    rng = np.random.default_rng(2)
    for _ in range(25):
        pts = np.column_stack([rng.uniform(-5e5, 5e5, 5), rng.uniform(-5e5, 5e5, 5)]).round()
        pts[0] = [3, 4]
        pts[1, 1] = -4e5  # 保证触发裁剪
        ref = np.zeros((12, 16), np.uint8)
        cv2.fillPoly(ref, [pts.astype(np.int32)], int(W))
        got = labelme_to_mask(_doc([_poly("walkable", pts)], 12, 16))
        assert np.array_equal(got, ref)
    t0 = time.perf_counter()
    m = labelme_to_mask(_doc([_poly("ignore", [(-1e12, -1e12), (1e12, -1e12), (0, 1e12)])]))
    m2 = labelme_to_mask(_doc([_poly("walkable", [(0, -3e9), (5, -3e9), (5, -2e9)])]))  # 全在上方
    assert time.perf_counter() - t0 < 1.0
    assert (m == I).all() and not m2.any()


def test_coordinates_round_half_up():
    # 1.5 → 2，2.49 → 2，0.5 → 1：四舍五入到整数像素坐标
    m = labelme_to_mask(_doc([_poly("walkable", [(0.5, 0.5), (2.49, 0.5), (2.49, 1.5), (0.5, 1.5)])], 4, 4))
    expected = np.zeros((4, 4), np.uint8)
    expected[1:3, 1:3] = W
    assert np.array_equal(m, expected)


def test_non_dict_inputs_raise():
    with pytest.raises(ValueError):
        labelme_to_mask([], 3, 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"shapes\[0\]"):
        labelme_to_mask(_doc(["not a shape"]))
    with pytest.raises(ValueError):
        labelme_to_mask(_doc({"label": "walkable"}))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- mask_to_shapes：精确往返


@pytest.mark.parametrize("name", sorted(STRUCTURED))
def test_roundtrip_exact_structured(name):
    mask = STRUCTURED[name]()
    shapes = _assert_exact(mask)
    assert shapes, "非空掩码至少有一个形状"


@pytest.mark.parametrize("name", sorted(STRUCTURED))
def test_roundtrip_exact_structured_transposed_and_flipped(name):
    # 同一结构的转置/翻转版本：轮廓走向和 findContours 起点都不同
    mask = STRUCTURED[name]()
    for variant in (mask.T, mask[::-1], mask[:, ::-1], mask[::-1, ::-1].T):
        _assert_exact(np.ascontiguousarray(variant))


def test_roundtrip_exact_random_masks():
    rng = np.random.default_rng(20241010)
    for i in range(400):
        h, w = (int(v) for v in rng.integers(1, 48, size=2))
        mask = _random_mask(rng, h, w, n_values=2 if i % 4 == 0 else 3)
        out, _ = _roundtrip(mask)
        assert np.array_equal(out, mask), f"随机掩码 #{i} ({h}x{w}) 不一致像素 {int((out != mask).sum())}"


def test_roundtrip_exact_random_masks_large():
    rng = np.random.default_rng(7)
    for _ in range(6):
        mask = _random_mask(rng, 120, 200)
        out, _ = _roundtrip(mask)
        assert np.array_equal(out, mask)


def test_roundtrip_through_json_file(tmp_path):
    mask = _m_diamond_rings()
    data = mask_to_labelme(mask, "../images/x.jpg", epsilon=0, min_area=0)
    path = tmp_path / "x.json"
    save_labelme(path, data)
    assert np.array_equal(labelme_to_mask(load_labelme(path)), mask)


def test_nested_structure_order_and_count():
    """可走区域 → 不可走孔洞 → 可走小岛：外层容器先画，内部区域后画，按面积降序。"""
    mask = _m_island_in_hole()
    shapes = _assert_exact(mask)
    assert [s["label"] for s in shapes] == ["non_walkable", "walkable", "non_walkable", "walkable"]
    areas = _areas(shapes)
    assert areas == sorted(areas, reverse=True)
    # 每个外轮廓都是“填满孔洞”的矩形：外框、可走区域、孔洞、小岛
    assert areas == [19 * 19, 15 * 15, 7 * 7, 1 * 1]


def test_ignore_band_order():
    shapes = _assert_exact(_m_ignore_ring())
    assert [s["label"] for s in shapes] == ["non_walkable", "ignore", "walkable", "non_walkable"]
    areas = _areas(shapes)
    assert areas == sorted(areas, reverse=True)


@pytest.mark.parametrize(
    "name", ["rectangles", "hole", "ignore_ring", "hud_borders", "border_touching", "concentric"]
)
def test_structured_without_diagonal_crossings_is_sorted_by_area(name):
    shapes = _assert_exact(STRUCTURED[name]())
    areas = _areas(shapes)
    assert areas == sorted(areas, reverse=True)


def test_all_one_value_gives_single_full_image_polygon():
    for v, label in VALUE_TO_LABEL.items():
        mask = np.full((6, 8), v, np.uint8)
        shapes = _assert_exact(mask)
        assert len(shapes) == 1 and shapes[0]["label"] == label
        pts = np.asarray(shapes[0]["points"])
        assert pts[:, 0].min() == 0 and pts[:, 0].max() == 7
        assert pts[:, 1].min() == 0 and pts[:, 1].max() == 5


def test_single_pixel_and_line_shapes_have_four_points():
    mask = np.zeros((7, 7), np.uint8)
    mask[3, 3] = W
    mask[0, 1:6] = I
    shapes = _assert_exact(mask)
    by_label = {s["label"]: s for s in shapes}
    assert by_label["walkable"]["points"] == [[3.0, 3.0]] * 4
    assert sorted(map(tuple, by_label["ignore"]["points"])) == sorted([(1.0, 0.0), (5.0, 0.0)] * 2)


def test_diagonal_line_is_exact_even_with_epsilon():
    mask = np.zeros((8, 8), np.uint8)
    for k in range(6):
        mask[1 + k, 1 + k] = W
    for eps in (0.0, 1.0, 3.0):
        out, shapes = _roundtrip(mask, epsilon=eps)
        assert np.array_equal(out, mask), eps


def test_diamond_crossing_needs_correction_shape():
    """菱形中心的不可走像素与外部背景对角相连：画家排序后还需追加一个修正形状。"""
    mask = _m_diamond()
    shapes = _assert_exact(mask)
    assert [s["label"] for s in shapes] == ["non_walkable", "walkable", "non_walkable"]
    assert shapes[-1]["points"] == [[2.0, 2.0]] * 4


def test_run_fallback_is_exact(monkeypatch):
    """修正轮数耗尽时退化为逐行游程线段，依然逐像素一致。"""
    monkeypatch.setattr(lio, "_MAX_FIX_PASSES", 0)
    for make in (_m_diamond, _m_diamond_rings, _m_checkerboard, _m_checkerboard3, _m_diagonal_touch):
        _assert_exact(make())
    rng = np.random.default_rng(3)
    for _ in range(60):
        mask = _random_mask(rng, 30, 40)
        out, _ = _roundtrip(mask)
        assert np.array_equal(out, mask)


def test_fix_passes_converge_without_fallback(monkeypatch):
    """4 连通修正轮次本身就能收敛（不需要游程兜底），棋盘格这种极端情况也只需 1 轮。"""

    def _fail(*args, **kwargs):
        raise AssertionError("不应退化到游程兜底")

    monkeypatch.setattr(lio, "_run_pieces", _fail)
    fix_calls: list[int] = []
    real_fix_pass = lio._fix_pass

    def _counting_fix_pass(mask, region):
        fix_calls.append(int(region.sum()))
        return real_fix_pass(mask, region)

    monkeypatch.setattr(lio, "_fix_pass", _counting_fix_pass)
    for name in sorted(STRUCTURED):
        _assert_exact(STRUCTURED[name]())
    rng = np.random.default_rng(99)
    for _ in range(150):
        mask = _random_mask(rng, 40, 40)
        out, _ = _roundtrip(mask)
        assert np.array_equal(out, mask)

    yy, xx = np.mgrid[0:60, 0:80]
    board = np.where((yy + xx) % 2 == 0, W, N).astype(np.uint8)
    fix_calls.clear()
    _assert_exact(board)
    assert len(fix_calls) == 1


def test_main_pass_matches_retr_tree_even_depth():
    """主轮次（RETR_LIST + 有向面积符号）选出的外轮廓与 RETR_TREE 偶数深度的轮廓逐点一致。"""

    def tree_outer(binary: np.ndarray) -> list[bytes]:
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []
        parent = hierarchy.reshape(-1, 4)[:, 3]
        out = []
        for i, c in enumerate(contours):
            depth, j = 0, parent[i]
            while j >= 0:
                depth, j = depth + 1, parent[j]
            if depth % 2 == 0:
                out.append(c.reshape(-1, 2).astype(np.int32).tobytes())
        return sorted(out)

    rng = np.random.default_rng(42)
    masks = [make() for make in STRUCTURED.values()]
    masks += [_random_mask(rng, *(int(v) for v in rng.integers(1, 50, size=2))) for _ in range(200)]
    for mask in masks:
        pieces = lio._painter_pass(mask)
        for v in (N, W, I):
            got = sorted(p.points.astype(np.int32).tobytes() for p in pieces if p.value == v)
            assert got == tree_outer((mask == v).astype(np.uint8))


def test_fill_count_matches_rendering():
    """内部按 Pick 定理算出的像素数与 fillPoly 实际覆盖的像素数一致（排序/min_area 依赖它）。"""
    rng = np.random.default_rng(11)
    for _ in range(30):
        mask = _random_mask(rng, 25, 30)
        diff = rng.random(mask.shape) < 0.3
        for piece in lio._painter_pass(mask) + lio._fix_pass(mask, diff):
            canvas = np.zeros(mask.shape, np.uint8)
            cv2.fillPoly(canvas, [piece.points], 1)
            assert piece.count == canvas.sum()
            assert piece.area == pytest.approx(cv2.contourArea(piece.points.reshape(-1, 1, 2)))


# --------------------------------------------------------------------------- mask_to_shapes：简化与过滤


def _blob_mask() -> np.ndarray:
    m = np.zeros((120, 160), np.uint8)
    cv2.circle(m, (60, 60), 40, int(W), -1)
    cv2.ellipse(m, (120, 80), (30, 18), 25, 0, 360, int(I), -1)
    cv2.circle(m, (60, 60), 12, int(N), -1)
    cv2.rectangle(m, (0, 100), (159, 119), int(W), -1)
    return m


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


@pytest.mark.parametrize("epsilon", [0.5, 1.0, 2.0])
def test_epsilon_simplifies_with_high_iou(epsilon):
    mask = _blob_mask()
    exact = mask_to_shapes(mask, epsilon=0)
    out, shapes = _roundtrip(mask, epsilon=epsilon)
    assert all(len(s["points"]) >= 3 for s in shapes)
    assert sum(len(s["points"]) for s in shapes) < sum(len(s["points"]) for s in exact)
    assert [s["label"] for s in shapes] == [s["label"] for s in exact]
    for v in (N, W, I):
        assert _iou(out == v, mask == v) > 0.93, v
    assert (out == mask).mean() > 0.97


def test_epsilon_degenerate_contours_keep_three_points():
    mask = np.zeros((20, 20), np.uint8)
    mask[5:7, 2:15] = W  # 2 像素宽的细条，大 epsilon 下会被简化成 2 个点
    mask[10, 3:12] = I
    mask[15, 15] = W
    mask[12:14, 16:18] = I
    out, shapes = _roundtrip(mask, epsilon=5.0)
    assert all(len(s["points"]) >= 3 for s in shapes)
    assert (out == mask).mean() > 0.97
    assert out[15, 15] == W and out[10, 5] == I


def test_epsilon_random_masks_reasonable():
    rng = np.random.default_rng(5)
    for _ in range(40):
        mask = _random_mask(rng, 40, 50)
        out, shapes = _roundtrip(mask, epsilon=1.0)
        assert all(len(s["points"]) >= 3 for s in shapes)
        assert (out == mask).mean() > 0.6


def test_min_area_drops_small_components():
    mask = np.zeros((40, 40), np.uint8)
    mask[5:30, 5:30] = W  # 大可走区域
    mask[10:12, 10:12] = N  # 2x2 小孔洞（4 像素）
    mask[33:35, 33:35] = W  # 2x2 可走斑点
    mask[37, 2] = I  # 单像素忽略
    mask[2, 2:12] = I  # 10 像素细线
    shapes = mask_to_shapes(mask, epsilon=0, min_area=16)
    _check_shapes_format(shapes)
    assert [s["label"] for s in shapes] == ["non_walkable", "walkable"]
    out = labelme_to_mask({"shapes": shapes, "imageHeight": 40, "imageWidth": 40})
    expected = np.zeros((40, 40), np.uint8)
    expected[5:30, 5:30] = W  # 小区域被周围吸收
    assert np.array_equal(out, expected)

    # min_area 比较的是填满后的像素数：4x4 = 16 像素刚好保留，min_area=17 时丢弃
    m2 = np.zeros((10, 10), np.uint8)
    m2[2:6, 2:6] = W
    assert [s["label"] for s in mask_to_shapes(m2, epsilon=0, min_area=16)] == ["non_walkable", "walkable"]
    assert [s["label"] for s in mask_to_shapes(m2, epsilon=0, min_area=17)] == ["non_walkable"]
    # 孔洞计入面积：外径 5x5 的环（16 像素 + 1 像素孔洞 = 25）
    m3 = np.zeros((10, 10), np.uint8)
    m3[2:7, 2:7] = W
    m3[4, 4] = N
    labels = [s["label"] for s in mask_to_shapes(m3, epsilon=0, min_area=25)]
    assert labels == ["non_walkable", "walkable"]


def test_min_area_zero_or_negative_keeps_everything():
    mask = _m_single_pixels()
    assert len(mask_to_shapes(mask, 0, 0)) == len(mask_to_shapes(mask, 0, -5))


# --------------------------------------------------------------------------- mask_to_shapes：输入校验


def test_invalid_masks():
    with pytest.raises(ValueError):
        mask_to_shapes(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        mask_to_shapes(np.zeros((4,), np.uint8))
    bad = np.zeros((4, 4), np.uint8)
    bad[1, 1] = 2
    with pytest.raises(ValueError, match="2"):
        mask_to_shapes(bad)
    with pytest.raises(ValueError):
        mask_to_shapes(np.zeros((3, 3), np.uint8), epsilon=float("nan"))


def test_empty_and_bool_and_other_dtypes():
    assert mask_to_shapes(np.zeros((0, 5), np.uint8)) == []
    b = np.zeros((6, 6), bool)
    b[1:4, 1:4] = True
    assert np.array_equal(mask_to_shapes(b, 0), mask_to_shapes(b.astype(np.uint8), 0))
    m = _m_hole()
    assert mask_to_shapes(m.astype(np.int64), 0) == mask_to_shapes(m, 0)
    # 非 C 连续视图
    assert mask_to_shapes(m[:, ::-1], 0) == mask_to_shapes(np.ascontiguousarray(m[:, ::-1]), 0)


def test_mask_to_shapes_does_not_modify_input():
    mask = _m_diagonal_touch()
    before = mask.copy()
    mask_to_shapes(mask, 0)
    mask_to_shapes(mask, 2.0, 4)
    assert np.array_equal(mask, before)


def test_deterministic():
    rng = np.random.default_rng(1)
    mask = _random_mask(rng, 50, 60)
    assert mask_to_shapes(mask, 1.0, 4) == mask_to_shapes(mask.copy(), 1.0, 4)


def test_performance_noisy_720p():
    rng = np.random.default_rng(0)
    h, w = 720, 1280
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    lo, hi = np.quantile(noise, [0.4, 0.75])
    mask = np.where(noise < lo, N, np.where(noise < hi, W, I)).astype(np.uint8)
    speck = rng.random((h, w)) < 0.02  # 2% 椒盐噪声
    mask[speck] = rng.choice(np.array([N, W, I], np.uint8), size=int(speck.sum()))

    t0 = time.perf_counter()
    shapes = mask_to_shapes(mask, epsilon=0, min_area=0)
    t_exact = time.perf_counter() - t0
    out = labelme_to_mask({"shapes": shapes, "imageHeight": h, "imageWidth": w})
    assert np.array_equal(out, mask)

    t0 = time.perf_counter()
    simplified = mask_to_shapes(mask, epsilon=1.0, min_area=16)
    t_simple = time.perf_counter() - t0
    assert len(simplified) < len(shapes)
    assert t_exact < 4.0, t_exact
    assert t_simple < 4.0, t_simple


def test_performance_speckled_background_720p():
    """大片可走区域里撒 10% 杂点：单个连通域有数万个孔洞（RETR_TREE 建层级在这里是平方复杂度）。"""
    rng = np.random.default_rng(1)
    h, w = 720, 1280
    mask = np.full((h, w), W, np.uint8)
    speck = rng.random((h, w)) < 0.10
    mask[speck] = rng.choice(np.array([N, I], np.uint8), size=int(speck.sum()))
    t0 = time.perf_counter()
    shapes = mask_to_shapes(mask, epsilon=1.0, min_area=16)
    elapsed = time.perf_counter() - t0
    assert [s["label"] for s in shapes] == ["walkable"]  # 杂点全部被 min_area 过滤
    assert elapsed < 4.0, elapsed


# --------------------------------------------------------------------------- mask_to_labelme / JSON 读写


def test_mask_to_labelme_structure():
    mask = _m_hole()
    data = mask_to_labelme(mask, "../images/frame_000001.jpg", epsilon=0, min_area=0)
    assert list(data.keys()) == [
        "version",
        "flags",
        "shapes",
        "imagePath",
        "imageData",
        "imageHeight",
        "imageWidth",
    ]
    assert data["version"] == LABELME_VERSION
    assert data["flags"] == {}
    assert data["imageData"] is None
    assert data["imagePath"] == "../images/frame_000001.jpg"
    assert (data["imageHeight"], data["imageWidth"]) == mask.shape
    assert type(data["imageHeight"]) is int and type(data["imageWidth"]) is int
    assert data["shapes"] == mask_to_shapes(mask, 0, 0)
    assert np.array_equal(labelme_to_mask(data), mask)


def test_mask_to_labelme_normalizes_backslashes():
    data = mask_to_labelme(np.zeros((3, 3), np.uint8), "..\\images\\a.jpg")
    assert data["imagePath"] == "../images/a.jpg"


def test_save_load_roundtrip_chinese(tmp_path):
    data = mask_to_labelme(_m_island_in_hole(), "../图片/第一帧.jpg", epsilon=0)
    data["shapes"][1]["label"] = "可走"
    data["shapes"][1]["description"] = "人工修正：楼梯"
    path = tmp_path / "深层" / "目录" / "第一帧.json"
    save_labelme(path, data)
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp"))
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert "可走" in text and "人工修正：楼梯" in text and "\\u" not in text  # ensure_ascii=False
    assert '\n  "version"' in text  # indent=2
    loaded = load_labelme(path)
    assert loaded == data
    assert np.array_equal(labelme_to_mask(loaded), _m_island_in_hole())


def test_save_overwrites_existing(tmp_path):
    path = tmp_path / "a.json"
    save_labelme(path, {"a": 1})
    save_labelme(path, {"b": "二"})
    assert load_labelme(path) == {"b": "二"}


def test_load_labelme_bom_and_errors(tmp_path):
    p = tmp_path / "bom.json"
    payload = json.dumps({"shapes": [], "imagePath": "图.jpg"}, ensure_ascii=False).encode("utf-8")
    p.write_bytes(b"\xef\xbb\xbf" + payload)
    assert load_labelme(p)["imagePath"] == "图.jpg"
    q = tmp_path / "list.json"
    q.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_labelme(q)
    r = tmp_path / "broken.json"
    r.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_labelme(r)
    with pytest.raises(FileNotFoundError):
        load_labelme(tmp_path / "missing.json")


# --------------------------------------------------------------------------- 批量导出 / 转换


def _rgb(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _setup_dataset(tmp_path):
    root = tmp_path / "data"
    images, masks = root / "images", root / "pseudo_masks"
    m_a = _m_island_in_hole()
    m_b = _m_hud_borders()
    m_cn = _m_diagonal_touch()
    write_image_rgb(images / "a.jpg", _rgb(*m_a.shape))
    write_image_rgb(images / "b.PNG", _rgb(*m_b.shape, seed=1))  # 大写后缀也能找到
    write_image_rgb(images / "中文帧_000001.png", _rgb(*m_cn.shape, seed=2))
    write_mask(masks / "a.png", m_a)
    write_mask(masks / "b.png", m_b)
    write_mask(masks / "中文帧_000001.png", m_cn)
    write_mask(masks / "c.png", m_a)  # 没有对应图片
    (masks / "notes.txt").write_text("ignored", encoding="utf-8")
    return root, {"a": m_a, "b": m_b, "中文帧_000001": m_cn}


def test_export_masks_to_labelme(tmp_path):
    root, masks = _setup_dataset(tmp_path)
    out_dir = root / "labelme"
    with pytest.warns(UserWarning, match="c.png"):
        written = export_masks_to_labelme(
            root / "pseudo_masks", root / "images", out_dir, epsilon=0, min_area=0
        )
    assert written == [out_dir / "a.json", out_dir / "b.json", out_dir / "中文帧_000001.json"]
    assert not (out_dir / "c.json").exists()
    expected_paths = {
        "a": "../images/a.jpg",
        "b": "../images/b.PNG",
        "中文帧_000001": "../images/中文帧_000001.png",
    }
    for stem, mask in masks.items():
        data = load_labelme(out_dir / f"{stem}.json")
        assert data["imagePath"] == expected_paths[stem]
        assert "\\" not in data["imagePath"]
        assert data["imageData"] is None
        assert (data["imageHeight"], data["imageWidth"]) == mask.shape
        assert np.array_equal(labelme_to_mask(data), mask)


def test_export_relative_path_nested_out_dir(tmp_path):
    root, _ = _setup_dataset(tmp_path)
    out_dir = root / "annotations" / "round1"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_masks_to_labelme(root / "pseudo_masks", root / "images", out_dir)
    assert load_labelme(out_dir / "a.json")["imagePath"] == "../../images/a.jpg"
    # 输出目录与图片同目录
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_masks_to_labelme(root / "pseudo_masks", root / "images", root / "images")
    assert load_labelme(root / "images" / "a.json")["imagePath"] == "a.jpg"


def test_export_overwrite_protection(tmp_path):
    root, masks = _setup_dataset(tmp_path)
    out_dir = root / "labelme"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = export_masks_to_labelme(root / "pseudo_masks", root / "images", out_dir)
    assert len(first) == 3

    # 模拟人工修正
    edited = load_labelme(out_dir / "a.json")
    edited["shapes"] = [_poly("walkable", [(0, 0), (19, 0), (19, 19), (0, 19)])]
    edited["flags"] = {"checked": True}
    save_labelme(out_dir / "a.json", edited)
    (out_dir / "b.json").unlink()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        second = export_masks_to_labelme(root / "pseudo_masks", root / "images", out_dir)
    assert second == [out_dir / "b.json"]  # 只补写缺失的
    assert load_labelme(out_dir / "a.json") == edited  # 人工修正未被覆盖

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        third = export_masks_to_labelme(root / "pseudo_masks", root / "images", out_dir, overwrite=True)
    assert len(third) == 3
    assert load_labelme(out_dir / "a.json") != edited


def test_export_skips_size_mismatch(tmp_path):
    images, masks, out = tmp_path / "images", tmp_path / "masks", tmp_path / "labelme"
    write_image_rgb(images / "x.jpg", _rgb(10, 12))
    write_mask(masks / "x.png", np.zeros((12, 10), np.uint8))
    with pytest.warns(UserWarning, match="x.png"):
        written = export_masks_to_labelme(masks, images, out)
    assert written == []
    assert not (out / "x.json").exists()


def test_export_respects_exif_orientation(tmp_path):
    """EXIF 方向为 90° 的 JPEG：cv2 / LabelMe 显示的尺寸是旋转后的，掩码按旋转后的尺寸匹配。"""
    from PIL import Image

    images, masks, out = tmp_path / "images", tmp_path / "masks", tmp_path / "labelme"
    images.mkdir()
    exif = Image.Exif()
    exif[0x0112] = 6  # 顺时针旋转 90° 显示
    Image.fromarray(_rgb(20, 40)).save(images / "rot.jpg", exif=exif.tobytes())
    mask = np.zeros((40, 20), np.uint8)  # 旋转后的 高 x 宽
    mask[5:30, 2:15] = W
    write_mask(masks / "rot.png", mask)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        written = export_masks_to_labelme(masks, images, out, epsilon=0, min_area=0)
    assert written == [out / "rot.json"]
    data = load_labelme(written[0])
    assert (data["imageHeight"], data["imageWidth"]) == (40, 20)
    assert np.array_equal(labelme_to_mask(data), mask)


def test_export_multiple_extensions_prefers_ext_order(tmp_path):
    images, masks, out = tmp_path / "images", tmp_path / "masks", tmp_path / "labelme"
    write_image_rgb(images / "x.png", _rgb(6, 8))
    write_image_rgb(images / "x.jpg", _rgb(6, 8))
    write_mask(masks / "x.png", np.zeros((6, 8), np.uint8))
    with pytest.warns(UserWarning, match="x"):
        written = export_masks_to_labelme(masks, images, out)
    assert written == [out / "x.json"]
    assert load_labelme(out / "x.json")["imagePath"] == "../images/x.jpg"


def test_export_missing_dirs_and_bad_mask(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_masks_to_labelme(tmp_path / "nope", tmp_path, tmp_path / "out")
    masks = tmp_path / "masks"
    masks.mkdir()
    with pytest.raises(FileNotFoundError):
        export_masks_to_labelme(masks, tmp_path / "nope", tmp_path / "out")
    images = tmp_path / "images"
    write_image_rgb(images / "bad.jpg", _rgb(4, 4))
    bad = np.zeros((4, 4), np.uint8)
    bad[0, 0] = 7
    write_mask(masks / "bad.png", bad)
    with pytest.raises(ValueError, match="bad.png"):
        export_masks_to_labelme(masks, images, tmp_path / "out")


def test_export_empty_mask_dir(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "i").mkdir()
    assert export_masks_to_labelme(tmp_path / "m", tmp_path / "i", tmp_path / "o") == []


def test_convert_labelme_dir_applies_hud_ignore(tmp_path):
    json_dir, out_dir = tmp_path / "labelme", tmp_path / "masks"
    full = [(0, 0), (39, 0), (39, 19), (0, 19)]
    save_labelme(json_dir / "f1.json", _doc([_poly("walkable", full)], h=20, w=40))
    tri = [(5, 5), (10, 5), (10, 10)]
    save_labelme(json_dir / "帧2.json", _doc([_poly("不可走", full), _poly("可走", tri)], 20, 40))
    (json_dir / "readme.txt").write_text("x", encoding="utf-8")
    hud = [[0.0, 0.0, 0.25, 0.5], [0.9, 0.9, 1.0, 1.0]]

    written = convert_labelme_dir(json_dir, out_dir, hud_rects=hud)
    assert written == [out_dir / "f1.png", out_dir / "帧2.png"]

    m1 = read_mask(out_dir / "f1.png")
    expected = np.full((20, 40), W, np.uint8)
    expected[:10, :10] = I
    expected[18:, 36:] = I
    assert np.array_equal(m1, expected)

    m2 = read_mask(out_dir / "帧2.png")
    assert m2.shape == (20, 40)
    assert (m2[:10, :10] == I).all() and (m2[18:, 36:] == I).all()
    assert m2[15, 20] == N and m2[10, 6] == N  # 三角形外
    assert m2[10, 10] == W and m2[7, 12] == N and m2[6, 9] == I  # 三角形顶点 / HUD 覆盖三角形

    # 无 HUD 时原样输出
    written = convert_labelme_dir(json_dir, tmp_path / "plain")
    assert np.array_equal(read_mask(written[0]), np.full((20, 40), W, np.uint8))


def test_convert_labelme_dir_error_contains_filename(tmp_path):
    json_dir = tmp_path / "labelme"
    save_labelme(json_dir / "good.json", _doc([], 5, 5))
    save_labelme(json_dir / "坏文件.json", _doc([_poly("car", [(0, 0), (2, 0), (2, 2)])], 5, 5))
    with pytest.raises(ValueError, match="坏文件.json") as ei:
        convert_labelme_dir(json_dir, tmp_path / "masks")
    assert "car" in str(ei.value)

    json_dir2 = tmp_path / "labelme2"
    json_dir2.mkdir()
    (json_dir2 / "broken.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.json"):
        convert_labelme_dir(json_dir2, tmp_path / "masks2")

    json_dir3 = tmp_path / "labelme3"
    save_labelme(json_dir3 / "nosize.json", {"shapes": []})
    with pytest.raises(ValueError, match="nosize.json"):
        convert_labelme_dir(json_dir3, tmp_path / "masks3")

    with pytest.raises(FileNotFoundError):
        convert_labelme_dir(tmp_path / "missing", tmp_path / "masks4")


def test_full_pipeline_roundtrip(tmp_path):
    """伪标签 → LabelMe JSON → 掩码，epsilon=0、min_area=0 时逐像素一致；再叠加 HUD 忽略。"""
    root, masks = _setup_dataset(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_masks_to_labelme(
            root / "pseudo_masks", root / "images", root / "labelme", epsilon=0, min_area=0
        )
    written = convert_labelme_dir(root / "labelme", root / "masks")
    assert sorted(p.stem for p in written) == sorted(masks)
    for stem, mask in masks.items():
        assert np.array_equal(read_mask(root / "masks" / f"{stem}.png"), mask)

    hud = [[0.5, 0.5, 1.0, 1.0]]
    written = convert_labelme_dir(root / "labelme", root / "masks_hud", hud_rects=hud)
    for p in written:
        m = read_mask(p)
        h, w = m.shape
        src = masks[p.stem]
        assert (m[h // 2 :, w // 2 :] == I).all()
        assert np.array_equal(m[: h // 2], src[: h // 2])
