"""掩码 ⇄ LabelMe JSON 互转，用于人工修正伪标签。

工作流：dfwalk autolabel 生成伪标签 → to-labelme 导出 JSON → 在 LabelMe 中修正多边形
       → from-labelme 转回训练用掩码。

LabelMe 中允许的标签名（大小写不敏感，去首尾空白）及其掩码值：
    walkable / 可走        → 1
    non_walkable / 不可走  → 0
    ignore / 忽略          → 255
形状按 JSON 中的顺序依次绘制，后画的覆盖先画的；未被任何形状覆盖的像素为 default_value（默认 0）。
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import cv2
import numpy as np

from . import IGNORE_INDEX, IMAGE_EXTS, NON_WALKABLE, WALKABLE
from .hud import apply_hud_ignore
from .utils import list_images, read_image_rgb, read_mask, write_mask

LABEL_TO_VALUE: dict[str, int] = {
    "walkable": 1,
    "可走": 1,
    "non_walkable": 0,
    "不可走": 0,
    "ignore": 255,
    "忽略": 255,
}
VALUE_TO_LABEL: dict[int, str] = {1: "walkable", 0: "non_walkable", 255: "ignore"}
LABELME_VERSION = "5.4.1"

# 掩码中允许出现的值（顺序也是排序的最后一级键，保证输出确定）
_MASK_VALUES: tuple[int, ...] = (NON_WALKABLE, WALKABLE, IGNORE_INDEX)
# 画家算法之后修正“对角交叉”残差的最大轮数；超过后退化为逐行游程线段（总能精确还原）
_MAX_FIX_PASSES = 64
# 多边形顶点坐标裁剪范围（int32），防止极端坐标转换溢出；fillPoly 在此范围内能正确裁剪
_COORD_LIMIT = 2**31 - 1
# 顶点在图像上方超过这么多行时先做几何裁剪（见 _fill_polygon），避免 fillPoly 空扫上亿行
_CLIP_ABOVE = 100_000


class _Piece(NamedTuple):
    """一个待输出的多边形（整数像素坐标的轮廓）。"""

    value: int  # 掩码值 0 / 1 / 255
    points: np.ndarray  # Nx2 int32，(x, y)，N >= 1
    area: float  # cv2.contourArea（顶点在像素中心的多边形面积）
    count: float  # fillPoly 填充后覆盖的像素数（含边界、含孔洞）


# --------------------------------------------------------------------------- 标签


def label_to_value(label: str) -> int:
    """标签名 → 掩码值；未知标签抛 ValueError，信息中列出允许的标签名。

    匹配时去掉首尾空白（含全角空格）并忽略大小写，例如 " Walkable " → 1、"忽略" → 255。
    """
    if isinstance(label, str):
        key = label.strip().casefold()
        if key in LABEL_TO_VALUE:
            return LABEL_TO_VALUE[key]
    allowed = ", ".join(LABEL_TO_VALUE)
    raise ValueError(f"未知标签 {label!r}；允许的标签名（大小写不敏感）: {allowed}")


# --------------------------------------------------------------------------- JSON → 掩码


def labelme_to_mask(
    data: dict[str, Any],
    height: int | None = None,
    width: int | None = None,
    default_value: int = 0,
) -> np.ndarray:
    """LabelMe JSON（已解析的 dict）→ HxW uint8 掩码。

    - height/width 缺省时取 data["imageHeight"] / data["imageWidth"]，都没有则抛 ValueError。
    - 支持 shape_type: "polygon"（缺省值也按 polygon）、"rectangle"（两个对角点，任意顺序）。
      其他类型（circle、line、point、linestrip、mask 等）抛 ValueError，信息包含形状序号和类型。
    - 多边形顶点四舍五入到整数后用 cv2.fillPoly 填充（含边界）；少于 3 个点的多边形抛 ValueError。
    - 坐标超出图像范围时由 fillPoly 自然裁剪，不报错。

    补充约定：
    - 形状序号从 0 开始，即 data["shapes"] 中的下标，错误信息形如 "shapes[3]: ..."。
    - “四舍五入”为 0.5 向上取整（floor(x + 0.5)），与 mask_to_shapes 输出的整数坐标严格互逆。
    - rectangle 取所有点的外接矩形（通常恰好 2 个点），左上/右下两角都包含在内。
    - 缺少 "shapes" 时视为没有形状，整幅图为 default_value。
    """
    if not isinstance(data, dict):
        raise ValueError(f"LabelMe 数据必须是 JSON 对象（dict），得到 {type(data).__name__}")
    h = _resolve_dim(height, data.get("imageHeight"), "height", "imageHeight")
    w = _resolve_dim(width, data.get("imageWidth"), "width", "imageWidth")
    default_value = int(default_value)
    if not 0 <= default_value <= 255:
        raise ValueError(f"default_value 必须在 0~255 之间，得到 {default_value}")

    mask = np.full((h, w), default_value, dtype=np.uint8)
    shapes = data.get("shapes") or []
    if not isinstance(shapes, list):
        raise ValueError(f'"shapes" 必须是列表，得到 {type(shapes).__name__}')
    for idx, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise ValueError(f"shapes[{idx}]: 形状必须是 JSON 对象，得到 {type(shape).__name__}")
        shape_type = shape.get("shape_type") or "polygon"
        if shape_type not in ("polygon", "rectangle"):
            raise ValueError(
                f"shapes[{idx}]: 不支持的 shape_type {shape_type!r}（label={shape.get('label')!r}），"
                "仅支持 polygon / rectangle"
            )
        try:
            value = label_to_value(shape.get("label"))
        except ValueError as e:
            raise ValueError(f"shapes[{idx}]（shape_type={shape_type!r}）: {e}") from None
        pts = _shape_points_array(shape.get("points"), idx, shape_type)
        if shape_type == "polygon":
            if len(pts) < 3:
                raise ValueError(f"shapes[{idx}]: polygon 至少需要 3 个点，得到 {len(pts)} 个")
            _fill_polygon(mask, pts, value)
        else:  # rectangle：取外接矩形，两角都包含在内
            if len(pts) < 2:
                raise ValueError(f"shapes[{idx}]: rectangle 需要 2 个对角点，得到 {len(pts)} 个")
            # 角点先夹到 [-1, W] x [-1, H]：轴对齐矩形这样裁剪是精确的，也避免极端坐标拖慢 fillPoly
            ipts = _round_points(pts).astype(np.int64)
            x0, x1 = np.clip([ipts[:, 0].min(), ipts[:, 0].max()], -1, w)
            y0, y1 = np.clip([ipts[:, 1].min(), ipts[:, 1].max()], -1, h)
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
            cv2.fillPoly(mask, [corners], int(value), lineType=cv2.LINE_8)
    return mask


def _resolve_dim(explicit: int | None, from_json: Any, name: str, json_key: str) -> int:
    value = explicit if explicit is not None else from_json
    if value is None:
        raise ValueError(f"缺少图像尺寸：未提供 {name}，JSON 中也没有 {json_key}")
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"图像尺寸 {name}={value!r} 不是整数") from None
    if ivalue <= 0 or ivalue != float(value):
        raise ValueError(f"图像尺寸 {name} 必须是正整数，得到 {value!r}")
    return ivalue


def _shape_points_array(points: Any, idx: int, shape_type: str) -> np.ndarray:
    """形状的 points → Nx2 float64；格式不对或含 NaN/inf 时抛 ValueError（含形状序号）。"""
    try:
        arr = np.asarray(points if points is not None else [], dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"shapes[{idx}]（{shape_type}）: points 必须是 [[x, y], ...] 数值列表") from None
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"shapes[{idx}]（{shape_type}）: points 必须是 [[x, y], ...]，得到形状 {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"shapes[{idx}]（{shape_type}）: points 含 NaN 或 inf")
    return arr


def _round_points(pts: np.ndarray) -> np.ndarray:
    """四舍五入（0.5 向上）到整数像素坐标，并裁剪到 int32 范围。"""
    rounded = np.floor(np.asarray(pts, dtype=np.float64) + 0.5)
    return np.clip(rounded, -_COORD_LIMIT, _COORD_LIMIT).astype(np.int32).reshape(-1, 2)


def _fill_polygon(canvas: np.ndarray, pts: np.ndarray, value: int) -> None:
    """与 LabelMe 解析完全相同的填充方式：顶点取整 → cv2.fillPoly（LINE_8，含边界）。

    cv2.fillPoly 会从多边形最上方的扫描线逐行扫到图像底部，即使那些行在图像上方很远处
    （顶点 y = -1e9 时要近十秒；右侧、左侧、下方的远距离坐标不影响速度）。因此只有当顶点在
    图像上方超过 _CLIP_ABOVE 行时，才先把取整后的多边形裁剪到 y >= -_CLIP_ABOVE 的半平面再填充。
    裁剪线离图像足够远，图像内的结果与直接填充一致（仅在斜边从极远处穿过图像时可能有个别像素
    的取整差异）；其余多边形（包括 mask_to_shapes 的全部输出）不做任何处理，保证精确往返。"""
    ipts = _round_points(pts)
    if len(ipts) and ipts[:, 1].min() < -_CLIP_ABOVE:
        clipped = _clip_above(ipts.astype(np.float64), -float(_CLIP_ABOVE))
        if len(clipped) < 3:
            return  # 完全在裁剪线上方（或只擦到裁剪线），图像内没有像素
        ipts = _round_points(clipped)
    cv2.fillPoly(canvas, [ipts], int(value), lineType=cv2.LINE_8)


def _clip_above(pts: np.ndarray, y_min: float) -> np.ndarray:
    """Sutherland–Hodgman：把多边形裁剪到半平面 y >= y_min，返回 Nx2 float64（可能为空）。"""
    out: list[tuple[float, float]] = []
    px, py = (float(v) for v in pts[-1])
    prev_in = py >= y_min
    for x, y in pts.tolist():
        cur_in = y >= y_min
        if cur_in != prev_in:  # 这条边穿过裁剪线：加入交点
            out.append((px + (y_min - py) * (x - px) / (y - py), y_min))
        if cur_in:
            out.append((x, y))
        px, py, prev_in = x, y, cur_in
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


# --------------------------------------------------------------------------- 掩码 → 形状


def mask_to_shapes(mask: np.ndarray, epsilon: float = 1.0, min_area: float = 0.0) -> list[dict[str, Any]]:
    """掩码 → LabelMe shapes 列表，保证按顺序绘制（后覆盖先）能还原掩码，包括嵌套结构
    （例如：可走区域中的不可走孔洞、孔洞中的可走小岛、可走区域外围的忽略带）。

    算法（“画家算法”）：
    1. 对每个值 v ∈ {0, 1, 255}：在二值图 (mask == v) 上 cv2.findContours(RETR_TREE, CHAIN_APPROX_SIMPLE)，
       取所有“外轮廓”（层级深度为偶数的轮廓 = 每个连通域的外边界，含嵌套在别的同值连通域孔洞里的连通域）。
       每个外轮廓生成一个 shape：label=VALUE_TO_LABEL[v]，points=该外轮廓（孔洞视为已填满）。
    2. 用 cv2.contourArea 计算填满后的面积（单像素/细线轮廓面积可能为 0，按像素数兜底排序），
       所有 shape 按面积从大到小排序输出：外层容器一定先于被它包含的区域绘制，后者覆盖前者。
    3. epsilon > 0 时用 cv2.approxPolyDP 简化；简化后少于 3 个点的轮廓用其外接矩形的 4 个角点代替。
       面积小于 min_area 的连通域丢弃。
    4. 每个 shape：{"label", "points": [[x, y], ...] (float), "group_id": None, "description": "",
       "shape_type": "polygon", "flags": {}}。

    epsilon=0、min_area=0 时，labelme_to_mask(mask_to_shapes(m)) 必须与 m 逐像素一致
    （cv2.fillPoly 填充 findContours 轮廓会精确覆盖该连通域的像素及其孔洞）。

    实现细节（对上述算法的补充，保证任意掩码都能精确还原）：
    - 第 1 步的外轮廓用 RETR_LIST + 有向面积符号（外轮廓 <= 0、孔洞 > 0）挑出，与 RETR_TREE
      偶数深度的轮廓集合逐点一致，但避免了 RETR_TREE 在单个连通域含大量孔洞时的平方复杂度。
    - “像素数”指该外轮廓被 fillPoly 填满后覆盖的像素数（连通域 + 孔洞），由 Pick 定理
      面积 + 棋盘距离周长/2 + 1 精确算出，无需逐个渲染。排序键为 (面积, 像素数) 降序，
      再按轮廓起点 (y, x)、值 0 → 1 → 255 升序，结果完全确定。
    - 对角交叉修正：findContours 对每个值都按 8 连通处理，两种值在 2x2 棋盘格处会“对角交叉”
      （例如四个可走像素围成菱形，中心的不可走像素与外面的不可走背景对角相连）。此时外层可走
      连通域把中心当作孔洞填满，而中心所属的大背景连通域更早绘制，单靠排序无法还原。
      因此先按上面的顺序在内存中渲染一遍，对仍不一致的像素按值取 4 连通域（4 连通不会再交叉），
      每个连通域取外轮廓、同样按面积降序，作为修正形状追加到列表末尾；修正形状的孔洞若盖错了
      更内层的像素，下一轮继续修正，直到完全一致（实践中 1~3 轮；棋盘格这类极端掩码也只需 1 轮）。
      超过 _MAX_FIX_PASSES 轮仍有残差时，用逐行游程线段兜底（单行线段没有孔洞，总能精确覆盖）。
      所以输出顺序为：主体形状按面积降序，其后是各轮修正形状（每轮内部同样按面积降序）。
      没有对角交叉的掩码（绝大多数真实掩码）不会产生修正形状。
    - epsilon 简化与 min_area 过滤作用在上述精确形状上（修正形状同样按自身大小过滤），
      排序依据始终是简化前的精确轮廓。min_area 比较的是“填满后像素数”。
    - 原始轮廓本身不足 3 个点（单个像素、水平/竖直/45° 斜向的单像素宽直线）时，
      无论 epsilon 取值都用首尾往返的 4 个点 [a, b, b, a] 表示，精确覆盖原像素
      （对单像素和水平/竖直线，这与外接矩形的 4 个角点完全相同）。
    - mask 必须是二维、值域 ⊆ {0, 1, 255}，否则抛 ValueError；空掩码返回 []。
    """
    m = _as_label_mask(mask)
    epsilon = float(epsilon)
    min_area = float(min_area)
    if np.isnan(epsilon) or np.isnan(min_area):
        raise ValueError("epsilon / min_area 不能是 NaN")
    if m.size == 0:
        return []

    pieces = [p for p in _exact_pieces(m) if not (min_area > 0 and p.count < min_area)]
    if not pieces:
        return []
    out_points = [_output_points(p.points, epsilon) for p in pieces]
    # 一次性转成 Python float 列表再切片，比逐个 astype().tolist() 快得多（噪声掩码可有上万个形状）
    flat = np.concatenate(out_points).astype(np.float64).tolist()
    shapes: list[dict[str, Any]] = []
    start = 0
    for piece, pts in zip(pieces, out_points):
        end = start + len(pts)
        shapes.append(
            {
                "label": VALUE_TO_LABEL[piece.value],
                "points": flat[start:end],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
            }
        )
        start = end
    return shapes


def _as_label_mask(mask: np.ndarray) -> np.ndarray:
    """校验并转换为 C 连续的 HxW uint8，值域必须 ⊆ {0, 1, 255}。"""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"掩码必须是 HxW，得到 shape={arr.shape}")
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    if arr.size:
        bad = ~np.isin(arr, _MASK_VALUES)
        if bad.any():
            extra = np.unique(arr[bad])[:10].tolist()
            raise ValueError(f"掩码含有非法值 {extra}，只允许 {list(_MASK_VALUES)}")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _contour_metrics(contours: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """批量计算轮廓的 (有向面积, 填满后像素数)，向量化以支持数万个轮廓。

    - 有向面积：鞋带公式，与 cv2.contourArea(oriented=True) 相同（整数坐标下精确）；
      findContours 的外轮廓有向面积 <= 0，孔洞轮廓 > 0（孔洞至少包住 1 个像素，面积必不为 0）。
    - 像素数：|面积| + 棋盘距离周长 / 2 + 1（Pick 定理）。findContours 轮廓相邻顶点之间只有
      水平/竖直/45° 三种方向，段上格点数 = max(|dx|, |dy|)；往返的细线（两次经过各计一半）、
      对角相接的“捏点”同样满足此式，结果与 fillPoly 实际覆盖的像素数一致。
    """
    lengths = np.fromiter((len(c) for c in contours), dtype=np.int64, count=len(contours))
    pts = np.concatenate(contours).reshape(-1, 2).astype(np.int64)
    starts = np.zeros(len(contours), dtype=np.int64)
    starts[1:] = np.cumsum(lengths)[:-1]
    nxt = np.arange(len(pts), dtype=np.int64) + 1
    nxt[starts + lengths - 1] = starts  # 每个轮廓首尾相连
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = x[nxt], y[nxt]
    signed_area = np.add.reduceat(x * yn - xn * y, starts) / 2.0
    perimeter = np.add.reduceat(np.maximum(np.abs(xn - x), np.abs(yn - y)), starts)
    return signed_area, np.abs(signed_area) + perimeter / 2.0 + 1.0


def _sorted_pieces(
    values: list[int], contours: list[np.ndarray], area: np.ndarray, count: np.ndarray
) -> list[_Piece]:
    """按 (面积, 像素数) 降序排列；再按起点 (y, x)、值排序，保证结果与轮廓检索顺序无关、完全确定。"""
    first = np.array([c.reshape(-1, 2)[0] for c in contours], dtype=np.int64).reshape(-1, 2)
    rank = np.array([_MASK_VALUES.index(v) for v in values], dtype=np.int64)
    order = np.lexsort((rank, first[:, 0], first[:, 1], -count, -area))  # 最后一个键优先
    return [_Piece(values[i], contours[i].reshape(-1, 2), float(area[i]), float(count[i])) for i in order]


def _painter_pass(mask: np.ndarray) -> list[_Piece]:
    """画家算法主轮次：每个值 v 在 (mask == v) 上的所有外轮廓（RETR_TREE 中深度为偶数的轮廓）。

    等价实现：用 RETR_LIST 一次取出全部轮廓，再按有向面积的符号区分外轮廓（<= 0）与孔洞（> 0）。
    得到的轮廓集合与 RETR_TREE 偶数深度完全相同（逐点一致，有测试校验），但 RETR_TREE/RETR_CCOMP
    在单个连通域含大量孔洞时建层级是平方复杂度（720x1280 棋盘格要 30 秒以上），RETR_LIST 是线性的。
    """
    values: list[int] = []
    outers: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    for v in _MASK_VALUES:
        binary = mask == v
        if not binary.any():
            continue
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        signed_area, count = _contour_metrics(contours)
        is_outer = signed_area <= 0  # 孔洞边界由外轮廓填满后再被内部形状覆盖，跳过
        idx = np.flatnonzero(is_outer)
        outers.extend(contours[i] for i in idx)
        values.extend([int(v)] * len(idx))
        areas.append(-signed_area[is_outer])
        counts.append(count[is_outer])
    if not outers:
        return []
    return _sorted_pieces(values, outers, np.concatenate(areas), np.concatenate(counts))


def _fix_pass(mask: np.ndarray, region: np.ndarray) -> list[_Piece]:
    """修正轮次：对 region（渲染结果与 mask 不一致的像素）按值取 4 连通域，每个连通域取外轮廓。

    用 4 连通是为了在修正轮次内彻底避免对角交叉：一个 4 连通域要么整个落在另一个连通域的孔洞里
    （面积更小、排在后面画），要么与它的填充区域不相交，所以本轮所有残差像素都会被改对；
    只有被修正形状的孔洞覆盖的像素可能再出错，它们严格嵌套在更小的范围里，下一轮继续处理。
    只在 region 的外接矩形内计算，坐标再平移回原图。"""
    rows = np.flatnonzero(region.any(axis=1))
    cols = np.flatnonzero(region.any(axis=0))
    if rows.size == 0:
        return []
    y0, x0 = int(rows[0]), int(cols[0])
    sub = mask[y0 : rows[-1] + 1, x0 : cols[-1] + 1]
    sub_region = region[y0 : rows[-1] + 1, x0 : cols[-1] + 1]
    offset = np.array([x0, y0], dtype=np.int32)

    values: list[int] = []
    contours: list[np.ndarray] = []
    for v in _MASK_VALUES:
        binary = ((sub == v) & sub_region).astype(np.uint8)
        if not binary.any():
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4, ltype=cv2.CV_32S)
        bx, by, bw, bh = (stats[1:, i].astype(np.int32) for i in range(4))
        straight = (bw == 1) | (bh == 1)
        # 单行/单列的 4 连通域就是一条直线段（含单像素）：两个端点即可精确表示，批量生成，无需 findContours
        starts = np.stack([bx, by], axis=1)
        stops = np.stack([bx + bw - 1, by + bh - 1], axis=1)
        ends = np.stack([starts, stops], axis=1)[straight] + offset
        contours.extend(ends)
        values.extend([int(v)] * len(ends))
        for k in np.flatnonzero(~straight):
            x, y, w, h = int(bx[k]), int(by[k]), int(bw[k]), int(bh[k])
            comp = (labels[y : y + h, x : x + w] == k + 1).astype(np.uint8)
            found, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 4 连通域也是一个 8 连通域，只有一个外轮廓
            contours.append(found[0].reshape(-1, 2) + np.array([x, y], dtype=np.int32) + offset)
            values.append(int(v))
    if not contours:
        return []
    signed_area, count = _contour_metrics(contours)
    return _sorted_pieces(values, contours, np.abs(signed_area), count)


def _run_pieces(mask: np.ndarray, region: np.ndarray) -> list[_Piece]:
    """兜底：把 region 内的像素按行拆成同值游程，每段生成一个单行线段（没有孔洞，绘制不影响别处）。"""
    pieces: list[_Piece] = []
    for y in np.flatnonzero(region.any(axis=1)):
        xs = np.flatnonzero(region[y])
        vals = mask[y, xs]
        # 游程断点：列不连续或值变化
        breaks = np.flatnonzero((np.diff(xs) != 1) | (np.diff(vals) != 0)) + 1
        for seg_x, seg_v in zip(np.split(xs, breaks), np.split(vals, breaks)):
            a, b = int(seg_x[0]), int(seg_x[-1])
            pts = np.array([[a, y], [b, y]], dtype=np.int32)
            pieces.append(_Piece(int(seg_v[0]), pts, 0.0, float(b - a + 1)))
    return pieces


def _render(canvas: np.ndarray, pieces: Sequence[_Piece]) -> None:
    """按顺序绘制整数轮廓（等价于 _fill_polygon：整数坐标取整不变，省去逐个转换的开销）。"""
    for piece in pieces:
        cv2.fillPoly(canvas, [piece.points], piece.value, lineType=cv2.LINE_8)


def _exact_pieces(mask: np.ndarray) -> list[_Piece]:
    """画家算法 + 对角交叉修正：返回按顺序绘制后与 mask 逐像素一致的整数轮廓列表。"""
    pieces = _painter_pass(mask)
    canvas = np.zeros_like(mask)  # 与 labelme_to_mask 的 default_value=0 一致（主体形状本就铺满全图）
    _render(canvas, pieces)
    diff = canvas != mask
    passes = 0
    while diff.any() and passes < _MAX_FIX_PASSES:
        fix = _fix_pass(mask, diff)
        _render(canvas, fix)
        pieces.extend(fix)
        diff = canvas != mask
        passes += 1
    if diff.any():
        pieces.extend(_run_pieces(mask, diff))
    return pieces


def _output_points(points: np.ndarray, epsilon: float) -> np.ndarray:
    """精确整数轮廓 → 输出顶点（Nx2，N >= 3）。"""
    if len(points) < 3:
        # 单像素 / 单像素宽直线：首尾往返，fillPoly 恰好画出这条线段
        a, b = points[0], points[-1]
        return np.array([a, b, b, a], dtype=np.int32)
    if epsilon > 0:
        approx = cv2.approxPolyDP(points.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)
        if len(approx) >= 3:
            return approx
        x, y, w, h = cv2.boundingRect(points.reshape(-1, 1, 2))
        x1, y1 = x + w - 1, y + h - 1  # 外接矩形的闭区间角点（顶点在像素中心）
        return np.array([[x, y], [x1, y], [x1, y1], [x, y1]], dtype=np.int32)
    return points


def mask_to_labelme(
    mask: np.ndarray,
    image_path: str,
    epsilon: float = 1.0,
    min_area: float = 0.0,
) -> dict[str, Any]:
    """生成完整的 LabelMe JSON dict：
    {"version": LABELME_VERSION, "flags": {}, "shapes": mask_to_shapes(...), "imagePath": image_path,
     "imageData": None, "imageHeight": H, "imageWidth": W}。
    image_path 为 JSON 文件到图片的相对路径（使用 '/' 分隔）。

    image_path 中的反斜杠会被统一替换为 '/'（Windows 路径也能被 LabelMe 正确解析）。"""
    m = _as_label_mask(mask)
    height, width = m.shape
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": mask_to_shapes(m, epsilon=epsilon, min_area=min_area),
        "imagePath": str(image_path).replace("\\", "/"),
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


# --------------------------------------------------------------------------- JSON 读写


def load_labelme(path: str | Path) -> dict[str, Any]:
    """读取 UTF-8 JSON。

    兼容带 BOM 的 UTF-8（Windows 记事本保存）。顶层不是 JSON 对象时抛 ValueError。"""
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"LabelMe JSON 顶层必须是对象: {path}")
    return data


def save_labelme(path: str | Path, data: dict[str, Any]) -> None:
    """写 UTF-8 JSON（ensure_ascii=False, indent=2），自动创建父目录。

    先写临时文件再原子替换，中途出错不会留下半截文件、也不会破坏已有的人工标注。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


# --------------------------------------------------------------------------- 批量转换


def export_masks_to_labelme(
    mask_dir: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    epsilon: float = 1.0,
    min_area: float = 16.0,
    overwrite: bool = False,
) -> list[Path]:
    """mask_dir 下每个 {stem}.png → out_dir/{stem}.json。
    对应图片按 stem 在 images_dir 中查找（任意 dfwalk.IMAGE_EXTS 后缀）；找不到则跳过并打印警告。
    imagePath 写为 os.path.relpath(图片, out_dir)，分隔符统一为 '/'。
    已存在的 JSON 在 overwrite=False 时跳过（防止覆盖人工修正结果）。返回新写出的路径列表。

    补充约定：
    - 警告用 warnings.warn 发出（与 autolabel 一致）。
    - 同一 stem 有多个后缀的图片时，按 IMAGE_EXTS 的顺序取第一个并警告。
    - 图片尺寸（考虑 EXIF 方向）与掩码不一致时跳过并警告，避免生成在 LabelMe 中错位的标注。
    - mask_dir / images_dir 不存在抛 FileNotFoundError；单个掩码出错时抛 ValueError，信息含文件名。
    """
    mask_dir, images_dir, out_dir = Path(mask_dir), Path(images_dir), Path(out_dir)
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")

    image_index = _index_images_by_stem(images_dir)
    mask_paths = sorted(p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    written: list[Path] = []
    for mask_path in mask_paths:
        stem = mask_path.stem
        out_path = out_dir / f"{stem}.json"
        if out_path.exists() and not overwrite:
            continue  # 可能已经人工修正过，绝不覆盖
        image_path = image_index.get(stem)
        if image_path is None:
            warnings.warn(f"跳过 {mask_path.name}：在 {images_dir} 中找不到同名图片（{stem}.*）", stacklevel=2)
            continue
        try:
            mask = read_mask(mask_path)
            image_hw = _image_size(image_path)
            if image_hw is None:
                warnings.warn(f"跳过 {mask_path.name}：无法读取图片 {image_path}", stacklevel=2)
                continue
            if image_hw != mask.shape:
                warnings.warn(
                    f"跳过 {mask_path.name}：掩码尺寸 {mask.shape} 与图片 {image_path.name} 尺寸 {image_hw} 不一致",
                    stacklevel=2,
                )
                continue
            rel = _relative_posix(image_path, out_dir)
            data = mask_to_labelme(mask, rel, epsilon=epsilon, min_area=min_area)
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"导出 LabelMe 失败 {mask_path.name}: {e}") from e
        save_labelme(out_path, data)
        written.append(out_path)
    return written


def _index_images_by_stem(images_dir: Path) -> dict[str, Path]:
    """stem → 图片路径；同一 stem 多个后缀时按 IMAGE_EXTS 顺序取第一个并警告。"""
    groups: dict[str, list[Path]] = {}
    for p in list_images(images_dir):
        groups.setdefault(p.stem, []).append(p)
    rank = {ext: i for i, ext in enumerate(IMAGE_EXTS)}
    index: dict[str, Path] = {}
    for stem, paths in groups.items():
        paths.sort(key=lambda p: (rank.get(p.suffix.lower(), len(rank)), p.name))
        if len(paths) > 1:
            names = ", ".join(p.name for p in paths)
            warnings.warn(f"图片 stem {stem!r} 有多个文件（{names}），使用 {paths[0].name}", stacklevel=3)
        index[stem] = paths[0]
    return index


def _image_size(path: Path) -> tuple[int, int] | None:
    """图片 (高, 宽)，按 EXIF 方向修正（与 cv2.imdecode / LabelMe 显示一致）。只读文件头，很快。"""
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            orientation = im.getexif().get(0x0112, 1)
        if orientation in (5, 6, 7, 8):  # 旋转 90°/270° 的方向，显示时宽高互换
            w, h = h, w
        return int(h), int(w)
    except Exception:
        pass
    try:
        return tuple(read_image_rgb(path).shape[:2])  # type: ignore[return-value]
    except (OSError, ValueError):
        return None


def _relative_posix(target: Path, start: Path) -> str:
    """target 相对 start 的路径，统一用 '/' 分隔；跨盘符（Windows）无法相对时退回绝对路径。"""
    try:
        rel = os.path.relpath(os.path.abspath(target), os.path.abspath(start))
    except ValueError:
        rel = os.path.abspath(target)
    return rel.replace(os.sep, "/").replace("\\", "/")


def convert_labelme_dir(
    json_dir: str | Path,
    out_mask_dir: str | Path,
    hud_rects: Sequence[Sequence[float]] = (),
) -> list[Path]:
    """json_dir 下每个 *.json → out_mask_dir/{stem}.png（labelme_to_mask 后再 hud.apply_hud_ignore）。
    返回写出的路径列表。单个文件出错时抛出异常，信息中包含文件名。

    补充约定：按文件名排序处理；已存在的掩码直接覆盖（掩码是由 JSON 派生的数据）。
    JSON 解析/标签/形状错误统一包装为 ValueError（信息含文件名），IO 错误原样抛出（自带路径）。"""
    json_dir, out_mask_dir = Path(json_dir), Path(out_mask_dir)
    if not json_dir.is_dir():
        raise FileNotFoundError(f"LabelMe JSON 目录不存在: {json_dir}")
    json_paths = sorted(p for p in json_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    written: list[Path] = []
    for json_path in json_paths:
        out_path = out_mask_dir / f"{json_path.stem}.png"
        try:
            mask = labelme_to_mask(load_labelme(json_path))
            mask = apply_hud_ignore(mask, hud_rects)
        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(f"转换 LabelMe 文件失败 {json_path.name}: {e}") from e
        write_mask(out_path, mask)
        written.append(out_path)
    return written
