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

from pathlib import Path
from typing import Any, Sequence

import numpy as np

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


def label_to_value(label: str) -> int:
    """标签名 → 掩码值；未知标签抛 ValueError，信息中列出允许的标签名。"""
    raise NotImplementedError


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
    """
    raise NotImplementedError


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
    """
    raise NotImplementedError


def mask_to_labelme(
    mask: np.ndarray,
    image_path: str,
    epsilon: float = 1.0,
    min_area: float = 0.0,
) -> dict[str, Any]:
    """生成完整的 LabelMe JSON dict：
    {"version": LABELME_VERSION, "flags": {}, "shapes": mask_to_shapes(...), "imagePath": image_path,
     "imageData": None, "imageHeight": H, "imageWidth": W}。
    image_path 为 JSON 文件到图片的相对路径（使用 '/' 分隔）。"""
    raise NotImplementedError


def load_labelme(path: str | Path) -> dict[str, Any]:
    """读取 UTF-8 JSON。"""
    raise NotImplementedError


def save_labelme(path: str | Path, data: dict[str, Any]) -> None:
    """写 UTF-8 JSON（ensure_ascii=False, indent=2），自动创建父目录。"""
    raise NotImplementedError


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
    已存在的 JSON 在 overwrite=False 时跳过（防止覆盖人工修正结果）。返回新写出的路径列表。"""
    raise NotImplementedError


def convert_labelme_dir(
    json_dir: str | Path,
    out_mask_dir: str | Path,
    hud_rects: Sequence[Sequence[float]] = (),
) -> list[Path]:
    """json_dir 下每个 *.json → out_mask_dir/{stem}.png（labelme_to_mask 后再 hud.apply_hud_ignore）。
    返回写出的路径列表。单个文件出错时抛出异常，信息中包含文件名。"""
    raise NotImplementedError
