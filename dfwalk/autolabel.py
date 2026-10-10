"""零样本自动标注：用 ADE20K 预训练的 SegFormer 生成可走区域伪标签。

ADE20K 有 150 类（floor、road、grass、stairs……）。对每个像素把“可走类别”的 softmax 概率求和，
得到可走概率 p，再用双阈值转换为 {0, 1, 255}：
    p >= hi → WALKABLE(1)，p <= lo → NON_WALKABLE(0)，其余 → IGNORE_INDEX(255)
游戏画面与真实照片存在域差异，伪标签必须人工抽查/修正（见 labelme_io）后再用于训练。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def normalize_label(name: str) -> str:
    """类别名规范化：去首尾空白并转小写（ADE20K 的 id2label 里有 'bed ' 这种带空格的名字）。"""
    raise NotImplementedError


def build_walkable_lookup(id2label: Mapping[int | str, str], walkable_labels: Sequence[str]) -> np.ndarray:
    """返回长度为 num_labels 的 bool 数组，lookup[i] 表示第 i 类是否可走。
    id2label 的键可能是 int 或数字字符串（HF config 反序列化后两者都可能出现）。
    walkable_labels 中有任何名字（规范化后）在 id2label 中找不到时抛 ValueError，
    信息里列出找不到的名字，防止拼写错误被静默忽略。"""
    raise NotImplementedError


def walkable_prob_to_label(prob: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """HxW 概率 → HxW uint8 标签：p >= hi → 1，p <= lo → 0，其余 → 255。要求 0 <= lo < hi <= 1。"""
    raise NotImplementedError


def compute_infer_size(height: int, width: int, short_side: int, multiple: int = 32) -> tuple[int, int]:
    """等比缩放使短边为 short_side，再把高、宽各自四舍五入到 multiple 的倍数（至少为 multiple）。
    返回 (new_h, new_w)。"""
    raise NotImplementedError


class AutoLabeler:
    """可走区域伪标签生成器。

    model: 可调用对象，接收 pixel_values=float32 张量 (1, 3, h, w)（已按 image_mean/std 归一化），
           返回带 .logits 属性 (1, num_labels, h', w') 的对象（兼容 HF SegformerForSemanticSegmentation；
           h', w' 可以小于 h, w，例如 SegFormer 输出 1/4 分辨率）。
           单元测试中会传入假模型，因此本类除 from_pretrained 外不得依赖 transformers。
    """

    def __init__(
        self,
        model: Callable[..., Any],
        id2label: Mapping[int | str, str],
        walkable_labels: Sequence[str],
        image_mean: Sequence[float],
        image_std: Sequence[float],
        short_side: int = 512,
        hi: float = 0.7,
        lo: float = 0.3,
        device: Any = "cpu",
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        walkable_labels: Sequence[str],
        short_side: int = 512,
        hi: float = 0.7,
        lo: float = 0.3,
        device: str = "auto",
    ) -> "AutoLabeler":
        """从 Hugging Face 加载 SegformerForSemanticSegmentation 及其 image processor 的 mean/std。
        transformers 延迟导入；未安装时抛 ImportError，提示 pip install "dfwalk[autolabel]"。
        device 经 dfwalk.utils.resolve_device 解析；模型 eval() 并移到该设备。"""
        raise NotImplementedError

    def walkable_prob(self, image_rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB → HxW float32 可走概率（0~1，原图分辨率）。

        步骤：compute_infer_size 缩放（cv2.INTER_AREA 缩小 / INTER_LINEAR 放大）→ /255 → 按 mean/std 归一化
        → torch.no_grad 推理 → 在 logits 分辨率上做类别维 softmax → 对可走类别求和得到 1 通道概率
        → 双线性插值 (align_corners=False) 回原图 HxW → clip 到 [0, 1]。
        注意：不要把 150 通道 logits 先上采样到原图分辨率（1080p 下会占用 >1GB 内存）。"""
        raise NotImplementedError

    def label(self, image_rgb: np.ndarray, hud_rects: Sequence[Sequence[float]] = ()) -> np.ndarray:
        """walkable_prob → walkable_prob_to_label(hi, lo) → hud.apply_hud_ignore，返回 HxW uint8。"""
        raise NotImplementedError


def autolabel_folder(
    labeler: AutoLabeler,
    images_dir: str | Path,
    out_dir: str | Path,
    hud_rects: Sequence[Sequence[float]] = (),
    overwrite: bool = False,
) -> list[Path]:
    """对 images_dir 下所有图片（dfwalk.utils.list_images）生成伪标签 out_dir/{stem}.png。
    已存在的输出在 overwrite=False 时跳过（不计入返回值）。显示 tqdm 进度条。返回新写出的路径列表。"""
    raise NotImplementedError
