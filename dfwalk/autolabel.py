"""零样本自动标注：用 ADE20K 预训练的 SegFormer 生成可走区域伪标签。

ADE20K 有 150 类（floor、road、grass、stairs……）。对每个像素把“可走类别”的 softmax 概率求和，
得到可走概率 p，再用双阈值转换为 {0, 1, 255}：
    p >= hi → WALKABLE(1)，p <= lo → NON_WALKABLE(0)，其余 → IGNORE_INDEX(255)
游戏画面与真实照片存在域差异，伪标签必须人工抽查/修正（见 labelme_io）后再用于训练。

本模块只处理已经保存在磁盘上的截图/抽帧图片，不接触游戏进程。
"""

from __future__ import annotations

import difflib
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from . import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD, NON_WALKABLE, WALKABLE, hud
from .utils import list_images, read_image_rgb, resolve_device, write_mask

# dfwalk 没有发布到 PyPI：只能在仓库目录中以可编辑方式安装可选依赖（与 README、CLI 帮助一致）
_INSTALL_HINT = 'pip install -e ".[autolabel]"（在仓库目录中执行）或 pip install transformers'


def normalize_label(name: str) -> str:
    """类别名规范化：去首尾空白并转小写（ADE20K 的 id2label 里有 'bed ' 这种带空格的名字）。"""
    return str(name).strip().lower()


def build_walkable_lookup(id2label: Mapping[int | str, str], walkable_labels: Sequence[str]) -> np.ndarray:
    """返回长度为 num_labels 的 bool 数组，lookup[i] 表示第 i 类是否可走。
    id2label 的键可能是 int 或数字字符串（HF config 反序列化后两者都可能出现）。
    walkable_labels 中有任何名字（规范化后）在 id2label 中找不到时抛 ValueError，
    信息里列出找不到的名字，防止拼写错误被静默忽略。

    补充约定：
    - num_labels = 最大类别 id + 1（HF 的 id 从 0 连续编号，此时即类别数）；
    - 同名的多个类别 id 全部标为可走；
    - walkable_labels 传入单个字符串时视为只有一个名字（避免被逐字符拆开）；
    - id2label 为空、键不是非负整数时抛 ValueError。
    """
    if not id2label:
        raise ValueError("id2label 为空，无法确定类别")
    if isinstance(walkable_labels, str):
        walkable_labels = [walkable_labels]

    name_to_ids: dict[str, list[int]] = {}
    max_id = -1
    for key, name in id2label.items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as e:
            raise ValueError(f"id2label 的键必须是整数或数字字符串，得到 {key!r}") from e
        if idx < 0:
            raise ValueError(f"id2label 的键必须是非负整数，得到 {key!r}")
        name_to_ids.setdefault(normalize_label(name), []).append(idx)
        max_id = max(max_id, idx)

    lookup = np.zeros(max_id + 1, dtype=bool)
    missing: list[str] = []
    for raw in walkable_labels:
        name = normalize_label(raw)
        ids = name_to_ids.get(name)
        if ids is None:
            if name not in missing:
                missing.append(name)
            continue
        lookup[ids] = True

    if missing:
        hints = []
        for name in missing:
            close = difflib.get_close_matches(name, list(name_to_ids), n=3, cutoff=0.6)
            hints.append(f"{name!r}" + (f"（是否想写 {', '.join(repr(c) for c in close)}？）" if close else ""))
        raise ValueError(
            f"walkable_labels 中有 {len(missing)} 个类别名在模型的 id2label 中找不到: "
            + "; ".join(hints)
            + f"。模型共有 {len(name_to_ids)} 个类别名，请检查 autolabel.walkable_labels 的拼写。"
        )
    return lookup


def _check_thresholds(hi: float, lo: float) -> None:
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"需要满足 0 <= lo < hi <= 1，得到 lo={lo}, hi={hi}")


def walkable_prob_to_label(prob: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """HxW 概率 → HxW uint8 标签：p >= hi → 1，p <= lo → 0，其余 → 255。要求 0 <= lo < hi <= 1。

    阈值先转换为 prob 的浮点类型再比较，因此 float32 概率恰好等于 hi/lo 时也落在“确定”一侧
    （float32(0.7) 与 Python 的 0.7 并不相等）。NaN 落入忽略(255)。
    """
    _check_thresholds(hi, lo)
    p = np.asarray(prob)
    if not np.issubdtype(p.dtype, np.floating):
        p = p.astype(np.float64)
    hi_t = p.dtype.type(hi)
    lo_t = p.dtype.type(lo)
    out = np.full(p.shape, IGNORE_INDEX, dtype=np.uint8)
    out[p >= hi_t] = WALKABLE
    out[p <= lo_t] = NON_WALKABLE
    return out


def compute_infer_size(height: int, width: int, short_side: int, multiple: int = 32) -> tuple[int, int]:
    """等比缩放使短边为 short_side，再把高、宽各自四舍五入到 multiple 的倍数（至少为 multiple）。
    返回 (new_h, new_w)。

    例：1080x1920、short_side=512 → 缩放后 512x910.2 → (512, 896)。
    四舍五入采用“.5 进位”，不使用 Python round 的银行家舍入。
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"图像尺寸必须为正，得到 {height}x{width}")
    if short_side <= 0:
        raise ValueError(f"short_side 必须为正，得到 {short_side}")
    if multiple <= 0:
        raise ValueError(f"multiple 必须为正，得到 {multiple}")
    scale = float(short_side) / float(min(height, width))

    def _round(x: float) -> int:
        return max(int(multiple), int(np.floor(x / multiple + 0.5)) * int(multiple))

    return _round(height * scale), _round(width * scale)


def _extract_logits(output: Any):
    """从模型输出中取出 logits：兼容 HF ModelOutput(.logits)、dict、tuple/list（第一个元素）、裸张量。"""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"无法从模型输出中取得 logits（需要 .logits 属性），得到 {type(output).__name__}")


def _channel_stats(values: Any, default: Sequence[float]) -> list[float]:
    """processor 的 image_mean/std → 3 个 float；缺失时用 default，标量展开成 3 通道。"""
    if values is None:
        return [float(x) for x in default]
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return [float(x) for x in default]
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    return [float(x) for x in arr]


class AutoLabeler:
    """可走区域伪标签生成器。

    model: 可调用对象，接收 pixel_values=float32 张量 (1, 3, h, w)（已按 image_mean/std 归一化），
           返回带 .logits 属性 (1, num_labels, h', w') 的对象（兼容 HF SegformerForSemanticSegmentation；
           h', w' 可以小于 h, w，例如 SegFormer 输出 1/4 分辨率）。
           单元测试中会传入假模型，因此本类除 from_pretrained 外不得依赖 transformers。

    公开属性：model、device、id2label（键统一为 int）、walkable_labels、walkable_lookup（bool 数组）、
    image_mean、image_std、short_side、hi、lo。
    model 若是 torch.nn.Module，会被移到 device 并切换到 eval()。
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
        import torch

        _check_thresholds(hi, lo)
        if int(short_side) <= 0:
            raise ValueError(f"short_side 必须为正，得到 {short_side}")
        mean = tuple(float(x) for x in image_mean)
        std = tuple(float(x) for x in image_std)
        if len(mean) != 3 or len(std) != 3:
            raise ValueError(f"image_mean / image_std 必须各有 3 个值，得到 {mean} / {std}")
        if any(s <= 0 for s in std):
            raise ValueError(f"image_std 必须全部为正，得到 {std}")

        self.device = resolve_device(device) if isinstance(device, str) else torch.device(device)
        self.id2label: dict[int, str] = {int(k): str(v) for k, v in id2label.items()}
        self.walkable_labels: list[str] = (
            [walkable_labels] if isinstance(walkable_labels, str) else [str(x) for x in walkable_labels]
        )
        self.walkable_lookup: np.ndarray = build_walkable_lookup(self.id2label, self.walkable_labels)
        self.image_mean = mean
        self.image_std = std
        self.short_side = int(short_side)
        self.hi = float(hi)
        self.lo = float(lo)

        if isinstance(model, torch.nn.Module):
            model = model.to(self.device)
            model.eval()
        self.model = model

        # 预先放到设备上的常量，避免每帧重复创建
        self._mean_t = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._std_t = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._walk_idx = torch.as_tensor(np.flatnonzero(self.walkable_lookup), dtype=torch.long, device=self.device)

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
        transformers 延迟导入；未安装时抛 ImportError，提示在仓库目录中 pip install -e ".[autolabel]"
        （或 pip install transformers）。
        device 经 dfwalk.utils.resolve_device 解析；模型 eval() 并移到该设备。

        name 可以是 Hub 上的模型名（如 nvidia/segformer-b2-finetuned-ade-512-512），也可以是
        save_pretrained 保存的本地目录。实际通过 AutoModelForSemanticSegmentation 加载，
        对 SegFormer 即得到 SegformerForSemanticSegmentation，同时也兼容其他输出 .logits 的语义分割模型。
        processor 缺少 image_mean/image_std 时退回 ImageNet 参数。
        """
        import torch

        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        except ImportError as e:
            raise ImportError(f"自动标注需要 transformers，请先安装：{_INSTALL_HINT}") from e

        processor = AutoImageProcessor.from_pretrained(name)
        mean = _channel_stats(getattr(processor, "image_mean", None), IMAGENET_MEAN)
        std = _channel_stats(getattr(processor, "image_std", None), IMAGENET_STD)

        model = AutoModelForSemanticSegmentation.from_pretrained(name)
        dev = resolve_device(device)
        # 统一为 float32：新版 transformers 可能按权重文件的精度（如 fp16）加载，而输入固定为 float32
        model.to(device=dev, dtype=torch.float32)
        model.eval()
        return cls(
            model,
            id2label=model.config.id2label,
            walkable_labels=walkable_labels,
            image_mean=mean,
            image_std=std,
            short_side=short_side,
            hi=hi,
            lo=lo,
            device=dev,
        )

    def _preprocess(self, image_rgb: np.ndarray):
        """校验输入并完成缩放、归一化，返回 ((1, 3, ih, iw) 张量, 原图高, 原图宽)。"""
        import torch

        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            shape = getattr(image_rgb, "shape", None)
            raise ValueError(f"需要 HxWx3 的 RGB 图像，得到 shape={shape}")
        if image_rgb.dtype != np.uint8:
            raise ValueError(f"需要 uint8 图像（0~255），得到 dtype={image_rgb.dtype}")
        h, w = image_rgb.shape[:2]
        if h == 0 or w == 0:
            raise ValueError(f"图像尺寸为空: {image_rgb.shape}")

        ih, iw = compute_infer_size(h, w, self.short_side)
        if (ih, iw) == (h, w):
            resized = image_rgb
        else:
            interp = cv2.INTER_AREA if ih * iw < h * w else cv2.INTER_LINEAR
            resized = cv2.resize(image_rgb, (iw, ih), interpolation=interp)
        x = torch.from_numpy(np.ascontiguousarray(resized)).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0)
        x = (x - self._mean_t) / self._std_t
        return x.contiguous(), h, w

    def walkable_prob(self, image_rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB → HxW float32 可走概率（0~1，原图分辨率）。

        步骤：compute_infer_size 缩放（cv2.INTER_AREA 缩小 / INTER_LINEAR 放大）→ /255 → 按 mean/std 归一化
        → torch.no_grad 推理 → 在 logits 分辨率上做类别维 softmax → 对可走类别求和得到 1 通道概率
        → 双线性插值 (align_corners=False) 回原图 HxW → clip 到 [0, 1]。
        注意：不要把 150 通道 logits 先上采样到原图分辨率（1080p 下会占用 >1GB 内存）。

        输入不是 HxWx3 uint8 时抛 ValueError；logits 的类别数与 id2label 不一致时抛 ValueError。
        """
        import torch
        import torch.nn.functional as F

        x, h, w = self._preprocess(image_rgb)
        with torch.no_grad():
            logits = _extract_logits(self.model(pixel_values=x))
            if logits.ndim != 4 or logits.shape[0] != 1:
                raise ValueError(f"模型 logits 形状应为 (1, num_labels, h, w)，得到 {tuple(logits.shape)}")
            num_labels = len(self.walkable_lookup)
            if logits.shape[1] != num_labels:
                raise ValueError(
                    f"模型输出 {logits.shape[1]} 个类别，但 id2label 推出 {num_labels} 个类别，二者不一致"
                )
            logits = logits.to(device=self.device, dtype=torch.float32)
            if self._walk_idx.numel() == 0:
                walk = logits.new_zeros((1, 1, logits.shape[2], logits.shape[3]))
            else:
                # 在 logits 分辨率上 softmax 并对可走类别求和，只把 1 通道结果上采样
                probs = torch.softmax(logits, dim=1)
                walk = probs.index_select(1, self._walk_idx).sum(dim=1, keepdim=True)
                del probs
            walk = F.interpolate(walk, size=(h, w), mode="bilinear", align_corners=False)
            walk = walk.clamp_(0.0, 1.0)
        return walk[0, 0].cpu().numpy().astype(np.float32, copy=False)

    def label(self, image_rgb: np.ndarray, hud_rects: Sequence[Sequence[float]] = ()) -> np.ndarray:
        """walkable_prob → walkable_prob_to_label(hi, lo) → hud.apply_hud_ignore，返回 HxW uint8。"""
        prob = self.walkable_prob(image_rgb)
        mask = walkable_prob_to_label(prob, self.hi, self.lo)
        if hud_rects is not None and len(hud_rects) > 0:
            mask = hud.apply_hud_ignore(mask, hud_rects)
        return mask


def autolabel_folder(
    labeler: AutoLabeler,
    images_dir: str | Path,
    out_dir: str | Path,
    hud_rects: Sequence[Sequence[float]] = (),
    overwrite: bool = False,
) -> list[Path]:
    """对 images_dir 下所有图片（dfwalk.utils.list_images）生成伪标签 out_dir/{stem}.png。
    已存在的输出在 overwrite=False 时跳过（不计入返回值）。显示 tqdm 进度条。返回新写出的路径列表。

    补充约定：
    - images_dir 不存在时抛 FileNotFoundError；out_dir 与 images_dir 是同一目录时抛 ValueError；
    - 同一 stem 有多个后缀（如 a.jpg 与 a.png）时，在写任何文件之前抛 ValueError（输出会互相覆盖）；
    - 无法解码的图片用 warnings.warn 提示后跳过（不计入返回值），不中断整批处理；
    - 掩码先写入同目录的临时文件再改名，中途中断不会留下“看似完整”的半个 PNG 被下次跳过。
    """
    from tqdm import tqdm

    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    if out_dir.exists() and out_dir.resolve() == images_dir.resolve():
        raise ValueError(f"out_dir 不能与 images_dir 相同（会用掩码覆盖同名 .png 图片）: {out_dir}")
    images = list_images(images_dir)

    by_stem: dict[str, list[Path]] = {}
    for p in images:
        by_stem.setdefault(p.stem, []).append(p)
    dups = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if dups:
        detail = "; ".join(f"{stem}: {', '.join(p.name for p in paths)}" for stem, paths in sorted(dups.items()))
        raise ValueError(f"同一文件名（stem）存在多个图片，伪标签会互相覆盖: {detail}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for img_path in tqdm(images, desc="autolabel", unit="img"):
        out_path = out_dir / f"{img_path.stem}.png"
        if out_path.exists() and not overwrite:
            continue
        try:
            image = read_image_rgb(img_path)
        except ValueError as e:
            warnings.warn(f"跳过无法读取的图片 {img_path}: {e}", stacklevel=2)
            continue
        mask = labeler.label(image, hud_rects)
        tmp_path = out_dir / f".{img_path.stem}.tmp.png"
        try:
            write_mask(tmp_path, mask)
            tmp_path.replace(out_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        written.append(out_path)
    return written
