"""推理：图片 / 图片目录 / 录像 → 可走区域掩码与叠加可视化。

只做离线分析：输入是截图或录像文件，输出是掩码 PNG 和叠加可视化图片/视频，不与游戏进程交互。

掩码约定与全项目一致：0 = 不可走，1 = 可走，255 = 忽略（HUD 区域）。
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD, NON_WALKABLE, WALKABLE
from .hud import apply_hud_ignore
from .transforms import EvalTransform
from .utils import (
    bgr_to_rgb,
    list_images,
    open_video,
    read_image_rgb,
    resolve_device,
    write_image_rgb,
    write_mask,
)

# 源视频 fps 读不到（0 / NaN / 负数）时的兜底值
DEFAULT_VIDEO_FPS = 30.0
# 叠加图中可走区域轮廓线宽（像素）
CONTOUR_THICKNESS = 2
# 忽略区（HUD）压暗系数
IGNORE_DIM_FACTOR = 0.5


# ---------------------------------------------------------------- 参数检查


def _check_image(image_rgb: np.ndarray) -> tuple[int, int]:
    """检查输入为 HxWx3 uint8 且尺寸非零，返回 (H, W)。"""
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        shape = getattr(image_rgb, "shape", None)
        raise ValueError(f"image 必须是 HxWx3 的 RGB 数组，得到 shape={shape}")
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"image 必须是 uint8（0~255），得到 dtype={image_rgb.dtype}")
    h, w = image_rgb.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"image 尺寸不能为 0，得到 shape={image_rgb.shape}")
    return h, w


def _check_threshold(threshold: float) -> float:
    t = float(threshold)
    if not 0.0 <= t <= 1.0:  # 同时排除 NaN
        raise ValueError(f"threshold 必须在 [0, 1] 之间，得到 {threshold}")
    return t


def _check_color(color: Sequence[int]) -> tuple[int, int, int]:
    """(R, G, B) → 裁剪到 0~255 的 int 三元组（与 hud.draw_hud_preview 一致：越界裁剪，个数不对报错）。"""
    try:
        values = [float(c) for c in color]
    except (TypeError, ValueError):
        raise ValueError(f"color 必须是 (R, G, B) 三个数，得到 {color!r}") from None
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"color 必须是 (R, G, B) 三个有限数，得到 {color!r}")
    return tuple(int(np.clip(round(v), 0, 255)) for v in values)  # type: ignore[return-value]


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not math.isfinite(a):
        raise ValueError(f"alpha 必须是有限数，得到 {alpha}")
    return min(max(a, 0.0), 1.0)


# ---------------------------------------------------------------- Predictor


class Predictor:
    """封装模型推理。输入任意分辨率 RGB 图像，内部缩放到 image_size，输出原分辨率结果。

    属性：model（已在 device 上、eval 模式）、image_size (H, W)、mean、std、device、transform（EvalTransform）。
    """

    def __init__(self, model: nn.Module, image_size: Sequence[int], mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD, device="cpu") -> None:
        """model 移到 device 并 eval()。

        - device 可以是字符串（"auto" 按 cuda > mps > cpu 选择，见 utils.resolve_device）或 torch.device。
        - image_size / mean / std 非法时由 EvalTransform 抛 ValueError。
        """
        # 预处理与训练时的验证集完全一致：直接复用 EvalTransform（同时完成参数检查）
        self.transform = EvalTransform(image_size, mean, std)
        self.image_size: tuple[int, int] = self.transform.size
        self.mean = tuple(float(v) for v in self.transform.mean)
        self.std = tuple(float(v) for v in self.transform.std)
        self.device = resolve_device(device) if isinstance(device, str) else torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Predictor":
        """dfwalk.checkpoint.load_checkpoint + model_from_checkpoint；image_size/mean/std 取自 checkpoint。

        checkpoint 先加载到 CPU（weights_only=True），再由 __init__ 移到目标 device。
        文件不存在抛 FileNotFoundError，不是 dfwalk checkpoint 抛 ValueError。
        """
        from .checkpoint import load_checkpoint, model_from_checkpoint

        state = load_checkpoint(path, map_location="cpu")
        model = model_from_checkpoint(state)
        return cls(
            model,
            state["image_size"],
            mean=state.get("mean", IMAGENET_MEAN),
            std=state.get("std", IMAGENET_STD),
            device=device,
        )

    @torch.inference_mode()
    def predict_prob(self, image_rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB → HxW float32 可走概率（softmax 第 WALKABLE 通道），原图分辨率。
        预处理与 transforms.EvalTransform 完全一致（复用 EvalTransform）；概率图双线性插值回原尺寸。

        - 插值为半像素中心对齐（align_corners=False，与 cv2.INTER_LINEAR 相同），结果裁剪到 [0, 1]。
        - 输入不是 HxWx3 uint8 时抛 ValueError；不修改输入数组。
        """
        h, w = _check_image(image_rgb)
        x, _ = self.transform(image_rgb)
        logits = self.model(x.unsqueeze(0).to(self.device))
        if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] <= WALKABLE:
            raise ValueError(f"模型输出应为 (1, C>={WALKABLE + 1}, h, w) 的 logits，得到 {tuple(logits.shape)}")
        # softmax 用 float32 计算（模型可能以半精度运行）
        prob = torch.softmax(logits.float(), dim=1)[:, WALKABLE:WALKABLE + 1]
        if tuple(prob.shape[-2:]) != (h, w):
            prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
        prob = prob[0, 0].clamp_(0.0, 1.0)
        return np.ascontiguousarray(prob.cpu().numpy(), dtype=np.float32)

    def predict_mask(self, image_rgb: np.ndarray, threshold: float = 0.5,
                     hud_rects: Sequence[Sequence[float]] = ()) -> np.ndarray:
        """HxW uint8：prob >= threshold → 1，否则 0；hud_rects 区域置为 255。

        threshold 不在 [0, 1] 之间时抛 ValueError；hud_rects 为归一化矩形 [x0, y0, x1, y1]（见 dfwalk.hud）。
        """
        t = _check_threshold(threshold)
        prob = self.predict_prob(image_rgb)
        mask = np.where(prob >= t, WALKABLE, NON_WALKABLE).astype(np.uint8)
        if hud_rects is not None and len(hud_rects):
            mask = apply_hud_ignore(mask, hud_rects)
        return mask


# ---------------------------------------------------------------- 可视化


def overlay(image_rgb: np.ndarray, mask: np.ndarray, color: Sequence[int] = (0, 255, 0),
            alpha: float = 0.45) -> np.ndarray:
    """返回叠加图（HxWx3 uint8 RGB，不修改输入）：mask == 1 的像素与 color 按 alpha 混合，
    并用同色 2px 描出可走区域轮廓；mask == 255 的像素整体压暗 50%（标示 HUD/忽略区）。

    - 混合公式：round(pixel * (1 - alpha) + color * alpha)；压暗：round(pixel * 0.5)。
    - 绘制顺序：压暗 → 混合 → 轮廓（轮廓包括可走区域内部“洞”的边界，线宽 2px，可能向外溢出 1px）。
    - alpha 裁剪到 [0, 1]，color 各分量裁剪到 0~255；color 不是 3 个数、alpha 非有限数时抛 ValueError。
    - image 必须是 HxWx3 uint8，mask 必须是同尺寸 HxW，否则抛 ValueError。即使 mask 全 0 也返回新数组。
    """
    h, w = _check_image(image_rgb)
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != (h, w):
        raise ValueError(f"mask 必须是与图像同尺寸的 HxW 数组：image {(h, w)}，mask {mask.shape}")
    color_px = _check_color(color)
    a = _check_alpha(alpha)

    out = np.array(image_rgb, dtype=np.uint8, copy=True, order="C")

    # 1) 忽略区压暗
    ignore = mask == IGNORE_INDEX
    if ignore.any():
        out[ignore] = np.rint(out[ignore].astype(np.float32) * IGNORE_DIM_FACTOR).astype(np.uint8)

    # 2) 可走区域半透明着色
    walk = mask == WALKABLE
    if walk.any():
        blended = out[walk].astype(np.float32) * (1.0 - a) + np.asarray(color_px, dtype=np.float32) * a
        out[walk] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)

        # 3) 同色轮廓（RETR_LIST：外边界与内部洞的边界都画）
        contours, _ = cv2.findContours(walk.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(out, contours, -1, color_px, CONTOUR_THICKNESS, lineType=cv2.LINE_8)
    return out


# ---------------------------------------------------------------- 批量图片


def _collect_images(inputs: Sequence[str | Path]) -> list[Path]:
    """展开输入：文件原样保留，目录用 list_images 展开（不递归）。按出现顺序去重（同一文件只处理一次）。
    路径不存在时抛 FileNotFoundError。"""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            candidates = list_images(p)
        elif p.is_file():
            candidates = [p]
        else:
            raise FileNotFoundError(f"输入不存在: {p}")
        for c in candidates:
            key = c.resolve()
            if key not in seen:
                seen.add(key)
                paths.append(c)
    return paths


def infer_images(
    predictor: Predictor,
    inputs: Sequence[str | Path],
    out_dir: str | Path,
    threshold: float = 0.5,
    alpha: float = 0.45,
    color: Sequence[int] = (0, 255, 0),
    hud_rects: Sequence[Sequence[float]] = (),
    save_mask: bool = True,
) -> list[Path]:
    """对每张图片写 out_dir/{stem}_overlay.jpg，save_mask 时再写 out_dir/{stem}_mask.png（0/1/255）。
    返回写出的叠加图路径列表。显示 tqdm 进度条。

    补充约定：
    - inputs 中的目录会用 utils.list_images 展开（不递归）；同一文件出现多次只处理一次；返回值按输入顺序。
    - 路径不存在时抛 FileNotFoundError；在写任何文件之前完成以下检查：
      threshold / color / alpha 非法抛 ValueError；
      不同图片的 stem 相同（大小写不敏感，如 a.jpg 与 A.png）时输出会互相覆盖，抛 ValueError；
      某个输出路径恰好是某张输入图片时（例如 out_dir 与输入目录相同且里面已有 *_overlay.jpg）抛 ValueError。
    - 无法解码的图片用 warnings.warn 提示后跳过（不计入返回值），不中断整批处理（与 autolabel 一致）。
    """
    from tqdm import tqdm

    t = _check_threshold(threshold)
    color_px = _check_color(color)
    a = _check_alpha(alpha)
    out_dir = Path(out_dir)
    paths = _collect_images(inputs)

    # stem 冲突检查（Windows 文件名大小写不敏感，按 casefold 比较）
    by_stem: dict[str, list[Path]] = {}
    for p in paths:
        by_stem.setdefault(p.stem.casefold(), []).append(p)
    dups = [ps for ps in by_stem.values() if len(ps) > 1]
    if dups:
        detail = "; ".join(", ".join(str(p) for p in ps) for ps in dups)
        raise ValueError(f"多张图片的文件名（stem）相同，输出会互相覆盖: {detail}")

    plan = [(p, out_dir / f"{p.stem}_overlay.jpg", out_dir / f"{p.stem}_mask.png") for p in paths]
    input_keys = {p.resolve() for p in paths}
    for _, overlay_path, mask_path in plan:
        for target in (overlay_path, mask_path) if save_mask else (overlay_path,):
            if target.resolve() in input_keys:
                raise ValueError(f"输出会覆盖输入图片: {target}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for img_path, overlay_path, mask_path in tqdm(plan, desc="infer", unit="img"):
        try:
            image = read_image_rgb(img_path)
        except ValueError as e:
            warnings.warn(f"跳过无法读取的图片 {img_path}: {e}", stacklevel=2)
            continue
        mask = predictor.predict_mask(image, t, hud_rects)
        write_image_rgb(overlay_path, overlay(image, mask, color_px, a))
        if save_mask:
            write_mask(mask_path, mask)
        written.append(overlay_path)
    return written


# ---------------------------------------------------------------- 视频


def infer_video(
    predictor: Predictor,
    video_path: str | Path,
    out_path: str | Path,
    threshold: float = 0.5,
    alpha: float = 0.45,
    color: Sequence[int] = (0, 255, 0),
    hud_rects: Sequence[Sequence[float]] = (),
    every_n: int = 1,
    max_frames: int | None = None,
) -> Path:
    """逐帧推理并写出叠加视频（cv2.VideoWriter，fourcc "mp4v"，与源视频相同 fps（读不到按 30）和分辨率）。
    every_n > 1 时每 every_n 帧推理一次，中间帧复用上一次的掩码。max_frames 限制处理的帧数。
    视频打不开或 writer 打不开时抛 ValueError。返回 out_path。

    补充约定：
    - 推理发生在第 0、every_n、2*every_n…… 帧；输出视频包含所有已处理的帧（帧数 = min(总帧数, max_frames)）。
    - every_n 必须是 >= 1 的整数，max_frames 必须是 None 或 >= 1 的整数，threshold / color / alpha 非法，
      或 out_path 与 video_path 是同一文件时，在打开视频之前抛 ValueError。
    - 视频文件不存在、无法解码、或一帧都读不出来时抛 ValueError。
    - 输出分辨率取第一帧的尺寸；个别帧尺寸不同（少见的变分辨率流）时先缩放到该尺寸再推理。
    - 自动创建 out_path 的父目录。
    """
    from tqdm import tqdm

    if isinstance(every_n, bool) or not isinstance(every_n, (int, np.integer)) or every_n < 1:
        raise ValueError(f"every_n 必须是 >= 1 的整数，得到 {every_n!r}")
    if max_frames is not None and (
        isinstance(max_frames, bool) or not isinstance(max_frames, (int, np.integer)) or max_frames < 1
    ):
        raise ValueError(f"max_frames 必须是 None 或 >= 1 的整数，得到 {max_frames!r}")
    every_n = int(every_n)
    t = _check_threshold(threshold)
    color_px = _check_color(color)
    a = _check_alpha(alpha)
    video_path = Path(video_path)
    out_path = Path(out_path)
    if out_path.exists() and video_path.exists() and out_path.resolve() == video_path.resolve():
        raise ValueError(f"out_path 不能与输入视频相同: {out_path}")

    cap = open_video(video_path)  # 打不开抛 ValueError；Windows 中文路径自动用短路径重试
    writer: cv2.VideoWriter | None = None
    pbar = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = DEFAULT_VIDEO_FPS

        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"视频中没有可读取的帧: {video_path}")
        height, width = frame.shape[:2]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"无法创建输出视频（mp4v 编码器不可用或路径不可写）: {out_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # 部分容器读不到或不准，只用于进度条
        if max_frames is not None:
            total = min(total, max_frames) if total > 0 else max_frames
        pbar = tqdm(total=total if total > 0 else None, desc="infer-video", unit="frame")

        index = 0
        mask: np.ndarray | None = None
        while ok and frame is not None:
            rgb = bgr_to_rgb(frame)
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            if mask is None or index % every_n == 0:
                mask = predictor.predict_mask(rgb, t, hud_rects)
            vis = overlay(rgb, mask, color_px, a)
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            index += 1
            pbar.update(1)
            if max_frames is not None and index >= max_frames:
                break
            ok, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if pbar is not None:
            pbar.close()
    return out_path
