"""训练/评估的数据变换（图像与掩码同步），只依赖 numpy/cv2/torch。

随机数一律使用 Python 标准库 random 模块：PyTorch DataLoader 会为每个 worker 重新设置 random 的种子，
而 numpy 的全局 RNG 在各 worker 中会重复，导致不同 worker 产生相同的增强。

输出约定：
    image: torch.float32 (3, H, W)，已按 mean/std 归一化
    mask:  torch.int64 (H, W)，值域 {0, 1, 255}；输入 mask 为 None 时输出 None

插值约定：
    图像：目标尺寸在两个方向上都不大于原图（缩小）时用 cv2.INTER_AREA，否则 cv2.INTER_LINEAR。
          EvalTransform 与 TrainTransform 使用同一规则，避免训练/推理之间的图像统计差异。
    掩码：最近邻（MASK_INTERP）。优先使用 cv2.INTER_NEAREST_EXACT——它与 INTER_NEAREST 一样只取原有像素值
          （不会产生 {0, 1, 255} 以外的标签），但按像素中心对齐采样；普通 INTER_NEAREST 在缩小时整体向左上偏移
          约半个原图像素，与 INTER_LINEAR / INTER_AREA 缩放后的图像错位。
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import cv2
import numpy as np
import torch

from . import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD

Transform = Callable[[np.ndarray, "np.ndarray | None"], tuple]

# 掩码缩放插值：按像素中心对齐的最近邻（旧版 OpenCV 没有时退回 INTER_NEAREST）
MASK_INTERP = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)

# ITU-R BT.601 亮度系数（与 cv2.COLOR_RGB2GRAY 一致），用于对比度/饱和度抖动
_GRAY_COEF = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ---------------------------------------------------------------- 参数检查


def _check_size(image_size: Sequence[int]) -> tuple[int, int]:
    """image_size → (H, W) 正整数元组。"""
    try:
        h, w = (int(v) for v in image_size)
    except (TypeError, ValueError):
        raise ValueError(f"image_size 必须是 [高, 宽] 两个正整数，得到 {image_size!r}") from None
    if h <= 0 or w <= 0:
        raise ValueError(f"image_size 必须是正整数，得到 {image_size!r}")
    return h, w


def _check_mean_std(mean: Sequence[float], std: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mean, dtype=np.float32).reshape(-1)
    s = np.asarray(std, dtype=np.float32).reshape(-1)
    if m.shape != (3,) or s.shape != (3,):
        raise ValueError(f"mean/std 必须各有 3 个数（RGB），得到 mean={mean!r}, std={std!r}")
    if not np.all(s > 0):
        raise ValueError(f"std 必须全部为正数，得到 {std!r}")
    return m, s


def _check_inputs(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    """检查 image 为 HxWx3、mask 为同尺寸 HxW；mask 统一转成 uint8（cv2.resize 不支持 int64/bool）。"""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        shape = getattr(image, "shape", None)
        raise ValueError(f"image 必须是 HxWx3 的 RGB 数组，得到 shape={shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image 尺寸不能为 0，得到 shape={image.shape}")
    # 切片/翻转得到的视图可能不连续，cv2 部分函数不接受负步长
    image = np.ascontiguousarray(image)
    if mask is None:
        return image, None
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        shape = getattr(mask, "shape", None)
        raise ValueError(f"mask 必须是 HxW 的单通道数组，得到 shape={shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"image 与 mask 尺寸不一致：image {image.shape[:2]}，mask {mask.shape}")
    if mask.dtype != np.uint8:
        if mask.size and (int(mask.min()) < 0 or int(mask.max()) > 255):
            raise ValueError(f"mask 值必须在 0~255 之间（约定为 0/1/{IGNORE_INDEX}），"
                             f"得到范围 [{mask.min()}, {mask.max()}]")
        mask = mask.astype(np.uint8)
    return image, np.ascontiguousarray(mask)


def _check_prob(name: str, p: float) -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} 必须在 [0, 1] 之间，得到 {p}")
    return p


# ---------------------------------------------------------------- 基础操作


def _image_interp(src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> int:
    """两个方向都不放大时用 INTER_AREA（抗锯齿），否则 INTER_LINEAR。"""
    return cv2.INTER_AREA if dst_hw[0] <= src_hw[0] and dst_hw[1] <= src_hw[1] else cv2.INTER_LINEAR


def _resize_pair(image: np.ndarray, mask: np.ndarray | None,
                 size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    """图像与掩码同步缩放到 size_hw=(H, W)；尺寸已一致时原样返回。"""
    h, w = size_hw
    src = image.shape[:2]
    if src == (h, w):
        return image, mask
    image = cv2.resize(image, (w, h), interpolation=_image_interp(src, (h, w)))
    if mask is not None:
        mask = cv2.resize(mask, (w, h), interpolation=MASK_INTERP)
    return image, mask


def _mask_to_tensor(mask: np.ndarray | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64))


def to_tensor_normalized(image_rgb: np.ndarray, mean: Sequence[float] = IMAGENET_MEAN,
                         std: Sequence[float] = IMAGENET_STD):
    """HxWx3 uint8 RGB → float32 张量 (3, H, W)：/255 后按 mean/std 归一化。

    也接受取值范围为 0~255 的其他数值类型（例如颜色抖动中间结果 float32）。形状不是 HxWx3 时抛 ValueError。
    返回的张量是连续内存，不与输入共享数据。
    """
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        shape = getattr(image_rgb, "shape", None)
        raise ValueError(f"image 必须是 HxWx3 的 RGB 数组，得到 shape={shape}")
    m, s = _check_mean_std(mean, std)
    # (x/255 - mean)/std == (x - 255*mean) / (255*std)，少一次整图遍历
    x = image_rgb.astype(np.float32)
    x -= m * 255.0
    x /= s * 255.0
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def _shift_hue(image: np.ndarray, turns: float) -> np.ndarray:
    """色相旋转 turns 圈（1 圈 = 360°），输入/输出均为取值 0~255 的 float32 HxWx3 RGB。

    用 float32 HSV（OpenCV 中 H ∈ [0, 360)）计算，避免 uint8 HSV（H ∈ [0, 180)，2° 一档）的量化误差；
    色相按 360° 取模环绕（例如 350° + 20° → 10°）。灰色像素（饱和度 0）不受影响。
    """
    if turns == 0.0:
        return image
    hsv = cv2.cvtColor(np.ascontiguousarray(image, dtype=np.float32) * (1.0 / 255.0), cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    h += float(turns) * 360.0
    np.mod(h, 360.0, out=h)
    # 浮点取模可能得到恰好 360.0（例如 -1e-7 % 360），归到 0
    h[h >= 360.0] = 0.0
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb *= 255.0
    return np.clip(rgb, 0.0, 255.0, out=rgb)


def _color_jitter(image: np.ndarray, brightness: float, contrast: float, saturation: float,
                  hue_turns: float) -> np.ndarray:
    """依次做亮度、对比度、饱和度（各为乘性因子，1 表示不变）和色相偏移，每步之后裁剪到 [0, 255]。
    返回 uint8 RGB；所有因子都是恒等时直接返回输入。"""
    if brightness == 1.0 and contrast == 1.0 and saturation == 1.0 and hue_turns == 0.0:
        return image
    img = image.astype(np.float32)
    if brightness != 1.0:
        img *= brightness
        np.clip(img, 0.0, 255.0, out=img)
    if contrast != 1.0:
        # 以整图平均亮度为中心拉伸/压缩
        mean_gray = float((img @ _GRAY_COEF).mean())
        img = (img - mean_gray) * contrast + mean_gray
        np.clip(img, 0.0, 255.0, out=img)
    if saturation != 1.0:
        # 与逐像素灰度混合：0 → 灰度图，1 → 原图，>1 → 更鲜艳
        gray = (img @ _GRAY_COEF)[..., None]
        img = (img - gray) * saturation + gray
        np.clip(img, 0.0, 255.0, out=img)
    if hue_turns != 0.0:
        img = _shift_hue(img, hue_turns)
    return np.rint(img).astype(np.uint8)


# ---------------------------------------------------------------- 变换类


class EvalTransform:
    """把图像缩放到 image_size=(H, W)（cv2.INTER_LINEAR；缩小时 INTER_AREA），掩码用 INTER_NEAREST。

    - “缩小”指目标高、宽都不大于原图；只要有一个方向放大就用 INTER_LINEAR。尺寸已一致时不做缩放。
    - 掩码实际使用 MASK_INTERP（像素中心对齐的最近邻，见模块说明），不会产生新的标签值。
    - 返回 (image float32 (3, H, W), mask int64 (H, W) 或 None)。无随机性。
    - image 不是 HxWx3、mask 不是同尺寸 HxW 时抛 ValueError。
    """

    def __init__(self, image_size: Sequence[int], mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD) -> None:
        self.size = _check_size(image_size)
        self.mean, self.std = _check_mean_std(mean, std)

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple:
        image, mask = _check_inputs(image, mask)
        image, mask = _resize_pair(image, mask, self.size)
        return to_tensor_normalized(image, self.mean, self.std), _mask_to_tensor(mask)

    def __repr__(self) -> str:
        return f"EvalTransform(image_size={list(self.size)})"


class TrainTransform:
    """训练增强，按顺序：
    1. 随机缩放：s ~ U(scale)，把图像整体缩放到 (round(H*s), round(W*s))（与原图宽高比无关，直接拉伸到目标比例），
       图像 INTER_LINEAR，掩码 INTER_NEAREST。
    2. 若小于 image_size 则在右/下方填充（图像填 0，掩码填 IGNORE_INDEX），再随机裁剪到 image_size。
    3. 以概率 hflip_p 水平翻转（图像与掩码同步）。
    4. 仅对图像做颜色抖动：亮度、对比度、饱和度各乘以 U(1-jitter, 1+jitter)，色相偏移 U(-hue, hue)（单位：圈，
       hue=0.03 约等于 ±11°），结果裁剪到 [0, 255]。
    5. 以概率 blur_p 做 3x3 高斯模糊（模拟运动模糊/低画质）。
    6. to_tensor_normalized。

    补充说明：
    - 第 1 步中 H、W 指 image_size（目标尺寸），而不是原图尺寸；缩放后的边长至少为 1 像素。
      图像插值与 EvalTransform 相同（两个方向都缩小时 INTER_AREA，否则 INTER_LINEAR），
      掩码用 MASK_INTERP（像素中心对齐的最近邻），保证掩码值仍在 {0, 1, 255} 内且与图像对齐。
    - 填充区域的图像在归一化后为 -mean/std，掩码为 IGNORE_INDEX，不参与 loss。翻转发生在填充之后，
      所以翻转时填充区域会出现在左侧。
    - 某项参数为 0（jitter=0、hue=0、blur_p=0、hflip_p=0）时完全跳过该步，图像像素不受影响。
    - 参数非法（scale 非正或下界大于上界、概率不在 [0, 1]、jitter 不在 [0, 1)、hue 不在 [0, 0.5]）抛 ValueError。
    - 只使用 random 模块的全局 RNG：random.seed(k) 后对相同输入的输出完全确定。
    """

    def __init__(
        self,
        image_size: Sequence[int],
        scale: tuple[float, float] = (0.75, 1.25),
        hflip_p: float = 0.5,
        jitter: float = 0.3,
        hue: float = 0.03,
        blur_p: float = 0.2,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.size = _check_size(image_size)
        try:
            lo, hi = (float(v) for v in scale)
        except (TypeError, ValueError):
            raise ValueError(f"scale 必须是 (下界, 上界)，得到 {scale!r}") from None
        if not 0.0 < lo <= hi:
            raise ValueError(f"scale 必须满足 0 < 下界 <= 上界，得到 {scale!r}")
        self.scale = (lo, hi)
        self.hflip_p = _check_prob("hflip_p", hflip_p)
        self.blur_p = _check_prob("blur_p", blur_p)
        self.jitter = float(jitter)
        if not 0.0 <= self.jitter < 1.0:
            raise ValueError(f"jitter 必须在 [0, 1) 之间，得到 {jitter}")
        self.hue = float(hue)
        if not 0.0 <= self.hue <= 0.5:
            raise ValueError(f"hue 必须在 [0, 0.5] 之间（单位：圈），得到 {hue}")
        self.mean, self.std = _check_mean_std(mean, std)

    def _jitter_factor(self) -> float:
        return random.uniform(1.0 - self.jitter, 1.0 + self.jitter) if self.jitter > 0 else 1.0

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> tuple:
        image, mask = _check_inputs(image, mask)
        th, tw = self.size

        # 1. 随机缩放（直接拉伸到目标宽高比）
        s = random.uniform(*self.scale)
        sh, sw = max(1, int(round(th * s))), max(1, int(round(tw * s)))
        image, mask = _resize_pair(image, mask, (sh, sw))

        # 2. 不足 image_size 的方向在右/下方填充，然后随机裁剪
        pad_h, pad_w = max(0, th - sh), max(0, tw - sw)
        if pad_h or pad_w:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if mask is not None:
                mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
        y0 = random.randint(0, image.shape[0] - th)
        x0 = random.randint(0, image.shape[1] - tw)
        image = image[y0:y0 + th, x0:x0 + tw]
        if mask is not None:
            mask = mask[y0:y0 + th, x0:x0 + tw]

        # 3. 水平翻转
        if self.hflip_p > 0 and random.random() < self.hflip_p:
            image = image[:, ::-1]
            if mask is not None:
                mask = mask[:, ::-1]

        # 4. 颜色抖动（仅图像）
        brightness = self._jitter_factor()
        contrast = self._jitter_factor()
        saturation = self._jitter_factor()
        hue_turns = random.uniform(-self.hue, self.hue) if self.hue > 0 else 0.0
        image = _color_jitter(np.ascontiguousarray(image), brightness, contrast, saturation, hue_turns)

        # 5. 高斯模糊
        if self.blur_p > 0 and random.random() < self.blur_p:
            image = cv2.GaussianBlur(np.ascontiguousarray(image), (3, 3), 0)

        # 6. 归一化 → 张量
        return to_tensor_normalized(image, self.mean, self.std), _mask_to_tensor(mask)

    def __repr__(self) -> str:
        return (f"TrainTransform(image_size={list(self.size)}, scale={self.scale}, hflip_p={self.hflip_p}, "
                f"jitter={self.jitter}, hue={self.hue}, blur_p={self.blur_p})")
