"""从自己录制的对局视频中抽帧，用于标注和训练。

帧文件命名约定（dataset.group_of 依赖它按视频分组）：
    {prefix}_{frame_index:06d}.jpg
    prefix 默认为 sanitize_stem(视频文件名不含后缀)，frame_index 为该帧在源视频中的帧序号（从 0 开始）。

只读取本地录像文件做离线分析，不涉及游戏进程。
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np

from .utils import bgr_to_rgb, open_video, write_image_rgb

# 源视频 fps 读不到时的兜底值
DEFAULT_SRC_FPS = 30.0
# 采样时刻比较的时间容差（秒），吸收 i / src_fps 与 k / fps 的浮点误差
_TIME_EPS = 1e-6

# 允许保留的字符：ASCII 字母数字、'-'、'_'、CJK 统一表意文字 (U+4E00 ~ U+9FFF)
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9\-_一-鿿]")
_MULTI_UNDERSCORE = re.compile(r"_{2,}")


def sanitize_stem(name: str) -> str:
    """把任意文件名主干转换为安全前缀：保留 ASCII 字母数字、'-'、'_' 和中日韩统一表意文字
    (\\u4e00-\\u9fff)，其他字符（空格、括号、点号等）替换为 '_'，连续的 '_' 合并为一个，
    去掉首尾 '_'。结果为空时返回 "video"。

    例："零号大坝 第1局" → "零号大坝_第1局"；"run (2).final" → "run_2_final"；"()" → "video"。
    """
    s = _UNSAFE_CHARS.sub("_", name)
    s = _MULTI_UNDERSCORE.sub("_", s).strip("_")
    return s or "video"


def downscale_gray(frame_rgb: np.ndarray, size: tuple[int, int] = (64, 36)) -> np.ndarray:
    """缩放到 size=(宽, 高)（cv2.INTER_AREA）并转灰度，返回 float32 HxW，用于去重比较。

    也接受 HxW 灰度图和 HxWx4 RGBA 输入。
    """
    small = cv2.resize(frame_rgb, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        channels = small.shape[2]
        if channels == 3:
            small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        elif channels == 4:
            small = cv2.cvtColor(small, cv2.COLOR_RGBA2GRAY)
        elif channels == 1:
            small = small[:, :, 0]
        else:
            raise ValueError(f"不支持的通道数: {channels}")
    return small.astype(np.float32)


def is_duplicate(prev_small: np.ndarray | None, cur_small: np.ndarray, threshold: float) -> bool:
    """prev_small 为 None 或 threshold <= 0 时返回 False；
    否则当 mean(|prev - cur|) < threshold 时返回 True。

    两者形状不同（例如视频中途换了分辨率）时无法比较，返回 False。
    """
    if prev_small is None or threshold <= 0:
        return False
    if prev_small.shape != cur_small.shape:
        return False
    diff = np.abs(prev_small.astype(np.float32) - cur_small.astype(np.float32))
    return float(diff.mean()) < float(threshold)


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    """最长边大于 max_side 时等比缩小到 max_side（cv2.INTER_AREA）；max_side <= 0 或已足够小则原样返回。

    缩小后最长边恰好等于 max_side，短边按比例四舍五入（至少 1 像素）。
    """
    h, w = frame.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return frame
    if w >= h:
        new_w = int(max_side)
        new_h = max(1, int(round(h * max_side / w)))
    else:
        new_h = int(max_side)
        new_w = max(1, int(round(w * max_side / h)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class _FrameSampler:
    """按目标 fps 决定哪些源帧需要采样（extract_frames 的采样规则）。

    第 i 帧时间戳 t = i / src_fps；下一个采样时刻 next_t 初始为 0。
    当 t >= next_t - 1e-6 时采样，然后 next_t += 1/fps（若仍 <= t 则继续累加直到 > t）。

    实现上用整数计数 k 表示 next_t = k / fps，与逐次累加 1/fps 语义相同，但不会在长视频
    （几小时、几十万帧）里累积浮点误差，也不依赖对帧序号四舍五入。
    fps <= 0 或 fps >= src_fps 时每帧都采样。
    """

    def __init__(self, src_fps: float, fps: float):
        if math.isnan(fps):
            raise ValueError("fps 不能是 NaN")
        self.src_fps = float(src_fps)
        self.fps = float(fps)
        self.every = self.fps <= 0 or self.fps >= self.src_fps
        self.k = 0

    @property
    def next_t(self) -> float:
        return 0.0 if self.every else self.k / self.fps

    def due(self, i: int) -> bool:
        """第 i 帧是否到了采样时刻。"""
        if self.every:
            return True
        return i / self.src_fps >= self.k / self.fps - _TIME_EPS

    def advance(self, i: int) -> None:
        """第 i 帧已被采样：推进到下一个采样时刻。"""
        if self.every:
            return
        t = i / self.src_fps
        self.k += 1
        while self.k / self.fps <= t:
            self.k += 1


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float = 2.0,
    max_side: int = 1280,
    dedup_threshold: float = 6.0,
    jpeg_quality: int = 95,
    prefix: str | None = None,
) -> list[Path]:
    """按目标 fps 从视频中抽帧并写成 JPEG，返回写出的文件路径列表（按时间顺序）。

    - 用 cv2.VideoCapture 顺序读取（不要用 seek，很多编码格式 seek 不精确）。
    - 源视频 fps 读不到（<=0 或 NaN）时按 30 处理。
    - 采样规则：第 i 帧的时间戳 t = i / src_fps；维护下一个采样时刻 next_t（初始 0），
      当 t >= next_t - 1e-6 时采样该帧，然后 next_t += 1/fps（若 next_t 仍 <= t，则继续累加直到 > t）。
      fps <= 0 或 fps >= src_fps 时每帧都采样。
    - 采样到的帧先 resize_max_side，再与“上一张保留帧”做 is_duplicate 去重，重复则丢弃。
    - 文件名 f"{prefix}_{i:06d}.jpg"，i 为源帧序号；prefix 为 None 时取 sanitize_stem(video_path.stem)。
    - 写图使用 dfwalk.utils.write_image_rgb（支持中文路径）。
    - 视频打不开时抛 ValueError（信息中包含路径；见 dfwalk.utils.open_video）。

    补充说明：
    - 未采样的帧只 grab() 不 retrieve()，省去颜色转换，仍是顺序读取。
    - 某个采样帧解码失败时跳过它，由下一帧顶替这个采样时刻。
    - out_dir 不存在时自动创建（即使最终一帧都没写）。
    - 显式传入的 prefix 原样使用，但不能为空或包含路径分隔符（否则抛 ValueError）。
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    if prefix is None:
        prefix = sanitize_stem(video_path.stem)
    elif not prefix or "/" in prefix or "\\" in prefix:
        raise ValueError(f"prefix 不能为空或包含路径分隔符: {prefix!r}")

    cap = open_video(video_path)  # 打不开抛 ValueError；Windows 中文路径自动用短路径重试
    written: list[Path] = []
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(src_fps) or src_fps <= 0:
            src_fps = DEFAULT_SRC_FPS
        sampler = _FrameSampler(src_fps, float(fps))
        out_dir.mkdir(parents=True, exist_ok=True)

        prev_small: np.ndarray | None = None
        i = 0
        while cap.grab():
            if sampler.due(i):
                ok, frame = cap.retrieve()
                if ok and frame is not None and frame.size > 0:
                    sampler.advance(i)
                    rgb = resize_max_side(bgr_to_rgb(frame), max_side)
                    small = downscale_gray(rgb)
                    if not is_duplicate(prev_small, small, dedup_threshold):
                        path = out_dir / f"{prefix}_{i:06d}.jpg"
                        write_image_rgb(path, rgb, jpeg_quality=jpeg_quality)
                        written.append(path)
                        prev_small = small
            i += 1
    finally:
        cap.release()
    return written
