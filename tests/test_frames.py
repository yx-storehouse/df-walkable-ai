"""dfwalk.frames 的单元测试：文件名清洗、缩放、去重、采样规则和视频抽帧。

测试视频在测试内用 cv2.VideoWriter 现场合成（优先 mp4v/.mp4，打不开时回退 MJPG/.avi），
不依赖网络和外部文件。
"""

from __future__ import annotations

import math
import random
import re
import shutil
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from dfwalk import frames
from dfwalk.frames import (
    _FrameSampler,
    downscale_gray,
    extract_frames,
    is_duplicate,
    resize_max_side,
    sanitize_stem,
)
from dfwalk.utils import read_image_rgb

H, W = 96, 160  # 合成视频尺寸（宽高都是 16 的倍数，编码器最友好）
SRC_FPS = 10.0


# ---------------------------------------------------------------- 合成数据工具


def pattern(k: int, h: int = H, w: int = W) -> np.ndarray:
    """第 k 种画面：深色背景 + 一个移动的亮色竖条。相邻 k 的竖条位置相差很大，平均灰度差约 60。"""
    img = np.full((h, w, 3), 30, np.uint8)
    x = (k * 37) % (w - 30)
    img[:, x : x + 30] = (230, 200, 40)
    return img


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.isOpened():
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def write_video(tmp_path: Path, dest: Path, frames_rgb: list[np.ndarray], fps: float = SRC_FPS) -> Path:
    """把 RGB 帧写成视频 dest + ".mp4"（回退编码器时为 ".avi"），返回实际路径。

    先写到纯 ASCII 的暂存目录再移动到 dest，避免 VideoWriter 在某些平台不支持中文路径。
    """
    h, w = frames_rgb[0].shape[:2]
    stage = tmp_path / "_stage"
    stage.mkdir(exist_ok=True)
    for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        tmp = stage / f"v{ext}"
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if not writer.isOpened():
            writer.release()
            continue
        for f in frames_rgb:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        if _count_frames(tmp) != len(frames_rgb):
            continue
        final = dest.parent / (dest.name + ext)  # 不用 with_suffix：文件名主干里可能本身带点号
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(final))
        return final
    pytest.skip("当前环境的 OpenCV 无法写出 mp4v / MJPG 视频")


def distinct_frames(n: int) -> list[np.ndarray]:
    return [pattern(k) for k in range(n)]


def jpg_index(path: Path) -> int:
    m = re.fullmatch(r".*_(\d{6})\.jpg", path.name)
    assert m, path.name
    return int(m.group(1))


class FakeCapture:
    """模拟 cv2.VideoCapture：返回给定的 BGR 帧，fps 可以是 0 / NaN 等异常值。"""

    def __init__(self, frames_bgr: list[np.ndarray | None], fps: float):
        self._frames = frames_bgr
        self._fps = fps
        self._pos = -1
        self.released = False
        self.retrieved: list[int] = []

    def isOpened(self) -> bool:
        return not self.released

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def grab(self) -> bool:
        if self._pos + 1 >= len(self._frames):
            return False
        self._pos += 1
        return True

    def retrieve(self):
        self.retrieved.append(self._pos)
        f = self._frames[self._pos]
        if f is None:
            return False, None
        return True, f.copy()

    def read(self):
        if not self.grab():
            return False, None
        return self.retrieve()

    def release(self) -> None:
        self.released = True


def use_fake(monkeypatch, frames_bgr, fps) -> FakeCapture:
    cap = FakeCapture(frames_bgr, fps)
    monkeypatch.setattr(frames, "open_video", lambda path: cap)
    return cap


def gray_frame(value: int, h: int = 18, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


# ---------------------------------------------------------------- sanitize_stem


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("零号大坝 第1局", "零号大坝_第1局"),
        ("run (2).final", "run_2_final"),
        ("[2024-01-01] 航天基地", "2024-01-01_航天基地"),
        ("a  b", "a_b"),
        ("a__b", "a_b"),
        ("a _ b", "a_b"),
        ("__a__", "a"),
        ("abc-DEF_123", "abc-DEF_123"),
        ("--a--", "--a--"),
        ("x.y.z", "x_y_z"),
        ("a/b\\c", "a_b_c"),
        ("a\tb\nc", "a_b_c"),
        ("巴克什（困难）#3", "巴克什_困难_3"),  # 全角括号不是 CJK 表意文字
        ("café", "caf"),  # 非 ASCII 字母被替换
        ("", "video"),
        ("   ", "video"),
        ("()", "video"),
        ("...", "video"),
        ("テスト", "video"),  # 假名不在 一-鿿 范围内
        ("一丿龥", "一丿龥"),  # 范围两端附近的表意文字保留
    ],
)
def test_sanitize_stem_cases(raw, expected):
    assert sanitize_stem(raw) == expected


def test_sanitize_stem_invariants_on_random_strings():
    rng = random.Random(1234)
    alphabet = "aZ09-_ .()[]{}#@!，。（）、　零号大坝第局テé\t/\\"
    allowed = re.compile(r"[A-Za-z0-9\-_一-鿿]+")
    for _ in range(500):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 20)))
        out = sanitize_stem(raw)
        assert out, raw
        assert allowed.fullmatch(out), (raw, out)
        assert "__" not in out
        assert not out.startswith("_") and not out.endswith("_")
        assert sanitize_stem(out) == out  # 幂等


# ---------------------------------------------------------------- downscale_gray


def test_downscale_gray_default_size_and_dtype():
    img = np.zeros((720, 1280, 3), np.uint8)
    small = downscale_gray(img)
    assert small.shape == (36, 64)  # size=(宽, 高) → 数组 HxW
    assert small.dtype == np.float32


def test_downscale_gray_custom_size_and_values():
    img = np.zeros((50, 80, 3), np.uint8)
    img[:, :, 1] = 200  # 纯绿
    small = downscale_gray(img, size=(10, 5))
    assert small.shape == (5, 10)
    expected = cv2.cvtColor(np.full((1, 1, 3), (0, 200, 0), np.uint8), cv2.COLOR_RGB2GRAY)[0, 0]
    np.testing.assert_allclose(small, float(expected), atol=1.0)


def test_downscale_gray_uses_rgb_order():
    red = np.zeros((20, 20, 3), np.uint8)
    red[:, :, 0] = 255  # RGB 中的红色
    blue = np.zeros((20, 20, 3), np.uint8)
    blue[:, :, 2] = 255
    # 灰度公式中 R 权重 (0.299) 大于 B 权重 (0.114)
    assert downscale_gray(red).mean() > downscale_gray(blue).mean()


def test_downscale_gray_accepts_gray_and_rgba():
    assert downscale_gray(np.zeros((40, 60), np.uint8)).shape == (36, 64)
    assert downscale_gray(np.zeros((40, 60, 4), np.uint8)).shape == (36, 64)


# ---------------------------------------------------------------- is_duplicate


def test_is_duplicate_none_prev_is_false():
    cur = np.zeros((36, 64), np.float32)
    assert is_duplicate(None, cur, 6.0) is False


@pytest.mark.parametrize("threshold", [0.0, -1.0])
def test_is_duplicate_disabled_threshold_is_false(threshold):
    a = np.zeros((36, 64), np.float32)
    assert is_duplicate(a, a.copy(), threshold) is False


def test_is_duplicate_strict_less_than():
    a = np.zeros((36, 64), np.float32)
    assert is_duplicate(a, a + 5.9, 6.0) is True
    assert is_duplicate(a, a + 6.0, 6.0) is False  # 恰好等于阈值不算重复
    assert is_duplicate(a, a - 10.0, 6.0) is False  # 取绝对值
    assert is_duplicate(a, a.copy(), 6.0) is True


def test_is_duplicate_uses_mean_not_max():
    a = np.zeros((10, 10), np.float32)
    b = a.copy()
    b[0, 0] = 255.0  # 单个像素差很大，但平均差 2.55
    assert is_duplicate(a, b, 6.0) is True


def test_is_duplicate_shape_mismatch_is_false():
    assert is_duplicate(np.zeros((36, 64), np.float32), np.zeros((10, 10), np.float32), 6.0) is False


# ---------------------------------------------------------------- resize_max_side


def test_resize_max_side_landscape():
    out = resize_max_side(np.zeros((1080, 1920, 3), np.uint8), 1280)
    assert out.shape == (720, 1280, 3)
    assert out.dtype == np.uint8


def test_resize_max_side_portrait():
    out = resize_max_side(np.zeros((1920, 1080, 3), np.uint8), 1280)
    assert out.shape == (1280, 720, 3)


def test_resize_max_side_odd_ratio_keeps_long_side_exact():
    out = resize_max_side(np.zeros((1000, 1333, 3), np.uint8), 640)
    assert out.shape[1] == 640
    assert out.shape[0] == round(1000 * 640 / 1333)


@pytest.mark.parametrize("max_side", [0, -5, 1920, 4000])
def test_resize_max_side_returns_original_when_not_needed(max_side):
    img = np.zeros((1080, 1920, 3), np.uint8)
    assert resize_max_side(img, max_side) is img


def test_resize_max_side_extreme_aspect_keeps_one_pixel():
    out = resize_max_side(np.zeros((1, 4000, 3), np.uint8), 100)
    assert out.shape == (1, 100, 3)


def test_resize_max_side_grayscale_and_area_average():
    img = np.zeros((4, 8), np.uint8)
    img[:, ::2] = 200  # 竖条纹，INTER_AREA 缩小后取平均
    out = resize_max_side(img, 4)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, 100, atol=1)


# ---------------------------------------------------------------- 采样规则


def _sampled(n: int, src_fps: float, fps: float) -> list[int]:
    s = _FrameSampler(src_fps, fps)
    out = []
    for i in range(n):
        if s.due(i):
            out.append(i)
            s.advance(i)
    return out


def test_sampler_integer_ratio():
    assert _sampled(40, 10.0, 2.0) == [0, 5, 10, 15, 20, 25, 30, 35]
    assert _sampled(31, 30.0, 10.0) == list(range(0, 31, 3))


@pytest.mark.parametrize("fps", [0.0, -1.0, 10.0, 10.0001, 60.0, float("inf")])
def test_sampler_every_frame(fps):
    assert _sampled(25, 10.0, fps) == list(range(25))


def test_sampler_nan_fps_rejected():
    with pytest.raises(ValueError):
        _FrameSampler(30.0, float("nan"))


def test_sampler_fractional_fps():
    # 源 10 fps，目标 3 fps：采样时刻 0, 1/3, 2/3, 1, ... → 帧 0, 4, 7, 10, 14, 17, 20
    assert _sampled(21, 10.0, 3.0) == [0, 4, 7, 10, 14, 17, 20]
    # 目标 fps 低于 1：每 2.5 秒一帧
    assert _sampled(100, 10.0, 0.4) == [0, 25, 50, 75]


@pytest.mark.parametrize(("src", "fps"), [(Fraction(2997, 100), 3), (Fraction(30000, 1001), 2), (60, 7)])
def test_sampler_no_drift_over_two_hours(src, fps):
    """两小时录像：采样结果必须与精确有理数计算完全一致（不随时间漂移）。"""
    n = int(2 * 3600 * src)
    got = _sampled(n, float(src), float(fps))
    # 第 k 个采样时刻 k/fps 对应的帧：第一个满足 i/src >= k/fps 的 i = ceil(k * src / fps)
    expected = []
    k = 0
    while True:
        i = math.ceil(Fraction(k) * Fraction(src) / Fraction(fps))
        if i >= n:
            break
        expected.append(i)
        k += 1
    assert got == expected
    # 总数约为 时长 × fps
    assert abs(len(got) - 2 * 3600 * fps) <= 1


def test_sampler_tolerates_float_rounding_at_exact_boundaries():
    # 29.97 fps 下第 2997 帧恰好是 100 秒，必须被采样（t 与 next_t 理论上相等）
    s = _FrameSampler(29.97, 1.0)
    s.k = 100
    assert s.due(2997)
    assert not s.due(2996)


# ---------------------------------------------------------------- extract_frames（真实视频）


def test_extract_sampling_fps2_from_10fps(tmp_path):
    video = write_video(tmp_path, tmp_path / "clip", distinct_frames(40))
    out = extract_frames(video, tmp_path / "out", fps=2.0, dedup_threshold=0)
    assert [p.name for p in out] == [f"clip_{i:06d}.jpg" for i in range(0, 40, 5)]
    assert all(p.is_file() for p in out)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [p.name for p in out]
    img = read_image_rgb(out[0])
    assert img.shape == (H, W, 3)


@pytest.mark.parametrize("fps", [0.0, -1.0, 10.0, 25.0])
def test_extract_keeps_every_frame_when_fps_nonpositive_or_too_high(tmp_path, fps):
    video = write_video(tmp_path, tmp_path / "clip", distinct_frames(12))
    out = extract_frames(video, tmp_path / "out", fps=fps, dedup_threshold=6.0)
    assert [jpg_index(p) for p in out] == list(range(12))


def test_extract_names_use_source_frame_index(tmp_path):
    video = write_video(tmp_path, tmp_path / "clip", distinct_frames(30))
    out = extract_frames(video, tmp_path / "out", fps=3.0, dedup_threshold=0)
    # 源 10 fps、目标 3 fps：帧 0, 4, 7, 10, 14, 17, 20, 24, 27
    assert [jpg_index(p) for p in out] == [0, 4, 7, 10, 14, 17, 20, 24, 27]


def test_extract_dedup_on_and_off(tmp_path):
    runs = [pattern(0)] * 4 + [pattern(1)] * 4 + [pattern(2)] * 4 + [pattern(3), pattern(4)]
    video = write_video(tmp_path, tmp_path / "dup", runs)

    on = extract_frames(video, tmp_path / "on", fps=0, dedup_threshold=6.0)
    assert [jpg_index(p) for p in on] == [0, 4, 8, 12, 13]

    off = extract_frames(video, tmp_path / "off", fps=0, dedup_threshold=0)
    assert [jpg_index(p) for p in off] == list(range(14))


def test_extract_dedup_after_sampling(tmp_path):
    # 帧 0..9 相同，10..19 相同；fps=2 采样 0,5,10,15 → 去重后剩 0,10
    seq = [pattern(0)] * 10 + [pattern(1)] * 10
    video = write_video(tmp_path, tmp_path / "dup2", seq)
    out = extract_frames(video, tmp_path / "out", fps=2.0, dedup_threshold=6.0)
    assert [jpg_index(p) for p in out] == [0, 10]


def test_extract_resize_max_side(tmp_path):
    video = write_video(tmp_path, tmp_path / "clip", distinct_frames(5))
    small = extract_frames(video, tmp_path / "small", fps=0, max_side=80, dedup_threshold=0)
    assert read_image_rgb(small[0]).shape == (48, 80, 3)
    full = extract_frames(video, tmp_path / "full", fps=0, max_side=0, dedup_threshold=0)
    assert read_image_rgb(full[0]).shape == (H, W, 3)


def test_extract_output_content_is_rgb(tmp_path):
    video = write_video(tmp_path, tmp_path / "clip", [pattern(0)] * 3)
    out = extract_frames(video, tmp_path / "out", fps=0, dedup_threshold=0)
    img = read_image_rgb(out[0]).astype(int)
    # 亮条颜色为 RGB (230, 200, 40)：R 通道必须明显大于 B 通道（没有把 BGR 当 RGB 写出）
    bar = img[:, 5:25].reshape(-1, 3).mean(axis=0)
    assert bar[0] > bar[2] + 100


def test_extract_jpeg_quality_affects_size(tmp_path):
    rng = np.random.default_rng(0)
    noisy = [rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8) for _ in range(2)]
    video = write_video(tmp_path, tmp_path / "noise", noisy)
    hi = extract_frames(video, tmp_path / "hi", fps=0, dedup_threshold=0, jpeg_quality=95)
    lo = extract_frames(video, tmp_path / "lo", fps=0, dedup_threshold=0, jpeg_quality=10)
    assert lo[0].stat().st_size < hi[0].stat().st_size


def test_extract_default_prefix_is_sanitized_stem(tmp_path):
    video = write_video(tmp_path, tmp_path / "run (2).final", distinct_frames(3))
    assert video.stem == "run (2).final"
    out = extract_frames(video, tmp_path / "out", fps=0, dedup_threshold=0)
    assert [p.name for p in out] == ["run_2_final_000000.jpg", "run_2_final_000001.jpg", "run_2_final_000002.jpg"]


def test_extract_explicit_prefix(tmp_path):
    video = write_video(tmp_path, tmp_path / "clip", distinct_frames(3))
    out = extract_frames(video, tmp_path / "out", fps=0, dedup_threshold=0, prefix="dam_run1")
    assert [p.name for p in out] == [f"dam_run1_{i:06d}.jpg" for i in range(3)]


@pytest.mark.parametrize("prefix", ["", "a/b", "a\\b"])
def test_extract_invalid_prefix_raises(tmp_path, prefix):
    video = write_video(tmp_path, tmp_path / "clip", distinct_frames(2))
    with pytest.raises(ValueError):
        extract_frames(video, tmp_path / "out", prefix=prefix)


def test_extract_chinese_paths(tmp_path):
    video = write_video(tmp_path, tmp_path / "录像" / "零号大坝 第1局", distinct_frames(20))
    assert video.parent.name == "录像" and video.stem == "零号大坝 第1局"
    out_dir = tmp_path / "数据" / "图片"
    out = extract_frames(str(video), str(out_dir), fps=2.0, dedup_threshold=6.0)
    assert [p.name for p in out] == ["零号大坝_第1局_000000.jpg", "零号大坝_第1局_000005.jpg",
                                     "零号大坝_第1局_000010.jpg", "零号大坝_第1局_000015.jpg"]
    for p in out:
        assert p.parent == out_dir
        assert read_image_rgb(p).shape == (H, W, 3)


def test_extract_creates_nested_out_dir(tmp_path):
    video = write_video(tmp_path, tmp_path / "clip", distinct_frames(2))
    out_dir = tmp_path / "a" / "b" / "c"
    out = extract_frames(video, out_dir, fps=0, dedup_threshold=0)
    assert out_dir.is_dir() and len(out) == 2


def test_extract_unopenable_video_raises_with_path(tmp_path):
    bad = tmp_path / "坏的 video.mp4"
    bad.write_bytes(b"this is definitely not a video file" * 10)
    with pytest.raises(ValueError) as exc:
        extract_frames(bad, tmp_path / "out")
    assert str(bad) in str(exc.value)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_extract_missing_video_raises_value_error(tmp_path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(ValueError) as exc:
        extract_frames(missing, tmp_path / "out")
    assert str(missing) in str(exc.value)


# ---------------------------------------------------------------- extract_frames（模拟 VideoCapture）


@pytest.mark.parametrize("bad_fps", [0.0, -1.0, float("nan"), float("inf")])
def test_extract_unknown_src_fps_defaults_to_30(tmp_path, monkeypatch, bad_fps):
    seq = [gray_frame((i * 41) % 256) for i in range(61)]
    use_fake(monkeypatch, seq, bad_fps)
    out = extract_frames(tmp_path / "x.mp4", tmp_path / "out", fps=2.0, dedup_threshold=0)
    # 按 30 fps 处理：每 15 帧采样一次
    assert [jpg_index(p) for p in out] == [0, 15, 30, 45, 60]


def test_extract_dedup_compares_with_last_kept_frame(tmp_path, monkeypatch):
    # 亮度每帧 +2：相邻帧差 2 < 6，但累计差会超过阈值。
    # 与“上一张保留帧”比较 → 每 3 帧保留一张（差 6 不小于 6，不算重复）；
    # 若错误地与“上一张采样帧”比较，则只会保留第 0 帧。
    seq = [gray_frame(50 + 2 * i) for i in range(10)]
    use_fake(monkeypatch, seq, 10.0)
    out = extract_frames(tmp_path / "x.mp4", tmp_path / "out", fps=0, dedup_threshold=6.0)
    assert [jpg_index(p) for p in out] == [0, 3, 6, 9]


def test_extract_only_retrieves_sampled_frames_and_releases(tmp_path, monkeypatch):
    seq = [gray_frame((i * 41) % 256) for i in range(30)]
    cap = use_fake(monkeypatch, seq, 10.0)
    out = extract_frames(tmp_path / "x.mp4", tmp_path / "out", fps=2.0, dedup_threshold=0)
    assert [jpg_index(p) for p in out] == [0, 5, 10, 15, 20, 25]
    assert cap.retrieved == [0, 5, 10, 15, 20, 25]
    assert cap.released


def test_extract_failed_decode_is_replaced_by_next_frame(tmp_path, monkeypatch):
    seq = [gray_frame((i * 41) % 256) for i in range(20)]
    seq[5] = None  # 第 5 帧解码失败
    use_fake(monkeypatch, seq, 10.0)
    out = extract_frames(tmp_path / "x.mp4", tmp_path / "out", fps=2.0, dedup_threshold=0)
    # 第 5 帧的采样时刻由第 6 帧顶替，后续采样时刻不受影响
    assert [jpg_index(p) for p in out] == [0, 6, 10, 15]


def test_extract_empty_video_returns_empty_list(tmp_path, monkeypatch):
    cap = use_fake(monkeypatch, [], 25.0)
    out_dir = tmp_path / "out"
    assert extract_frames(tmp_path / "x.mp4", out_dir) == []
    assert out_dir.is_dir()
    assert cap.released


def test_extract_releases_capture_on_write_error(tmp_path, monkeypatch):
    cap = use_fake(monkeypatch, [gray_frame(10)] * 3, 10.0)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(frames, "write_image_rgb", boom)
    with pytest.raises(OSError):
        extract_frames(tmp_path / "x.mp4", tmp_path / "out", fps=0)
    assert cap.released


def test_extract_handles_grayscale_and_bgra_frames(tmp_path, monkeypatch):
    seq = [np.full((18, 32), 10, np.uint8), np.full((18, 32, 4), 200, np.uint8)]
    use_fake(monkeypatch, seq, 10.0)
    out = extract_frames(tmp_path / "x.mp4", tmp_path / "out", fps=0, dedup_threshold=0)
    assert len(out) == 2
    for p in out:
        assert read_image_rgb(p).shape == (18, 32, 3)


def test_extract_long_fake_video_matches_sampler(tmp_path, monkeypatch):
    # 29.97 fps、约 3 分钟的“视频”，抽帧序号与纯采样规则一致
    n = int(29.97 * 180)
    tiny = np.zeros((2, 2, 3), np.uint8)
    use_fake(monkeypatch, [tiny] * n, 29.97)
    monkeypatch.setattr(frames, "write_image_rgb", lambda *a, **k: None)  # 只验证序号，不真正写盘
    out = extract_frames(tmp_path / "x.mp4", tmp_path / "out", fps=0.5, dedup_threshold=0)
    assert [jpg_index(p) for p in out] == _sampled(n, 29.97, 0.5)
    assert len(out) == 90
