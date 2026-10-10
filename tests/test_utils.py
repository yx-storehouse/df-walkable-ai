"""dfwalk.utils 的单元测试：中文路径图片/掩码读写、掩码位深检查、打开视频、BGR → RGB、设备选择。

全部离线、只用 CPU，测试数据现场合成。
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest
import torch

from dfwalk import IMAGE_EXTS
from dfwalk.utils import (
    bgr_to_rgb,
    list_images,
    open_video,
    read_image_rgb,
    read_mask,
    resolve_device,
    seed_everything,
    write_image_rgb,
    write_mask,
)

H, W = 24, 40


def _gradient_rgb() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = np.linspace(0, 255, W, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 255, H, dtype=np.uint8)[:, None]
    img[..., 2] = 77
    return img


def _write_raw_png(path, array: np.ndarray) -> None:
    """绕过 write_mask 直接写 PNG（用来构造 16 位 / 多通道等“不合规”的掩码文件）。"""
    ok, enc = cv2.imencode(".png", array)
    assert ok
    enc.tofile(str(path))


# ---------------------------------------------------------------- 图片 / 掩码读写


def test_image_roundtrip_png_chinese_path(tmp_path):
    img = _gradient_rgb()
    path = tmp_path / "零号大坝" / "截图 1.png"
    write_image_rgb(path, img)  # 自动创建父目录
    out = read_image_rgb(path)
    assert out.dtype == np.uint8 and out.shape == (H, W, 3)
    np.testing.assert_array_equal(out, img)  # PNG 无损，且通道顺序保持 RGB


def test_image_jpeg_keeps_rgb_order(tmp_path):
    img = np.zeros((32, 32, 3), np.uint8)
    img[..., 0] = 220  # 纯红
    path = tmp_path / "红.jpg"
    write_image_rgb(path, img, jpeg_quality=95)
    out = read_image_rgb(path).astype(int)
    assert out[..., 0].mean() > 200 and out[..., 1].mean() < 30 and out[..., 2].mean() < 30


def test_read_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image_rgb(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="bad.png"):
        read_image_rgb(bad)


def test_mask_roundtrip_chinese_path(tmp_path):
    mask = np.zeros((H, W), np.uint8)
    mask[5:10, 5:20] = 1
    mask[:3] = 255
    path = tmp_path / "掩码" / "帧_000001.png"
    write_mask(path, mask)
    out = read_mask(path)
    assert out.dtype == np.uint8 and out.shape == (H, W)
    np.testing.assert_array_equal(out, mask)


def test_write_mask_requires_png_and_2d(tmp_path):
    with pytest.raises(ValueError, match=".png"):
        write_mask(tmp_path / "m.jpg", np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError):
        write_mask(tmp_path / "m.png", np.zeros((4, 4, 3), np.uint8))


def test_read_mask_rejects_16bit_png(tmp_path):
    # 16 位 PNG 若直接转 uint8，256 会变成 0、257 会变成 1，值域检查发现不了，必须显式拒绝
    mask16 = np.array([[0, 1, 256, 257]], dtype=np.uint16)
    path = tmp_path / "m16.png"
    _write_raw_png(path, mask16)
    with pytest.raises(ValueError, match="8 位"):
        read_mask(path)


def test_read_mask_rejects_multichannel_png(tmp_path):
    path = tmp_path / "rgb.png"
    _write_raw_png(path, np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError, match="单通道"):
        read_mask(path)


def test_read_mask_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_mask(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG broken")
    with pytest.raises(ValueError):
        read_mask(bad)


def test_list_images_sorted_non_recursive_case_insensitive(tmp_path):
    img = _gradient_rgb()
    for name in ("b.jpg", "a.PNG", "c.webp"):
        write_image_rgb(tmp_path / name, img)
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "video.mp4").write_bytes(b"")
    write_image_rgb(tmp_path / "sub" / "d.jpg", img)
    names = [p.name for p in list_images(tmp_path)]
    assert names == ["a.PNG", "b.jpg", "c.webp"]
    assert all(p.suffix.lower() in IMAGE_EXTS for p in list_images(tmp_path))


# ---------------------------------------------------------------- 视频


def _write_video(path, n: int = 4, fps: float = 10.0):
    """写一个小视频，第 i 帧为纯色 (B, G, R) = (10*i, 100, 200)。mp4v 不可用时退回 MJPG/.avi。"""
    for fourcc, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        target = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*fourcc), fps, (32, 24))
        if not writer.isOpened():
            writer.release()
            continue
        for i in range(n):
            writer.write(np.full((24, 32, 3), (10 * i, 100, 200), np.uint8))
        writer.release()
        if target.is_file() and target.stat().st_size > 0:
            return target
    pytest.skip("没有可用的视频编码器")


def test_open_video_reads_frames_chinese_path(tmp_path):
    path = _write_video(tmp_path / "长弓溪谷 第2局")
    cap = open_video(path)
    try:
        assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, abs=0.01)
        ok, frame = cap.read()
        assert ok and frame.shape == (24, 32, 3)
        rgb = bgr_to_rgb(frame).astype(int)
        # 第 0 帧 BGR = (0, 100, 200) → RGB ≈ (200, 100, 0)
        assert abs(rgb[..., 0].mean() - 200) < 12 and abs(rgb[..., 2].mean() - 0) < 12
    finally:
        cap.release()


def test_open_video_missing_or_corrupt_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="missing.mp4"):
        open_video(tmp_path / "missing.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(ValueError, match="bad.mp4"):
        open_video(bad)


def test_open_video_directory_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        open_video(tmp_path)


def test_bgr_to_rgb_handles_bgr_gray_and_bgra():
    bgr = np.zeros((2, 3, 3), np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 2] = 200  # R
    rgb = bgr_to_rgb(bgr)
    assert rgb.shape == (2, 3, 3) and rgb[0, 0].tolist() == [200, 0, 10]

    gray = np.full((2, 3), 50, np.uint8)
    assert bgr_to_rgb(gray).shape == (2, 3, 3)
    assert bgr_to_rgb(gray)[0, 0].tolist() == [50, 50, 50]

    bgra = np.zeros((2, 3, 4), np.uint8)
    bgra[..., 0], bgra[..., 2], bgra[..., 3] = 10, 200, 128
    out = bgr_to_rgb(bgra)
    assert out.shape == (2, 3, 3) and out[0, 0].tolist() == [200, 0, 10]


def test_bgr_to_rgb_does_not_modify_input():
    bgr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    before = bgr.copy()
    bgr_to_rgb(bgr)
    np.testing.assert_array_equal(bgr, before)


# ---------------------------------------------------------------- 设备 / 随机种子


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda:1") == torch.device("cuda:1")  # 只构造 device，不要求真的有 GPU
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(123)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b
