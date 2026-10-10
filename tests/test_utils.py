"""dfwalk.utils 的单元测试：中文路径图片/掩码读写、掩码位深检查、打开视频、BGR → RGB、设备选择。

全部离线、只用 CPU，测试数据现场合成。
"""

from __future__ import annotations

import random
import re
from pathlib import Path

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
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")
    if torch.cuda.is_available():
        assert resolve_device("cuda:0") == torch.device("cuda:0")


def test_resolve_device_cuda_unavailable_gives_actionable_error(monkeypatch):
    """CPU 版 PyTorch 上指定 cuda：立即抛带安装提示的 ValueError，而不是等到 model.to() 抛 AssertionError。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None)
    for name in ("cuda", "cuda:1"):
        with pytest.raises(ValueError, match="CPU 版 PyTorch") as info:
            resolve_device(name)
        assert "第 3 步" in str(info.value) and "--device auto / cpu" in str(info.value)
    # CUDA 版 PyTorch 但没有可用显卡/驱动
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    with pytest.raises(ValueError, match="驱动") as info:
        resolve_device("cuda")
    assert "uninstall" not in str(info.value)
    assert resolve_device("cpu") == torch.device("cpu")


def test_cpu_torch_remedy_uninstalls_before_step_3(monkeypatch):
    """已装 CPU 版 PyTorch 时，第 3 步的 pip install（无版本号、无 --force-reinstall）只会提示
    “Requirement already satisfied” 而不替换，所以报错信息和 README 都必须先让用户卸载 CPU 版，再执行第 3 步。"""
    uninstall = "pip uninstall -y torch torchvision"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None)
    with pytest.raises(ValueError) as info:
        resolve_device("cuda")
    message = str(info.value)
    assert uninstall in message and message.index(uninstall) < message.index("第 3 步")

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    # 安装步骤第 3 步：在 CUDA 版的安装命令之前提示先卸载
    step3 = readme[readme.index("# 3. "):readme.index("# 4. ")]
    assert uninstall in step3 and step3.index(uninstall) < step3.index("pip install torch torchvision")
    # 常见问题 “训练没有用上显卡？”（报错信息让用户去 README 找的解决办法）
    faq = readme[readme.index("**训练没有用上显卡？**"):]
    faq = faq[:faq.index("\n**", 1)]
    assert uninstall in faq and faq.index(uninstall) < faq.index("第 3 步")


def test_faq_gpu_check_distinguishes_cpu_build_from_missing_driver(monkeypatch, capsys):
    """常见问题 “训练没有用上显卡？” 要和 resolve_device 一样按 torch.version.cuda 区分两种 is_available() == False：
    只有 CPU 版（version.cuda 为 None）才卸载后按第 3 步重装；CUDA 版但检测不到显卡/驱动时卸载重装没用，
    只会白白重新下载几 GB 的 CUDA 版，应该去查驱动。这里实际执行 FAQ 里的检查命令，按它的输出找到 FAQ 对应的那一条。"""
    uninstall = "pip uninstall -y torch torchvision"
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    faq = readme[readme.index("**训练没有用上显卡？**"):]
    faq = faq[:faq.index("\n**", 1)]
    command = re.search(r'python -c "([^"]+)"', faq)
    assert command is not None
    items = faq.split("\n- ")[1:]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    found = {}
    for cuda_version in (None, "12.8"):
        monkeypatch.setattr(torch.version, "cuda", cuda_version)
        exec(command.group(1), {})
        printed = capsys.readouterr().out.strip()
        matching = [item for item in items if f"`{printed}`" in item]
        assert len(matching) == 1, f"FAQ 没有（或不止一条）说明检查命令输出 {printed!r} 的情况"
        with pytest.raises(ValueError) as info:
            resolve_device("cuda")
        found[cuda_version] = (matching[0], str(info.value))

    # CPU 版：先卸载，再按第 3 步装 CUDA 版（与报错信息一致）
    cpu_item, cpu_message = found[None]
    assert uninstall in cpu_message
    assert uninstall in cpu_item and cpu_item.index(uninstall) < cpu_item.index("第 3 步")
    # CUDA 版但用不了显卡：先查驱动 / 独显，不让用户先卸载重装（与报错信息一致）
    cuda_item, cuda_message = found["12.8"]
    assert "驱动" in cuda_message and "uninstall" not in cuda_message
    assert "驱动" in cuda_item and "nvidia-smi" in cuda_item and "不要" in cuda_item
    if "uninstall" in cuda_item:
        assert cuda_item.index("驱动") < cuda_item.index("uninstall")


def test_resolve_device_mps_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS"):
        resolve_device("mps")


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(123)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b
