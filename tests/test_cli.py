"""dfwalk.cli 的端到端测试：在 tmp_path 中合成数据集，直接调用 main([...]) 驱动各个子命令。

全部离线、只用 CPU：
- 训练用 resnet18 + encoder_weights=null、64x96 小图、1 个 epoch（整个模块只训练一次，eval/infer/export 共用）；
- autolabel 不下载真实模型：把 AutoLabeler.from_pretrained 替换为返回“假模型”的标注器；
- 测试视频用 cv2.VideoWriter 现场合成（mp4v/.mp4，失败时回退 MJPG/.avi）；
- 测试期间禁止任何网络连接。
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml

from dfwalk import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD, NON_WALKABLE, WALKABLE, __version__, cli
from dfwalk.cli import DataPaths, format_metrics_table, main
from dfwalk.config import Config
from dfwalk.utils import read_image_rgb, read_mask, write_image_rgb, write_mask

H, W = 64, 96
GROUPS = ("dam", "valley", "base")
FRAMES_PER_GROUP = 4
SUBCOMMANDS = (
    "extract-frames", "hud-preview", "autolabel", "to-labelme", "from-labelme", "split",
    "check-data", "train", "eval", "infer", "export-onnx",
)


# ---------------------------------------------------------------- 公共夹具与工具


def _guard(*args, **kwargs):
    raise RuntimeError("测试中禁止联网")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _guard)
    monkeypatch.setattr(socket, "create_connection", _guard)


@pytest.fixture(autouse=True)
def _threads():
    """限制 torch 线程数，避免和其他进程抢核时小模型变慢几十倍。"""
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, min(2, old)))
    yield
    torch.set_num_threads(old)


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """在空的临时目录中运行：不会误用仓库里的 configs/default.yaml，默认输出（hud_preview.jpg 等）也落在这里。"""
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.delenv(cli.DEBUG_ENV, raising=False)
    return work


def run(capsys, *argv) -> tuple[int, str, str]:
    """调用 main，返回 (退出码, stdout, stderr)。"""
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def make_image(seed: int, h: int = H, w: int = W) -> np.ndarray:
    """上半部分灰色（不可走），下半部分绿色地面（可走），带少量随机纹理。"""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.uint8)
    img[: h // 2] = (120, 120, 130)
    img[h // 2:] = (60, 150, 60)
    noise = rng.integers(-15, 16, size=(h, w, 3))
    return np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)


def make_mask(h: int = H, w: int = W) -> np.ndarray:
    mask = np.full((h, w), NON_WALKABLE, np.uint8)
    mask[h // 2:] = WALKABLE
    mask[h // 2 - 2: h // 2 + 2] = IGNORE_INDEX  # 边界处一条忽略带
    return mask


def make_dataset(root: Path, groups=GROUPS, n: int = FRAMES_PER_GROUP, masks_subdir: str = "masks",
                 with_masks: bool = True) -> list[str]:
    """写 root/images/{group}_{i:06d}.jpg 与 root/{masks_subdir}/{stem}.png，返回 stem 列表。"""
    stems = []
    for gi, group in enumerate(groups):
        for i in range(n):
            stem = f"{group}_{i * 10:06d}"
            write_image_rgb(root / "images" / f"{stem}.jpg", make_image(gi * 100 + i))
            if with_masks:
                write_mask(root / masks_subdir / f"{stem}.png", make_mask())
            stems.append(stem)
    return sorted(stems)


def make_video(path_no_ext: Path, frames: list[np.ndarray], fps: float = 10.0) -> Path:
    """合成测试视频（RGB 帧），优先 mp4v/.mp4，打不开时回退 MJPG/.avi。"""
    h, w = frames[0].shape[:2]
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    for ext, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        path = path_no_ext.with_name(path_no_ext.name + ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if not writer.isOpened():
            writer.release()
            continue
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        if path.is_file() and path.stat().st_size > 0:
            return path
    pytest.skip("当前 OpenCV 没有可用的视频编码器")


def noise_frames(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8) for _ in range(n)]


def count_video_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        n = 0
        while cap.grab():
            n += 1
        return n
    finally:
        cap.release()


def write_config(path: Path, root: Path, **sections) -> Path:
    data = {"data": {"root": str(root), "image_size": [H, W]},
            "train": {"encoder": "resnet18", "encoder_weights": None, "num_workers": 0}}
    for name, values in sections.items():
        data.setdefault(name, {}).update(values)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 帮助、版本与参数解析


def test_help_lists_all_subcommands(capsys):
    code, out, _ = run(capsys, "--help")
    assert code == 0
    assert "用法" in out and "离线" in out
    for name in SUBCOMMANDS:
        assert name in out


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_subcommand_help(capsys, name):
    code, out, _ = run(capsys, name, "-h")
    assert code == 0
    assert "用法" in out
    assert "--config" in out and "--set" in out


def test_version(capsys):
    code, out, _ = run(capsys, "--version")
    assert code == 0
    assert __version__ in out


def test_no_subcommand_prints_help(capsys):
    code, out, _ = run(capsys)
    assert code == 2
    assert "子命令" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["nosuch-command"],
        ["train", "--epochs", "abc"],
        ["train", "--bogus"],
        ["infer", "x.jpg"],  # 缺少 --ckpt
        ["eval"],
        ["extract-frames"],  # 缺少 INPUT
    ],
)
def test_argument_errors_exit_2(capsys, argv):
    code, _, err = run(capsys, *argv)
    assert code == 2
    assert "错误" in err


def test_python_m_entry_point():
    proc = subprocess.run([sys.executable, "-m", "dfwalk", "--version"], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert proc.returncode == 0
    assert __version__ in proc.stdout


def test_data_paths_match_dataset_constants():
    from dfwalk import dataset

    assert cli.IMAGES_SUBDIR == dataset.IMAGES_SUBDIR
    assert cli.MASKS_SUBDIR == dataset.MASKS_SUBDIR
    assert cli.SPLITS_SUBDIR == dataset.SPLITS_SUBDIR
    cfg = Config()
    cfg.data.root = "some/root"
    p = DataPaths.from_config(cfg)
    assert p.root == Path("some/root")
    assert (p.images, p.pseudo_masks, p.labelme, p.masks, p.splits) == tuple(
        Path("some/root") / s for s in ("images", "pseudo_masks", "labelme", "masks", "splits")
    )


# ---------------------------------------------------------------- 配置加载


@pytest.fixture
def hud_image(tmp_path) -> Path:
    path = tmp_path / "shot.png"
    write_image_rgb(path, make_image(0))
    return path


def test_default_config_in_cwd_is_used_with_notice(capsys, _isolated_cwd, hud_image):
    cfg_dir = _isolated_cwd / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "default.yaml").write_text("hud:\n  rects:\n    - [0.0, 0.0, 0.25, 0.25]\n", encoding="utf-8")
    code, out, err = run(capsys, "hud-preview", hud_image)
    assert code == 0
    assert "configs/default.yaml" in err
    assert "1 个 HUD 矩形" in out
    assert (_isolated_cwd / "hud_preview.jpg").is_file()


def test_without_default_config_uses_dataclass_defaults(capsys, hud_image):
    code, out, err = run(capsys, "hud-preview", hud_image)
    assert code == 0
    assert "configs/default.yaml" not in err
    assert "0 个 HUD 矩形" in out
    assert "hud.rects 为空" in err  # 空 rects 警告


def test_explicit_config_overrides_default_and_skips_notice(capsys, _isolated_cwd, tmp_path, hud_image):
    (_isolated_cwd / "configs").mkdir()
    (_isolated_cwd / "configs" / "default.yaml").write_text("hud:\n  rects: []\n", encoding="utf-8")
    cfg_path = tmp_path / "my.yaml"
    cfg_path.write_text("hud:\n  rects: [[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]\n", encoding="utf-8")
    code, out, err = run(capsys, "hud-preview", hud_image, "-c", cfg_path)
    assert code == 0
    assert "configs/default.yaml" not in err
    assert "2 个 HUD 矩形" in out


def test_common_options_before_and_after_subcommand(capsys, tmp_path, hud_image):
    cfg_path = tmp_path / "my.yaml"
    cfg_path.write_text("hud:\n  rects: [[0, 0, 0.5, 0.5]]\n", encoding="utf-8")
    # -c 写在子命令之前，--set 前后都有：按出现顺序生效（后面的覆盖前面的）
    code, out, _ = run(capsys, "-c", cfg_path, "--set", "hud.rects=[]", "hud-preview", hud_image,
                       "--set", "hud.rects=[[0, 0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3], [0.5, 0.5, 0.6, 0.6]]")
    assert code == 0
    assert "3 个 HUD 矩形" in out
    code, out, _ = run(capsys, "-c", cfg_path, "hud-preview", hud_image)
    assert code == 0
    assert "1 个 HUD 矩形" in out


def test_config_given_twice_conflict(capsys, tmp_path, hud_image):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("{}\n", encoding="utf-8")
    b.write_text("{}\n", encoding="utf-8")
    code, _, err = run(capsys, "-c", a, "hud-preview", hud_image, "-c", b)
    assert code == 1
    assert "错误" in err
    code, _, _ = run(capsys, "-c", a, "hud-preview", hud_image, "-c", a)  # 两处相同则允许
    assert code == 0


@pytest.mark.parametrize(
    "extra, expected",
    [
        (["-c", "nope.yaml"], "配置文件不存在"),
        (["--set", "nosuch.key=1"], "未知配置段"),
        (["--set", "train.nosuch=1"], "未知配置项: train.nosuch"),
        (["--set", "train.epochs"], "section.key=value"),
        (["--set", "data.image_size=[1,"], "YAML"),
        (["--set", "data.image_size=5"], "data.image_size"),
        (["--set", "train.lr=abc"], "train.lr"),
        (["--set", "infer.color=[0,255]"], "infer.color"),
        (["--set", "data.image_size=[100,100]"], "image_size"),
        (["--set", "hud.rects=[[0,0,2,2]]"], "hud.rects"),
    ],
)
def test_config_errors_exit_1(capsys, hud_image, extra, expected):
    code, out, err = run(capsys, "hud-preview", hud_image, *extra)
    assert code == 1
    assert err.startswith("错误: ") or "\n错误: " in err
    assert expected in err
    assert "Traceback" not in err
    if "未知配置" in expected:
        assert "'" not in err.split("错误: ", 1)[1].splitlines()[0]  # KeyError 不带引号


def test_bad_config_file_yaml(capsys, tmp_path, hud_image):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data: [unclosed\n", encoding="utf-8")
    code, _, err = run(capsys, "hud-preview", hud_image, "-c", bad)
    assert code == 1
    assert "YAML" in err


def test_debug_env_reraises(capsys, monkeypatch, hud_image):
    monkeypatch.setenv(cli.DEBUG_ENV, "1")
    with pytest.raises(KeyError):
        main(["hud-preview", str(hud_image), "--set", "nosuch.key=1"])
    monkeypatch.setenv(cli.DEBUG_ENV, "0")  # "0" 视为关闭
    assert main(["hud-preview", str(hud_image), "--set", "nosuch.key=1"]) == 1


def test_keyboard_interrupt_exit_130(capsys, monkeypatch, tmp_path):
    import dfwalk.frames

    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(dfwalk.frames, "extract_frames", boom)
    video = make_video(tmp_path / "clip", noise_frames(3))
    code, _, err = run(capsys, "extract-frames", video, "--set", f"data.root={tmp_path / 'data'}")
    assert code == 130
    assert "中断" in err


def test_runtime_error_prints_debug_hint(capsys, monkeypatch, tmp_path):
    import dfwalk.frames

    def boom(*args, **kwargs):
        raise RuntimeError("解码器崩溃了")

    monkeypatch.setattr(dfwalk.frames, "extract_frames", boom)
    video = make_video(tmp_path / "clip", noise_frames(3))
    code, _, err = run(capsys, "extract-frames", video, "--set", f"data.root={tmp_path / 'data'}")
    assert code == 1
    assert "错误: 解码器崩溃了" in err
    assert cli.DEBUG_ENV in err


def test_unhandled_exception_types_propagate(monkeypatch, tmp_path):
    """非预期的异常类型（例如程序 bug 导致的 AttributeError）不应被吞掉。"""
    import dfwalk.frames

    def boom(*args, **kwargs):
        raise AttributeError("bug")

    monkeypatch.setattr(dfwalk.frames, "extract_frames", boom)
    video = make_video(tmp_path / "clip", noise_frames(3))
    with pytest.raises(AttributeError):
        main(["extract-frames", str(video), "--set", f"data.root={tmp_path / 'data'}"])


def test_error_message_formatting(tmp_path):
    assert cli._error_message(KeyError("未知配置项: a.b")) == "未知配置项: a.b"
    assert cli._error_message(FileNotFoundError(tmp_path / "x.jpg")) == f"文件或目录不存在: {tmp_path / 'x.jpg'}"
    try:
        open(tmp_path / "missing.txt", encoding="utf-8")
    except FileNotFoundError as e:
        assert cli._error_message(e) == f"文件或目录不存在: {tmp_path / 'missing.txt'}"
    assert cli._error_message(FileNotFoundError("图片目录不存在: x")) == "图片目录不存在: x"
    assert cli._error_message(ValueError()) == "ValueError"


def test_user_warnings_are_printed_concisely(capsys, tmp_path):
    """各模块 warnings.warn 的 UserWarning 显示为 “警告: …”。这里用单视频分组时 make_splits 的提醒验证。"""
    root = tmp_path / "data"
    make_dataset(root, groups=("only",), n=5)
    code, _, err = run(capsys, "split", "--set", f"data.root={root}")
    assert code == 0
    assert "警告: " in err and "同一个视频分组" in err
    assert "UserWarning" not in err


# ---------------------------------------------------------------- extract-frames


def test_extract_frames_video_file(capsys, tmp_path):
    root = tmp_path / "data"
    video = make_video(tmp_path / "rec" / "clip", noise_frames(20), fps=10.0)
    code, out, _ = run(capsys, "extract-frames", video, "--fps", "2", "--set", f"data.root={root}")
    assert code == 0
    names = sorted(p.name for p in (root / "images").iterdir())
    assert names == [f"clip_{i:06d}.jpg" for i in (0, 5, 10, 15)]
    assert "写出 4 帧" in out
    assert "共写出 4 帧（1 个视频）" in out


def test_extract_frames_directory_and_flags(capsys, tmp_path):
    rec = tmp_path / "rec"
    make_video(rec / "零号大坝 第1局", noise_frames(10, seed=1), fps=10.0)
    make_video(rec / "valley", [noise_frames(1, seed=2)[0]] * 10, fps=10.0)  # 全部相同的帧
    (rec / "notes.txt").write_text("not a video", encoding="utf-8")
    out_dir = tmp_path / "frames_out"
    code, out, _ = run(capsys, "extract-frames", rec, "-o", out_dir, "--fps", "0", "--max-side", "48",
                       "--dedup", "6", "--set", f"data.root={tmp_path / 'unused'}")
    assert code == 0
    files = sorted(out_dir.iterdir())
    dam = [p for p in files if p.name.startswith("零号大坝_第1局_")]
    valley = [p for p in files if p.name.startswith("valley_")]
    assert len(dam) == 10  # fps<=0 每帧都取，随机噪声帧不会被去重
    assert len(valley) == 1  # 相同帧被去重
    assert read_image_rgb(dam[0]).shape == (32, 48, 3)  # --max-side 生效
    assert "共写出 11 帧（2 个视频）" in out
    assert not (tmp_path / "unused").exists()

    # --dedup 0 关闭去重
    out_dir2 = tmp_path / "frames_out2"
    vid = next(p for p in rec.iterdir() if p.stem == "valley")
    code, _, _ = run(capsys, "extract-frames", vid, "-o", out_dir2, "--fps", "0", "--dedup", "0")
    assert code == 0
    assert len(list(out_dir2.iterdir())) == 10


def test_extract_frames_rerun_warns_existing(capsys, tmp_path):
    root = tmp_path / "data"
    video = make_video(tmp_path / "clip", noise_frames(6), fps=10.0)
    assert run(capsys, "extract-frames", video, "--fps", "0", "--set", f"data.root={root}")[0] == 0
    code, _, err = run(capsys, "extract-frames", video, "--fps", "0", "--set", f"data.root={root}")
    assert code == 0
    assert "已有 6 张 clip_*.jpg" in err


def test_extract_frames_prefix_collision(capsys, tmp_path):
    rec = tmp_path / "rec"
    make_video(rec / "a b", noise_frames(3), fps=10.0)
    make_video(rec / "a_b", noise_frames(3, seed=1), fps=10.0)
    root = tmp_path / "data"
    code, _, err = run(capsys, "extract-frames", rec, "--set", f"data.root={root}")
    assert code == 1
    assert "a_b" in err and "重命名" in err
    assert not (root / "images").exists()  # 冲突在写任何文件之前发现


def test_extract_frames_same_file_twice_processed_once(capsys, tmp_path):
    root = tmp_path / "data"
    video = make_video(tmp_path / "clip", noise_frames(4), fps=10.0)
    code, out, _ = run(capsys, "extract-frames", video, tmp_path, "--fps", "0", "--set", f"data.root={root}")
    assert code == 0
    assert "（1 个视频）" in out


@pytest.mark.parametrize("kind", ["missing", "empty_dir", "not_video"])
def test_extract_frames_errors(capsys, tmp_path, kind):
    root = tmp_path / "data"
    if kind == "missing":
        target, expected = tmp_path / "nope.mp4", "输入不存在"
    elif kind == "empty_dir":
        target = tmp_path / "empty"
        target.mkdir()
        expected = "没有找到视频"
    else:
        target = tmp_path / "fake.mp4"
        target.write_bytes(b"this is not a video")
        expected = "无法打开视频"
    code, _, err = run(capsys, "extract-frames", target, "--set", f"data.root={root}")
    assert code == 1
    assert expected in err


# ---------------------------------------------------------------- hud-preview


def test_hud_preview_draws_rects(capsys, tmp_path, hud_image):
    out_path = tmp_path / "preview.png"
    code, out, _ = run(capsys, "hud-preview", hud_image, "-o", out_path,
                       "--set", "hud.rects=[[0.0, 0.0, 0.25, 0.5]]")
    assert code == 0
    src = read_image_rgb(hud_image)
    prev = read_image_rgb(out_path)
    assert prev.shape == src.shape
    c1, r1 = int(W * 0.25), int(H * 0.5)
    assert np.abs(prev[:r1, :c1].astype(int) - src[:r1, :c1]).mean() > 10  # 矩形内被着色
    np.testing.assert_array_equal(prev[:, c1 + 1:], src[:, c1 + 1:])  # 矩形外不变
    np.testing.assert_array_equal(prev[r1 + 1:], src[r1 + 1:])
    assert f"像素 x 0~{c1}, y 0~{r1}" in out
    assert str(out_path) in out


def test_hud_preview_errors(capsys, tmp_path, hud_image):
    before = hud_image.read_bytes()
    code, _, err = run(capsys, "hud-preview", hud_image, "-o", hud_image)
    assert code == 1 and "覆盖原图" in err
    assert hud_image.read_bytes() == before
    code, _, err = run(capsys, "hud-preview", hud_image, "-o", tmp_path / "preview.txt")
    assert code == 1 and "后缀" in err
    code, _, err = run(capsys, "hud-preview", tmp_path / "nope.jpg")
    assert code == 1 and "不存在" in err
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    code, _, err = run(capsys, "hud-preview", bad)
    assert code == 1 and "无法解码" in err


# ---------------------------------------------------------------- autolabel（假模型）


class FakeSegModel:
    """假 SegFormer：1/4 分辨率 logits，下半部分 'floor'（可走），上半部分 'sky'。"""

    def __call__(self, pixel_values):
        _, _, h, w = pixel_values.shape
        hh, ww = max(1, h // 4), max(1, w // 4)
        logits = torch.zeros(1, 2, hh, ww)
        logits[:, 0, hh // 2:, :] = 10.0
        logits[:, 1, : hh // 2, :] = 10.0
        return SimpleNamespace(logits=logits)


@pytest.fixture
def fake_pretrained(monkeypatch):
    from dfwalk.autolabel import AutoLabeler

    calls = []

    def fake(cls, name, walkable_labels, short_side=512, hi=0.7, lo=0.3, device="auto"):
        calls.append(dict(name=name, walkable_labels=list(walkable_labels), short_side=short_side,
                          hi=hi, lo=lo, device=device))
        return cls(FakeSegModel(), {0: "floor", 1: "sky"}, ["floor"], IMAGENET_MEAN, IMAGENET_STD,
                   short_side=short_side, hi=hi, lo=lo, device="cpu")

    monkeypatch.setattr(AutoLabeler, "from_pretrained", classmethod(fake))
    return calls


def test_autolabel_with_fake_model(capsys, tmp_path, fake_pretrained):
    root = tmp_path / "data"
    stems = make_dataset(root, with_masks=False)
    args = ["autolabel", "--model", "my/fake-model", "--device", "cpu",
            "--set", f"data.root={root}", "--set", "autolabel.short_side=64",
            "--set", "autolabel.hi_threshold=0.8", "--set", "hud.rects=[[0.75, 0.0, 1.0, 0.25]]",
            "--set", "autolabel.walkable_labels=[floor, road]"]
    code, out, _ = run(capsys, *args)
    assert code == 0
    call = fake_pretrained[0]
    assert call["name"] == "my/fake-model"
    assert call["device"] == "cpu"
    assert call["short_side"] == 64
    assert call["hi"] == 0.8 and call["lo"] == 0.3
    # 配置里的可走类别要原样传给标注器（README 让用户靠增删它来修正伪标签），而不是内置默认值
    assert call["walkable_labels"] == ["floor", "road"]

    out_dir = root / "pseudo_masks"
    assert sorted(p.stem for p in out_dir.glob("*.png")) == stems
    mask = read_mask(out_dir / f"{stems[0]}.png")
    assert mask.shape == (H, W)
    assert (mask[: H // 4, : W // 2] == NON_WALKABLE).all()
    assert (mask[H * 3 // 4:] == WALKABLE).all()
    assert (mask[: H // 4, W * 3 // 4:] == IGNORE_INDEX).all()  # HUD 矩形
    assert f"生成伪标签 {len(stems)} 张" in out

    # 再跑一次：全部跳过；--overwrite 重新生成
    code, out, _ = run(capsys, *args)
    assert code == 0 and "生成伪标签 0 张" in out and "--overwrite" in out
    code, out, _ = run(capsys, *args, "--overwrite")
    assert code == 0 and f"生成伪标签 {len(stems)} 张" in out


def test_autolabel_custom_dirs(capsys, tmp_path, fake_pretrained):
    src = tmp_path / "shots"
    write_image_rgb(src / "a.png", make_image(1))
    out_dir = tmp_path / "out"
    code, _, _ = run(capsys, "autolabel", "-i", src, "-o", out_dir, "--set", "autolabel.short_side=64")
    assert code == 0
    assert (out_dir / "a.png").is_file()
    assert fake_pretrained[0]["name"] == Config().autolabel.model
    assert fake_pretrained[0]["device"] == "auto"
    assert fake_pretrained[0]["walkable_labels"] == Config().autolabel.walkable_labels


def test_autolabel_errors(capsys, tmp_path, monkeypatch, fake_pretrained):
    from dfwalk.autolabel import AutoLabeler

    root = tmp_path / "data"
    code, _, err = run(capsys, "autolabel", "--set", f"data.root={root}")
    assert code == 1 and "图片目录不存在" in err
    (root / "images").mkdir(parents=True)
    code, _, err = run(capsys, "autolabel", "--set", f"data.root={root}")
    assert code == 1 and "没有图片" in err
    assert fake_pretrained == []  # 没有图片时不加载模型

    write_image_rgb(root / "images" / "a.jpg", make_image(0))

    def offline(cls, *a, **k):
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr(AutoLabeler, "from_pretrained", classmethod(offline))
    code, _, err = run(capsys, "autolabel", "--set", f"data.root={root}")
    assert code == 1 and "HF_ENDPOINT=https://hf-mirror.com" in err

    def no_transformers(cls, *a, **k):
        raise ImportError('自动标注需要 transformers，请先安装：pip install -e ".[autolabel]"')

    monkeypatch.setattr(AutoLabeler, "from_pretrained", classmethod(no_transformers))
    code, _, err = run(capsys, "autolabel", "--set", f"data.root={root}")
    assert code == 1 and "transformers" in err


# ---------------------------------------------------------------- to-labelme / from-labelme


def complex_mask() -> np.ndarray:
    m = make_mask()
    m[40:50, 20:30] = NON_WALKABLE  # 可走区域中的不可走孔洞
    m[43:46, 23:26] = WALKABLE  # 孔洞中的可走小岛
    m[5:15, 70:90] = IGNORE_INDEX
    return m


def test_labelme_roundtrip(capsys, tmp_path):
    root = tmp_path / "data"
    stems = make_dataset(root, masks_subdir="pseudo_masks")
    for s in stems:
        write_mask(root / "pseudo_masks" / f"{s}.png", complex_mask())

    code, out, err = run(capsys, "to-labelme", "--epsilon", "0", "--min-area", "0", "--set", f"data.root={root}")
    assert code == 0
    jsons = sorted((root / "labelme").glob("*.json"))
    assert [p.stem for p in jsons] == stems
    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert data["imagePath"] == f"../images/{stems[0]}.jpg"
    assert (data["imageHeight"], data["imageWidth"]) == (H, W)
    assert f"写出 {len(stems)} 个 LabelMe JSON" in out
    assert "labelme " in err and "--labels walkable,non_walkable,ignore" in err

    code, out, _ = run(capsys, "from-labelme", "--set", f"data.root={root}")
    assert code == 0
    assert f"转换 {len(stems)} 个 LabelMe 标注" in out
    for s in stems:
        np.testing.assert_array_equal(read_mask(root / "masks" / f"{s}.png"), complex_mask())

    # HUD 矩形在 from-labelme 时置为忽略
    code, _, _ = run(capsys, "from-labelme", "-o", tmp_path / "hud_masks", "--set", f"data.root={root}",
                     "--set", "hud.rects=[[0, 0, 0.5, 0.5]]")
    assert code == 0
    m = read_mask(tmp_path / "hud_masks" / f"{stems[0]}.png")
    assert (m[: H // 2, : W // 2] == IGNORE_INDEX).all()


def test_to_labelme_never_overwrites_without_flag(capsys, tmp_path):
    root = tmp_path / "data"
    stems = make_dataset(root, masks_subdir="pseudo_masks")
    assert run(capsys, "to-labelme", "--set", f"data.root={root}")[0] == 0
    target = root / "labelme" / f"{stems[0]}.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data["shapes"] = []  # 模拟人工修正
    target.write_text(json.dumps(data), encoding="utf-8")

    code, out, _ = run(capsys, "to-labelme", "--set", f"data.root={root}")
    assert code == 0
    assert "写出 0 个" in out and "--overwrite" in out
    assert json.loads(target.read_text(encoding="utf-8"))["shapes"] == []

    code, out, _ = run(capsys, "to-labelme", "--overwrite", "--set", f"data.root={root}")
    assert code == 0 and f"写出 {len(stems)} 个" in out
    assert json.loads(target.read_text(encoding="utf-8"))["shapes"] != []


def test_to_labelme_hint_command_is_valid_for_current_labelme(capsys, tmp_path):
    """下一步提示里的 labelme 命令：不能带 LabelMe 7.x 已删除的 --nodata / --autosave（会直接报错退出），
    data.root 含空格时路径要加引号，复制到终端后参数不会被拆开。"""
    import os
    import shlex

    root = tmp_path / "my data"
    make_dataset(root, masks_subdir="pseudo_masks")
    code, _, err = run(capsys, "to-labelme", "--set", f"data.root={root}")
    assert code == 0
    line = next(s.strip() for s in err.splitlines() if s.strip().startswith("labelme "))
    assert "--nodata" not in line and "--autosave" not in line
    if os.name != "nt":
        assert shlex.split(line) == ["labelme", (root / "images").as_posix(), "--output",
                                     (root / "labelme").as_posix(), "--labels", "walkable,non_walkable,ignore"]


def test_quote_arg_for_both_shells():
    import shlex

    assert cli._quote_arg("data/images", windows=True) == "data/images"
    assert cli._quote_arg("D:/我的 数据/images", windows=True) == '"D:/我的 数据/images"'
    assert cli._quote_arg("data/images", windows=False) == "data/images"
    assert shlex.split(cli._quote_arg("/tmp/my data/images", windows=False)) == ["/tmp/my data/images"]


def test_readme_labelme_command_has_no_removed_flags():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    commands = [line for line in readme.splitlines() if line.startswith("labelme ")]
    assert commands
    for cmd in commands:
        assert "--nodata" not in cmd and "--autosave" not in cmd, cmd
    assert 'pipx install "labelme>=6"' in readme


def test_to_labelme_passes_flags_and_default_paths(capsys, tmp_path, monkeypatch):
    import dfwalk.labelme_io

    calls = []
    monkeypatch.setattr(dfwalk.labelme_io, "export_masks_to_labelme",
                        lambda *a, **k: calls.append((a, k)) or [])
    root = tmp_path / "data"
    assert run(capsys, "to-labelme", "--set", f"data.root={root}")[0] == 0
    args, kwargs = calls[-1]
    assert [Path(a) for a in args] == [root / "pseudo_masks", root / "images", root / "labelme"]
    assert kwargs == {"epsilon": 1.0, "min_area": 16.0, "overwrite": False}

    assert run(capsys, "to-labelme", "--masks", tmp_path / "m", "--images", tmp_path / "i", "-o", tmp_path / "o",
               "--epsilon", "2.5", "--min-area", "4", "--overwrite")[0] == 0
    args, kwargs = calls[-1]
    assert [Path(a) for a in args] == [tmp_path / "m", tmp_path / "i", tmp_path / "o"]
    assert kwargs == {"epsilon": 2.5, "min_area": 4.0, "overwrite": True}


def test_labelme_errors(capsys, tmp_path):
    root = tmp_path / "data"
    code, _, err = run(capsys, "to-labelme", "--set", f"data.root={root}")
    assert code == 1 and "掩码目录不存在" in err
    code, _, err = run(capsys, "from-labelme", "--set", f"data.root={root}")
    assert code == 1 and "目录不存在" in err

    (root / "labelme").mkdir(parents=True)
    code, out, err = run(capsys, "from-labelme", "--set", f"data.root={root}")
    assert code == 0 and "没有 LabelMe JSON" in err and "转换 0 个" in out

    bad = {"imageHeight": H, "imageWidth": W,
           "shapes": [{"label": "水面", "points": [[0, 0], [10, 0], [10, 10]], "shape_type": "polygon"}]}
    (root / "labelme" / "bad_000001.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    code, _, err = run(capsys, "from-labelme", "--set", f"data.root={root}")
    assert code == 1
    assert "bad_000001.json" in err and "水面" in err


# ---------------------------------------------------------------- split


def read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_split_and_refuse_overwrite(capsys, tmp_path):
    from dfwalk.dataset import group_of

    root = tmp_path / "data"
    stems = make_dataset(root)
    code, out, _ = run(capsys, "split", "--set", f"data.root={root}")
    assert code == 0
    train = read_lines(root / "splits" / "train.txt")
    val = read_lines(root / "splits" / "val.txt")
    assert sorted(train + val) == stems
    assert {group_of(s) for s in train}.isdisjoint({group_of(s) for s in val})
    assert f"划分 {len(stems)} 帧" in out
    assert f"train {len(train):>6} 帧" in out and f"val   {len(val):>6} 帧" in out
    assert "段视频" in out

    before = (root / "splits" / "train.txt").read_bytes(), (root / "splits" / "val.txt").read_bytes()
    code, _, err = run(capsys, "split", "--set", f"data.root={root}", "--val-ratio", "0.5")
    assert code == 1
    assert "--force" in err
    assert ((root / "splits" / "train.txt").read_bytes(), (root / "splits" / "val.txt").read_bytes()) == before

    code, out, _ = run(capsys, "split", "--force", "--val-ratio", "0.6", "--seed", "3", "--set", f"data.root={root}")
    assert code == 0
    assert "val_ratio=0.6, seed=3" in out
    val2 = read_lines(root / "splits" / "val.txt")
    assert len(val2) == 2 * FRAMES_PER_GROUP  # 3 组各 4 帧，目标 7.2 帧 → 最接近的是 2 组


def test_split_refuses_when_only_one_file_exists(capsys, tmp_path):
    root = tmp_path / "data"
    make_dataset(root)
    (root / "splits").mkdir()
    (root / "splits" / "val.txt").write_text("x\n", encoding="utf-8")
    code, _, err = run(capsys, "split", "--set", f"data.root={root}")
    assert code == 1 and "val.txt" in err


@pytest.mark.parametrize(
    "setup, args, expected",
    [
        ("no_masks", [], "没有同时具有图片"),
        ("none", [], "图片目录不存在"),
        ("ok", ["--val-ratio", "1.5"], "val_ratio"),
        ("one_frame", [], "只有 1 个样本"),
    ],
)
def test_split_errors(capsys, tmp_path, setup, args, expected):
    root = tmp_path / "data"
    if setup == "no_masks":
        make_dataset(root, with_masks=False)
    elif setup == "ok":
        make_dataset(root)
    elif setup == "one_frame":
        make_dataset(root, groups=("solo",), n=1)
    code, _, err = run(capsys, "split", "--set", f"data.root={root}", *args)
    assert code == 1
    assert expected in err
    assert not (root / "splits" / "train.txt").exists()


# ---------------------------------------------------------------- check-data


def test_check_data_clean(capsys, tmp_path):
    root = tmp_path / "data"
    stems = make_dataset(root)
    write_mask(root / "pseudo_masks" / f"{stems[0]}.png", make_mask())
    (root / "labelme").mkdir()
    (root / "labelme" / f"{stems[0]}.json").write_text("{}", encoding="utf-8")
    write_image_rgb(root / "images" / "extra_000001.jpg", make_image(9))  # 还没标注的图片：只提示，不是错误

    code, out, _ = run(capsys, "check-data", "--set", f"data.root={root}")
    assert code == 0, out
    assert f"{len(stems) + 1} 张图片" in out
    assert f"{len(stems)} 个掩码" in out
    assert "1 个伪标签" in out and "1 个 LabelMe JSON" in out
    assert "有图片但还没有掩码（未标注）1 张: extra_000001" in out
    assert "尚未划分" in out
    assert "全部" in out and f"{len(stems)} 帧" in out
    assert "未发现错误" in out
    # 标签占比：make_mask 中 4/64 行为忽略，可走 30/64，不可走 30/64
    assert "可走  46.9%" in out and "忽略   6.2%" in out


def test_check_data_with_splits_and_vis(capsys, tmp_path):
    root = tmp_path / "data"
    stems = make_dataset(root)
    assert run(capsys, "split", "--set", f"data.root={root}")[0] == 0
    vis = tmp_path / "vis"
    code, out, _ = run(capsys, "check-data", "--vis-dir", vis, "--vis-limit", "5", "--set", f"data.root={root}")
    assert code == 0, out
    assert "  train " in out and "  val " in out
    files = sorted(vis.glob("*_label.jpg"))
    assert len(files) == 5
    # 均匀抽样：第一个和最后一个样本都在其中
    assert files[0].name == f"{stems[0]}_label.jpg"
    assert f"{stems[-1]}_label.jpg" in {p.name for p in files}
    vis_img = read_image_rgb(files[0])
    assert vis_img.shape == (H, W, 3)
    # 可走区域偏绿，忽略带被压暗
    src = read_image_rgb(root / "images" / f"{stems[0]}.jpg")
    assert vis_img[H * 3 // 4:, :, 1].mean() > src[H * 3 // 4:, :, 1].mean()
    assert vis_img[H // 2 - 1].mean() < src[H // 2 - 1].mean()

    # --vis-limit 0 表示全部
    code, _, _ = run(capsys, "check-data", "--vis-dir", tmp_path / "vis_all", "--vis-limit", "0",
                     "--set", f"data.root={root}")
    assert code == 0
    assert len(list((tmp_path / "vis_all").glob("*.jpg"))) == len(stems)

    # 新增样本不在划分中 → 提示
    write_image_rgb(root / "images" / "new_000000.jpg", make_image(5))
    write_mask(root / "masks" / "new_000000.png", make_mask())
    code, out, _ = run(capsys, "check-data", "--set", f"data.root={root}")
    assert code == 0
    assert "不在任何划分中" in out and "new_000000" in out


@pytest.mark.parametrize("problem", ["invalid_value", "mask_without_image", "size_mismatch", "rgb_mask",
                                     "split_missing", "split_overlap", "duplicate_stem"])
def test_check_data_hard_errors(capsys, tmp_path, problem):
    root = tmp_path / "data"
    stems = make_dataset(root)
    expected = {
        "invalid_value": "非法值",
        "mask_without_image": "有掩码但没有对应图片",
        "size_mismatch": "尺寸不一致",
        "rgb_mask": "无法读取",
        "split_missing": "缺图片或掩码",
        "split_overlap": "数据泄漏",
        "duplicate_stem": "多个图片文件",
    }[problem]
    if problem == "invalid_value":
        m = make_mask()
        m[0, 0] = 7
        write_mask(root / "masks" / f"{stems[1]}.png", m)
    elif problem == "mask_without_image":
        write_mask(root / "masks" / "ghost_000001.png", make_mask())
    elif problem == "size_mismatch":
        write_mask(root / "masks" / f"{stems[2]}.png", make_mask(H // 2, W // 2))
    elif problem == "rgb_mask":
        write_image_rgb(root / "masks" / f"{stems[0]}.png", make_image(1))  # 3 通道 PNG 不是合法掩码
    elif problem == "split_missing":
        run(capsys, "split", "--set", f"data.root={root}")
        with open(root / "splits" / "train.txt", "a", encoding="utf-8") as f:
            f.write("deleted_000001\n")
    elif problem == "split_overlap":
        run(capsys, "split", "--set", f"data.root={root}")
        with open(root / "splits" / "val.txt", "a", encoding="utf-8") as f:
            f.write(read_lines(root / "splits" / "train.txt")[0] + "\n")
    elif problem == "duplicate_stem":
        write_image_rgb(root / "images" / f"{stems[0]}.png", make_image(3))
    capsys.readouterr()
    code, out, _ = run(capsys, "check-data", "--set", f"data.root={root}")
    assert code == 1, out
    assert "[错误]" in out and expected in out
    assert "发现" in out


def test_check_data_notes_video_shared_between_train_and_val(capsys, tmp_path):
    """手工编辑的划分把同一段视频的不同帧放在 train 和 val 两边：不是错误，但必须提示（验证指标会偏乐观）。"""
    root = tmp_path / "data"
    make_dataset(root)
    (root / "splits").mkdir()
    (root / "splits" / "train.txt").write_text("base_000000\ndam_000000\ndam_000010\n", encoding="utf-8")
    (root / "splits" / "val.txt").write_text("base_000010\nvalley_000000\n", encoding="utf-8")
    code, out, _ = run(capsys, "check-data", "--set", f"data.root={root}")
    assert code == 0, out
    notes = [line for line in out.splitlines() if "同时出现在 train 和 val" in line]
    assert len(notes) == 1
    assert "1 段视频" in notes[0] and "base" in notes[0]
    assert "dam" not in notes[0] and "valley" not in notes[0]

    # 按视频分组的正常划分不会出现该提示
    run(capsys, "split", "--force", "--set", f"data.root={root}")
    code, out, _ = run(capsys, "check-data", "--set", f"data.root={root}")
    assert code == 0 and "同时出现在 train 和 val" not in out


def test_check_data_bad_args(capsys, tmp_path):
    root = tmp_path / "data"
    code, _, err = run(capsys, "check-data", "--set", f"data.root={root}")
    assert code == 1 and "数据目录不存在" in err
    make_dataset(root)
    code, _, err = run(capsys, "check-data", "--vis-dir", root / "images", "--set", f"data.root={root}")
    assert code == 1 and "--vis-dir" in err
    code, _, err = run(capsys, "check-data", "--vis-limit", "-1", "--set", f"data.root={root}")
    assert code == 1 and "--vis-limit" in err


def test_check_data_empty_root(capsys, tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    code, out, _ = run(capsys, "check-data", "--set", f"data.root={root}")
    assert code == 0
    assert "0 张图片" in out and "0 个掩码" in out


def test_evenly_spaced():
    items = [str(i) for i in range(10)]
    assert cli._evenly_spaced(items, 0) == items
    assert cli._evenly_spaced(items, 20) == items
    assert cli._evenly_spaced(items, 1) == ["0"]
    assert cli._evenly_spaced(items, 2) == ["0", "9"]
    assert cli._evenly_spaced(items, 4) == ["0", "3", "6", "9"]
    assert cli._evenly_spaced([], 5) == []


# ---------------------------------------------------------------- train / eval / infer / export（共用一次训练）


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """合成数据集并通过 CLI 训练 1 个 epoch，返回 (配置文件, 数据根目录, best.pt, 训练输出目录, stdout)。"""
    base = tmp_path_factory.mktemp("cli_train")
    root = base / "data"
    make_dataset(root)
    cfg_path = write_config(base / "train.yaml", root, train={"epochs": 3, "batch_size": 8, "device": "auto"})
    out_dir = base / "runs" / "exp"
    old_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, min(2, old_threads)))
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(socket.socket, "connect", _guard)
            mp.setattr(socket, "create_connection", _guard)
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                code = main(["train", "-c", str(cfg_path), "--epochs", "1", "--batch-size", "4",
                             "--out-dir", str(out_dir), "--device", "cpu"])
    finally:
        torch.set_num_threads(old_threads)
    assert code == 0, buf_err.getvalue()
    return SimpleNamespace(cfg=cfg_path, root=root, ckpt=out_dir / "best.pt", out_dir=out_dir,
                           out=buf_out.getvalue(), err=buf_err.getvalue())


def test_train_outputs(trained):
    assert trained.ckpt.is_file()
    assert (trained.out_dir / "last.pt").is_file()
    assert (trained.out_dir / "history.csv").is_file()
    assert f"最佳 checkpoint: {trained.ckpt}" in trained.out
    # 子命令参数覆盖了配置文件中的值
    saved = yaml.safe_load((trained.out_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved["train"]["epochs"] == 1
    assert saved["train"]["batch_size"] == 4
    assert saved["train"]["device"] == "cpu"
    assert saved["train"]["out_dir"] == str(trained.out_dir)
    assert saved["train"]["encoder"] == "resnet18"
    # 自动划分了训练/验证集
    assert (trained.root / "splits" / "train.txt").is_file()
    assert "eval --ckpt" in trained.err


def test_train_flag_precedence(capsys, tmp_path, monkeypatch):
    import dfwalk.train

    seen = []

    def fake_train(cfg):
        seen.append(cfg)
        return Path(cfg.train.out_dir) / "best.pt"

    monkeypatch.setattr(dfwalk.train, "train", fake_train)
    code, out, _ = run(capsys, "train", "--set", "train.epochs=5", "--set", "train.batch_size=3")
    assert code == 0
    assert (seen[-1].train.epochs, seen[-1].train.batch_size) == (5, 3)
    assert seen[-1].train.out_dir == "runs/exp" and seen[-1].train.device == "auto"
    assert "最佳 checkpoint: runs" in out
    code, _, _ = run(capsys, "train", "--set", "train.epochs=5", "--epochs", "2", "--batch-size", "1",
                     "--out-dir", tmp_path / "o", "--device", "cpu")
    assert code == 0
    t = seen[-1].train
    assert (t.epochs, t.batch_size, t.out_dir, t.device) == (2, 1, str(tmp_path / "o"), "cpu")


def test_train_refuses_to_overwrite_previous_run(capsys, tmp_path, monkeypatch):
    """重新训练会在第 1 个 epoch 后替换 best.pt：输出目录已有 best.pt / last.pt 时默认拒绝（与 split 一致），
    提示换 --out-dir 或加 --force。"""
    import dfwalk.train

    calls = []
    monkeypatch.setattr(dfwalk.train, "train", lambda cfg: calls.append(cfg) or Path(cfg.train.out_dir) / "best.pt")
    out_dir = tmp_path / "runs" / "exp"
    out_dir.mkdir(parents=True)
    (out_dir / "best.pt").write_bytes(b"v1 model")
    code, _, err = run(capsys, "train", "--out-dir", out_dir)
    assert code == 1
    assert "best.pt" in err and "--out-dir" in err and "--force" in err
    assert calls == []
    assert (out_dir / "best.pt").read_bytes() == b"v1 model"

    (out_dir / "best.pt").unlink()
    (out_dir / "last.pt").write_bytes(b"interrupted run")  # 只有 last.pt（上次训练中断）同样拒绝
    code, _, err = run(capsys, "train", "--out-dir", out_dir)
    assert code == 1 and "last.pt" in err and calls == []

    assert run(capsys, "train", "--out-dir", out_dir, "--force")[0] == 0
    assert len(calls) == 1
    assert run(capsys, "train", "--out-dir", tmp_path / "runs" / "exp2")[0] == 0  # 新目录不受影响
    assert len(calls) == 2


def test_train_errors(capsys, tmp_path):
    root = tmp_path / "data"
    make_dataset(root)
    cfg = write_config(tmp_path / "c.yaml", root)
    code, _, err = run(capsys, "train", "-c", cfg, "--epochs", "0", "--out-dir", tmp_path / "o")
    assert code == 1 and "epochs" in err
    code, _, err = run(capsys, "train", "-c", cfg, "--set", f"data.root={tmp_path / 'nope'}",
                       "--out-dir", tmp_path / "o2")
    assert code == 1 and "图片目录不存在" in err


def test_eval_prints_metrics_table(capsys, trained):
    code, out, _ = run(capsys, "eval", "-c", trained.cfg, "--ckpt", trained.ckpt, "--device", "cpu")
    assert code == 0
    n_val = len(read_lines(trained.root / "splits" / "val.txt"))
    assert f"划分: val（{n_val} 帧）" in out
    lines = {line.split()[0]: line.split() for line in out.splitlines() if line.strip()}
    for key in ("iou_walkable", "iou_non_walkable", "miou", "pixel_acc", "precision_walkable",
                "recall_walkable", "f1_walkable", "loss"):
        assert key in lines, out
    for key in ("iou_walkable", "miou", "pixel_acc"):
        value = lines[key][1]
        assert value == "nan" or 0.0 <= float(value) <= 1.0
    # 表格对齐：数值列起始位置一致
    rows = [line for line in out.splitlines() if line.split() and line.split()[0] in METRIC_KEYS_SET]
    starts = {len(r) - len(r[len(r.split()[0]):].lstrip()) for r in rows}
    assert len(starts) == 1

    code, out, _ = run(capsys, "eval", "-c", trained.cfg, "--ckpt", trained.ckpt, "--split", "train")
    assert code == 0 and "划分: train" in out


METRIC_KEYS_SET = {"iou_walkable", "iou_non_walkable", "miou", "pixel_acc", "precision_walkable",
                   "recall_walkable", "f1_walkable", "loss"}


def test_eval_split_selects_the_evaluated_frames(capsys, trained, monkeypatch):
    """--split 不只是表头：指标必须真的在该划分上计算（表头的帧数来自另一处 read_split）。"""
    import dfwalk.train

    real = dfwalk.train.evaluate_checkpoint
    splits = []

    def spy(cfg, ckpt_path, split="val"):
        splits.append(split)
        return real(cfg, ckpt_path, split=split)

    monkeypatch.setattr(dfwalk.train, "evaluate_checkpoint", spy)
    assert run(capsys, "eval", "-c", trained.cfg, "--ckpt", trained.ckpt, "--device", "cpu")[0] == 0
    code, out, _ = run(capsys, "eval", "-c", trained.cfg, "--ckpt", trained.ckpt, "--device", "cpu",
                       "--split", "train")
    assert code == 0
    assert splits == ["val", "train"]
    cfg = cli.load_cli_config(trained.cfg, ["train.device=cpu"], notice=False)
    expected = real(cfg, trained.ckpt, split="train")
    printed = {line.split()[0]: line.split()[1] for line in out.splitlines()
               if line.split() and line.split()[0] in METRIC_KEYS_SET}
    assert printed["loss"] == f"{expected['loss']:.4f}"


def test_eval_errors(capsys, trained, tmp_path):
    code, _, err = run(capsys, "eval", "-c", trained.cfg, "--ckpt", tmp_path / "nope.pt")
    assert code == 1 and "checkpoint 不存在" in err
    code, _, err = run(capsys, "eval", "-c", trained.cfg, "--ckpt", trained.ckpt, "--split", "nosuch")
    assert code == 1 and "nosuch.txt" in err
    not_ckpt = tmp_path / "x.pt"
    torch.save({"hello": torch.zeros(1)}, not_ckpt)
    code, _, err = run(capsys, "eval", "-c", trained.cfg, "--ckpt", not_ckpt)
    assert code == 1 and "不是 dfwalk checkpoint" in err


@pytest.mark.parametrize("name", ["model.onnx", "config.yaml", "history.csv", "shot.jpg"])
def test_non_checkpoint_ckpt_is_a_clean_error(capsys, trained, tmp_path, name):
    """误把导出的 model.onnx、配置、图片等传给 --ckpt：打印 “错误: 不是 dfwalk checkpoint …”，
    而不是 UnpicklingError 堆栈和“改用 weights_only=False”的误导建议。"""
    path = tmp_path / name
    if name == "model.onnx":
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper

        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g",
                                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        onnx.save(helper.make_model(graph), str(path))
    elif name == "config.yaml":
        path.write_bytes(trained.cfg.read_bytes())
    elif name == "history.csv":
        path.write_bytes((trained.out_dir / "history.csv").read_bytes())
    else:
        write_image_rgb(path, make_image(3))
    img = tmp_path / "a.jpg"
    write_image_rgb(img, make_image(4))
    for argv in (["infer", "--ckpt", path, img, "-o", tmp_path / "out"], ["eval", "--ckpt", path],
                 ["export-onnx", "--ckpt", path, "-o", tmp_path / "m.onnx"]):
        code, _, err = run(capsys, *argv, "-c", trained.cfg)
        assert code == 1, argv
        assert "错误: 不是 dfwalk checkpoint" in err and "Traceback" not in err and "weights_only" not in err
        if name == "model.onnx":
            assert "onnxruntime" in err


def test_format_metrics_table():
    table = format_metrics_table({"loss": 0.5, "miou": 0.25, "iou_walkable": float("nan"), "iou_class2": 0.1})
    lines = table.splitlines()
    names = [line.split()[0] for line in lines[2:]]
    assert names == ["iou_walkable", "miou", "iou_class2", "loss"]
    assert "nan" in lines[2] and "0.2500" in lines[3] and "0.5000" in lines[5]
    assert "可走区域 IoU" in lines[2]
    assert format_metrics_table({}) == "（没有指标）"


def test_infer_images(capsys, trained, tmp_path):
    shots = tmp_path / "shots"
    for i in range(3):
        write_image_rgb(shots / f"shot{i}.jpg", make_image(50 + i))
    single = tmp_path / "single.png"
    write_image_rgb(single, make_image(60, h=72, w=128))  # 与训练尺寸不同的分辨率
    out_dir = tmp_path / "out"
    code, out, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, shots, single,
                         "-o", out_dir, "--device", "cpu", "--threshold", "0",
                         "--set", "hud.rects=[[0, 0, 0.25, 0.25]]")
    assert code == 0, err
    assert "图片: 4 张" in out
    for stem in ("shot0", "shot1", "shot2", "single"):
        assert (out_dir / f"{stem}_overlay.jpg").is_file()
        mask = read_mask(out_dir / f"{stem}_mask.png")
        h, w = mask.shape
        # threshold=0 时概率 >= 0 恒成立：除 HUD 外全部为可走
        assert (mask[: h // 4, : w // 4] == IGNORE_INDEX).all()
        assert (mask[h // 2:] == WALKABLE).all()
    assert read_mask(out_dir / "single_mask.png").shape == (72, 128)
    assert "cpu" in err


def test_infer_no_mask_and_threshold(capsys, trained, tmp_path):
    img = tmp_path / "a.jpg"
    write_image_rgb(img, make_image(70))
    out_dir = tmp_path / "out"
    code, out, _ = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, img, "-o", out_dir, "--no-mask")
    assert code == 0
    assert (out_dir / "a_overlay.jpg").is_file()
    assert not (out_dir / "a_mask.png").exists()

    code, _, _ = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, img, "-o", out_dir,
                     "--threshold", "1.0", "--set", "infer.threshold=0")
    assert code == 0
    # --threshold 优先于 --set infer.threshold；softmax 概率不会严格达到 1.0 以上，几乎全为不可走
    assert (read_mask(out_dir / "a_mask.png") == WALKABLE).mean() < 0.5


def test_infer_video_and_directory(capsys, trained, tmp_path):
    media = tmp_path / "media"
    video = make_video(media / "clip", [make_image(80 + i) for i in range(8)], fps=10.0)
    write_image_rgb(media / "frame.jpg", make_image(90))
    out_dir = tmp_path / "out"
    code, out, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, media, "-o", out_dir,
                         "--every-n", "2", "--max-frames", "5")
    assert code == 0, err
    out_video = out_dir / "clip_overlay.mp4"
    assert out_video.is_file()
    assert count_video_frames(out_video) == 5
    assert (out_dir / "frame_overlay.jpg").is_file()
    assert f"视频: {video} → {out_video}" in out


def test_infer_video_receives_cli_and_config_settings(capsys, trained, tmp_path, monkeypatch):
    """视频推理要用上 --threshold / --every-n / --max-frames 以及配置中的 hud.rects、infer.alpha、infer.color
    （图片路径有内容检查，视频路径之前只检查了帧数和分辨率）。"""
    import inspect

    import dfwalk.infer

    real = dfwalk.infer.infer_video
    calls = []

    def spy(*args, **kwargs):
        bound = inspect.signature(real).bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        return real(*args, **kwargs)

    monkeypatch.setattr(dfwalk.infer, "infer_video", spy)
    video = make_video(tmp_path / "media" / "clip", [make_image(80 + i) for i in range(6)], fps=10.0)
    out_dir = tmp_path / "out"
    code, _, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, video, "-o", out_dir,
                       "--threshold", "0.7", "--every-n", "3", "--max-frames", "4",
                       "--set", "hud.rects=[[0.75, 0.0, 1.0, 0.25]]", "--set", "infer.alpha=0.3",
                       "--set", "infer.color=[255, 0, 0]")
    assert code == 0, err
    assert len(calls) == 1
    kw = calls[0]
    assert Path(kw["video_path"]) == video and Path(kw["out_path"]) == out_dir / "clip_overlay.mp4"
    assert kw["threshold"] == pytest.approx(0.7)
    assert (kw["every_n"], kw["max_frames"]) == (3, 4)
    assert kw["alpha"] == pytest.approx(0.3)
    assert list(kw["color"]) == [255, 0, 0]
    assert [list(r) for r in kw["hud_rects"]] == [[0.75, 0.0, 1.0, 0.25]]
    assert count_video_frames(out_dir / "clip_overlay.mp4") == 4


@pytest.mark.parametrize("suffix", [".ts", ".m4v"])
def test_infer_treats_ts_and_m4v_recordings_as_videos(capsys, trained, tmp_path, suffix):
    """OBS 的 .ts、.m4v 等录像（extract-frames 能处理）不能被当成图片跳过：之前会提示“无法读取的图片”并以 0 退出。"""
    src = make_video(tmp_path / "src" / "clip", [make_image(80 + i) for i in range(4)], fps=10.0)
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir()
    rec = rec_dir / f"rec{suffix}"
    rec.write_bytes(src.read_bytes())  # OpenCV 按内容识别容器格式

    out_dir = tmp_path / "out"
    code, out, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, rec, "-o", out_dir)
    assert code == 0, err
    assert "无法读取的图片" not in err
    assert count_video_frames(out_dir / "rec_overlay.mp4") == 4
    assert f"视频: {rec}" in out

    # 目录输入同样会找到这类录像
    code, _, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, rec_dir, "-o", tmp_path / "out2")
    assert code == 0, err
    assert (tmp_path / "out2" / "rec_overlay.mp4").is_file()


@pytest.mark.parametrize(
    "extra, expected",
    [
        (["--every-n", "0"], "--every-n"),
        (["--max-frames", "0"], "--max-frames"),
        (["--threshold", "1.5"], "infer.threshold"),
    ],
)
def test_infer_bad_flags_fail_before_writing(capsys, trained, tmp_path, extra, expected):
    img = tmp_path / "a.jpg"
    write_image_rgb(img, make_image(1))
    out_dir = tmp_path / "out"
    code, _, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, img, "-o", out_dir, *extra)
    assert code == 1
    assert expected in err
    assert not out_dir.exists()


def test_device_cuda_with_cpu_only_torch_is_a_clean_error(capsys, trained, tmp_path, monkeypatch):
    """CPU 版 PyTorch 上 --device cuda：打印一行 “错误: …” 和安装提示，而不是 30 多行 AssertionError 堆栈。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None)
    img = tmp_path / "a.jpg"
    write_image_rgb(img, make_image(1))
    code, _, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, img, "-o", tmp_path / "out",
                       "--device", "cuda")
    assert code == 1
    assert "错误: " in err and "CPU 版 PyTorch" in err and "Traceback" not in err
    code, _, err = run(capsys, "eval", "-c", trained.cfg, "--ckpt", trained.ckpt, "--device", "cuda")
    assert code == 1 and "CPU 版 PyTorch" in err


def test_infer_input_errors(capsys, trained, tmp_path):
    code, _, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, tmp_path / "nope.jpg")
    assert code == 1 and "输入不存在" in err
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, empty)
    assert code == 1 and "没有找到可推理" in err
    code, _, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", tmp_path / "nope.pt", empty)
    assert code == 1 and "checkpoint 不存在" in err
    # 不同目录下同名视频：输出会互相覆盖
    v1 = make_video(tmp_path / "a" / "clip", noise_frames(2))
    v2 = make_video(tmp_path / "b" / "clip", noise_frames(2))
    code, _, err = run(capsys, "infer", "-c", trained.cfg, "--ckpt", trained.ckpt, v1, v2, "-o", tmp_path / "o")
    assert code == 1 and "stem" in err
    assert not (tmp_path / "o").exists()


def test_export_onnx(capsys, trained, tmp_path):
    pytest.importorskip("onnx")
    out = tmp_path / "导出" / "model.onnx"
    code, stdout, err = run(capsys, "export-onnx", "--ckpt", trained.ckpt, "-o", out)
    assert code == 0, err
    assert out.is_file()
    assert f"已导出 ONNX: {out}" in stdout
    assert "opset: 17" in stdout  # 打印文件实际的 opset

    import onnx

    model = onnx.load(str(out))
    dims = model.graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param == "batch"
    assert [d.dim_value for d in dims[1:]] == [3, H, W]

    static = tmp_path / "static.onnx"
    code, stdout, _ = run(capsys, "export-onnx", "--ckpt", trained.ckpt, "-o", static, "--static-batch",
                          "--opset", "17")
    assert code == 0
    dims = onnx.load(str(static)).graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_value == 1
    assert "(1, 3, H, W)" in stdout


def test_export_onnx_default_output_in_cwd(capsys, trained, _isolated_cwd):
    pytest.importorskip("onnx")
    code, _, err = run(capsys, "export-onnx", "--ckpt", trained.ckpt, "--static-batch")
    assert code == 0, err
    assert (_isolated_cwd / "model.onnx").is_file()


def test_export_onnx_errors(capsys, tmp_path, trained):
    code, _, err = run(capsys, "export-onnx", "--ckpt", tmp_path / "nope.pt")
    assert code == 1 and "checkpoint 不存在" in err
    code, _, err = run(capsys, "export-onnx", "--ckpt", trained.ckpt, "-o", tmp_path / "m.onnx", "--opset", "0")
    assert code == 1 and "opset" in err


# ---------------------------------------------------------------- README 与命令/配置保持同步


def test_readme_documents_commands_config_and_boundary():
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    for name in SUBCOMMANDS:
        assert f"dfwalk {name}" in readme, name
    for section, values in Config().to_dict().items():
        for key in values:
            assert f"`{section}.{key}`" in readme, f"{section}.{key} 不在配置参考表中"
    for phrase in ("不做外挂", "用户协议", "反作弊", "HF_ENDPOINT=https://hf-mirror.com", "MIT"):
        assert phrase in readme
