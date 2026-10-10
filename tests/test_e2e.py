"""端到端测试：只通过 dfwalk.cli.main 走一遍 README 中的完整流程。

    extract-frames → hud-preview → autolabel（假模型）→ to-labelme → from-labelme → split → check-data
    → train → eval → infer（图片目录 + 视频）→ export-onnx

- 输入是现场合成的两段“类游戏”第一人称录像（天空、带纹理的地面、移动的箱子、左上角 HUD 小地图），
  文件名含中文和空格，工作目录也是中文路径；
- autolabel 不下载真实模型：AutoLabeler.from_pretrained 被替换为按颜色“分割”的假 SegFormer
  （绿色地面 → floor，其余 → sky），因此伪标签与画面内容一致，训练出的模型可以被检验；
- 全程离线（禁止 socket 连接）、只用 CPU，torch 线程数限制为 2；整个模块约 15~30 秒。

流程只在模块级 fixture 中跑一次（到 train 为止），各个测试检查对应步骤的产物；
eval / infer / export-onnx 在各自的测试中基于训练好的 best.pt 运行。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import re
import socket
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from dfwalk import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD, NON_WALKABLE, WALKABLE
from dfwalk.cli import main
from dfwalk.config import load_config
from dfwalk.dataset import group_of, read_split
from dfwalk.hud import hud_mask
from dfwalk.utils import list_images, read_image_rgb, read_mask

# 合成录像：VH x VW，10 fps；抽帧时最长边缩到 MAX_SIDE
VH, VW = 144, 256
VIDEO_FPS = 10.0
MAX_SIDE = 192
FH, FW = 108, 192  # 抽帧后的尺寸（256 → 192，144 → 108）
VIDEOS = {"零号大坝 第1局": 20, "长弓溪谷_run2": 15}  # 文件名 → 帧数
PREFIXES = {"零号大坝 第1局": "零号大坝_第1局", "长弓溪谷_run2": "长弓溪谷_run2"}  # sanitize_stem 的结果
HORIZON = 0.42  # 地平线位置（占画面高度的比例）
HUD_RECT = [0.0, 0.0, 0.25, 0.3]  # 左上角小地图（归一化）
IMAGE_SIZE = [64, 128]

SKY = (90, 140, 220)
GROUND = (70, 150, 60)
CRATE = (140, 95, 55)


# ---------------------------------------------------------------- 合成数据


def render_frame(i: int, seed: int) -> np.ndarray:
    """第 i 帧（RGB）：天空渐变 + 向前滚动的地面纹理 + 横向移动的箱子 + 左上角 HUD。"""
    rng = np.random.default_rng(seed * 1000 + i)
    img = np.empty((VH, VW, 3), np.float32)
    hz = int(VH * HORIZON)
    t = np.linspace(0.0, 1.0, hz, dtype=np.float32)[:, None, None]
    img[:hz] = np.asarray(SKY, np.float32) * (0.85 + 0.15 * t)
    # 地面：条纹纹理随帧号向下滚动（模拟前进），再叠加噪声
    rows = np.arange(VH - hz, dtype=np.float32)[:, None, None]
    stripes = 12.0 * np.sin((rows + 3 * i) / 4.0)
    img[hz:] = np.asarray(GROUND, np.float32) + stripes
    img += rng.normal(0.0, 6.0, img.shape).astype(np.float32)
    # 箱子：立在地面上，随帧号横向移动
    x0 = int(VW * 0.55) + 3 * i if seed == 1 else int(VW * 0.15) + 2 * i
    cv2.rectangle(img, (x0, hz - 12), (x0 + 34, hz + 26), CRATE, -1)
    # HUD：左上角深色小地图 + 圆环
    c1, r1 = int(VW * HUD_RECT[2]) - 6, int(VH * HUD_RECT[3]) - 6
    img[4:r1, 4:c1] = img[4:r1, 4:c1] * 0.3 + np.asarray((25, 30, 25), np.float32) * 0.7
    cv2.circle(img, ((4 + c1) // 2, (4 + r1) // 2), 14, (90, 200, 120), 1)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_video(path_no_ext: Path, frames: list[np.ndarray]) -> Path:
    """mp4v/.mp4 优先，当前 OpenCV 不支持时退回 MJPG/.avi。"""
    for ext, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        path = path_no_ext.with_name(path_no_ext.name + ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), VIDEO_FPS, (VW, VH))
        if not writer.isOpened():
            writer.release()
            continue
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        if path.is_file() and path.stat().st_size > 0:
            return path
    pytest.skip("当前 OpenCV 没有可用的视频编码器")


class ColorSegFormer:
    """假 SegFormer（ADE20K 风格接口）：输出 1/4 分辨率 logits，类别 0="floor"、1="sky"。

    floor 的 logit = 8 * (G - max(R, B))（在归一化后的像素值上计算）：绿色地面 → 可走，
    蓝色天空、棕色箱子、深色 HUD → 不可走。这样伪标签与画面内容一致，训练结果可检验。
    """

    def __call__(self, pixel_values: torch.Tensor):
        x = F.avg_pool2d(pixel_values, 4)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        floor = 8.0 * (g - torch.maximum(r, b))
        return SimpleNamespace(logits=torch.cat([floor, torch.zeros_like(floor)], dim=1))


def _guard(*args, **kwargs):
    raise RuntimeError("端到端测试中禁止联网")


def run(*argv) -> tuple[int, str, str]:
    """调用 main，返回 (退出码, stdout, stderr)。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def ok(*argv) -> tuple[str, str]:
    code, out, err = run(*argv)
    assert code == 0, f"dfwalk {' '.join(map(str, argv))} 退出码 {code}\nstdout:\n{out}\nstderr:\n{err}"
    return out, err


# ---------------------------------------------------------------- 模块级流程（跑一次）


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """在中文工作目录中跑 extract-frames → … → train，返回各步骤的输出和路径。"""
    from dfwalk.autolabel import AutoLabeler

    work = tmp_path_factory.mktemp("e2e") / "工作区"
    work.mkdir()
    old_threads = torch.get_num_threads()
    autolabel_calls: list[dict] = []

    def fake_from_pretrained(cls, name, walkable_labels, short_side=512, hi=0.7, lo=0.3, device="auto"):
        autolabel_calls.append(dict(name=name, walkable_labels=list(walkable_labels), short_side=short_side,
                                    hi=hi, lo=lo, device=device))
        return cls(ColorSegFormer(), {0: "floor", 1: "sky"}, ["floor"], IMAGENET_MEAN, IMAGENET_STD,
                   short_side=short_side, hi=hi, lo=lo, device="cpu")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", _guard)
        mp.setattr(socket, "create_connection", _guard)
        mp.setattr(AutoLabeler, "from_pretrained", classmethod(fake_from_pretrained))
        mp.delenv("DFWALK_DEBUG", raising=False)
        mp.chdir(work)
        torch.set_num_threads(max(1, min(2, old_threads)))
        try:
            yield _run_pipeline(work, autolabel_calls)
        finally:
            torch.set_num_threads(old_threads)


def _run_pipeline(work: Path, autolabel_calls: list[dict]) -> SimpleNamespace:
    videos_dir = work / "录像"
    videos_dir.mkdir()
    videos = {}
    for seed, (name, n) in enumerate(VIDEOS.items(), start=1):
        videos[name] = write_video(videos_dir / name, [render_frame(i, seed) for i in range(n)])

    cfg_path = work / "my.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {"root": "data", "image_size": IMAGE_SIZE, "val_ratio": 0.3, "seed": 0},
                "hud": {"rects": [HUD_RECT]},
                "autolabel": {"model": "fake/segformer", "short_side": 64, "device": "cpu"},
                "train": {"arch": "Unet", "encoder": "resnet18", "encoder_weights": None, "batch_size": 4,
                          "num_workers": 0, "device": "cpu", "out_dir": "runs/exp", "amp": True},
                "infer": {"device": "cpu"},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    c = ["-c", cfg_path]
    steps: dict[str, tuple[str, str]] = {}
    steps["extract"] = ok("extract-frames", videos_dir, *c, "--fps", 5, "--max-side", MAX_SIDE, "--dedup", 0)
    first = sorted((work / "data" / "images").glob("*.jpg"))[0]
    steps["hud"] = ok("hud-preview", first, "-o", work / "hud_preview.png", *c)
    steps["autolabel"] = ok("autolabel", *c)
    steps["to_labelme"] = ok("to-labelme", *c, "--epsilon", 0, "--min-area", 0)
    steps["from_labelme"] = ok("from-labelme", *c)
    steps["split"] = ok("split", *c)
    steps["split_again"] = run("split", *c)  # 已有划分：应拒绝覆盖
    steps["check"] = ok("check-data", *c, "--vis-dir", work / "vis", "--vis-limit", 4)
    # --set 的 1e-3 会被 PyYAML 解析成字符串，配置层需要把它转成 float
    steps["train"] = ok("train", *c, "--epochs", 3, "--set", "train.lr=1e-3")
    return SimpleNamespace(
        work=work,
        cfg_path=cfg_path,
        root=work / "data",
        videos=videos,
        steps=steps,
        autolabel_calls=autolabel_calls,
        best=work / "runs" / "exp" / "best.pt",
    )


@pytest.fixture
def offline(monkeypatch, e2e):
    """单个测试内同样禁止联网，并在 e2e 的工作目录中运行。"""
    monkeypatch.setattr(socket.socket, "connect", _guard)
    monkeypatch.setattr(socket, "create_connection", _guard)
    monkeypatch.chdir(e2e.work)
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, min(2, old)))
    yield e2e
    torch.set_num_threads(old)


def _expected_frame_indices(n_frames: int) -> list[int]:
    # 10 fps 的源视频按 5 fps 抽帧：每 2 帧取 1 帧，去重关闭
    return list(range(0, n_frames, 2))


def _region_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(HUD 区域, 天空区域（不含 HUD）, 地面区域中远离地平线和箱子的下沿部分)。"""
    hud = hud_mask(h, w, [HUD_RECT])
    sky = np.zeros((h, w), bool)
    sky[: int(h * HORIZON) - 2] = True
    sky &= ~hud
    ground = np.zeros((h, w), bool)
    ground[int(h * 0.75):] = True
    return hud, sky, ground


# ---------------------------------------------------------------- 各步骤产物


def test_extract_frames_names_counts_and_size(e2e):
    images = list_images(e2e.root / "images")
    by_group: dict[str, list[int]] = {}
    for p in images:
        by_group.setdefault(group_of(p.stem), []).append(int(p.stem.rsplit("_", 1)[1]))
    expected = {PREFIXES[name]: _expected_frame_indices(n) for name, n in VIDEOS.items()}
    assert {g: sorted(v) for g, v in by_group.items()} == expected
    for p in images[:3]:
        assert read_image_rgb(p).shape == (FH, FW, 3)  # 最长边缩到 MAX_SIDE
    out, _ = e2e.steps["extract"]
    assert f"共写出 {len(images)} 帧（2 个视频）" in out


def test_hud_preview_written_and_reports_pixels(e2e):
    out, _ = e2e.steps["hud"]
    preview = read_image_rgb(e2e.work / "hud_preview.png")
    assert preview.shape == (FH, FW, 3)
    # 终点向上取整，保证 HUD 被完整覆盖：0.3 * 108 = 32.4 → 33
    c1, r1 = math.ceil(HUD_RECT[2] * FW), math.ceil(HUD_RECT[3] * FH)
    assert (c1, r1) == (48, 33)
    assert f"像素 x 0~{c1}, y 0~{r1}" in out


def test_autolabel_uses_config_and_labels_content(e2e):
    (call,) = e2e.autolabel_calls
    assert call["name"] == "fake/segformer" and call["short_side"] == 64 and call["device"] == "cpu"
    assert (call["hi"], call["lo"]) == (0.7, 0.3) and "floor" in call["walkable_labels"]

    images = list_images(e2e.root / "images")
    pseudo = sorted((e2e.root / "pseudo_masks").glob("*.png"))
    assert [p.stem for p in pseudo] == [p.stem for p in images]
    hud, sky, ground = _region_masks(FH, FW)
    for p in pseudo:
        m = read_mask(p)
        assert m.shape == (FH, FW) and set(np.unique(m).tolist()) <= {NON_WALKABLE, WALKABLE, IGNORE_INDEX}
        assert (m[hud] == IGNORE_INDEX).all()  # HUD 一律忽略
        assert (m[sky] == NON_WALKABLE).mean() > 0.95  # 天空不可走
        assert (m[ground] == WALKABLE).mean() > 0.8  # 近处地面可走（箱子遮挡的部分除外）
    out, _ = e2e.steps["autolabel"]
    assert f"生成伪标签 {len(images)} 张" in out


def test_labelme_roundtrip_is_exact(e2e):
    jsons = sorted((e2e.root / "labelme").glob("*.json"))
    masks = sorted((e2e.root / "masks").glob("*.png"))
    assert len(jsons) == len(masks) == len(list_images(e2e.root / "images"))
    data = json.loads(jsons[0].read_text(encoding="utf-8"))
    # imagePath 相对 JSON 所在目录，LabelMe 打开时能找到图片
    assert (jsons[0].parent / data["imagePath"]).resolve() == (e2e.root / "images" / f"{jsons[0].stem}.jpg").resolve()
    assert {s["label"] for s in data["shapes"]} <= {"walkable", "non_walkable", "ignore"}
    # epsilon=0、min_area=0：伪标签 → JSON → 掩码 逐像素一致（HUD 区域本来就是 255）
    for m in masks:
        np.testing.assert_array_equal(read_mask(m), read_mask(e2e.root / "pseudo_masks" / m.name))


def test_split_groups_by_video_and_refuses_overwrite(e2e):
    train, val = read_split(e2e.root, "train"), read_split(e2e.root, "val")
    assert train and val and not set(train) & set(val)
    assert {group_of(s) for s in train}.isdisjoint({group_of(s) for s in val})  # 同一段录像不会跨两边
    assert sorted(train + val) == sorted(p.stem for p in list_images(e2e.root / "images"))
    code, _, err = e2e.steps["split_again"]
    assert code == 1 and "--force" in err


def test_check_data_clean_with_visualizations(e2e):
    out, _ = e2e.steps["check"]
    n = len(list_images(e2e.root / "images"))
    assert f"有效的图片+掩码对: {n} / {n}" in out
    assert "检查完成：未发现错误" in out
    vis = sorted((e2e.work / "vis").glob("*_label.jpg"))
    assert len(vis) == 4
    groups = {group_of(p.stem[: -len("_label")]) for p in vis}
    assert groups == set(PREFIXES.values())  # 均匀抽样覆盖两段录像


def test_train_outputs_history_and_config(e2e):
    out_dir = e2e.work / "runs" / "exp"
    for name in ("best.pt", "last.pt", "history.csv", "config.yaml"):
        assert (out_dir / name).is_file(), name
    lines = (out_dir / "history.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("epoch,lr,train_loss,val_loss,")
    assert len(lines) == 1 + 3
    cfg = load_config(out_dir / "config.yaml")
    assert cfg.train.lr == pytest.approx(1e-3) and isinstance(cfg.train.lr, float)
    assert cfg.train.epochs == 3 and cfg.data.image_size == IMAGE_SIZE and cfg.hud.rects == [HUD_RECT]
    out, _ = e2e.steps["train"]
    assert "最佳 checkpoint:" in out and "amp=False" in out  # CPU 上 AMP 自动关闭


# ---------------------------------------------------------------- 基于 best.pt 的后续命令


def test_eval_reports_good_walkable_iou(offline):
    out, _ = ok("eval", "-c", offline.cfg_path, "--ckpt", offline.best)
    n_val = len(read_split(offline.root, "val"))
    assert f"划分: val（{n_val} 帧）" in out
    m = re.search(r"^iou_walkable\s+([0-9.]+|nan)", out, re.M)
    assert m is not None, out
    # 伪标签按颜色生成、规律简单：3 个 epoch 后验证集（另一段录像）上可走 IoU 应当很高
    assert float(m.group(1)) > 0.7, out
    assert re.search(r"^loss\s+[0-9.]+", out, re.M)


def test_infer_images_and_video(offline):
    images_dir = offline.root / "images"
    val = read_split(offline.root, "val")
    shots = offline.work / "截图"
    shots.mkdir(exist_ok=True)
    for stem in val[:3]:
        (shots / f"{stem}.jpg").write_bytes((images_dir / f"{stem}.jpg").read_bytes())
    video = offline.videos["长弓溪谷_run2"]
    out_dir = offline.work / "out" / "infer"
    out, _ = ok("infer", "-c", offline.cfg_path, "--ckpt", offline.best, shots, video, "-o", out_dir,
                "--every-n", 2, "--max-frames", 7)
    assert "图片: 3 张" in out

    hud = hud_mask(FH, FW, [HUD_RECT])
    agree = []
    for stem in val[:3]:
        overlay = read_image_rgb(out_dir / f"{stem}_overlay.jpg")
        mask = read_mask(out_dir / f"{stem}_mask.png")
        assert overlay.shape == (FH, FW, 3) and mask.shape == (FH, FW)
        assert (mask[hud] == IGNORE_INDEX).all() and set(np.unique(mask[~hud]).tolist()) <= {0, 1}
        target = read_mask(offline.root / "masks" / f"{stem}.png")
        valid = target != IGNORE_INDEX
        agree.append(float((mask[valid] == target[valid]).mean()))
    assert min(agree) > 0.85, agree

    out_video = out_dir / f"{video.stem}_overlay.mp4"
    cap = cv2.VideoCapture(str(out_video))
    try:
        n = 0
        while cap.grab():
            n += 1
        assert n == 7  # --max-frames
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == VW and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == VH
    finally:
        cap.release()


def test_export_onnx_matches_pytorch(offline):
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("未安装 onnx，跳过 export-onnx")
    out_path = offline.work / "out" / "model.onnx"
    out, _ = ok("export-onnx", "-c", offline.cfg_path, "--ckpt", offline.best, "-o", out_path)
    assert out_path.is_file() and "已导出 ONNX" in out

    import onnx

    model = onnx.load(str(out_path))
    meta = {p.key: json.loads(p.value) for p in model.metadata_props}
    assert meta["dfwalk.image_size"] == IMAGE_SIZE and meta["dfwalk.encoder"] == "resnet18"
    dims = [d.dim_param or d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    assert dims == ["batch", 2, *IMAGE_SIZE]

    if importlib.util.find_spec("onnxruntime") is None:
        return
    import onnxruntime as ort

    from dfwalk.infer import Predictor

    predictor = Predictor.from_checkpoint(offline.best, device="cpu")
    stem = read_split(offline.root, "val")[0]
    x, _ = predictor.transform(read_image_rgb(offline.root / "images" / f"{stem}.jpg"))
    batch = torch.stack([x, x]).numpy()  # 动态 batch = 2
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    (logits,) = sess.run(None, {"image": batch})
    with torch.no_grad():
        ref = predictor.model(torch.from_numpy(batch)).numpy()
    assert logits.shape == (2, 2, *IMAGE_SIZE)
    assert np.abs(logits - ref).max() < 1e-3
