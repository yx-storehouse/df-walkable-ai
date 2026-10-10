"""dfwalk.infer 的单元测试：Predictor（概率/掩码/HUD）、叠加可视化、批量图片推理、视频推理。

全部离线、CPU、合成数据。真实模型用随机初始化的 Unet-resnet18（image_size [64, 96]）；
需要精确控制输出的用例用小的桩模型（输出固定 logits）。
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from dfwalk import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD, NON_WALKABLE, WALKABLE
from dfwalk.checkpoint import save_checkpoint
from dfwalk.hud import apply_hud_ignore, hud_mask
from dfwalk.infer import Predictor, infer_images, infer_video, overlay
from dfwalk.model import build_model, model_config
from dfwalk.transforms import EvalTransform
from dfwalk.utils import read_image_rgb, read_mask, write_image_rgb

IMAGE_SIZE = (64, 96)
CUSTOM_MEAN = (0.5, 0.4, 0.3)
CUSTOM_STD = (0.2, 0.25, 0.3)


# ---------------------------------------------------------------- 桩模型与工具


class ConstModel(nn.Module):
    """输出常数 logits：通道 WALKABLE 为 c，通道 NON_WALKABLE 为 0 → 可走概率 sigmoid(c)。记录调用次数。"""

    def __init__(self, c: float = 0.0) -> None:
        super().__init__()
        self.c = float(c)
        self.calls = 0
        self.anchor = nn.Parameter(torch.zeros(()))  # 让 .to(device) / .eval() 有参数可处理

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        n, _, h, w = x.shape
        out = torch.zeros(n, 2, h, w, dtype=x.dtype, device=x.device)
        out[:, WALKABLE] = self.c
        return out + self.anchor * 0


class HalfModel(nn.Module):
    """axis="x"：左半边可走、右半边不可走；axis="y"：上半边可走、下半边不可走（logit ±20）。"""

    def __init__(self, axis: str) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        out = torch.zeros(n, 2, h, w)
        if self.axis == "x":
            out[:, WALKABLE, :, : w // 2] = 20.0
            out[:, WALKABLE, :, w // 2:] = -20.0
        else:
            out[:, WALKABLE, : h // 2] = 20.0
            out[:, WALKABLE, h // 2:] = -20.0
        return out


class FixedLogitsModel(nn.Module):
    """返回固定的 logits 张量（与输入无关），并记录最近一次输入。"""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return self.logits.expand(x.shape[0], -1, -1, -1).clone()


def rand_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def softmax_walkable(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1)[0, WALKABLE].numpy()


@pytest.fixture(scope="module")
def ckpt_path(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    model = build_model("Unet", "resnet18", encoder_weights=None)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    save_checkpoint(path, model, model_config("Unet", "resnet18"), list(IMAGE_SIZE), epoch=3, metrics={"iou": 0.5},
                    mean=CUSTOM_MEAN, std=CUSTOM_STD)
    return path


@pytest.fixture(scope="module")
def predictor(ckpt_path) -> Predictor:
    return Predictor.from_checkpoint(ckpt_path, device="cpu")


# ---------------------------------------------------------------- Predictor 构造


def test_init_moves_to_device_and_sets_eval():
    model = ConstModel().train()
    p = Predictor(model, [64, 96])
    assert p.model is model
    assert not model.training
    assert p.device == torch.device("cpu")
    assert next(model.parameters()).device.type == "cpu"
    assert p.image_size == (64, 96)
    assert p.mean == pytest.approx(IMAGENET_MEAN) and p.std == pytest.approx(IMAGENET_STD)


def test_init_accepts_auto_and_torch_device():
    # 测试环境无 GPU：auto → cpu
    assert Predictor(ConstModel(), IMAGE_SIZE, device="auto").device.type == "cpu"
    assert Predictor(ConstModel(), IMAGE_SIZE, device=torch.device("cpu")).device == torch.device("cpu")


def test_init_rejects_bad_size_and_stats():
    with pytest.raises(ValueError):
        Predictor(ConstModel(), [64])
    with pytest.raises(ValueError):
        Predictor(ConstModel(), [0, 96])
    with pytest.raises(ValueError):
        Predictor(ConstModel(), IMAGE_SIZE, std=(0.2, 0.0, 0.2))


def test_from_checkpoint_reads_size_mean_std_and_weights(ckpt_path):
    p = Predictor.from_checkpoint(ckpt_path)  # 默认 device="auto"
    assert p.device.type == "cpu"
    assert p.image_size == IMAGE_SIZE
    assert p.mean == pytest.approx(CUSTOM_MEAN)
    assert p.std == pytest.approx(CUSTOM_STD)
    assert not p.model.training
    # 权重与保存时一致（相同种子重建同结构模型）
    torch.manual_seed(0)
    ref = build_model("Unet", "resnet18", encoder_weights=None)
    for (k, v), (k2, v2) in zip(ref.state_dict().items(), p.model.state_dict().items(), strict=True):
        assert k == k2
        assert torch.equal(v, v2), k


def test_from_checkpoint_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor.from_checkpoint(tmp_path / "missing.pt")
    bogus = tmp_path / "bogus.pt"
    torch.save({"hello": 1}, bogus)
    with pytest.raises(ValueError):
        Predictor.from_checkpoint(bogus)


# ---------------------------------------------------------------- predict_prob


def test_predict_prob_odd_size_real_model(predictor):
    img = rand_image(101, 157)
    before = img.copy()
    prob = predictor.predict_prob(img)
    assert prob.shape == (101, 157)
    assert prob.dtype == np.float32
    assert prob.flags.c_contiguous
    assert np.all(np.isfinite(prob))
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    np.testing.assert_array_equal(img, before)  # 不修改输入
    # eval 模式下确定性
    np.testing.assert_array_equal(prob, predictor.predict_prob(img))


def test_predict_prob_matches_reference_pipeline(predictor):
    """EvalTransform（checkpoint 的 mean/std）→ 模型 → softmax 第 1 通道 → 双线性（半像素对齐）回原尺寸。"""
    img = rand_image(101, 157, seed=1)
    x, _ = EvalTransform(IMAGE_SIZE, CUSTOM_MEAN, CUSTOM_STD)(img)
    with torch.no_grad():
        low = softmax_walkable(predictor.model(x.unsqueeze(0)))
    assert low.shape == IMAGE_SIZE
    expected = cv2.resize(low, (157, 101), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_allclose(predictor.predict_prob(img), expected, atol=1e-5)


def test_predict_prob_uses_eval_transform_preprocessing():
    logits = torch.zeros(1, 2, *IMAGE_SIZE)
    model = FixedLogitsModel(logits)
    p = Predictor(model, IMAGE_SIZE, mean=CUSTOM_MEAN, std=CUSTOM_STD)
    for shape in [(101, 157), (30, 20), (64, 96), (200, 50)]:
        img = rand_image(*shape, seed=shape[0])
        p.predict_prob(img)
        expected, _ = EvalTransform(IMAGE_SIZE, CUSTOM_MEAN, CUSTOM_STD)(img)
        assert model.last_input.shape == (1, 3, *IMAGE_SIZE)
        torch.testing.assert_close(model.last_input[0], expected, rtol=0, atol=1e-6)


def test_predict_prob_same_size_is_plain_softmax():
    torch.manual_seed(1)
    logits = torch.randn(1, 2, *IMAGE_SIZE) * 3
    p = Predictor(FixedLogitsModel(logits), IMAGE_SIZE)
    prob = p.predict_prob(rand_image(*IMAGE_SIZE))
    np.testing.assert_allclose(prob, softmax_walkable(logits), atol=1e-6)


@pytest.mark.parametrize("shape", [(101, 157), (31, 47), (480, 854)])
def test_predict_prob_bilinear_upsampling_matches_cv2(shape):
    torch.manual_seed(2)
    logits = torch.randn(1, 2, *IMAGE_SIZE) * 4
    p = Predictor(FixedLogitsModel(logits), IMAGE_SIZE)
    prob = p.predict_prob(rand_image(*shape))
    expected = cv2.resize(softmax_walkable(logits), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    assert prob.shape == shape
    np.testing.assert_allclose(prob, expected, atol=1e-5)
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_predict_prob_orientation_columns_and_rows():
    # 101x157：模型分辨率 64x96 的左/右、上/下半边映射回原图的左/右、上/下半边（不转置）
    img = rand_image(101, 157)
    prob_x = Predictor(HalfModel("x"), IMAGE_SIZE).predict_prob(img)
    assert np.all(prob_x[:, :75] > 0.99) and np.all(prob_x[:, 82:] < 0.01)
    prob_y = Predictor(HalfModel("y"), IMAGE_SIZE).predict_prob(img)
    assert np.all(prob_y[:48] > 0.99) and np.all(prob_y[54:] < 0.01)


def test_predict_prob_constant_logits():
    for c in (-3.0, 0.0, 2.5):
        prob = Predictor(ConstModel(c), IMAGE_SIZE).predict_prob(rand_image(101, 157))
        np.testing.assert_allclose(prob, 1.0 / (1.0 + math.exp(-c)), atol=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((32, 48), np.uint8),  # 灰度
        np.zeros((32, 48, 4), np.uint8),  # RGBA
        np.zeros((32, 48, 3), np.float32),  # 非 uint8
        np.zeros((0, 48, 3), np.uint8),  # 空图
        [[1, 2, 3]],  # 不是数组
    ],
)
def test_predict_prob_rejects_bad_input(bad):
    p = Predictor(ConstModel(), IMAGE_SIZE)
    with pytest.raises(ValueError):
        p.predict_prob(bad)


def test_predict_prob_rejects_single_channel_model():
    class OneChannel(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, *x.shape[2:])

    with pytest.raises(ValueError):
        Predictor(OneChannel(), IMAGE_SIZE).predict_prob(rand_image(*IMAGE_SIZE))


# ---------------------------------------------------------------- predict_mask


def test_predict_mask_threshold_is_inclusive():
    # 输入尺寸等于 image_size 时不插值，c=0 → 概率恰好 0.5
    p = Predictor(ConstModel(0.0), IMAGE_SIZE)
    img = rand_image(*IMAGE_SIZE)
    assert np.all(p.predict_prob(img) == 0.5)
    m = p.predict_mask(img, threshold=0.5)
    assert m.dtype == np.uint8 and m.shape == IMAGE_SIZE
    assert np.all(m == WALKABLE)
    assert np.all(p.predict_mask(img, threshold=0.5001) == NON_WALKABLE)
    assert np.all(p.predict_mask(img, threshold=0.0) == WALKABLE)
    assert np.all(p.predict_mask(img, threshold=1.0) == NON_WALKABLE)


def test_predict_mask_split_and_values(predictor):
    img = rand_image(101, 157, seed=3)
    prob = predictor.predict_prob(img)
    t = float(np.median(prob))
    m = predictor.predict_mask(img, threshold=t)
    assert m.shape == (101, 157) and m.dtype == np.uint8
    assert set(np.unique(m)) <= {NON_WALKABLE, WALKABLE}
    np.testing.assert_array_equal(m, (prob >= t).astype(np.uint8))


def test_predict_mask_hud_rects_set_to_ignore():
    p = Predictor(HalfModel("x"), IMAGE_SIZE)
    img = rand_image(101, 157)
    rects = [[0.0, 0.0, 0.2, 0.3], [0.8, 0.85, 1.0, 1.0], [0.48, 0.47, 0.52, 0.53]]
    plain = p.predict_mask(img, 0.5)
    m = p.predict_mask(img, 0.5, hud_rects=rects)
    region = hud_mask(101, 157, rects)
    assert np.all(m[region] == IGNORE_INDEX)
    np.testing.assert_array_equal(m[~region], plain[~region])
    np.testing.assert_array_equal(m, apply_hud_ignore(plain, rects))
    assert set(np.unique(m)) == {NON_WALKABLE, WALKABLE, IGNORE_INDEX}
    # 空 rects / None 不改变结果
    np.testing.assert_array_equal(p.predict_mask(img, 0.5, hud_rects=[]), plain)
    np.testing.assert_array_equal(p.predict_mask(img, 0.5, hud_rects=None), plain)


@pytest.mark.parametrize("t", [-0.1, 1.5, float("nan")])
def test_predict_mask_rejects_bad_threshold(t):
    with pytest.raises(ValueError):
        Predictor(ConstModel(), IMAGE_SIZE).predict_mask(rand_image(*IMAGE_SIZE), threshold=t)


def test_predict_mask_rejects_bad_hud_rect():
    with pytest.raises(ValueError):
        Predictor(ConstModel(), IMAGE_SIZE).predict_mask(rand_image(*IMAGE_SIZE), hud_rects=[[0.1, 0.2, 0.3]])


# ---------------------------------------------------------------- overlay


def _scene():
    """80x120 图：可走矩形 rows 20..59 × cols 30..89；HUD 忽略区 rows 0..9 × cols 100..119。"""
    img = rand_image(80, 120, seed=7)
    mask = np.zeros((80, 120), np.uint8)
    mask[20:60, 30:90] = WALKABLE
    mask[0:10, 100:120] = IGNORE_INDEX
    return img, mask


def test_overlay_does_not_mutate_and_returns_new_array():
    img, mask = _scene()
    img0, mask0 = img.copy(), mask.copy()
    out = overlay(img, mask)
    np.testing.assert_array_equal(img, img0)
    np.testing.assert_array_equal(mask, mask0)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out is not img and not np.shares_memory(out, img)
    # 掩码全 0：内容与原图相同，但仍是新数组
    out0 = overlay(img, np.zeros((80, 120), np.uint8))
    np.testing.assert_array_equal(out0, img)
    assert not np.shares_memory(out0, img)


def test_overlay_colors_walkable_interior():
    img, mask = _scene()
    color, alpha = (0, 255, 0), 0.45
    out = overlay(img, mask, color=color, alpha=alpha)
    interior = np.zeros_like(mask, bool)
    interior[23:57, 33:87] = True  # 远离 2px 轮廓
    expected = np.rint(img.astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha)
    np.testing.assert_array_equal(out[interior], expected[interior].astype(np.uint8))
    # 绿色分量整体提升
    assert out[interior][:, 1].mean() > img[interior][:, 1].mean() + 50


def test_overlay_draws_contour_in_same_color():
    img, mask = _scene()
    color = (255, 0, 255)
    out = overlay(img, mask, color=color, alpha=0.3)
    # 可走矩形的四条边（轮廓线经过的像素）都是纯 color
    for r, c in [(20, 50), (59, 50), (40, 30), (40, 89), (20, 30), (59, 89)]:
        assert tuple(out[r, c]) == color, (r, c)
    # 内部有 "洞" 时洞的边界也描出来（RETR_LIST）
    mask2 = mask.copy()
    mask2[35:45, 55:65] = NON_WALKABLE
    out2 = overlay(img, mask2, color=color, alpha=0.3)
    assert tuple(out2[35, 60]) == color
    assert tuple(out2[40, 55]) == color


def test_overlay_leaves_far_non_walkable_unchanged():
    img, mask = _scene()
    out = overlay(img, mask)
    far = np.ones_like(mask, bool)
    far[17:63, 27:93] = False  # 去掉可走区域及其 3px 邻域
    far[mask == IGNORE_INDEX] = False
    np.testing.assert_array_equal(out[far], img[far])


def test_overlay_darkens_ignore_region_by_half():
    img, mask = _scene()
    out = overlay(img, mask)
    ign = mask == IGNORE_INDEX
    diff = np.abs(out[ign].astype(np.float32) - img[ign].astype(np.float32) * 0.5)
    assert diff.max() <= 0.5 + 1e-6
    # 纯白 HUD → 约 128
    white = np.full((10, 10, 3), 255, np.uint8)
    m = np.full((10, 10), IGNORE_INDEX, np.uint8)
    assert np.all(np.abs(overlay(white, m).astype(int) - 128) <= 1)


def test_overlay_alpha_extremes_and_clipping():
    img = rand_image(40, 40, seed=9)
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = WALKABLE
    inner = (slice(13, 27), slice(13, 27))
    np.testing.assert_array_equal(overlay(img, mask, alpha=0.0)[inner], img[inner])
    assert np.all(overlay(img, mask, color=(10, 20, 30), alpha=1.0)[inner] == (10, 20, 30))
    # alpha 越界裁剪，color 越界裁剪
    np.testing.assert_array_equal(overlay(img, mask, alpha=5.0), overlay(img, mask, alpha=1.0))
    np.testing.assert_array_equal(overlay(img, mask, color=(300, -5, 0)), overlay(img, mask, color=(255, 0, 0)))


def test_overlay_accepts_bool_and_int_masks():
    img = rand_image(30, 30)
    m_u8 = np.zeros((30, 30), np.uint8)
    m_u8[5:25, 5:25] = 1
    ref = overlay(img, m_u8)
    np.testing.assert_array_equal(overlay(img, m_u8.astype(bool)), ref)
    np.testing.assert_array_equal(overlay(img, m_u8.astype(np.int64)), ref)


def test_overlay_rejects_bad_inputs():
    img = rand_image(30, 40)
    with pytest.raises(ValueError):
        overlay(img, np.zeros((30, 41), np.uint8))
    with pytest.raises(ValueError):
        overlay(img, np.zeros((30, 40, 1), np.uint8))
    with pytest.raises(ValueError):
        overlay(img[:, :, 0], np.zeros((30, 40), np.uint8))
    with pytest.raises(ValueError):
        overlay(img, np.zeros((30, 40), np.uint8), color=(1, 2))
    with pytest.raises(ValueError):
        overlay(img, np.zeros((30, 40), np.uint8), alpha=float("nan"))


# ---------------------------------------------------------------- infer_images


def test_infer_images_writes_overlay_and_mask(tmp_path, predictor):
    src = tmp_path / "src"
    a, b = rand_image(101, 157, seed=11), rand_image(64, 96, seed=12)
    write_image_rgb(src / "a.png", a)
    write_image_rgb(src / "b.jpg", b)
    out_dir = tmp_path / "out" / "nested"
    rects = [[0.0, 0.0, 0.25, 0.25]]

    written = infer_images(predictor, [src / "a.png", str(src / "b.jpg")], out_dir, threshold=0.5, hud_rects=rects)
    assert written == [out_dir / "a_overlay.jpg", out_dir / "b_overlay.jpg"]
    assert sorted(p.name for p in out_dir.iterdir()) == ["a_mask.png", "a_overlay.jpg", "b_mask.png", "b_overlay.jpg"]

    for stem, src_path in [("a", src / "a.png"), ("b", src / "b.jpg")]:
        image = read_image_rgb(src_path)
        mask = read_mask(out_dir / f"{stem}_mask.png")
        vis = read_image_rgb(out_dir / f"{stem}_overlay.jpg")
        assert mask.shape == image.shape[:2]
        assert vis.shape == image.shape
        assert set(np.unique(mask)) <= {NON_WALKABLE, WALKABLE, IGNORE_INDEX}
        assert np.all(mask[hud_mask(*image.shape[:2], rects)] == IGNORE_INDEX)
        np.testing.assert_array_equal(mask, predictor.predict_mask(image, 0.5, rects))


def test_infer_images_without_mask_and_directory_input(tmp_path):
    src = tmp_path / "frames"
    for i in range(3):
        write_image_rgb(src / f"f{i:02d}.png", rand_image(40, 60, seed=i))
    (src / "notes.txt").write_text("not an image", encoding="utf-8")
    p = Predictor(ConstModel(5.0), IMAGE_SIZE)
    out_dir = tmp_path / "vis"
    # 目录展开 + 与目录中某个文件重复列出 → 只处理一次，按出现顺序
    written = infer_images(p, [src, src / "f01.png"], out_dir, save_mask=False)
    assert [w.name for w in written] == ["f00_overlay.jpg", "f01_overlay.jpg", "f02_overlay.jpg"]
    assert not list(out_dir.glob("*_mask.png"))
    assert p.model.calls == 3


def test_infer_images_colors_walkable(tmp_path):
    write_image_rgb(tmp_path / "gray.png", np.full((50, 70, 3), 100, np.uint8))
    p = Predictor(ConstModel(10.0), IMAGE_SIZE)  # 全部可走
    [ov] = infer_images(p, [tmp_path / "gray.png"], tmp_path / "out", color=(255, 0, 0), alpha=0.5)
    vis = read_image_rgb(ov)
    center = vis[10:40, 10:60].astype(int)
    # (100*0.5 + 255*0.5, 100*0.5, 100*0.5) ≈ (178, 50, 50)，JPEG 有损，放宽容差；同时验证 RGB 通道顺序
    assert abs(center[..., 0].mean() - 178) < 6
    assert abs(center[..., 1].mean() - 50) < 6 and abs(center[..., 2].mean() - 50) < 6
    assert np.all(read_mask(tmp_path / "out" / "gray_mask.png") == WALKABLE)


def test_infer_images_unicode_paths(tmp_path):
    src = tmp_path / "截图"
    write_image_rgb(src / "画面一.png", rand_image(33, 55))
    out_dir = tmp_path / "输出"
    written = infer_images(Predictor(ConstModel(), IMAGE_SIZE), [src / "画面一.png"], out_dir)
    assert written == [out_dir / "画面一_overlay.jpg"]
    assert read_mask(out_dir / "画面一_mask.png").shape == (33, 55)
    assert read_image_rgb(written[0]).shape == (33, 55, 3)


def test_infer_images_duplicate_stems_raise_before_writing(tmp_path):
    write_image_rgb(tmp_path / "x" / "a.jpg", rand_image(20, 20))
    write_image_rgb(tmp_path / "y" / "A.png", rand_image(20, 20))
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="stem"):
        infer_images(Predictor(ConstModel(), IMAGE_SIZE), [tmp_path / "x" / "a.jpg", tmp_path / "y" / "A.png"],
                     out_dir)
    assert not out_dir.exists()


def test_infer_images_refuses_to_overwrite_inputs(tmp_path):
    write_image_rgb(tmp_path / "a.png", rand_image(20, 20))
    write_image_rgb(tmp_path / "a_overlay.jpg", rand_image(20, 20))
    before = (tmp_path / "a_overlay.jpg").read_bytes()
    with pytest.raises(ValueError):
        infer_images(Predictor(ConstModel(), IMAGE_SIZE), [tmp_path], tmp_path)
    assert (tmp_path / "a_overlay.jpg").read_bytes() == before


def test_infer_images_skips_undecodable_with_warning(tmp_path):
    write_image_rgb(tmp_path / "good.png", rand_image(20, 30))
    (tmp_path / "broken.jpg").write_bytes(b"definitely not a jpeg")
    out_dir = tmp_path / "out"
    with pytest.warns(UserWarning, match="broken"):
        written = infer_images(Predictor(ConstModel(), IMAGE_SIZE), [tmp_path / "broken.jpg", tmp_path / "good.png"],
                               out_dir)
    assert written == [out_dir / "good_overlay.jpg"]
    assert not (out_dir / "broken_overlay.jpg").exists()


def test_infer_images_missing_input_and_bad_args(tmp_path):
    p = Predictor(ConstModel(), IMAGE_SIZE)
    with pytest.raises(FileNotFoundError):
        infer_images(p, [tmp_path / "nope.png"], tmp_path / "out")
    write_image_rgb(tmp_path / "a.png", rand_image(20, 20))
    with pytest.raises(ValueError):
        infer_images(p, [tmp_path / "a.png"], tmp_path / "out", threshold=2.0)
    with pytest.raises(ValueError):
        infer_images(p, [tmp_path / "a.png"], tmp_path / "out", color=(1, 2, 3, 4))
    assert not (tmp_path / "out").exists()


def test_infer_images_empty_inputs(tmp_path):
    assert infer_images(Predictor(ConstModel(), IMAGE_SIZE), [], tmp_path / "out") == []


# ---------------------------------------------------------------- infer_video


VID_H, VID_W, VID_FPS = 72, 120, 12.0


def make_video(folder: Path, n_frames: int, value_fn=None, name: str = "clip") -> Path:
    """写合成视频：优先 mp4v/.mp4，不可用时退回 MJPG/.avi。每帧为纯色（便于检查颜色）。"""
    folder.mkdir(parents=True, exist_ok=True)
    value_fn = value_fn or (lambda i: (40 + 15 * i) % 256)
    for ext, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        path = folder / f"{name}{ext}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), VID_FPS, (VID_W, VID_H))
        if not writer.isOpened():
            writer.release()
            continue
        for i in range(n_frames):
            frame = np.full((VID_H, VID_W, 3), value_fn(i), np.uint8)
            frame[:, : VID_W // 4] = 200  # 给编码器一点结构
            writer.write(frame)
        writer.release()
        return path
    pytest.skip("当前 OpenCV 没有可用的视频编码器")


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    """读出全部帧（RGB）和 fps。"""
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def test_infer_video_frame_count_size_fps(tmp_path, predictor):
    src = make_video(tmp_path, 10)
    out = tmp_path / "out" / "overlay.mp4"
    result = infer_video(predictor, src, out, hud_rects=[[0.0, 0.0, 0.2, 0.2]])
    assert result == out and out.is_file()
    frames, fps = read_frames(out)
    assert len(frames) == 10
    assert all(f.shape == (VID_H, VID_W, 3) for f in frames)
    assert fps == pytest.approx(VID_FPS, abs=0.01)


def test_infer_video_every_n_reuses_masks(tmp_path, monkeypatch):
    import dfwalk.infer as infer_mod

    src = make_video(tmp_path, 7)
    model = ConstModel(1.0)
    p = Predictor(model, IMAGE_SIZE)
    seen: list[np.ndarray] = []
    real_overlay = infer_mod.overlay

    def recording_overlay(image, mask, *args, **kwargs):
        seen.append(mask)
        return real_overlay(image, mask, *args, **kwargs)

    monkeypatch.setattr(infer_mod, "overlay", recording_overlay)
    infer_video(p, src, tmp_path / "o.mp4", every_n=3)
    assert model.calls == 3  # 第 0、3、6 帧
    assert len(seen) == 7
    assert seen[1] is seen[0] and seen[2] is seen[0]
    assert seen[3] is not seen[0] and seen[4] is seen[3] and seen[5] is seen[3]
    assert seen[6] is not seen[3]
    assert len(read_frames(tmp_path / "o.mp4")[0]) == 7


@pytest.mark.parametrize("every_n, expected_calls", [(1, 5), (2, 3), (10, 1)])
def test_infer_video_max_frames(tmp_path, every_n, expected_calls):
    src = make_video(tmp_path, 12)
    model = ConstModel()
    out = infer_video(Predictor(model, IMAGE_SIZE), src, tmp_path / "o.mp4", every_n=every_n, max_frames=5)
    assert len(read_frames(out)[0]) == 5
    assert model.calls == expected_calls


def test_infer_video_max_frames_larger_than_video(tmp_path):
    src = make_video(tmp_path, 4)
    out = infer_video(Predictor(ConstModel(), IMAGE_SIZE), src, tmp_path / "o.mp4", max_frames=100)
    assert len(read_frames(out)[0]) == 4


def test_infer_video_overlay_colors_and_hud(tmp_path):
    src = make_video(tmp_path, 3, value_fn=lambda i: 120)
    # 全部可走 + 红色：输出偏红（同时验证 RGB/BGR 顺序正确）
    out = infer_video(Predictor(ConstModel(10.0), IMAGE_SIZE), src, tmp_path / "red.mp4", color=(255, 0, 0),
                      alpha=0.6)
    frames, _ = read_frames(out)
    center = np.stack(frames)[:, 20:50, 50:100].astype(float)
    assert center[..., 0].mean() > center[..., 2].mean() + 80
    # 全部不可走 + HUD 左上角：HUD 区压暗约一半，其余基本不变
    out2 = infer_video(Predictor(ConstModel(-10.0), IMAGE_SIZE), src, tmp_path / "hud.mp4",
                       hud_rects=[[0.5, 0.0, 1.0, 0.5]])
    frames2, _ = read_frames(out2)
    f = np.stack(frames2).astype(float)
    hud_px = f[:, 4:30, 66:116].mean()
    other_px = f[:, 42:70, 66:116].mean()
    assert abs(other_px - 120) < 10
    assert abs(hud_px - 60) < 10


def test_infer_video_fps_fallback(tmp_path, monkeypatch):
    import dfwalk.infer as infer_mod

    src = make_video(tmp_path, 3)
    real_open = infer_mod.open_video

    class NoFps:
        def __init__(self, cap):
            self.cap = cap

        def get(self, prop):
            return 0.0 if prop == cv2.CAP_PROP_FPS else self.cap.get(prop)

        def __getattr__(self, name):
            return getattr(self.cap, name)

    monkeypatch.setattr(infer_mod, "open_video", lambda path: NoFps(real_open(path)))
    out = infer_video(Predictor(ConstModel(), IMAGE_SIZE), src, tmp_path / "o.mp4")
    frames, fps = read_frames(out)
    assert len(frames) == 3
    assert fps == pytest.approx(30.0, abs=0.01)


def test_infer_video_bad_input_raises(tmp_path):
    p = Predictor(ConstModel(), IMAGE_SIZE)
    with pytest.raises(ValueError):
        infer_video(p, tmp_path / "missing.mp4", tmp_path / "o.mp4")
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"this is not a video at all")
    with pytest.raises(ValueError):
        infer_video(p, fake, tmp_path / "o2.mp4")
    assert not (tmp_path / "o.mp4").exists() and not (tmp_path / "o2.mp4").exists()


def test_infer_video_writer_failure_raises(tmp_path):
    src = make_video(tmp_path, 3)
    blocked = tmp_path / "taken.mp4"
    blocked.mkdir()  # 输出路径是已存在的目录 → writer 打不开
    with pytest.raises(ValueError):
        infer_video(Predictor(ConstModel(), IMAGE_SIZE), src, blocked)


@pytest.mark.parametrize("kwargs", [
    {"every_n": 0}, {"every_n": -2}, {"every_n": 1.5}, {"every_n": True},
    {"max_frames": 0}, {"max_frames": -1}, {"max_frames": 2.5},
    {"threshold": 1.2}, {"color": (0, 255)}, {"alpha": float("inf")},
])
def test_infer_video_rejects_bad_args(tmp_path, kwargs):
    src = make_video(tmp_path, 2)
    p = Predictor(ConstModel(), IMAGE_SIZE)
    with pytest.raises(ValueError):
        infer_video(p, src, tmp_path / "o.mp4", **kwargs)
    assert p.model.calls == 0
    assert not (tmp_path / "o.mp4").exists()


def test_infer_video_refuses_to_overwrite_source(tmp_path):
    src = make_video(tmp_path, 2)
    before = src.read_bytes()
    with pytest.raises(ValueError):
        infer_video(Predictor(ConstModel(), IMAGE_SIZE), src, src)
    assert src.read_bytes() == before


def test_infer_video_unicode_output_path(tmp_path):
    src = make_video(tmp_path / "录像", 3, name="对局")
    out = infer_video(Predictor(ConstModel(), IMAGE_SIZE), src, tmp_path / "结果" / "叠加.mp4")
    assert len(read_frames(out)[0]) == 3


def test_no_warnings_on_normal_run(tmp_path, predictor):
    write_image_rgb(tmp_path / "a.png", rand_image(40, 60))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        infer_images(predictor, [tmp_path / "a.png"], tmp_path / "out")
