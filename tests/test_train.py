"""dfwalk.train 的单元测试：学习率调度、DataLoader 构建、训练/评估循环、完整训练流程与 checkpoint 评估。

全部离线、只用 CPU：编码器用 resnet18 + encoder_weights=None，数据为测试内合成的 64x96 小图
（下方为“可走”的绿色地面，上方为灰色“不可走”区域，中间一条 255 忽略带）。
测试期间禁止任何网络连接。
"""

from __future__ import annotations

import copy
import csv
import math
import socket
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from dfwalk import IGNORE_INDEX, NON_WALKABLE, NUM_CLASSES, WALKABLE
from dfwalk import train as train_mod
from dfwalk.checkpoint import CKPT_FORMAT, load_checkpoint, model_from_checkpoint
from dfwalk.config import Config, load_config
from dfwalk.dataset import find_pairs, group_of, make_splits, read_split, write_split
from dfwalk.losses import SegLoss
from dfwalk.metrics import ConfusionMatrix
from dfwalk.model import model_config
from dfwalk.train import (
    build_loaders,
    default_warmup_steps,
    evaluate,
    evaluate_checkpoint,
    make_lr_lambda,
    train,
    train_one_epoch,
)
from dfwalk.transforms import EvalTransform, TrainTransform
from dfwalk.utils import write_image_rgb, write_mask

H, W = 64, 96
ENCODER = "resnet18"
METRIC_KEYS = list(ConfusionMatrix().compute().keys())
HISTORY_COLUMNS = ["epoch", "lr", "train_loss", "val_loss", *METRIC_KEYS]

CPU = torch.device("cpu")


# ---------------------------------------------------------------- 公共夹具与工具


def _guard(*args, **kwargs):
    raise RuntimeError("测试中禁止联网")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """任何尝试联网（例如误下载预训练权重）都直接失败。"""
    monkeypatch.setattr(socket.socket, "connect", _guard)
    monkeypatch.setattr(socket, "create_connection", _guard)


def _limit_threads():
    """限制 torch 线程数：CI/多进程并发跑测试时，满核 OpenMP 线程互相抢占会让小模型慢几十倍。"""
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, min(2, old)))
    return old


@pytest.fixture(autouse=True)
def _threads():
    old = _limit_threads()
    yield
    torch.set_num_threads(old)


def make_dataset(
    root: Path,
    videos: tuple[str, ...] = ("vidA", "vidB"),
    frames: int = 4,
    h: int = H,
    w: int = W,
    seed: int = 0,
) -> list[str]:
    """在 root/images 与 root/masks 下合成数据：每帧上方灰色（不可走）、下方绿色（可走），
    分界线位置逐帧变化，分界处 4 行标为 IGNORE_INDEX。返回 stem 列表（排序）。"""
    rng = np.random.default_rng(seed)
    stems = []
    for video in videos:
        for i in range(frames):
            stem = f"{video}_{i:06d}"
            boundary = h // 2 - 6 + (i * 5) % 14
            image = np.empty((h, w, 3), dtype=np.int16)
            image[:boundary] = (110, 110, 120)
            image[boundary:] = (40, 190, 60)
            image += rng.integers(-12, 13, size=image.shape, dtype=np.int16)
            write_image_rgb(root / "images" / f"{stem}.jpg", np.clip(image, 0, 255).astype(np.uint8))
            mask = np.full((h, w), NON_WALKABLE, dtype=np.uint8)
            mask[boundary:] = WALKABLE
            mask[boundary - 2:boundary + 2] = IGNORE_INDEX
            write_mask(root / "masks" / f"{stem}.png", mask)
            stems.append(stem)
    return sorted(stems)


def make_cfg(root: Path, out_dir: Path, **train_overrides) -> Config:
    cfg = Config()
    cfg.data.root = root  # 故意传 Path（而非 str），train 写 config.yaml 时需要能处理
    cfg.data.image_size = [H, W]
    cfg.train.encoder = ENCODER
    cfg.train.encoder_weights = None
    cfg.train.epochs = 2
    cfg.train.batch_size = 2
    cfg.train.num_workers = 0
    cfg.train.device = "cpu"
    cfg.train.out_dir = str(out_dir)
    for key, value in train_overrides.items():
        setattr(cfg.train, key, value)
    return cfg


def read_history(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def assert_metrics_equal(a: dict[str, float], b: dict[str, float]) -> None:
    assert set(a) == set(b)
    for key in a:
        if math.isnan(a[key]):
            assert math.isnan(b[key]), key
        else:
            assert a[key] == pytest.approx(b[key], rel=1e-6, abs=1e-9), key


# ---------------------------------------------------------------- 学习率调度


@pytest.mark.parametrize(
    "total, expected",
    [(0, 0), (1, 0), (9, 0), (10, 1), (19, 1), (1000, 100), (4999, 499), (5000, 500), (10**6, 500)],
)
def test_default_warmup_steps(total, expected):
    assert default_warmup_steps(total) == expected


def test_lr_lambda_linear_warmup_then_cosine():
    f = make_lr_lambda(1000)  # warmup = 100
    # warmup：(step + 1) / 100，线性、第一步非零、最后一步到 1
    assert f(0) == pytest.approx(0.01)
    assert f(49) == pytest.approx(0.5)
    assert f(99) == pytest.approx(1.0)
    warm = np.array([f(s) for s in range(100)])
    assert np.allclose(np.diff(warm), 0.01)
    # 余弦段：从 1 单调下降到 0
    assert f(100) == pytest.approx(1.0)
    assert f(550) == pytest.approx(0.5)
    assert f(325) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.25)))
    assert 0.0 < f(999) < 1e-4
    assert f(1000) == pytest.approx(0.0, abs=1e-12)
    decay = np.array([f(s) for s in range(100, 1001)])
    assert np.all(np.diff(decay) <= 1e-12)
    assert np.all((decay >= 0.0) & (decay <= 1.0))
    # 超出总步数保持 0
    assert f(1001) == pytest.approx(0.0, abs=1e-12)
    assert f(10**6) == pytest.approx(0.0, abs=1e-12)


def test_lr_lambda_warmup_capped_at_500():
    f = make_lr_lambda(20000)
    assert f(0) == pytest.approx(1 / 500)
    assert f(498) < 1.0
    assert f(499) == pytest.approx(1.0)
    assert f(500) == pytest.approx(1.0)
    assert f(501) < 1.0


def test_lr_lambda_without_warmup_for_tiny_runs():
    f = make_lr_lambda(5)  # 5 // 10 == 0 → 没有 warmup，直接余弦
    values = [f(s) for s in range(6)]
    assert values[0] == pytest.approx(1.0)
    assert values[5] == pytest.approx(0.0, abs=1e-12)
    assert all(a > b for a, b in zip(values, values[1:]))


def test_lr_lambda_explicit_warmup():
    f = make_lr_lambda(10, warmup_steps=4)
    assert [f(s) for s in range(4)] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert f(4) == pytest.approx(1.0)
    assert f(7) == pytest.approx(0.5)
    assert f(10) == pytest.approx(0.0, abs=1e-12)
    g = make_lr_lambda(10, warmup_steps=0)
    assert g(0) == pytest.approx(1.0)


@pytest.mark.parametrize("total, warmup", [(0, None), (-5, None), (10, 10), (10, 11), (10, -1)])
def test_lr_lambda_invalid_arguments(total, warmup):
    with pytest.raises(ValueError):
        make_lr_lambda(total, warmup_steps=warmup)


def test_lr_lambda_with_lambdalr_steps_per_iteration():
    param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(20))  # warmup = 2
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.05)
    assert lrs[1] == pytest.approx(0.1)
    assert lrs[2] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(lrs[1:], lrs[2:]))
    assert lrs[-1] > 0.0
    # 训练结束（第 20 次 step 之后）学习率退火到 0
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- build_loaders


def test_build_loaders_creates_splits_by_video(tmp_path, capsys):
    stems = make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run")
    assert not (tmp_path / "splits").exists()

    train_loader, val_loader = build_loaders(cfg)

    assert (tmp_path / "splits" / "train.txt").is_file()
    assert (tmp_path / "splits" / "val.txt").is_file()
    train_stems, val_stems = read_split(tmp_path, "train"), read_split(tmp_path, "val")
    expected = make_splits(find_pairs(tmp_path), cfg.data.val_ratio, cfg.data.seed)
    assert (train_stems, val_stems) == expected
    assert sorted(train_stems + val_stems) == stems
    # 按视频分组：同一视频的帧不会同时出现在两边
    assert not {group_of(s) for s in train_stems} & {group_of(s) for s in val_stems}
    assert train_stems and val_stems
    assert "train.txt" in capsys.readouterr().out

    assert train_loader.dataset.stems == train_stems
    assert val_loader.dataset.stems == val_stems
    assert isinstance(train_loader.dataset.transform, TrainTransform)
    assert train_loader.dataset.transform.size == (H, W)
    assert isinstance(val_loader.dataset.transform, EvalTransform)
    assert val_loader.dataset.transform.size == (H, W)


def test_build_loaders_loader_settings_and_batches(tmp_path):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run")
    train_loader, val_loader = build_loaders(cfg)

    assert train_loader.batch_size == val_loader.batch_size == 2
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert train_loader.drop_last is True  # 4 个训练样本 > batch_size 2
    assert val_loader.drop_last is False
    for loader in (train_loader, val_loader):
        assert loader.num_workers == 0
        assert loader.persistent_workers is False
        assert loader.pin_memory == torch.cuda.is_available()

    images, masks = next(iter(train_loader))
    assert images.shape == (2, 3, H, W) and images.dtype == torch.float32
    assert masks.shape == (2, H, W) and masks.dtype == torch.int64
    assert set(torch.unique(masks).tolist()) <= {NON_WALKABLE, WALKABLE, IGNORE_INDEX}
    images, masks = next(iter(val_loader))
    assert images.shape == (2, 3, H, W)
    assert set(torch.unique(masks).tolist()) == {NON_WALKABLE, WALKABLE, IGNORE_INDEX}


def test_build_loaders_drop_last_false_when_train_set_fits_in_one_batch(tmp_path):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run", batch_size=4)  # 训练集正好 4 帧
    train_loader, _ = build_loaders(cfg)
    assert len(train_loader.dataset) == 4
    assert train_loader.drop_last is False
    assert len(train_loader) == 1

    cfg = make_cfg(tmp_path, tmp_path / "run", batch_size=8)
    train_loader, _ = build_loaders(cfg)
    assert train_loader.drop_last is False
    assert len(train_loader) == 1  # 不足一个 batch 也不会被丢掉


def test_build_loaders_persistent_workers_when_num_workers_positive(tmp_path):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run", num_workers=2)
    train_loader, val_loader = build_loaders(cfg)  # 不迭代，不会真正启动子进程
    for loader in (train_loader, val_loader):
        assert loader.num_workers == 2
        assert loader.persistent_workers is True


def test_build_loaders_respects_existing_splits(tmp_path, capsys):
    make_dataset(tmp_path)
    train_stems = ["vidA_000000", "vidA_000001", "vidB_000003"]
    val_stems = ["vidB_000000"]
    write_split(tmp_path, "train", train_stems)
    write_split(tmp_path, "val", val_stems)
    before = (tmp_path / "splits" / "train.txt").read_bytes(), (tmp_path / "splits" / "val.txt").read_bytes()

    train_loader, val_loader = build_loaders(make_cfg(tmp_path, tmp_path / "run"))

    assert train_loader.dataset.stems == train_stems
    assert val_loader.dataset.stems == val_stems
    after = (tmp_path / "splits" / "train.txt").read_bytes(), (tmp_path / "splits" / "val.txt").read_bytes()
    assert after == before
    assert capsys.readouterr().out == ""


def test_build_loaders_regenerates_both_when_one_split_missing(tmp_path):
    make_dataset(tmp_path)
    write_split(tmp_path, "train", ["vidA_000000"])  # val.txt 缺失
    cfg = make_cfg(tmp_path, tmp_path / "run")

    train_loader, val_loader = build_loaders(cfg)

    expected_train, expected_val = make_splits(find_pairs(tmp_path), cfg.data.val_ratio, cfg.data.seed)
    assert read_split(tmp_path, "train") == expected_train
    assert read_split(tmp_path, "val") == expected_val
    assert train_loader.dataset.stems == expected_train
    assert val_loader.dataset.stems == expected_val


def test_build_loaders_split_uses_data_seed_and_ratio(tmp_path):
    make_dataset(tmp_path, videos=("a", "b", "c", "d", "e"), frames=2)
    cfg = make_cfg(tmp_path, tmp_path / "run")
    cfg.data.val_ratio = 0.4
    cfg.data.seed = 7
    build_loaders(cfg)
    expected = make_splits(find_pairs(tmp_path), 0.4, 7)
    assert (read_split(tmp_path, "train"), read_split(tmp_path, "val")) == expected


@pytest.mark.parametrize("empty", ["train", "val"])
def test_build_loaders_empty_split_raises(tmp_path, empty):
    make_dataset(tmp_path)
    write_split(tmp_path, "train", [] if empty == "train" else ["vidA_000000"])
    write_split(tmp_path, "val", [] if empty == "val" else ["vidB_000000"])
    with pytest.raises(ValueError):
        build_loaders(make_cfg(tmp_path, tmp_path / "run"))


def test_build_loaders_without_masks_raises_value_error(tmp_path):
    make_dataset(tmp_path)
    for p in (tmp_path / "masks").iterdir():
        p.unlink()
    with pytest.raises(ValueError):
        build_loaders(make_cfg(tmp_path, tmp_path / "run"))
    assert not (tmp_path / "splits" / "train.txt").exists()


def test_build_loaders_without_images_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_loaders(make_cfg(tmp_path, tmp_path / "run"))


def test_build_loaders_split_with_missing_files_raises(tmp_path):
    make_dataset(tmp_path)
    write_split(tmp_path, "train", ["vidA_000000", "ghost_000001"])
    write_split(tmp_path, "val", ["vidB_000000"])
    with pytest.raises(FileNotFoundError, match="ghost_000001"):
        build_loaders(make_cfg(tmp_path, tmp_path / "run"))


@pytest.mark.parametrize("key, value", [("batch_size", 0), ("num_workers", -1)])
def test_build_loaders_invalid_loader_settings(tmp_path, key, value):
    make_dataset(tmp_path)
    with pytest.raises(ValueError):
        build_loaders(make_cfg(tmp_path, tmp_path / "run", **{key: value}))


# ---------------------------------------------------------------- train_one_epoch


def tiny_loader(n: int = 5, batch_size: int = 2, size: int = 8, seed: int = 0) -> DataLoader:
    """n 个样本的 TensorDataset：图像通道 0 为正的位置是可走（1），并带一些忽略像素。"""
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, size, size, generator=g)
    masks = (images[:, 0] > 0).long()
    masks[:, 0, :] = IGNORE_INDEX
    return DataLoader(TensorDataset(images, masks), batch_size=batch_size, shuffle=False)


def tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Conv2d(3, NUM_CLASSES, kernel_size=1)


class BatchSizeLoss(nn.Module):
    """loss 值恰为 batch 中的样本数（且挂在计算图上），用来检验“按样本数加权平均”。"""

    def forward(self, logits, target):
        return logits.sum() * 0.0 + float(logits.shape[0])


class NanAtLoss(nn.Module):
    """第 bad_call 次调用（从 1 计数）返回 bad_value，其余调用为普通交叉熵。"""

    def __init__(self, bad_call: int, bad_value: float = math.nan) -> None:
        super().__init__()
        self.bad_call = bad_call
        self.bad_value = bad_value
        self.calls = 0

    def forward(self, logits, target):
        self.calls += 1
        loss = nn.functional.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
        if self.calls == self.bad_call:
            return loss * self.bad_value
        return loss


class FakeScaler:
    """记录 GradScaler 协议调用顺序的替身。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def scale(self, loss):
        self.calls.append("scale")
        return loss * 2.0

    def step(self, optimizer):
        self.calls.append("step")
        optimizer.step()

    def update(self):
        self.calls.append("update")


def snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def params_equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_train_one_epoch_returns_sample_weighted_mean():
    model = tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    # batch 大小 2, 2, 1 → loss 2, 2, 1；按样本加权 (2*2 + 2*2 + 1*1) / 5 = 1.8（简单平均会是 5/3）
    loss = train_one_epoch(model, tiny_loader(5, 2), BatchSizeLoss(), opt, CPU)
    assert loss == pytest.approx(1.8)


def test_train_one_epoch_updates_params_steps_scheduler_and_sets_train_mode():
    model = tiny_model()
    model.eval()
    before = snapshot(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(100))
    loader = tiny_loader(5, 2)

    loss = train_one_epoch(model, loader, SegLoss(), opt, CPU, scheduler=sched)

    assert model.training is True
    assert math.isfinite(loss) and loss > 0
    assert not params_equal(before, snapshot(model))
    assert sched.last_epoch == len(loader) == 3  # 每个 iteration step 一次


def test_train_one_epoch_uses_fresh_gradients_each_step():
    """每个 iteration 前都要清零梯度：结果必须与“每步 zero_grad → backward → step”的参考循环完全一致。
    漏掉 zero_grad 时梯度跨 iteration 累积，参数就会不同（而 loss 照样下降，其他测试发现不了）。"""
    loader = tiny_loader(6, 2)
    model = tiny_model()
    reference = copy.deepcopy(model)

    train_one_epoch(model, loader, SegLoss(), torch.optim.SGD(model.parameters(), lr=0.5), CPU)

    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.5)
    ref_loss = SegLoss()
    reference.train()
    for images, masks in loader:
        ref_opt.zero_grad()
        ref_loss(reference(images), masks).backward()
        ref_opt.step()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q)


def test_train_one_epoch_reduces_loss_over_epochs():
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
    loader = tiny_loader(8, 4)
    losses = [train_one_epoch(model, loader, SegLoss(), opt, CPU) for _ in range(15)]
    assert losses[-1] < 0.5 * losses[0]


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_train_one_epoch_non_finite_loss_raises_with_iteration(bad_value):
    model = tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    loss_fn = NanAtLoss(bad_call=2, bad_value=bad_value)
    with pytest.raises(FloatingPointError, match=r"iteration 2/3"):
        train_one_epoch(model, tiny_loader(5, 2), loss_fn, opt, CPU, scheduler=sched)
    assert sched.last_epoch == 1  # 只有第 1 个 iteration 完成了
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_train_one_epoch_nan_on_first_iteration_does_not_touch_weights():
    model = tiny_model()
    before = snapshot(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    with pytest.raises(FloatingPointError, match="iteration 1"):
        train_one_epoch(model, tiny_loader(5, 2), NanAtLoss(bad_call=1), opt, CPU)
    assert params_equal(before, snapshot(model))
    assert all(p.grad is None for p in model.parameters())


def test_train_one_epoch_accepts_loader_without_len():
    model = tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = list(tiny_loader(5, 2))
    assert train_one_epoch(model, iter(batches), BatchSizeLoss(), opt, CPU) == pytest.approx(1.8)
    with pytest.raises(FloatingPointError, match=r"iteration 1\b"):
        train_one_epoch(model, iter(batches), NanAtLoss(bad_call=1), opt, CPU)


def test_train_one_epoch_nan_from_model_output_raises():
    class NanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, NUM_CLASSES, 1)

        def forward(self, x):
            return self.conv(x) * math.nan

    model = NanModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(FloatingPointError):
        train_one_epoch(model, tiny_loader(4, 2), SegLoss(), opt, CPU)


def test_train_one_epoch_scaler_protocol():
    model = tiny_model()
    before = snapshot(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    scaler = FakeScaler()
    with warnings.catch_warnings():
        # 纯 CPU 环境下 torch.autocast("cuda") 会提示 CUDA 不可用并自动禁用
        warnings.simplefilter("ignore", UserWarning)
        loss = train_one_epoch(model, tiny_loader(5, 2), BatchSizeLoss(), opt, CPU,
                               scheduler=sched, scaler=scaler)
    assert scaler.calls == ["scale", "step", "update"] * 3
    assert loss == pytest.approx(1.8)  # 返回的是未缩放的 loss
    assert sched.last_epoch == 3
    assert params_equal(before, snapshot(model))  # BatchSizeLoss 的梯度为 0（SGD 无 weight_decay）

    model = tiny_model()
    before = snapshot(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        train_one_epoch(model, tiny_loader(5, 2), SegLoss(), opt, CPU, scaler=FakeScaler())
    assert not params_equal(before, snapshot(model))


@pytest.mark.skipif(torch.cuda.is_available(), reason="只在无 CUDA 时验证 GradScaler 自动禁用后的路径")
def test_train_one_epoch_with_real_grad_scaler_on_cpu():
    model = tiny_model()
    before = snapshot(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scaler = torch.amp.GradScaler("cuda")  # 无 CUDA 时自动禁用：scale 原样返回，step 直接调用 optimizer.step
        loss = train_one_epoch(model, tiny_loader(5, 2), SegLoss(), opt, CPU, scaler=scaler)
    assert math.isfinite(loss)
    assert not params_equal(before, snapshot(model))


def test_train_one_epoch_empty_loader_raises():
    model = tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(TensorDataset(torch.zeros(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.long)),
                        batch_size=2, drop_last=True)
    with pytest.raises(ValueError):
        train_one_epoch(model, loader, SegLoss(), opt, CPU)


# ---------------------------------------------------------------- evaluate


class EchoModel(nn.Module):
    """logits = [-x0, x0]：argmax 为 1 当且仅当图像通道 0 > 0。记录前向时的 training / grad 状态。"""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)  # 只用于检验子模块的 train/eval 状态
        self.seen_training: list[bool] = []
        self.seen_grad: list[bool] = []

    def forward(self, x):
        self.seen_training.append(self.training)
        self.seen_grad.append(torch.is_grad_enabled())
        c0 = x[:, :1]
        return torch.cat([-c0, c0], dim=1)


def echo_loader(n: int = 5, batch_size: int = 2, size: int = 8, seed: int = 1) -> tuple[DataLoader, torch.Tensor,
                                                                                       torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, size, size, generator=g)
    pred = (images[:, 0] > 0).long()
    # 真值：与预测大部分一致，第一行翻转制造错误，最后一行忽略
    masks = pred.clone()
    masks[:, 0, :] = 1 - masks[:, 0, :]
    masks[:, -1, :] = IGNORE_INDEX
    loader = DataLoader(TensorDataset(images, masks), batch_size=batch_size, shuffle=False)
    return loader, pred, masks


def test_evaluate_matches_confusion_matrix():
    loader, pred, masks = echo_loader()
    expected = ConfusionMatrix()
    expected.update(pred, masks)

    metrics = evaluate(EchoModel(), loader, CPU)

    assert list(metrics) == METRIC_KEYS  # 没有 loss_fn 时不含 "loss"
    assert all(isinstance(v, float) for v in metrics.values())
    assert_metrics_equal(metrics, expected.compute())
    assert 0.0 < metrics["iou_walkable"] < 1.0


def test_evaluate_loss_is_sample_weighted():
    loader, _, _ = echo_loader(n=5, batch_size=2)
    metrics = evaluate(EchoModel(), loader, CPU, loss_fn=BatchSizeLoss())
    assert metrics["loss"] == pytest.approx(1.8)
    assert set(metrics) == set(METRIC_KEYS) | {"loss"}


def test_evaluate_runs_without_grad_in_eval_mode():
    model = EchoModel()
    model.train()
    evaluate(model, echo_loader()[0], CPU, loss_fn=SegLoss())
    assert model.seen_training == [False, False, False]
    assert model.seen_grad == [False, False, False]


@pytest.mark.parametrize("initial_training", [True, False])
def test_evaluate_restores_train_eval_mode(initial_training):
    model = EchoModel()
    model.train(initial_training)
    evaluate(model, echo_loader()[0], CPU)
    assert model.training is initial_training
    assert model.bn.training is initial_training


def test_evaluate_restores_mixed_submodule_modes():
    model = EchoModel()
    model.train()
    model.bn.eval()  # 例如单独冻结的 BatchNorm
    evaluate(model, echo_loader()[0], CPU)
    assert model.training is True
    assert model.bn.training is False


def test_evaluate_restores_mode_on_exception():
    class Boom(EchoModel):
        def forward(self, x):
            super().forward(x)
            raise RuntimeError("boom")

    model = Boom()
    model.train()
    with pytest.raises(RuntimeError, match="boom"):
        evaluate(model, echo_loader()[0], CPU)
    assert model.training is True and model.bn.training is True
    assert model.seen_training == [False]


def test_evaluate_empty_loader_returns_nan():
    loader = DataLoader(TensorDataset(torch.zeros(0, 3, 8, 8), torch.zeros(0, 8, 8, dtype=torch.long)),
                        batch_size=2)
    metrics = evaluate(EchoModel(), loader, CPU, loss_fn=SegLoss())
    assert set(metrics) == set(METRIC_KEYS) | {"loss"}
    assert all(math.isnan(v) for v in metrics.values())


def test_evaluate_all_ignored_targets():
    images = torch.randn(3, 3, 8, 8)
    masks = torch.full((3, 8, 8), IGNORE_INDEX, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, masks), batch_size=2)
    metrics = evaluate(EchoModel(), loader, CPU, loss_fn=SegLoss())
    assert math.isnan(metrics["iou_walkable"]) and math.isnan(metrics["pixel_acc"])
    assert metrics["loss"] == pytest.approx(0.0)


# ---------------------------------------------------------------- train()：完整流程（真实模型）


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """真实训练 2 个 epoch（resnet18 Unet，64x96，8 帧 / 2 段视频），整个模块共用一次结果。"""
    old = _limit_threads()
    try:
        root = tmp_path_factory.mktemp("df_data")
        make_dataset(root)
        cfg = make_cfg(root, root / "runs" / "exp")
        best = train(cfg)
    finally:
        torch.set_num_threads(old)
    return cfg, best


def test_train_writes_all_outputs(trained_run):
    cfg, best = trained_run
    out_dir = Path(cfg.train.out_dir)
    assert best == out_dir / "best.pt"
    for name in ("best.pt", "last.pt", "history.csv", "config.yaml"):
        assert (out_dir / name).is_file(), name
    assert not list(out_dir.glob("*.tmp"))
    # splits 自动生成
    root = Path(cfg.data.root)
    assert read_split(root, "train") and read_split(root, "val")


def test_train_history_csv(trained_run):
    cfg, _ = trained_run
    columns, rows = read_history(Path(cfg.train.out_dir) / "history.csv")
    assert columns == HISTORY_COLUMNS
    assert len(rows) == 2
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    for row in rows:
        for key in ("train_loss", "val_loss", "pixel_acc"):
            assert math.isfinite(float(row[key])), key
        assert 0.0 <= float(row["pixel_acc"]) <= 1.0
    # 每个 epoch 2 个 iteration，总 4 步、无 warmup：epoch 1 用 base lr，epoch 2 起点在余弦一半处
    assert float(rows[0]["lr"]) == pytest.approx(cfg.train.lr)
    assert float(rows[1]["lr"]) == pytest.approx(cfg.train.lr * make_lr_lambda(4)(2))


def test_train_config_yaml_roundtrip(trained_run):
    cfg, _ = trained_run
    path = Path(cfg.train.out_dir) / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["data"]["root"] == str(cfg.data.root)
    assert raw["train"]["encoder_weights"] is None
    loaded = load_config(path)
    expected = cfg.to_dict()
    expected["data"]["root"] = str(cfg.data.root)
    assert loaded.to_dict() == expected


def test_train_best_checkpoint_loads_and_predicts(trained_run):
    cfg, best = trained_run
    state = load_checkpoint(best)
    assert state["format"] == CKPT_FORMAT
    assert state["image_size"] == [H, W]
    assert state["model_cfg"] == model_config("Unet", ENCODER, NUM_CLASSES)
    assert state["epoch"] in (1, 2)
    assert {"train_loss", "val_loss", *METRIC_KEYS} <= set(state["metrics"])

    model = model_from_checkpoint(state)
    assert model.training is False
    with torch.no_grad():
        logits = model(torch.randn(1, 3, H, W))
    assert logits.shape == (1, NUM_CLASSES, H, W)
    assert torch.isfinite(logits).all()

    # best.pt 指向 iou_walkable 最高（并列取最早）的 epoch，指标与 history 一致
    _, rows = read_history(Path(cfg.train.out_dir) / "history.csv")
    scores = [float(r["iou_walkable"]) for r in rows]
    scores = [-math.inf if math.isnan(s) else s for s in scores]
    best_epoch = 1 + int(np.argmax(scores))
    assert state["epoch"] == best_epoch
    row = rows[best_epoch - 1]
    assert state["metrics"]["val_loss"] == pytest.approx(float(row["val_loss"]))
    assert state["metrics"]["train_loss"] == pytest.approx(float(row["train_loss"]))


def test_train_last_checkpoint_is_final_epoch(trained_run):
    cfg, _ = trained_run
    state = load_checkpoint(Path(cfg.train.out_dir) / "last.pt")
    assert state["epoch"] == cfg.train.epochs
    _, rows = read_history(Path(cfg.train.out_dir) / "history.csv")
    assert state["metrics"]["val_loss"] == pytest.approx(float(rows[-1]["val_loss"]))


def test_evaluate_checkpoint_on_val_matches_manual_evaluation(trained_run):
    cfg, best = trained_run
    metrics = evaluate_checkpoint(cfg, best)
    assert set(metrics) == set(METRIC_KEYS) | {"loss"}
    assert 0.0 <= metrics["pixel_acc"] <= 1.0

    state = load_checkpoint(best)
    model = model_from_checkpoint(state)
    from dfwalk.dataset import WalkableDataset

    ds = WalkableDataset(cfg.data.root, read_split(cfg.data.root, "val"), EvalTransform(state["image_size"]))
    # Dice 项按 batch 计算，loss 与 batch 划分有关，因此用与 cfg 相同的 batch_size
    manual = evaluate(model, DataLoader(ds, batch_size=cfg.train.batch_size), CPU, loss_fn=SegLoss())
    assert_metrics_equal(metrics, manual)
    # 与训练时记录在 checkpoint 里的验证指标一致
    assert metrics["loss"] == pytest.approx(state["metrics"]["val_loss"], rel=1e-5)
    assert metrics["miou"] == pytest.approx(state["metrics"]["miou"], rel=1e-6, nan_ok=True)


def test_evaluate_checkpoint_other_split(trained_run):
    cfg, best = trained_run
    metrics = evaluate_checkpoint(cfg, best, split="train")
    assert set(metrics) == set(METRIC_KEYS) | {"loss"}


def test_evaluate_checkpoint_uses_checkpoint_image_size(trained_run, monkeypatch):
    cfg, best = trained_run
    cfg2 = copy.deepcopy(cfg)
    cfg2.data.image_size = [32, 32]  # 与 checkpoint 不同，不应被使用
    sizes = []
    real = train_mod.EvalTransform

    def spy(image_size, *args, **kwargs):
        sizes.append(list(image_size))
        return real(image_size, *args, **kwargs)

    monkeypatch.setattr(train_mod, "EvalTransform", spy)
    metrics = evaluate_checkpoint(cfg2, best)
    assert sizes == [[H, W]]
    assert_metrics_equal(metrics, evaluate_checkpoint(cfg, best))


def test_evaluate_checkpoint_missing_or_empty_split(trained_run, tmp_path):
    cfg, best = trained_run
    with pytest.raises(FileNotFoundError):
        evaluate_checkpoint(cfg, best, split="does_not_exist")

    cfg2 = copy.deepcopy(cfg)
    make_dataset(tmp_path, frames=1)
    cfg2.data.root = tmp_path
    write_split(tmp_path, "val", [])
    with pytest.raises(ValueError):
        evaluate_checkpoint(cfg2, best)


def test_evaluate_checkpoint_rejects_foreign_file(trained_run, tmp_path):
    cfg, _ = trained_run
    bogus = tmp_path / "bogus.pt"
    torch.save({"model_state": {}}, bogus)
    with pytest.raises(ValueError):
        evaluate_checkpoint(cfg, bogus)


# ---------------------------------------------------------------- train()：流程控制（打桩训练/评估）


def _scripted_metrics(iou: float) -> dict[str, float]:
    metrics = {k: 0.5 for k in METRIC_KEYS}
    metrics["iou_walkable"] = iou
    metrics["miou"] = iou
    metrics["loss"] = 0.25
    return metrics


def _stub_loops(monkeypatch, ious: list[float]) -> dict:
    """把 train_one_epoch / evaluate 换成脚本化的假实现，记录 save_checkpoint 调用。"""
    record: dict = {"train_args": [], "saves": [], "eval_calls": 0}
    script = iter(ious)

    def fake_train_one_epoch(model, loader, loss_fn, optimizer, device, scheduler=None, scaler=None):
        record["train_args"].append({"scheduler": scheduler, "scaler": scaler, "loss_fn": loss_fn,
                                     "device": device, "model_training": model.training})
        for _ in range(len(loader)):
            optimizer.step()
            scheduler.step()
        return 1.0

    def fake_evaluate(model, loader, device, loss_fn=None):
        record["eval_calls"] += 1
        assert loss_fn is not None
        return _scripted_metrics(next(script))

    real_save = train_mod.save_checkpoint

    def spy_save(path, model, model_cfg, image_size, epoch, metrics, *args, **kwargs):
        record["saves"].append((Path(path).name, epoch, dict(metrics)))
        return real_save(path, model, model_cfg, image_size, epoch, metrics, *args, **kwargs)

    monkeypatch.setattr(train_mod, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_mod, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_mod, "save_checkpoint", spy_save)
    return record


def test_train_best_selection_rules(tmp_path, monkeypatch, capsys):
    make_dataset(tmp_path)
    ious = [math.nan, 0.5, 0.4, 0.5, 0.7, math.nan]
    cfg = make_cfg(tmp_path, tmp_path / "run", epochs=len(ious))
    record = _stub_loops(monkeypatch, ious)

    best = train(cfg)

    best_epochs = [e for name, e, _ in record["saves"] if name == "best.pt"]
    last_epochs = [e for name, e, _ in record["saves"] if name == "last.pt"]
    # epoch 1 一定保存（即使 NaN）；之后严格大于历史最佳才保存；NaN 视为 -inf；并列不保存
    assert best_epochs == [1, 2, 5]
    assert last_epochs == [1, 2, 3, 4, 5, 6]
    # 每个 epoch 先存 last.pt 再（可能）存 best.pt
    assert [n for n, e, _ in record["saves"] if e == 2] == ["last.pt", "best.pt"]
    assert record["eval_calls"] == len(ious)
    assert load_checkpoint(best)["epoch"] == 5
    assert load_checkpoint(tmp_path / "run" / "last.pt")["epoch"] == 6

    metrics = record["saves"][0][2]
    assert metrics["train_loss"] == 1.0 and metrics["val_loss"] == 0.25
    assert "loss" not in metrics and set(METRIC_KEYS) <= set(metrics)

    columns, rows = read_history(tmp_path / "run" / "history.csv")
    assert columns == HISTORY_COLUMNS
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3, 4, 5, 6]
    assert rows[0]["iou_walkable"] == "nan"
    assert float(rows[4]["iou_walkable"]) == pytest.approx(0.7)
    assert all(float(r["val_loss"]) == 0.25 for r in rows)

    out = capsys.readouterr().out
    for epoch in range(1, len(ious) + 1):
        assert f"epoch {epoch}/{len(ious)}" in out
    for word in ("train_loss", "val_loss", "iou_walkable", "miou"):
        assert word in out


def test_train_all_nan_scores_keep_first_epoch_as_best(tmp_path, monkeypatch):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run", epochs=3)
    record = _stub_loops(monkeypatch, [math.nan] * 3)
    best = train(cfg)
    assert [e for name, e, _ in record["saves"] if name == "best.pt"] == [1]
    assert load_checkpoint(best)["epoch"] == 1


def test_train_wiring_cpu_disables_amp_and_builds_scheduler(tmp_path, monkeypatch):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run", epochs=3, amp=True, lr=1e-3)
    record = _stub_loops(monkeypatch, [0.1, 0.2, 0.3])
    train(cfg)

    assert len(record["train_args"]) == 3
    for args in record["train_args"]:
        assert args["scaler"] is None  # AMP 只在 CUDA 上启用
        assert args["device"] == CPU
        assert isinstance(args["loss_fn"], SegLoss)
        assert isinstance(args["scheduler"], torch.optim.lr_scheduler.LambdaLR)
    sched = record["train_args"][0]["scheduler"]
    assert sched is record["train_args"][-1]["scheduler"]  # 整个训练共用一个调度器
    assert sched.last_epoch == 3 * 2  # 3 个 epoch × 每个 epoch 2 个 iteration
    opt = sched.optimizer
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(cfg.train.weight_decay)
    assert opt.param_groups[0]["initial_lr"] == pytest.approx(1e-3)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)  # 余弦退火到 0

    _, rows = read_history(tmp_path / "run" / "history.csv")
    lam = make_lr_lambda(6)
    assert [float(r["lr"]) for r in rows] == pytest.approx([1e-3 * lam(0), 1e-3 * lam(2), 1e-3 * lam(4)])


def test_train_non_default_arch_checkpoint_rebuilds_same_model(tmp_path, monkeypatch):
    """checkpoint 的 model_cfg 必须记录实际训练的架构（这里是 FPN），否则 eval / infer / export 会按 Unet 重建，
    严格加载权重时失败。训练/评估循环打桩，只验证保存与重建。"""
    from dfwalk.infer import Predictor

    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run", arch="FPN", epochs=1)
    _stub_loops(monkeypatch, [0.5])
    best = train(cfg)
    for path in (best, tmp_path / "run" / "last.pt"):
        state = load_checkpoint(path)
        assert state["model_cfg"] == model_config("FPN", ENCODER, NUM_CLASSES)
        model = model_from_checkpoint(state)  # load_state_dict 为严格模式：结构不一致会抛 RuntimeError
        assert type(model).__name__ == "FPN"
    predictor = Predictor.from_checkpoint(best, device="cpu")
    assert predictor.predict_mask(np.zeros((H, W, 3), np.uint8)).shape == (H, W)


def test_train_loss_weights_passed_to_segloss(tmp_path, monkeypatch):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run", epochs=1, ce_weight=0.3, dice_weight=2.0)
    record = _stub_loops(monkeypatch, [0.5])
    train(cfg)
    loss_fn = record["train_args"][0]["loss_fn"]
    assert loss_fn.ce_weight == pytest.approx(0.3)
    assert loss_fn.dice_weight == pytest.approx(2.0)


def test_train_rerun_overwrites_history(tmp_path, monkeypatch):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run", epochs=2)
    _stub_loops(monkeypatch, [0.1, 0.2])
    train(cfg)
    _stub_loops(monkeypatch, [0.3, 0.4])
    train(cfg)
    _, rows = read_history(tmp_path / "run" / "history.csv")
    assert [float(r["iou_walkable"]) for r in rows] == pytest.approx([0.3, 0.4])


def test_train_nan_loss_raises_floating_point_error(tmp_path, monkeypatch):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run", epochs=2)

    class NanSegLoss(SegLoss):
        def forward(self, logits, target):
            return super().forward(logits, target) * math.nan

    monkeypatch.setattr(train_mod, "SegLoss", NanSegLoss)
    with pytest.raises(FloatingPointError, match="iteration 1"):
        train(cfg)
    out_dir = tmp_path / "run"
    assert (out_dir / "config.yaml").is_file()  # 配置在训练前就已写入
    columns, rows = read_history(out_dir / "history.csv")
    assert columns == HISTORY_COLUMNS and rows == []
    assert not (out_dir / "best.pt").exists()


@pytest.mark.parametrize(
    "arch, batch_size, one_frame_train",
    [("DeepLabV3Plus", 1, False), ("UPerNet", 1, False), ("PAN", 1, False), ("deeplabv3", 2, True)],
)
def test_train_rejects_batch_of_one_for_bn_after_global_pool(tmp_path, monkeypatch, arch, batch_size,
                                                              one_frame_train):
    """这些架构在全局池化后接 BatchNorm，1 个样本的 batch 会在第一个 iteration 崩溃：
    应在构建模型之前抛出带 batch_size 提示的 ValueError，而不是 PyTorch 的英文报错。"""
    make_dataset(tmp_path)
    if one_frame_train:
        write_split(tmp_path, "train", ["vidA_000000"])
        write_split(tmp_path, "val", ["vidB_000000"])
    cfg = make_cfg(tmp_path, tmp_path / "run", arch=arch, batch_size=batch_size, epochs=1)
    built = []
    real_build = train_mod.build_model
    monkeypatch.setattr(train_mod, "build_model", lambda *a, **k: built.append(a) or real_build(*a, **k))
    with pytest.raises(ValueError, match="batch_size") as info:
        train(cfg)
    assert arch in str(info.value) and "Unet" in str(info.value)
    assert built == []  # 在构建模型（下载预训练权重）之前就失败
    assert not (tmp_path / "run" / "best.pt").exists()


def test_check_batch_of_one_only_for_affected_archs_and_sizes():
    def loader(n: int, batch_size: int, drop_last: bool) -> DataLoader:
        return DataLoader(TensorDataset(torch.zeros(n, 1)), batch_size=batch_size, drop_last=drop_last)

    assert train_mod._smallest_batch(loader(5, 2, True)) == 2
    assert train_mod._smallest_batch(loader(4, 2, False)) == 2
    assert train_mod._smallest_batch(loader(5, 2, False)) == 1
    assert train_mod._smallest_batch(loader(1, 8, False)) == 1
    train_mod._check_batch_of_one("Unet", loader(3, 1, True))  # Unet 不受影响
    train_mod._check_batch_of_one("Segformer", loader(1, 8, False))
    train_mod._check_batch_of_one("DeepLabV3Plus", loader(5, 2, True))  # 每个 batch 都是 2 个样本
    with pytest.raises(ValueError, match="batch_size"):
        train_mod._check_batch_of_one("DeepLabV3Plus", loader(5, 1, True))
    with pytest.raises(ValueError, match="batch_size"):
        train_mod._check_batch_of_one("upernet", loader(1, 8, False))


@pytest.mark.parametrize(
    "section, key, value",
    [("train", "epochs", 0), ("train", "epochs", -1), ("data", "image_size", [60, 96]), ("data", "val_ratio", 1.0)],
)
def test_train_rejects_invalid_config(tmp_path, section, key, value):
    make_dataset(tmp_path)
    cfg = make_cfg(tmp_path, tmp_path / "run")
    setattr(getattr(cfg, section), key, value)
    with pytest.raises(ValueError):
        train(cfg)
    assert not (tmp_path / "run" / "best.pt").exists()


# ---------------------------------------------------------------- 学习能力冒烟测试


def test_learns_trivially_separable_dataset(tmp_path):
    """颜色完全可分的合成数据：几个 epoch 内验证集 iou_walkable 应超过 0.8（CPU 上约数秒）。"""
    make_dataset(tmp_path, videos=("vidA", "vidB", "vidC"), frames=6)
    cfg = make_cfg(tmp_path, tmp_path / "run", epochs=4, lr=1e-3)
    cfg.data.val_ratio = 0.25

    best = train(cfg)

    state = load_checkpoint(best)
    assert state["metrics"]["iou_walkable"] > 0.8, state["metrics"]
    _, rows = read_history(tmp_path / "run" / "history.csv")
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])
    metrics = evaluate_checkpoint(cfg, best)
    assert metrics["iou_walkable"] > 0.8
    assert metrics["pixel_acc"] > 0.9
