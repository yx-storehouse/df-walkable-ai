"""训练与评估。

输出目录 cfg.train.out_dir：
    best.pt       验证集 iou_walkable 最高的 checkpoint（dfwalk.checkpoint 格式）
    last.pt       最后一个 epoch 的 checkpoint
    history.csv   每个 epoch 一行：epoch, lr, train_loss, val_loss, 以及 ConfusionMatrix.compute() 的所有指标
    config.yaml   本次训练使用的完整配置（Config.to_dict()，yaml.safe_dump, allow_unicode=True）

补充约定：
    - history.csv 在训练开始时重写（只写表头），之后每个 epoch 追加一行并立即落盘，训练中断也能看到已完成的
      epoch；lr 列为该 epoch 第一个 iteration 使用的学习率；NaN 指标写作 "nan"。
    - config.yaml 在训练开始前写入（训练崩溃时也能知道用的是什么配置），Path/tuple 等转成 str/list，
      可被 dfwalk.config.load_config 直接读回。
    - checkpoint 的 metrics 为 {"train_loss", "val_loss", 以及 compute() 的全部指标}。
    - 这里只处理离线的截图/录像帧，不涉及任何游戏进程或输入控制。
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import NUM_CLASSES
from .checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint
from .config import Config, validate
from .dataset import WalkableDataset, find_pairs, make_splits, read_split, write_split
from .losses import SegLoss
from .metrics import ConfusionMatrix
from .model import build_model, model_config
from .transforms import EvalTransform, TrainTransform
from .utils import resolve_device, seed_everything

BEST_NAME = "best.pt"
LAST_NAME = "last.pt"
HISTORY_NAME = "history.csv"
CONFIG_NAME = "config.yaml"

# warmup 步数上限；实际 warmup = min(MAX_WARMUP_STEPS, 总步数 // 10)
MAX_WARMUP_STEPS = 500
# history.csv 中位于指标列之前的固定列
HISTORY_LEADING_COLUMNS = ("epoch", "lr", "train_loss", "val_loss")


# ---------------------------------------------------------------- 学习率调度


def default_warmup_steps(total_steps: int) -> int:
    """默认 warmup 步数：min(500, total_steps // 10)。总步数不足 10 时为 0（不做 warmup）。"""
    return min(MAX_WARMUP_STEPS, max(0, int(total_steps)) // 10)


def make_lr_lambda(total_steps: int, warmup_steps: int | None = None) -> Callable[[int], float]:
    """返回给 torch.optim.lr_scheduler.LambdaLR 用的系数函数 f(step) → 学习率倍数（相对 base lr）。

    step 为 LambdaLR 内部计数（第 i 个 iteration 使用 f(i)，i 从 0 开始，每个 iteration 后 step 一次）：
    - warmup 阶段 step < W：线性上升 f = (step + 1) / W，第一个 iteration 就有非零学习率，
      warmup 最后一步恰好到 1.0。
    - 之后余弦退火：progress = (step - W) / (total_steps - W)，f = 0.5 * (1 + cos(pi * progress))，
      step = W 时为 1.0，step = total_steps（最后一个 iteration 之后）时为 0；超出 total_steps 保持 0。
    - warmup_steps 为 None 时取 default_warmup_steps(total_steps)。
    - total_steps 必须是正整数，warmup_steps 必须在 [0, total_steps) 内，否则抛 ValueError。
    """
    total = int(total_steps)
    if total <= 0:
        raise ValueError(f"total_steps 必须是正整数，得到 {total_steps!r}")
    warmup = default_warmup_steps(total) if warmup_steps is None else int(warmup_steps)
    if not 0 <= warmup < total:
        raise ValueError(f"warmup_steps 必须在 [0, {total}) 内，得到 {warmup_steps!r}")
    decay = total - warmup  # >= 1

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = min(max((step - warmup) / decay, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# ---------------------------------------------------------------- 数据


def _make_loader(dataset: Dataset, cfg: Config, shuffle: bool, drop_last: bool) -> DataLoader:
    """按 cfg.train 的 batch_size / num_workers 创建 DataLoader。"""
    num_workers = int(cfg.train.num_workers)
    if num_workers < 0:
        raise ValueError(f"train.num_workers 不能为负，得到 {num_workers}")
    batch_size = int(cfg.train.batch_size)
    if batch_size < 1:
        raise ValueError(f"train.batch_size 必须 >= 1，得到 {cfg.train.batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _load_or_create_splits(cfg: Config) -> tuple[list[str], list[str]]:
    """读取 splits/train.txt 与 val.txt；任一不存在时重新划分并写入两个文件。"""
    root = Path(cfg.data.root)
    try:
        return read_split(root, "train"), read_split(root, "val")
    except FileNotFoundError:
        pass
    stems = find_pairs(root)
    train_stems, val_stems = make_splits(stems, cfg.data.val_ratio, cfg.data.seed)
    train_path = write_split(root, "train", train_stems)
    val_path = write_split(root, "val", val_stems)
    print(
        f"未找到完整的划分文件，已按视频分组自动划分（val_ratio={cfg.data.val_ratio}, seed={cfg.data.seed}）："
        f"训练 {len(train_stems)} 帧 → {train_path}，验证 {len(val_stems)} 帧 → {val_path}",
        flush=True,
    )
    return train_stems, val_stems


def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """读取 {root}/splits/train.txt 与 val.txt；任一不存在时用 dataset.find_pairs + make_splits(val_ratio, seed)
    生成并写入（打印提示）。训练集用 TrainTransform(image_size)，验证集用 EvalTransform(image_size)。
    train loader: shuffle=True, drop_last=(len(train_ds) > batch_size)；val loader: shuffle=False。
    num_workers 取 cfg.train.num_workers；pin_memory 仅在 CUDA 可用时为 True；
    num_workers > 0 时 persistent_workers=True。训练集或验证集为空时抛 ValueError。

    补充说明：
    - 自动划分使用 cfg.data.val_ratio 与 cfg.data.seed；只缺其中一个文件时两个文件都会重新生成（覆盖），
      保证训练/验证不重叠。
    - 划分文件存在但为空（或数据目录里根本没有图片+掩码对）时抛 ValueError；
      划分文件里的 stem 缺图片/掩码时由 WalkableDataset 抛 FileNotFoundError。
    - batch_size < 1 或 num_workers < 0 时抛 ValueError。
    """
    root = Path(cfg.data.root)
    train_stems, val_stems = _load_or_create_splits(cfg)
    if not train_stems:
        raise ValueError(f"训练集为空：{root / 'splits' / 'train.txt'} 没有任何样本")
    if not val_stems:
        raise ValueError(f"验证集为空：{root / 'splits' / 'val.txt'} 没有任何样本")

    image_size = cfg.data.image_size
    train_ds = WalkableDataset(root, train_stems, TrainTransform(image_size))
    val_ds = WalkableDataset(root, val_stems, EvalTransform(image_size))

    train_loader = _make_loader(train_ds, cfg, shuffle=True, drop_last=len(train_ds) > int(cfg.train.batch_size))
    val_loader = _make_loader(val_ds, cfg, shuffle=False, drop_last=False)
    return train_loader, val_loader


# ---------------------------------------------------------------- 训练 / 评估循环


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> float:
    """训练一个 epoch，返回按样本数加权的平均 loss。scheduler 不为 None 时每个 iteration 调用 scheduler.step()。
    scaler 不为 None（CUDA AMP）时使用 torch.autocast("cuda") + scaler；否则普通 FP32。
    loss 出现 NaN/Inf 时抛 FloatingPointError（信息中包含 iteration 序号）。

    补充说明：
    - 进入时调用 model.train()。loader 每个 batch 为 (images (N,3,H,W) float, masks (N,H,W) int64)。
    - NaN/Inf 检查发生在 backward / optimizer.step() 之前：出错的那个 batch 不会更新任何参数。
      iteration 序号从 1 开始计数，形如 "iteration 3/10"。
    - AMP 路径：autocast 只包住前向和 loss；backward 走 scaler.scale(loss)，再 scaler.step(optimizer)、
      scaler.update()（梯度含 inf 时 GradScaler 会自动跳过这一步）。
    - loader 一个 batch 都没有时抛 ValueError（通常是 drop_last=True 且样本数小于 batch_size）。
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    try:
        n_iters: int | None = len(loader)
    except TypeError:  # IterableDataset 等没有长度的 loader
        n_iters = None
    # 非终端（重定向到文件、pytest 捕获）时 tqdm 自动关闭，不污染日志；with 保证出错时也会关闭进度条
    with tqdm(loader, desc="train", leave=False, disable=None, dynamic_ncols=True) as bar:
        for it, (images, masks) in enumerate(bar, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                # CUDA AMP：autocast 只包住前向和 loss，backward 在 autocast 之外
                with torch.autocast("cuda"):
                    logits = model(images)
                    loss = loss_fn(logits, masks)
            else:
                logits = model(images)
                loss = loss_fn(logits, masks)

            # 在 backward / step 之前检查，出错的 batch 不会污染参数和优化器状态
            value = float(loss.detach())
            if not math.isfinite(value):
                where = f"iteration {it}/{n_iters}" if n_iters is not None else f"iteration {it}"
                raise FloatingPointError(
                    f"训练 loss 为 {value}（{where}）：可能是学习率过大、数据异常或 AMP 溢出，已中止训练"
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)  # 梯度含 inf/NaN 时 GradScaler 自动跳过本次更新
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            n = int(images.shape[0])
            total_loss += value * n
            total_samples += n
            bar.set_postfix(loss=f"{value:.4f}")

    if total_samples == 0:
        raise ValueError("训练 loader 没有产生任何 batch（检查样本数、batch_size 与 drop_last）")
    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> dict[str, float]:
    """在 loader 上评估：pred = logits.argmax(1)，累计 ConfusionMatrix；返回 compute() 的结果，
    loss_fn 不为 None 时额外包含按样本数加权的 "loss"。调用前后保持 model 原来的 train/eval 状态。

    补充说明：
    - 评估期间 model 处于 eval 模式（BatchNorm 用滑动统计量、Dropout 关闭）；结束后逐个子模块恢复原来的
      training 标志（例如 train 模式下被单独冻结成 eval 的 BatchNorm 仍保持 eval），即使中途抛异常也会恢复。
    - 混淆矩阵的类别数取模型输出的通道数（通常为 NUM_CLASSES）；IGNORE_INDEX 像素不计入指标和 loss。
    - loader 为空时各指标为 NaN（"loss" 同样为 NaN）。
    """
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    cm: ConfusionMatrix | None = None
    total_loss = 0.0
    total_samples = 0
    try:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            if cm is None:
                cm = ConfusionMatrix(num_classes=int(logits.shape[1]))
            cm.update(logits.argmax(1), masks)
            n = int(images.shape[0])
            if loss_fn is not None:
                total_loss += float(loss_fn(logits, masks)) * n
            total_samples += n
    finally:
        for module, training in modes:
            module.training = training

    metrics = (cm if cm is not None else ConfusionMatrix(NUM_CLASSES)).compute()
    if loss_fn is not None:
        metrics["loss"] = total_loss / total_samples if total_samples else math.nan
    return metrics


# ---------------------------------------------------------------- 完整训练流程


def _plain(obj: Any) -> Any:
    """把配置字典中的 Path / tuple 等转成 yaml.safe_dump 能输出的基础类型。"""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_config(path: Path, cfg: Config) -> None:
    text = yaml.safe_dump(_plain(cfg.to_dict()), allow_unicode=True, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def _history_columns() -> list[str]:
    return [*HISTORY_LEADING_COLUMNS, *ConfusionMatrix(NUM_CLASSES).compute().keys()]


def _append_history(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=columns, extrasaction="ignore").writerow(row)


def _score(value: float) -> float:
    """best.pt 的比较依据：NaN 视为 -inf。"""
    value = float(value)
    return -math.inf if math.isnan(value) else value


def train(cfg: Config) -> Path:
    """完整训练流程，返回 best.pt 路径。

    - seed_everything(cfg.train.seed)；device = resolve_device(cfg.train.device)。
    - 模型 build_model(arch, encoder, encoder_weights)；损失 SegLoss(ce_weight, dice_weight)。
    - 优化器 AdamW(lr, weight_decay)；学习率：前 min(500, 总步数//10) 步线性 warmup，之后余弦退火到 0
      （用 torch.optim.lr_scheduler.LambdaLR 实现，每个 iteration step 一次）。
    - AMP：cfg.train.amp 为 True 且 device.type == "cuda" 时启用 torch.amp.GradScaler("cuda")。
    - 每个 epoch：train_one_epoch → evaluate(val, loss_fn) → 追加 history.csv → 保存 last.pt；
      iou_walkable 为 NaN 视为 -inf；严格大于历史最佳时保存 best.pt。第一个 epoch 一定保存 best.pt。
    - checkpoint 用 dfwalk.checkpoint.save_checkpoint，model_cfg = model.model_config(...)，
      image_size = cfg.data.image_size。
    - 每个 epoch 打印一行摘要（epoch、train_loss、val_loss、iou_walkable、miou）。

    补充说明：
    - 开始前调用 dfwalk.config.validate(cfg)；train.epochs 必须 >= 1，否则抛 ValueError。
    - 学习率调度见 make_lr_lambda（总步数 = epochs * len(train_loader)）。
    - 训练 loss 出现 NaN/Inf 时 FloatingPointError 直接向上抛出，已保存的 last.pt / best.pt 保留。
    """
    validate(cfg)
    epochs = int(cfg.train.epochs)
    if epochs < 1:
        raise ValueError(f"train.epochs 必须 >= 1，得到 {cfg.train.epochs}")

    seed_everything(int(cfg.train.seed))
    device = resolve_device(cfg.train.device)
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_config(out_dir / CONFIG_NAME, cfg)

    train_loader, val_loader = build_loaders(cfg)

    model = build_model(cfg.train.arch, cfg.train.encoder, encoder_weights=cfg.train.encoder_weights,
                        num_classes=NUM_CLASSES).to(device)
    model_cfg = model_config(cfg.train.arch, cfg.train.encoder, NUM_CLASSES)
    loss_fn = SegLoss(cfg.train.ce_weight, cfg.train.dice_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr),
                                  weight_decay=float(cfg.train.weight_decay))
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_steps))

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if device.type == "cuda":
        # 输入尺寸固定，让 cuDNN 选最快的卷积算法
        torch.backends.cudnn.benchmark = True

    history_path = out_dir / HISTORY_NAME
    columns = _history_columns()
    with open(history_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(columns)

    best_path = out_dir / BEST_NAME
    last_path = out_dir / LAST_NAME
    image_size = [int(x) for x in cfg.data.image_size]
    print(
        f"开始训练：device={device}, amp={use_amp}, {cfg.train.arch}/{cfg.train.encoder}, "
        f"训练 {len(train_loader.dataset)} 帧 / 验证 {len(val_loader.dataset)} 帧, "
        f"{epochs} epochs × {len(train_loader)} iters, 输出 → {out_dir}",
        flush=True,
    )

    best_score = -math.inf
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device,
                                     scheduler=scheduler, scaler=scaler)
        val_metrics = evaluate(model, val_loader, device, loss_fn=loss_fn)
        val_loss = float(val_metrics.pop("loss"))

        row: dict[str, Any] = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss}
        row.update(val_metrics)
        _append_history(history_path, columns, row)

        ckpt_metrics = {"train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        save_checkpoint(last_path, model, model_cfg, image_size, epoch, ckpt_metrics)

        score = _score(val_metrics["iou_walkable"])
        improved = epoch == 1 or score > best_score
        if improved:
            best_score = max(best_score, score)
            save_checkpoint(best_path, model, model_cfg, image_size, epoch, ckpt_metrics)

        print(
            f"epoch {epoch}/{epochs}  lr {lr:.2e}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"iou_walkable {val_metrics['iou_walkable']:.4f}  miou {val_metrics['miou']:.4f}  "
            f"({time.time() - t0:.1f}s){'  *best' if improved else ''}",
            flush=True,
        )

    print(f"训练完成：best.pt（iou_walkable={best_score:.4f}）→ {best_path}", flush=True)
    return best_path


def evaluate_checkpoint(cfg: Config, ckpt_path: str | Path, split: str = "val") -> dict[str, float]:
    """加载 checkpoint，在 {root}/splits/{split}.txt 上用 EvalTransform(checkpoint 中的 image_size) 评估，返回指标。

    补充说明：
    - 归一化参数同样取自 checkpoint（mean/std），与训练时一致；cfg.data.image_size 不参与。
    - 设备取 cfg.train.device，batch_size / num_workers 取 cfg.train；返回值除 compute() 的指标外还包含
      "loss"（SegLoss(cfg.train.ce_weight, cfg.train.dice_weight)，按样本数加权）。
    - 划分文件不存在时抛 FileNotFoundError（不会自动划分），为空时抛 ValueError；
      checkpoint 格式不对时由 load_checkpoint 抛 ValueError。
    """
    state = load_checkpoint(ckpt_path)
    device = resolve_device(cfg.train.device)
    model = model_from_checkpoint(state).to(device)

    root = Path(cfg.data.root)
    stems = read_split(root, split)
    if not stems:
        raise ValueError(f"划分 {split!r} 为空：{root / 'splits' / f'{split}.txt'} 没有任何样本")
    transform = EvalTransform(state["image_size"], mean=state["mean"], std=state["std"])
    dataset = WalkableDataset(root, stems, transform)
    loader = _make_loader(dataset, cfg, shuffle=False, drop_last=False)
    loss_fn = SegLoss(cfg.train.ce_weight, cfg.train.dice_weight).to(device)
    return evaluate(model, loader, device, loss_fn=loss_fn)
