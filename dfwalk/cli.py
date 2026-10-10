"""命令行入口：dfwalk <子命令> [-c CONFIG] [--set section.key=value ...]

子命令：
    extract-frames  从录像抽帧            hud-preview    在截图上预览 HUD 矩形
    autolabel       生成伪标签            to-labelme     伪标签 → LabelMe JSON
    from-labelme    LabelMe JSON → 掩码   split          划分训练/验证集
    check-data      检查数据并可视化标签   train          训练
    eval            评估 checkpoint       infer          图片/目录/视频推理
    export-onnx     导出 ONNX

补充约定：
- -c/--config 与 --set 既可以写在子命令之后，也可以写在子命令之前；两处都给了 --set 时按出现顺序依次生效。
- 未指定 -c 时，若当前工作目录下存在 configs/default.yaml 就使用它（并在 stderr 打印一行提示），
  否则使用 dfwalk.config 中 dataclass 的默认值。
- 优先级：dataclass 默认值 < 配置文件 < --set < 子命令自己的参数（如 train --epochs）。
- 默认数据路径都由 data.root 推出：images/、pseudo_masks/、labelme/、masks/、splits/。
- 用户可处理的错误（FileNotFoundError、ValueError、KeyError、ImportError、OSError 等）打印为
  “错误: <信息>” 到 stderr，退出码 1；环境变量 DFWALK_DEBUG=1 时原样抛出以便查看完整堆栈。
  参数错误退出码 2，Ctrl+C 退出码 130。main() 总是返回退出码而不是调用 sys.exit。
- 结果打印到 stdout；提示、警告（包括各模块 warnings.warn 的 UserWarning）打印到 stderr。

本工具只做离线分析（自己录制的视频 / 截图文件），不读取游戏进程、不注入输入、不控制角色。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import IGNORE_INDEX, IMAGE_EXTS, NON_WALKABLE, VIDEO_EXTS, WALKABLE, __version__
from .config import Config, load_config, validate

PROG = "dfwalk"
# 未指定 -c 时自动使用的配置文件（相对当前工作目录）
DEFAULT_CONFIG_PATH = Path("configs") / "default.yaml"
# 设为 1 时不拦截异常，打印完整堆栈
DEBUG_ENV = "DFWALK_DEBUG"

# data.root 下的目录约定（与 dfwalk.dataset 的 IMAGES_SUBDIR / MASKS_SUBDIR / SPLITS_SUBDIR 一致，有测试校验）
IMAGES_SUBDIR = "images"
PSEUDO_MASKS_SUBDIR = "pseudo_masks"
LABELME_SUBDIR = "labelme"
MASKS_SUBDIR = "masks"
SPLITS_SUBDIR = "splits"

# 列举文件名时最多显示的个数
_MAX_LISTED = 5
_VALID_MASK_VALUES = (NON_WALKABLE, WALKABLE, IGNORE_INDEX)

# eval 表格中指标的显示顺序与说明（未列出的指标排在后面，说明留空）
METRIC_ORDER = (
    "iou_walkable",
    "iou_non_walkable",
    "miou",
    "pixel_acc",
    "precision_walkable",
    "recall_walkable",
    "f1_walkable",
)
METRIC_DESCRIPTIONS: dict[str, str] = {
    "iou_walkable": "可走区域 IoU（主要指标，best.pt 按它挑选）",
    "iou_non_walkable": "不可走区域 IoU",
    "miou": "各类 IoU 的平均值",
    "pixel_acc": "像素准确率（不含忽略像素）",
    "precision_walkable": "可走精确率：预测为可走的像素中真正可走的比例",
    "recall_walkable": "可走召回率：真正可走的像素中被找出来的比例",
    "f1_walkable": "可走 F1：精确率与召回率的调和平均",
    "loss": "验证 loss（交叉熵 + Dice）",
}

# 被 main 拦截并打印为 “错误: …” 的异常类型
_HANDLED_ERRORS: tuple[type[BaseException], ...] = (
    FileNotFoundError,
    ValueError,
    KeyError,
    ImportError,
    OSError,
    RuntimeError,
    FloatingPointError,
)
# 这些类型多半不是简单的输入问题，额外提示如何查看堆栈
_UNEXPECTED_ERRORS: tuple[type[BaseException], ...] = (RuntimeError, FloatingPointError)


class CLIError(Exception):
    """命令行层面的用户错误（例如拒绝覆盖已有划分文件）：打印 “错误: …” 并以退出码 1 结束。"""


# ---------------------------------------------------------------- 输出


def _info(message: str) -> None:
    """提示信息 → stderr（不混入 stdout 的结果输出）。"""
    print(message, file=sys.stderr, flush=True)


def _warn(message: str) -> None:
    print(f"警告: {message}", file=sys.stderr, flush=True)


def _preview(names: Sequence[Any], limit: int = _MAX_LISTED) -> str:
    """列出前几个名字："a, b, c …（共 N 个）"。"""
    items = [str(x) for x in names]
    shown = ", ".join(items[:limit])
    return shown + (f" …（共 {len(items)} 个）" if len(items) > limit else "")


def _percent(x: float) -> str:
    return f"{100.0 * float(x):5.1f}%"


def _make_showwarning(original: Callable[..., Any]) -> Callable[..., None]:
    """warnings.warn 发出的 UserWarning 简洁地显示为 “警告: …”；其他类别保持 Python 默认格式。"""

    def show(message, category, filename, lineno, file=None, line=None) -> None:
        if category is UserWarning:
            stream = file if file is not None else sys.stderr
            try:
                stream.write(f"警告: {message}\n")
                stream.flush()
            except (OSError, ValueError, AttributeError):  # pragma: no cover - 流已关闭等极端情况
                pass
        else:
            original(message, category, filename, lineno, file, line)

    return show


def _relax_stream_encoding() -> None:
    """控制台编码（如 Windows 重定向到文件时的 GBK）无法表示某个字符时用 '?' 代替，而不是抛 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError, TypeError, AttributeError):  # pragma: no cover - 非标准流
            pass


def _error_message(exc: BaseException) -> str:
    """把异常转换成适合给用户看的一行信息。"""
    if isinstance(exc, KeyError) and exc.args:
        # str(KeyError("x")) 会带引号，直接取参数
        return str(exc.args[0])
    if isinstance(exc, FileNotFoundError):
        if exc.filename is not None:  # 操作系统抛出的：[Errno 2] No such file or directory: 'x'
            return f"文件或目录不存在: {exc.filename}"
        if len(exc.args) == 1 and isinstance(exc.args[0], os.PathLike):  # FileNotFoundError(path)
            return f"文件或目录不存在: {os.fspath(exc.args[0])}"
    text = str(exc).strip()
    return text or type(exc).__name__


def _debug_enabled() -> bool:
    return os.environ.get(DEBUG_ENV, "").strip().lower() not in ("", "0", "false", "no", "off")


# ---------------------------------------------------------------- 配置


def load_cli_config(config_path: str | Path | None, overrides: Iterable[str] = (), notice: bool = True) -> Config:
    """按命令行约定加载配置。

    - config_path 为 None 且当前目录存在 configs/default.yaml 时使用它（notice=True 时在 stderr 打印一行提示）；
      否则使用 dataclass 默认值。
    - 指定的配置文件不存在抛 FileNotFoundError；YAML 语法错误、--set 的值无法解析、配置项类型不对
      （例如 data.image_size=5）统一转换为 ValueError；未知配置段/配置项为 KeyError（来自 dfwalk.config）。
    """
    import yaml

    path: Path | None = None
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {path}")
    elif DEFAULT_CONFIG_PATH.is_file():
        path = DEFAULT_CONFIG_PATH
        if notice:
            _info(f"未指定 -c/--config，使用当前目录下的 {DEFAULT_CONFIG_PATH.as_posix()}")
    try:
        return load_config(path, list(overrides))
    except yaml.YAMLError as e:
        where = f"配置文件 {path} 或 --set 覆盖项" if path is not None else "--set 覆盖项"
        raise ValueError(f"{where}的 YAML 语法错误: {e}") from e
    except TypeError as e:
        raise ValueError(f"配置无效: {e}") from e


def _validate(cfg: Config) -> None:
    """重新校验配置（应用子命令参数之后）；类型错误同样转换为 ValueError。"""
    try:
        validate(cfg)
    except TypeError as e:
        raise ValueError(f"配置无效: {e}") from e


def _override(section: Any, **values: Any) -> None:
    """把不为 None 的子命令参数写进配置段（子命令参数优先于配置文件和 --set）。"""
    for key, value in values.items():
        if value is not None:
            setattr(section, key, value)


@dataclass(frozen=True)
class DataPaths:
    """data.root 下各目录的默认路径。"""

    root: Path
    images: Path
    pseudo_masks: Path
    labelme: Path
    masks: Path
    splits: Path

    @classmethod
    def from_config(cls, cfg: Config) -> "DataPaths":
        root = Path(cfg.data.root)
        return cls(
            root=root,
            images=root / IMAGES_SUBDIR,
            pseudo_masks=root / PSEUDO_MASKS_SUBDIR,
            labelme=root / LABELME_SUBDIR,
            masks=root / MASKS_SUBDIR,
            splits=root / SPLITS_SUBDIR,
        )


def _path_or(value: str | None, default: Path) -> Path:
    return Path(value) if value else default


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:  # pragma: no cover - 无法解析的路径
        return False


def _require_file(path: str | Path, what: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{what}不存在: {p}")
    return p


def _list_files(folder: Path, suffix: str) -> list[Path]:
    """目录下（不递归）指定后缀（大小写不敏感）的文件；目录不存在时返回空列表。"""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == suffix)


# ---------------------------------------------------------------- extract-frames


def _collect_videos(inputs: Sequence[str]) -> list[Path]:
    """展开输入：视频文件原样保留（不检查后缀，打不开时由 extract_frames 报错）；
    目录取其第一层中后缀属于 VIDEO_EXTS 的文件。按出现顺序去重。路径不存在抛 FileNotFoundError。"""
    videos: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = sorted(c for c in p.iterdir() if c.is_file() and c.suffix.lower() in VIDEO_EXTS)
            if not found:
                _warn(f"目录中没有视频文件（支持 {', '.join(VIDEO_EXTS)}）: {p}")
        elif p.is_file():
            found = [p]
        else:
            raise FileNotFoundError(f"输入不存在: {p}")
        for v in found:
            key = v.resolve()
            if key not in seen:
                seen.add(key)
                videos.append(v)
    return videos


def _existing_frames(out_dir: Path, prefix: str) -> int:
    """out_dir 中已有的 {prefix}_{数字}.jpg 帧数（不会误算 {prefix}_xxx_{数字}.jpg 这类其他视频的帧）。"""
    if not out_dir.is_dir():
        return 0
    pattern = re.compile(rf"^{re.escape(prefix)}_[0-9]+\.jpg$")
    return sum(1 for p in out_dir.iterdir() if p.is_file() and pattern.match(p.name))


def _cmd_extract_frames(args: argparse.Namespace, cfg: Config) -> int:
    from .frames import extract_frames, sanitize_stem

    _override(cfg.frames, fps=args.fps, max_side=args.max_side, dedup_threshold=args.dedup)
    _validate(cfg)
    paths = DataPaths.from_config(cfg)
    out_dir = _path_or(args.output, paths.images)

    videos = _collect_videos(args.inputs)
    if not videos:
        raise CLIError(f"没有找到视频文件（支持的后缀: {', '.join(VIDEO_EXTS)}）")

    # 不同视频清洗后的前缀相同（如 "a b.mp4" 与 "a_b.mp4"）时帧文件会互相覆盖，且会被当成同一段视频分组
    by_prefix: dict[str, list[Path]] = {}
    for v in videos:
        by_prefix.setdefault(sanitize_stem(v.stem), []).append(v)
    clashes = {p: vs for p, vs in by_prefix.items() if len(vs) > 1}
    if clashes:
        detail = "；".join(f"{p}: {', '.join(str(v) for v in vs)}" for p, vs in clashes.items())
        raise CLIError(f"以下视频的文件名清洗后得到相同的帧前缀，抽出的帧会互相覆盖，请先重命名: {detail}")

    f = cfg.frames
    _info(f"抽帧参数: fps={f.fps}, max_side={f.max_side}, dedup_threshold={f.dedup_threshold} → {out_dir}")
    total = 0
    for video in videos:
        prefix = sanitize_stem(video.stem)
        existing = _existing_frames(out_dir, prefix)
        if existing:
            _warn(f"{out_dir} 中已有 {existing} 张 {prefix}_*.jpg（可能来自之前的抽帧），同名帧将被覆盖")
        written = extract_frames(
            video,
            out_dir,
            fps=f.fps,
            max_side=f.max_side,
            dedup_threshold=f.dedup_threshold,
            jpeg_quality=f.jpeg_quality,
        )
        total += len(written)
        print(f"{video}: 写出 {len(written)} 帧（前缀 {prefix}）", flush=True)
    print(f"共写出 {total} 帧（{len(videos)} 个视频）→ {out_dir}")
    _info("下一步: dfwalk hud-preview <一张抽出的图片> 核对 HUD 区域，然后 dfwalk autolabel")
    return 0


# ---------------------------------------------------------------- hud-preview


def _cmd_hud_preview(args: argparse.Namespace, cfg: Config) -> int:
    from .hud import draw_hud_preview, rect_to_pixels
    from .utils import read_image_rgb, write_image_rgb

    image_path = _require_file(args.image, "图片")
    out_path = Path(args.output)
    if out_path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"输出文件后缀必须是图片格式（{', '.join(IMAGE_EXTS)}），得到 {out_path}")
    if _same_path(out_path, image_path):
        raise CLIError(f"输出路径与输入图片相同，会覆盖原图: {out_path}")

    image = read_image_rgb(image_path)
    rects = cfg.hud.rects
    if not rects:
        _warn(
            "hud.rects 为空，当前没有任何 HUD 忽略区域。请在配置文件的 hud.rects 中填写归一化矩形 "
            "[x0, y0, x1, y1]（0~1，左上角为原点），再运行本命令核对"
        )
    preview = draw_hud_preview(image, rects)
    write_image_rgb(out_path, preview)

    h, w = image.shape[:2]
    print(f"{image_path}: {w}x{h}，{len(rects)} 个 HUD 矩形")
    for i, rect in enumerate(rects):
        c0, r0, c1, r1 = rect_to_pixels(rect, h, w)
        print(f"  [{i}] {list(rect)} → 像素 x {c0}~{c1}, y {r0}~{r1}（{c1 - c0}x{r1 - r0}）")
    print(f"预览图已保存: {out_path}")
    return 0


# ---------------------------------------------------------------- autolabel


def _cmd_autolabel(args: argparse.Namespace, cfg: Config) -> int:
    from .autolabel import AutoLabeler, autolabel_folder
    from .utils import list_images

    _override(cfg.autolabel, model=args.model, device=args.device)
    _validate(cfg)
    paths = DataPaths.from_config(cfg)
    images_dir = _path_or(args.input, paths.images)
    out_dir = _path_or(args.output, paths.pseudo_masks)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}（先运行 dfwalk extract-frames）")
    n_images = len(list_images(images_dir))
    if n_images == 0:
        raise CLIError(f"图片目录中没有图片: {images_dir}")

    a = cfg.autolabel
    _info(f"加载自动标注模型 {a.model}（首次使用需从 Hugging Face 下载；国内网络可设置 HF_ENDPOINT=https://hf-mirror.com）")
    try:
        labeler = AutoLabeler.from_pretrained(
            a.model,
            a.walkable_labels,
            short_side=a.short_side,
            hi=a.hi_threshold,
            lo=a.lo_threshold,
            device=a.device,
        )
    except OSError as e:
        raise CLIError(
            f"无法加载自动标注模型 {a.model}: {e}。若是网络问题，可设置环境变量 HF_ENDPOINT=https://hf-mirror.com "
            "后重试，或先把模型下载到本地目录，再用 --model 指向该目录"
        ) from e
    _info(f"设备: {labeler.device}，可走类别 {int(labeler.walkable_lookup.sum())} 个，"
          f"阈值 hi={labeler.hi} / lo={labeler.lo}，HUD 矩形 {len(cfg.hud.rects)} 个")

    written = autolabel_folder(labeler, images_dir, out_dir, hud_rects=cfg.hud.rects, overwrite=args.overwrite)
    skipped = n_images - len(written)
    msg = f"生成伪标签 {len(written)} 张 → {out_dir}"
    if skipped:
        msg += f"（另有 {skipped} 张已存在或无法读取，已跳过；加 --overwrite 可重新生成已存在的）"
    print(msg)
    _info("下一步: dfwalk to-labelme 导出为 LabelMe 标注后人工修正（伪标签必须人工检查）")
    return 0


# ---------------------------------------------------------------- to-labelme / from-labelme


def _cmd_to_labelme(args: argparse.Namespace, cfg: Config) -> int:
    from .labelme_io import export_masks_to_labelme

    paths = DataPaths.from_config(cfg)
    masks_dir = _path_or(args.masks, paths.pseudo_masks)
    images_dir = _path_or(args.images, paths.images)
    out_dir = _path_or(args.output, paths.labelme)
    n_masks = len(_list_files(masks_dir, ".png"))

    written = export_masks_to_labelme(
        masks_dir, images_dir, out_dir, epsilon=args.epsilon, min_area=args.min_area, overwrite=args.overwrite
    )
    msg = f"写出 {len(written)} 个 LabelMe JSON → {out_dir}"
    skipped = n_masks - len(written)
    if skipped:
        msg += f"（{skipped} 个掩码被跳过：JSON 已存在（不会覆盖人工修正结果，需要时加 --overwrite）或找不到对应图片）"
    print(msg)
    _info(
        "下一步: 用 LabelMe 修正标注，例如\n"
        f"  labelme {images_dir.as_posix()} --output {out_dir.as_posix()} "
        "--labels walkable,non_walkable,ignore --nodata --autosave\n"
        "修正完成后运行 dfwalk from-labelme"
    )
    return 0


def _cmd_from_labelme(args: argparse.Namespace, cfg: Config) -> int:
    from .labelme_io import convert_labelme_dir

    paths = DataPaths.from_config(cfg)
    json_dir = _path_or(args.input, paths.labelme)
    out_dir = _path_or(args.output, paths.masks)
    if json_dir.is_dir() and not _list_files(json_dir, ".json"):
        _warn(f"目录中没有 LabelMe JSON: {json_dir}")
    written = convert_labelme_dir(json_dir, out_dir, hud_rects=cfg.hud.rects)
    print(f"转换 {len(written)} 个 LabelMe 标注 → {out_dir}（HUD 矩形 {len(cfg.hud.rects)} 个已置为忽略）")
    _info("下一步: dfwalk split 划分训练/验证集，然后 dfwalk check-data 检查数据")
    return 0


# ---------------------------------------------------------------- split


def _cmd_split(args: argparse.Namespace, cfg: Config) -> int:
    from .dataset import find_pairs, group_of, make_splits, write_split

    _override(cfg.data, val_ratio=args.val_ratio, seed=args.seed)
    _validate(cfg)
    paths = DataPaths.from_config(cfg)
    existing = [p for p in (paths.splits / "train.txt", paths.splits / "val.txt") if p.exists()]
    if existing and not args.force:
        raise CLIError(
            f"划分文件已存在: {', '.join(str(p) for p in existing)}。重新划分会改变训练/验证集"
            "（之前的验证指标将不可比），确认要覆盖请加 --force"
        )

    stems = find_pairs(paths.root)
    if not stems:
        raise CLIError(
            f"{paths.root} 中没有同时具有图片（{IMAGES_SUBDIR}/）和掩码（{MASKS_SUBDIR}/{{stem}}.png）的样本，"
            "请先完成标注并运行 dfwalk from-labelme"
        )
    train_stems, val_stems = make_splits(stems, cfg.data.val_ratio, cfg.data.seed)
    write_split(paths.root, "train", train_stems)
    write_split(paths.root, "val", val_stems)

    print(f"按视频分组划分 {len(stems)} 帧（val_ratio={cfg.data.val_ratio}, seed={cfg.data.seed}）→ {paths.splits}")
    for name, split_stems in (("train", train_stems), ("val", val_stems)):
        groups = sorted({group_of(s) for s in split_stems})
        print(f"  {name:<5} {len(split_stems):>6} 帧  {len(groups):>4} 段视频: {_preview(groups)}")
    return 0


# ---------------------------------------------------------------- check-data


def _index_by_stem(files: Sequence[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """stem → 文件；同一 stem 有多个文件时记入 duplicates（取排序后的第一个）。"""
    groups: dict[str, list[Path]] = {}
    for p in files:
        groups.setdefault(p.stem, []).append(p)
    index = {stem: ps[0] for stem, ps in groups.items()}
    dups = {stem: ps for stem, ps in groups.items() if len(ps) > 1}
    return index, dups


def _evenly_spaced(items: Sequence[str], limit: int) -> list[str]:
    """从 items 中均匀取 limit 个（limit <= 0 或不少于总数时全部返回），覆盖不同视频而不是只看开头。"""
    n = len(items)
    if limit <= 0 or limit >= n:
        return list(items)
    if limit == 1:
        return [items[0]]
    idx = sorted({round(i * (n - 1) / (limit - 1)) for i in range(limit)})
    return [items[i] for i in idx]


def _cmd_check_data(args: argparse.Namespace, cfg: Config) -> int:
    import numpy as np
    from tqdm import tqdm

    from .dataset import group_of, label_stats, read_split
    from .utils import list_images, read_image_rgb, read_mask

    if args.vis_limit < 0:
        raise ValueError(f"--vis-limit 不能为负，得到 {args.vis_limit}")
    paths = DataPaths.from_config(cfg)
    if not paths.root.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {paths.root}（可用 --set data.root=... 指定）")
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir is not None and any(_same_path(vis_dir, d) for d in (paths.images, paths.masks, paths.pseudo_masks)):
        raise CLIError(f"--vis-dir 不能是数据目录本身（会混入可视化图片）: {vis_dir}")

    image_files = list_images(paths.images) if paths.images.is_dir() else []
    images, dup_images = _index_by_stem(image_files)
    masks, _ = _index_by_stem(_list_files(paths.masks, ".png"))
    n_pseudo = len(_list_files(paths.pseudo_masks, ".png"))
    n_labelme = len(_list_files(paths.labelme, ".json"))

    print(f"数据目录: {paths.root}")
    print(f"  {IMAGES_SUBDIR + '/':<14}{len(image_files):>7} 张图片")
    print(f"  {MASKS_SUBDIR + '/':<14}{len(masks):>7} 个掩码")
    print(f"  {PSEUDO_MASKS_SUBDIR + '/':<14}{n_pseudo:>7} 个伪标签")
    print(f"  {LABELME_SUBDIR + '/':<14}{n_labelme:>7} 个 LabelMe JSON")

    errors: list[str] = []
    notes: list[str] = []
    if dup_images:
        detail = "; ".join(f"{s}: {', '.join(p.name for p in ps)}" for s, ps in list(dup_images.items())[:_MAX_LISTED])
        errors.append(f"同一 stem 有多个图片文件（只能保留一个）{len(dup_images)} 个: {detail}")

    masks_without_images = sorted(set(masks) - set(images))
    images_without_masks = sorted(set(images) - set(masks))
    paired = sorted(set(masks) & set(images))
    if masks_without_images:
        errors.append(f"有掩码但没有对应图片 {len(masks_without_images)} 个: {_preview(masks_without_images)}")
    if images_without_masks:
        notes.append(f"有图片但还没有掩码（未标注）{len(images_without_masks)} 张: {_preview(images_without_masks)}")

    # 逐个检查图片 + 掩码对：能否读取、值域、尺寸
    unreadable: list[str] = []
    invalid_values: list[str] = []
    size_mismatch: list[str] = []
    valid: list[str] = []
    for stem in tqdm(paired, desc="check-data", unit="img", leave=False, disable=None):
        try:
            mask = read_mask(masks[stem])
        except ValueError as e:
            unreadable.append(f"{masks[stem].name}（{e}）")
            continue
        ok = True
        counts = np.bincount(mask.ravel(), minlength=256)
        counts[list(_VALID_MASK_VALUES)] = 0
        bad = np.flatnonzero(counts)
        if bad.size:
            values = ", ".join(str(int(v)) for v in bad[:5]) + (" …" if bad.size > 5 else "")
            invalid_values.append(f"{stem}（{values}）")
            ok = False
        try:
            image = read_image_rgb(images[stem])
        except ValueError as e:
            unreadable.append(f"{images[stem].name}（{e}）")
            continue
        if image.shape[:2] != mask.shape:
            size_mismatch.append(
                f"{stem}（图片 {image.shape[1]}x{image.shape[0]}，掩码 {mask.shape[1]}x{mask.shape[0]}）"
            )
            ok = False
        if ok:
            valid.append(stem)
    if unreadable:
        errors.append(f"无法读取的图片/掩码 {len(unreadable)} 个: {_preview(unreadable)}")
    if invalid_values:
        errors.append(f"掩码含非法值（只允许 0/1/255）{len(invalid_values)} 个: {_preview(invalid_values)}")
    if size_mismatch:
        errors.append(f"掩码与图片尺寸不一致 {len(size_mismatch)} 个: {_preview(size_mismatch)}")
    print(f"  有效的图片+掩码对: {len(valid)} / {len(paired)}")

    # 训练/验证划分
    valid_set = set(valid)
    train_txt, val_txt = paths.splits / "train.txt", paths.splits / "val.txt"
    stats_targets: list[tuple[str, list[str], int]] = []  # (名称, 参与统计的有效样本, 划分中列出的样本数)
    if train_txt.is_file() and val_txt.is_file():
        splits = {name: read_split(paths.root, name) for name in ("train", "val")}
        for name, stems in splits.items():
            missing = [s for s in stems if s not in images or s not in masks]
            if missing:
                errors.append(f"splits/{name}.txt 中有 {len(missing)} 个样本缺图片或掩码: {_preview(missing)}")
            stats_targets.append((name, [s for s in stems if s in valid_set], len(stems)))
        overlap = sorted(set(splits["train"]) & set(splits["val"]))
        if overlap:
            errors.append(f"train 与 val 有 {len(overlap)} 个相同样本（数据泄漏）: {_preview(overlap)}")
        shared_groups = sorted({group_of(s) for s in splits["train"]} & {group_of(s) for s in splits["val"]})
        if shared_groups:
            notes.append(
                f"有 {len(shared_groups)} 段视频同时出现在 train 和 val 中（验证指标会偏乐观）: "
                f"{_preview(shared_groups)}"
            )
        in_split = set(splits["train"]) | set(splits["val"])
        unsplit = [s for s in valid if s not in in_split]
        if unsplit:
            notes.append(
                f"{len(unsplit)} 个样本不在任何划分中（新标注的数据需要 dfwalk split --force）: {_preview(unsplit)}"
            )
    else:
        if train_txt.is_file() or val_txt.is_file():
            notes.append("splits/ 中只有 train.txt / val.txt 之一，训练时会重新划分")
        else:
            notes.append("尚未划分训练/验证集（dfwalk split；train 时也会自动划分）")
        stats_targets.append(("全部", valid, len(valid)))

    print("标签统计（像素占比）:")
    for name, stems, n_listed in stats_targets:
        s = label_stats(paths.root, stems)
        excluded = f"（另有 {n_listed - len(stems)} 个无效/缺失样本未计入）" if n_listed > len(stems) else ""
        print(
            f"  {name:<5} {s['frames']:>6} 帧  可走 {_percent(s['walkable_ratio'])}  "
            f"不可走 {_percent(s['non_walkable_ratio'])}  忽略 {_percent(s['ignore_ratio'])}{excluded}"
        )

    # 可视化：可走区域绿色、忽略区域压暗（灰），均匀抽样覆盖不同视频
    if vis_dir is not None:
        from .infer import overlay
        from .utils import write_image_rgb

        chosen = _evenly_spaced(valid, args.vis_limit)
        for stem in chosen:
            image = read_image_rgb(images[stem])
            mask = read_mask(masks[stem])
            write_image_rgb(vis_dir / f"{stem}_label.jpg", overlay(image, mask, (0, 255, 0), cfg.infer.alpha))
        print(f"可视化 {len(chosen)} 张 → {vis_dir}（绿色 = 可走，压暗 = 忽略）")

    for note in notes:
        print(f"[提示] {note}")
    for err in errors:
        print(f"[错误] {err}")
    if errors:
        print(f"检查完成：发现 {len(errors)} 类错误，请修复后再训练")
        return 1
    print("检查完成：未发现错误")
    return 0


# ---------------------------------------------------------------- train / eval


def format_metrics_table(metrics: Mapping[str, float]) -> str:
    """把指标字典格式化为对齐的表格（指标名、数值、说明）。按 METRIC_ORDER 排序，其余指标随后，loss 最后。"""
    keys = [k for k in METRIC_ORDER if k in metrics]
    keys += [k for k in metrics if k not in keys and k != "loss"]
    if "loss" in metrics:
        keys.append("loss")
    if not keys:
        return "（没有指标）"
    width = max(len("指标"), *(len(k) for k in keys))

    def fmt(v: Any) -> str:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return str(v)
        return "nan" if math.isnan(x) else f"{x:.4f}"

    lines = [f"{'指标':<{width - 2}}  {'数值':>8}  说明", "-" * (width + 30)]
    for k in keys:
        lines.append(f"{k:<{width}}  {fmt(metrics[k]):>10}  {METRIC_DESCRIPTIONS.get(k, '')}".rstrip())
    return "\n".join(lines)


def _cmd_train(args: argparse.Namespace, cfg: Config) -> int:
    from .train import train

    _override(cfg.train, epochs=args.epochs, batch_size=args.batch_size, out_dir=args.out_dir, device=args.device)
    _validate(cfg)
    best = train(cfg)
    print(f"最佳 checkpoint: {best}")
    _info(f"下一步: dfwalk eval --ckpt {best.as_posix()}，或 dfwalk infer --ckpt {best.as_posix()} <图片/目录/视频>")
    return 0


def _cmd_eval(args: argparse.Namespace, cfg: Config) -> int:
    from .dataset import read_split
    from .train import evaluate_checkpoint

    _override(cfg.train, device=args.device)
    _validate(cfg)
    ckpt = _require_file(args.ckpt, "checkpoint ")
    n = len(read_split(cfg.data.root, args.split))
    metrics = evaluate_checkpoint(cfg, ckpt, split=args.split)
    print(f"checkpoint: {ckpt}  划分: {args.split}（{n} 帧）")
    print(format_metrics_table(metrics))
    return 0


# ---------------------------------------------------------------- infer


def _collect_infer_inputs(inputs: Sequence[str]) -> tuple[list[Path], list[Path]]:
    """输入 → (图片列表, 视频列表)。文件按后缀区分（VIDEO_EXTS 为视频，其余当作图片）；
    目录取其第一层的图片和视频。按出现顺序去重；路径不存在抛 FileNotFoundError。"""
    from .utils import list_images

    images: list[Path] = []
    videos: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        key = p.resolve()
        if key in seen:
            return
        seen.add(key)
        (videos if p.suffix.lower() in VIDEO_EXTS else images).append(p)

    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = list_images(p) + sorted(c for c in p.iterdir() if c.is_file() and c.suffix.lower() in VIDEO_EXTS)
            if not found:
                _warn(f"目录中没有图片或视频: {p}")
            for c in found:
                add(c)
        elif p.is_file():
            add(p)
        else:
            raise FileNotFoundError(f"输入不存在: {p}")
    return images, videos


def _check_positive_int(value: Any, name: str, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是 >= 1 的整数，得到 {value!r}")


def _cmd_infer(args: argparse.Namespace, cfg: Config) -> int:
    from .infer import Predictor, infer_images, infer_video

    _override(cfg.infer, threshold=args.threshold, device=args.device)
    _validate(cfg)
    _check_positive_int(args.every_n, "--every-n")
    _check_positive_int(args.max_frames, "--max-frames", allow_none=True)
    ckpt = _require_file(args.ckpt, "checkpoint ")
    images, videos = _collect_infer_inputs(args.inputs)
    if not images and not videos:
        raise CLIError("没有找到可推理的图片或视频")
    out_dir = Path(args.output)

    # 同名视频的输出会互相覆盖：在开始（耗时的）推理前检查
    by_stem: dict[str, list[Path]] = {}
    for v in videos:
        by_stem.setdefault(v.stem.casefold(), []).append(v)
    dups = [vs for vs in by_stem.values() if len(vs) > 1]
    if dups:
        detail = "；".join(", ".join(str(v) for v in vs) for vs in dups)
        raise CLIError(f"多个视频的文件名（stem）相同，输出会互相覆盖: {detail}")

    inf = cfg.infer
    predictor = Predictor.from_checkpoint(ckpt, device=inf.device)
    h, w = predictor.image_size
    _info(f"模型: {ckpt}（输入 {h}x{w}），设备: {predictor.device}，阈值: {inf.threshold}")

    if images:
        written = infer_images(
            predictor,
            images,
            out_dir,
            threshold=inf.threshold,
            alpha=inf.alpha,
            color=inf.color,
            hud_rects=cfg.hud.rects,
            save_mask=not args.no_mask,
        )
        what = "叠加图" if args.no_mask else "叠加图（*_overlay.jpg）和掩码（*_mask.png）"
        print(f"图片: {len(written)} 张{what} → {out_dir}", flush=True)
    for video in videos:
        out_path = out_dir / f"{video.stem}_overlay.mp4"
        infer_video(
            predictor,
            video,
            out_path,
            threshold=inf.threshold,
            alpha=inf.alpha,
            color=inf.color,
            hud_rects=cfg.hud.rects,
            every_n=args.every_n,
            max_frames=args.max_frames,
        )
        print(f"视频: {video} → {out_path}", flush=True)
    return 0


# ---------------------------------------------------------------- export-onnx


def _cmd_export_onnx(args: argparse.Namespace, cfg: Config) -> int:
    from .export import export_onnx

    ckpt = _require_file(args.ckpt, "checkpoint ")
    out = export_onnx(ckpt, args.output, opset=args.opset, dynamic_batch=not args.static_batch)
    batch = "1" if args.static_batch else "batch（动态）"
    print(f"已导出 ONNX: {out}")
    print(
        f"  输入 image: float32 ({batch}, 3, H, W)，RGB，按 (x/255 - mean)/std 归一化（H/W/mean/std 见模型 metadata）\n"
        f"  输出 logits: ({batch}, 2, H, W)，未经 softmax，第 {WALKABLE} 通道为可走"
    )
    return 0


# ---------------------------------------------------------------- 参数解析


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """中文 “用法:” 前缀，保留描述文字中的换行。"""

    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, prefix if prefix is not None else "用法: ")


class _Parser(argparse.ArgumentParser):
    """错误信息使用中文前缀；退出码沿用 argparse 的 2。"""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: 错误: {message}\n")


def _localize(parser: argparse.ArgumentParser) -> None:
    """把 argparse 默认的英文分组标题换成中文。"""
    parser._positionals.title = "位置参数"
    parser._optionals.title = "选项"


def _add_help(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="显示此帮助信息并退出")


def _add_common_options(parser: argparse.ArgumentParser, dest_prefix: str = "") -> None:
    """-c/--config 与 --set。主命令上用不同的 dest，避免被子命令的默认值覆盖。"""
    group = parser.add_argument_group("通用选项")
    group.add_argument(
        "-c",
        "--config",
        dest=f"{dest_prefix}config",
        metavar="PATH",
        default=None,
        help="YAML 配置文件。不指定时，若当前目录有 configs/default.yaml 则使用它，否则使用内置默认值",
    )
    group.add_argument(
        "--set",
        dest=f"{dest_prefix}overrides",
        action="append",
        default=None,
        metavar="SECTION.KEY=VALUE",
        help="覆盖单个配置项，可重复；值按 YAML 解析，例如 --set train.epochs=5 --set \"data.image_size=[384,672]\"",
    )


_ROOT_NOTE = "（默认路径都在 data.root 下，可用 --set data.root=... 修改）"


def _add_command(
    sub: argparse._SubParsersAction,
    name: str,
    handler: Callable[[argparse.Namespace, Config], int],
    help_text: str,
    description: str,
) -> argparse.ArgumentParser:
    p = sub.add_parser(
        name,
        help=help_text,
        description=description,
        formatter_class=_HelpFormatter,
        add_help=False,
    )
    _localize(p)
    _add_help(p)
    _add_common_options(p)
    p.set_defaults(handler=handler)
    return p


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器（main 使用；单独暴露便于测试和生成文档）。"""
    parser = _Parser(
        prog=PROG,
        description=(
            "《三角洲行动》第一人称画面可走区域识别（离线分析工具）。\n"
            "只处理自己录制的视频和截图文件：不读取游戏进程、不注入输入、不控制角色。\n\n"
            "典型流程: extract-frames → hud-preview → autolabel → to-labelme → (LabelMe 修正) → from-labelme\n"
            "          → split → check-data → train → eval → infer → export-onnx"
        ),
        epilog="每个子命令的详细参数: dfwalk <子命令> -h。设置环境变量 DFWALK_DEBUG=1 可在出错时查看完整堆栈。",
        formatter_class=_HelpFormatter,
        add_help=False,
    )
    _localize(parser)
    _add_help(parser)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}", help="显示版本号并退出")
    _add_common_options(parser, dest_prefix="global_")
    sub = parser.add_subparsers(dest="command", title="子命令", metavar="<子命令>")

    # extract-frames
    p = _add_command(
        sub, "extract-frames", _cmd_extract_frames, "从录像抽帧",
        "按目标帧率从自己录制的视频中抽帧并去重，写成 {视频名}_{帧序号:06d}.jpg。\n"
        "同一视频的帧按文件名前缀分组，划分训练/验证集时不会被拆开。" + _ROOT_NOTE,
    )
    p.add_argument("inputs", nargs="+", metavar="INPUT",
                   help=f"视频文件，或包含视频的目录（处理目录第一层中 {'/'.join(VIDEO_EXTS)} 后缀的文件）")
    p.add_argument("-o", "--output", metavar="DIR", help="输出目录（默认 data.root/images）")
    p.add_argument("--fps", type=float, metavar="FPS", help="每秒抽取的帧数（默认取 frames.fps；<= 0 表示每帧都取）")
    p.add_argument("--max-side", type=int, metavar="PX",
                   help="最长边缩放到该值（默认取 frames.max_side；0 表示不缩放）")
    p.add_argument("--dedup", type=float, metavar="THRESHOLD",
                   help="去重阈值：与上一张保留帧的平均灰度差低于它则丢弃（默认取 frames.dedup_threshold；0 关闭去重）")

    # hud-preview
    p = _add_command(
        sub, "hud-preview", _cmd_hud_preview, "在截图上预览 HUD 矩形",
        "把配置中的 hud.rects 画到一张截图上（半透明填充 + 序号），用于核对小地图、血条、弹药等 HUD 区域是否覆盖完整。",
    )
    p.add_argument("image", metavar="IMAGE", help="一张截图或抽出的帧")
    p.add_argument("-o", "--output", default="hud_preview.jpg", metavar="PATH",
                   help="预览图输出路径（默认 hud_preview.jpg）")

    # autolabel
    p = _add_command(
        sub, "autolabel", _cmd_autolabel, "生成伪标签",
        "用 ADE20K 预训练的 SegFormer 零样本生成可走区域伪标签（0 不可走 / 1 可走 / 255 忽略），HUD 区域置为忽略。\n"
        "游戏画面与真实照片有差异，伪标签必须人工检查修正。需要 transformers（pip install -e \".[autolabel]\"）。"
        + _ROOT_NOTE,
    )
    p.add_argument("-i", "--input", metavar="DIR", help="图片目录（默认 data.root/images）")
    p.add_argument("-o", "--output", metavar="DIR", help="伪标签输出目录（默认 data.root/pseudo_masks）")
    p.add_argument("--model", metavar="NAME_OR_DIR", help="Hugging Face 模型名或本地目录（默认取 autolabel.model）")
    p.add_argument("--device", metavar="DEVICE", help="auto / cpu / cuda / cuda:0 / mps（默认取 autolabel.device）")
    p.add_argument("--overwrite", action="store_true", help="重新生成已存在的伪标签")

    # to-labelme
    p = _add_command(
        sub, "to-labelme", _cmd_to_labelme, "伪标签 → LabelMe JSON",
        "把掩码转换为 LabelMe 多边形标注（walkable / non_walkable / ignore），以便在 LabelMe 中人工修正。\n"
        "已存在的 JSON 默认不会被覆盖（保护人工修正结果）。" + _ROOT_NOTE,
    )
    p.add_argument("--masks", metavar="DIR", help="掩码目录（默认 data.root/pseudo_masks）")
    p.add_argument("--images", metavar="DIR", help="图片目录（默认 data.root/images）")
    p.add_argument("-o", "--output", metavar="DIR", help="JSON 输出目录（默认 data.root/labelme）")
    p.add_argument("--epsilon", type=float, default=1.0, metavar="PX",
                   help="多边形简化程度（像素，默认 1.0；0 表示不简化、精确还原）")
    p.add_argument("--min-area", type=float, default=16.0, metavar="PX", help="丢弃小于该像素数的碎片区域（默认 16）")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的 JSON（会丢失人工修正！）")

    # from-labelme
    p = _add_command(
        sub, "from-labelme", _cmd_from_labelme, "LabelMe JSON → 掩码",
        "把（人工修正后的）LabelMe JSON 转回训练用掩码 PNG，并把 hud.rects 区域置为忽略。" + _ROOT_NOTE,
    )
    p.add_argument("-i", "--input", metavar="DIR", help="LabelMe JSON 目录（默认 data.root/labelme）")
    p.add_argument("-o", "--output", metavar="DIR", help="掩码输出目录（默认 data.root/masks）")

    # split
    p = _add_command(
        sub, "split", _cmd_split, "划分训练/验证集",
        "对同时有图片和掩码的样本按视频分组划分训练/验证集，写入 splits/train.txt 与 val.txt。\n"
        "同一段录像的帧不会同时出现在两边。已有划分文件时默认拒绝覆盖。" + _ROOT_NOTE,
    )
    p.add_argument("--val-ratio", type=float, metavar="R", help="验证集占比（默认取 data.val_ratio）")
    p.add_argument("--seed", type=int, metavar="N", help="随机种子（默认取 data.seed）")
    p.add_argument("--force", action="store_true", help="覆盖已有的划分文件")

    # check-data
    p = _add_command(
        sub, "check-data", _cmd_check_data, "检查数据并可视化标签",
        "统计图片/掩码/伪标签/LabelMe 数量，检查缺失、非法掩码值、尺寸不一致和划分文件，输出标签像素占比；\n"
        "可选输出标签叠加图用于抽查。发现错误时退出码为 1。" + _ROOT_NOTE,
    )
    p.add_argument("--vis-dir", metavar="DIR", help="把标签叠加图（绿色 = 可走，压暗 = 忽略）写到该目录")
    p.add_argument("--vis-limit", type=int, default=50, metavar="N",
                   help="最多可视化多少张（在全部样本中均匀抽取，默认 50；0 表示全部）")

    # train
    p = _add_command(
        sub, "train", _cmd_train, "训练",
        "训练可走区域分割模型。输出目录中有 best.pt（验证集 iou_walkable 最高）、last.pt、history.csv、config.yaml。\n"
        "没有划分文件时会自动按视频分组划分。其余超参数用 --set train.xxx=... 修改。",
    )
    p.add_argument("--epochs", type=int, metavar="N", help="训练轮数（默认取 train.epochs）")
    p.add_argument("--batch-size", type=int, metavar="N", help="batch 大小（默认取 train.batch_size）")
    p.add_argument("--out-dir", metavar="DIR", help="输出目录（默认取 train.out_dir）")
    p.add_argument("--device", metavar="DEVICE", help="auto / cpu / cuda / cuda:0 / mps（默认取 train.device）")

    # eval
    p = _add_command(
        sub, "eval", _cmd_eval, "评估 checkpoint",
        "在 splits/{split}.txt 上评估 checkpoint，打印各项指标。",
    )
    p.add_argument("--ckpt", required=True, metavar="PATH", help="checkpoint 路径（如 runs/exp/best.pt）")
    p.add_argument("--split", default="val", metavar="NAME", help="评估的划分名（默认 val）")
    p.add_argument("--device", metavar="DEVICE", help="auto / cpu / cuda（默认取 train.device）")

    # infer
    p = _add_command(
        sub, "infer", _cmd_infer, "图片/目录/视频推理",
        "对截图、图片目录或录像推理可走区域，输出叠加可视化：\n"
        "  图片 → {stem}_overlay.jpg（及 {stem}_mask.png，0/1/255）；视频 → {stem}_overlay.mp4。\n"
        "按后缀区分图片和视频；目录处理其第一层的图片和视频。",
    )
    p.add_argument("--ckpt", required=True, metavar="PATH", help="checkpoint 路径（如 runs/exp/best.pt）")
    p.add_argument("inputs", nargs="+", metavar="INPUT", help="图片文件、图片目录或视频文件")
    p.add_argument("-o", "--output", default="runs/infer", metavar="DIR", help="输出目录（默认 runs/infer）")
    p.add_argument("--threshold", type=float, metavar="T", help="可走概率阈值（默认取 infer.threshold；越大越保守）")
    p.add_argument("--every-n", type=int, default=1, metavar="N",
                   help="视频每 N 帧推理一次，中间帧复用上一次结果（默认 1）")
    p.add_argument("--max-frames", type=int, metavar="N", help="视频最多处理的帧数（默认全部）")
    p.add_argument("--no-mask", action="store_true", help="图片只输出叠加图，不输出掩码 PNG")
    p.add_argument("--device", metavar="DEVICE", help="auto / cpu / cuda / cuda:0 / mps（默认取 infer.device）")

    # export-onnx
    p = _add_command(
        sub, "export-onnx", _cmd_export_onnx, "导出 ONNX",
        "把 checkpoint 导出为 ONNX（输入 image: (N,3,H,W) 已归一化的 RGB；输出 logits: (N,2,H,W)），\n"
        "预处理参数写在模型 metadata 中。需要 onnx 包（pip install -e \".[onnx]\"）。",
    )
    p.add_argument("--ckpt", required=True, metavar="PATH", help="checkpoint 路径（如 runs/exp/best.pt）")
    p.add_argument("-o", "--output", default="model.onnx", metavar="PATH", help="输出路径（默认 model.onnx）")
    p.add_argument("--opset", type=int, default=17, metavar="N", help="ONNX opset 版本（默认 17）")
    p.add_argument("--static-batch", action="store_true", help="batch 维固定为 1（默认动态 batch）")
    return parser


def _merge_common(args: argparse.Namespace) -> tuple[str | None, list[str]]:
    """合并子命令前后的 -c 与 --set：-c 两处都给且不同时报错；--set 先主命令后子命令。"""
    global_cfg = getattr(args, "global_config", None)
    local_cfg = getattr(args, "config", None)
    if global_cfg and local_cfg and global_cfg != local_cfg:
        raise CLIError(f"-c/--config 指定了两次且不一致: {global_cfg} 与 {local_cfg}")
    overrides = list(getattr(args, "global_overrides", None) or []) + list(getattr(args, "overrides", None) or [])
    return local_cfg or global_cfg, overrides


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口，返回退出码（0 成功，1 运行错误或 check-data 发现问题，2 参数错误，130 被中断）。"""
    _relax_stream_encoding()
    parser = build_parser()
    try:
        args = parser.parse_args(None if argv is None else list(argv))
    except SystemExit as e:  # --help / --version / 参数错误
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 2

    debug = _debug_enabled()
    with warnings.catch_warnings():
        warnings.showwarning = _make_showwarning(warnings.showwarning)
        try:
            config_path, overrides = _merge_common(args)
            cfg = load_cli_config(config_path, overrides)
            return int(args.handler(args, cfg) or 0)
        except (CLIError, *_HANDLED_ERRORS) as e:
            if debug:
                raise
            print(f"错误: {_error_message(e)}", file=sys.stderr, flush=True)
            if isinstance(e, _UNEXPECTED_ERRORS) or (isinstance(e, OSError) and not isinstance(e, FileNotFoundError)):
                print(f"（设置环境变量 {DEBUG_ENV}=1 可查看完整错误堆栈）", file=sys.stderr, flush=True)
            return 1
        except KeyboardInterrupt:
            if debug:
                raise
            print("\n已中断", file=sys.stderr, flush=True)
            return 130
