"""导出 ONNX，便于在其他推理框架（onnxruntime、TensorRT 等）中部署做离线分析。

导出器选择：
    优先使用 TorchScript（legacy）导出器：不需要额外依赖、速度快，原生支持 opset 17，且对 smp 的
    Unet / FPN / DeepLabV3Plus / UnetPlusPlus / Segformer 等结构都能正确导出。
    PyTorch 2.9 起它已被标记为弃用；若当前 torch 中 legacy 导出失败（例如未来版本移除了它，或 opset 超出其支持范围），
    且安装了 onnxscript，则自动改用 torch.export（dynamo）导出器并给出警告。

导出后（需要 onnx 包）：
    - 用 PyTorch 前向得到的真实输出形状改写 "logits" 的形状声明——legacy 导出器对 Unet / Segformer 会给出
      ["batch", 2, "batch", "..."] 这类错误的符号维度，部分下游工具（TensorRT 等）会因此推错形状；
    - 写入 metadata_props（image_size / mean / std / 结构等），部署时可直接读出预处理参数；
    - 运行 onnx.checker.check_model。
"""

from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path

from . import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, __version__

INPUT_NAME = "image"
OUTPUT_NAME = "logits"
BATCH_DIM = "batch"
# metadata_props 键名前缀
META_PREFIX = "dfwalk."

_ONNX_MISSING_HINT = "导出 ONNX 需要 onnx 包：pip install onnx（或 pip install -e \".[onnx]\"）"


def _onnx_available() -> bool:
    try:
        import onnx  # noqa: F401
    except ImportError:
        return False
    return True


def _supports_dynamo_kwarg() -> bool:
    """torch.onnx.export 是否有 dynamo 参数（torch >= 2.5）。"""
    import torch

    try:
        return "dynamo" in inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):  # pragma: no cover - 签名不可获取的极端情况
        return False


def _dynamo_available() -> bool:
    """能否使用 torch.export（dynamo）导出器：torch 支持 dynamo 参数且安装了 onnxscript。"""
    if not _supports_dynamo_kwarg():
        return False
    try:
        import onnxscript  # noqa: F401
    except ImportError:
        return False
    return True


def _export_legacy(model, dummy, path: Path, opset: int, dynamic_batch: bool) -> None:
    """TorchScript（legacy）导出器。"""
    import torch

    kwargs: dict = dict(
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=opset,
        do_constant_folding=True,
    )
    if dynamic_batch:
        kwargs["dynamic_axes"] = {INPUT_NAME: {0: BATCH_DIM}, OUTPUT_NAME: {0: BATCH_DIM}}
    if _supports_dynamo_kwarg():
        kwargs["dynamo"] = False
    tracer_warning = getattr(getattr(torch, "jit", None), "TracerWarning", None)
    with warnings.catch_warnings():
        # legacy 导出器的弃用提示；以及 smp 解码器中针对空间尺寸的 TracerWarning——
        # 导出时 H/W 固定为 checkpoint 的 image_size，只有 batch 是动态维度，这些警告不影响结果
        warnings.simplefilter("ignore", DeprecationWarning)
        if tracer_warning is not None:
            warnings.simplefilter("ignore", tracer_warning)
        torch.onnx.export(model, (dummy,), str(path), **kwargs)


def _export_dynamo(model, dummy, path: Path, opset: int, dynamic_batch: bool) -> None:
    """torch.export（dynamo）导出器，需要 onnxscript。"""
    import torch

    kwargs: dict = dict(
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=opset,
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    if dynamic_batch:
        kwargs["dynamic_shapes"] = ({0: torch.export.Dim(BATCH_DIM, min=1)},)
    torch.onnx.export(model, (dummy,), str(path), **kwargs)


def _default_opset(model) -> int | None:
    """onnx.ModelProto 默认域（"" / "ai.onnx"）的 opset 版本；没有声明时返回 None。"""
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return int(entry.version)
    return None


def onnx_opset(path: str | Path) -> int | None:
    """读取 ONNX 文件实际的 opset 版本（默认域）；没有安装 onnx 包或文件中没有声明时返回 None。"""
    if not _onnx_available():
        return None
    import onnx

    return _default_opset(onnx.load(str(path), load_external_data=False))


def _finalize(path: Path, in_shape: tuple[int, ...], out_shape: tuple[int, ...], dynamic_batch: bool,
              metadata: dict[str, str]) -> int | None:
    """改写输入/输出形状声明（dynamic_batch 时第 0 维写成符号 "batch"）、写入 metadata，
    并运行 onnx.checker（需要 onnx）。返回文件实际的 opset 版本（默认域）。"""
    import onnx

    model = onnx.load(str(path))

    def _set_shape(value_info, dims: tuple[int, ...]) -> None:
        shape = value_info.type.tensor_type.shape
        shape.ClearField("dim")
        for i, size in enumerate(dims):
            d = shape.dim.add()
            if i == 0 and dynamic_batch:
                d.dim_param = BATCH_DIM
            else:
                d.dim_value = int(size)

    for inp in model.graph.input:
        if inp.name == INPUT_NAME:
            _set_shape(inp, in_shape)
    for out in model.graph.output:
        if out.name == OUTPUT_NAME:
            _set_shape(out, out_shape)

    existing = {p.key: p for p in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            prop = model.metadata_props.add()
            prop.key, prop.value = key, value

    onnx.save(model, str(path))
    onnx.checker.check_model(str(path))
    return _default_opset(model)


def export_onnx(checkpoint_path: str | Path, out_path: str | Path, opset: int = 17,
                dynamic_batch: bool = True) -> Path:
    """加载 checkpoint，导出 ONNX：
    输入名 "image"：float32 (N, 3, H, W)，H/W 为 checkpoint 的 image_size，已按 checkpoint 的 mean/std 归一化；
    输出名 "logits"：(N, 2, H, W)。dynamic_batch 时 N 为动态维度。
    若安装了 onnx，导出后运行 onnx.checker.check_model。返回 out_path。

    补充约定：
    - 动态 batch 维度命名为 "batch"；dynamic_batch=False 时 N 固定为 1。H/W 始终是固定值。
    - 输入是 RGB、(x / 255 - mean) / std 归一化后的张量（与 Predictor / EvalTransform 一致），
      ONNX 模型本身不含归一化；输出是未经 softmax 的 logits，第 1 通道（WALKABLE）为可走。
    - 安装了 onnx 时，在 metadata_props 中写入 "dfwalk.image_size"、"dfwalk.mean"、"dfwalk.std"、
      "dfwalk.arch"、"dfwalk.encoder"、"dfwalk.num_classes"、"dfwalk.class_names"、"dfwalk.version"
      （值为 JSON 字符串），并按真实形状改写输入/输出形状声明（见模块说明）。
    - 先写入同目录临时文件，全部成功后再改名为 out_path；失败时不留下半成品。自动创建父目录。
    - 文件实际的 opset 与要求的不同（dynamo 导出器无法降到要求的 opset，只能给出 18）时发出 UserWarning，
      写明实际 opset；需要时可用 onnx_opset(out_path) 读取。
    - checkpoint 不存在抛 FileNotFoundError，不是 dfwalk checkpoint 抛 ValueError；
      opset 不是正整数抛 ValueError；导出需要 onnx 包而未安装时抛 ImportError；导出失败抛 RuntimeError。
    """
    import torch

    from .checkpoint import load_checkpoint, model_from_checkpoint

    if isinstance(opset, bool) or not isinstance(opset, int) or opset < 1:
        raise ValueError(f"opset 必须是正整数，得到 {opset!r}")
    checkpoint_path = Path(checkpoint_path)
    out_path = Path(out_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    state = load_checkpoint(checkpoint_path, map_location="cpu")
    model = model_from_checkpoint(state)  # eval 模式，CPU
    h, w = (int(v) for v in state["image_size"])
    # 动态 batch 时用 batch=2 的示例输入，避免导出器把 1 当作常量特化
    dummy = torch.zeros(2 if dynamic_batch else 1, 3, h, w, dtype=torch.float32)
    with torch.no_grad():
        out_shape = tuple(int(s) for s in model(dummy).shape)
    if dynamic_batch:
        out_shape = (1,) + out_shape[1:]  # 第 0 维会被写成符号维度 "batch"

    model_cfg = state.get("model_cfg", {})
    metadata = {
        f"{META_PREFIX}version": json.dumps(__version__),
        f"{META_PREFIX}image_size": json.dumps([h, w]),
        f"{META_PREFIX}mean": json.dumps([float(v) for v in state.get("mean", IMAGENET_MEAN)]),
        f"{META_PREFIX}std": json.dumps([float(v) for v in state.get("std", IMAGENET_STD)]),
        f"{META_PREFIX}arch": json.dumps(model_cfg.get("arch")),
        f"{META_PREFIX}encoder": json.dumps(model_cfg.get("encoder")),
        f"{META_PREFIX}num_classes": json.dumps(model_cfg.get("num_classes")),
        f"{META_PREFIX}class_names": json.dumps(list(CLASS_NAMES), ensure_ascii=False),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    have_onnx = _onnx_available()
    actual_opset: int | None = None
    try:
        try:
            _export_legacy(model, dummy, tmp, opset, dynamic_batch)
        except Exception as legacy_err:
            if not have_onnx:
                raise ImportError(f"{_ONNX_MISSING_HINT}（导出器报错：{legacy_err}）") from legacy_err
            if not _dynamo_available():
                raise RuntimeError(
                    f"ONNX 导出失败（TorchScript 导出器）：{legacy_err}。"
                    "可安装 onnxscript 以启用 torch.export（dynamo）导出器后重试"
                ) from legacy_err
            warnings.warn(
                f"TorchScript 导出器失败（{type(legacy_err).__name__}: {legacy_err}），"
                "改用 torch.export（dynamo）导出器",
                stacklevel=2,
            )
            tmp.unlink(missing_ok=True)
            try:
                _export_dynamo(model, dummy, tmp, opset, dynamic_batch)
            except Exception as dynamo_err:
                raise RuntimeError(
                    f"ONNX 导出失败：TorchScript 导出器 {type(legacy_err).__name__}: {legacy_err}；"
                    f"dynamo 导出器 {type(dynamo_err).__name__}: {dynamo_err}"
                ) from dynamo_err
        if have_onnx:
            actual_opset = _finalize(tmp, (1, 3, h, w), out_shape, dynamic_batch, metadata)
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)
    if actual_opset is not None and actual_opset != opset:
        # torch.export（dynamo）导出器只能生成 opset >= 18，降级失败时会保留 18（例如 UPerNet 在默认尺寸下总是走到这里）
        warnings.warn(
            f"导出的 ONNX 模型实际 opset 为 {actual_opset}，不是要求的 {opset}："
            f"改用的 torch.export（dynamo）导出器无法把模型转换到 opset {opset}。"
            f"只支持 opset {opset} 及以下的推理引擎（例如较旧的 TensorRT）可能无法加载它；"
            "必须使用该 opset 时，请换用 Unet / FPN 等 TorchScript 导出器能处理的结构",
            stacklevel=2,
        )
    return out_path
