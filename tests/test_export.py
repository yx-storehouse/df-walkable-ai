"""dfwalk.export 的单元测试：ONNX 导出的输入/输出名与形状、动态 batch、metadata、
onnxruntime 数值一致性（需要 onnxruntime）、dynamo 兜底（需要 onnxscript）以及各种错误路径。

全部离线、CPU：模型为随机初始化的 Unet-resnet18，image_size [64, 96]。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")

import dfwalk.export as export_mod  # noqa: E402
from dfwalk import CLASS_NAMES, WALKABLE, __version__  # noqa: E402
from dfwalk.checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint  # noqa: E402
from dfwalk.export import export_onnx  # noqa: E402
from dfwalk.infer import Predictor  # noqa: E402
from dfwalk.model import build_model, model_config  # noqa: E402
from dfwalk.transforms import to_tensor_normalized  # noqa: E402

IMAGE_SIZE = (64, 96)
CUSTOM_MEAN = (0.45, 0.4, 0.35)
CUSTOM_STD = (0.2, 0.22, 0.25)


# ---------------------------------------------------------------- 工具


def dims(value_info) -> list:
    """ONNX 张量形状 → 列表（符号维度为字符串，固定维度为 int）。"""
    return [d.dim_param if d.dim_param else d.dim_value for d in value_info.type.tensor_type.shape.dim]


def opset_of(model) -> int:
    return next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))


def metadata_of(model) -> dict[str, str]:
    return {p.key: p.value for p in model.metadata_props}


def torch_logits(ckpt: Path, x: np.ndarray) -> np.ndarray:
    model = model_from_checkpoint(load_checkpoint(ckpt))
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def ort_session(path: Path):
    ort = pytest.importorskip("onnxruntime")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def assert_no_leftovers(folder: Path, keep: set[str]) -> None:
    names = {p.name for p in folder.iterdir()}
    assert names == keep, names


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    model = build_model("Unet", "resnet18", encoder_weights=None)
    # 让 BatchNorm 的 running stats 不是默认值，数值对比更有意义
    with torch.no_grad():
        model.train()
        model(torch.randn(4, 3, *IMAGE_SIZE))
        model.eval()
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    save_checkpoint(path, model, model_config("Unet", "resnet18"), list(IMAGE_SIZE), epoch=7,
                    metrics={"iou_walkable": 0.5}, mean=CUSTOM_MEAN, std=CUSTOM_STD)
    return path


@pytest.fixture(scope="module")
def dynamic_onnx(ckpt, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("onnx") / "model.onnx"
    # 正常导出不应产生任何警告（legacy 导出器的弃用提示与 TracerWarning 已在内部屏蔽）
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = export_onnx(ckpt, out)
    assert result == out
    return out


@pytest.fixture(scope="module")
def static_onnx(ckpt, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("onnx_static") / "static.onnx"
    assert export_onnx(ckpt, out, opset=13, dynamic_batch=False) == out
    return out


# ---------------------------------------------------------------- 结构


def test_dynamic_export_names_shapes_opset(dynamic_onnx):
    model = onnx.load(str(dynamic_onnx))
    onnx.checker.check_model(model)
    assert [i.name for i in model.graph.input] == ["image"]
    assert [o.name for o in model.graph.output] == ["logits"]
    assert dims(model.graph.input[0]) == ["batch", 3, 64, 96]
    # 输出形状声明被修正为真实形状（legacy 导出器原本会给出 ["batch", 2, "batch", ...]）
    assert dims(model.graph.output[0]) == ["batch", 2, 64, 96]
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert opset_of(model) == 17
    # 权重以 initializer 保存，不是额外输入
    assert len(model.graph.initializer) > 0
    assert_no_leftovers(dynamic_onnx.parent, {"model.onnx"})


def test_static_export_fixed_batch_and_opset(static_onnx):
    model = onnx.load(str(static_onnx))
    onnx.checker.check_model(model)
    assert dims(model.graph.input[0]) == [1, 3, 64, 96]
    assert dims(model.graph.output[0]) == [1, 2, 64, 96]
    assert opset_of(model) == 13


def test_metadata_describes_preprocessing(dynamic_onnx):
    meta = metadata_of(onnx.load(str(dynamic_onnx)))
    assert json.loads(meta["dfwalk.image_size"]) == list(IMAGE_SIZE)
    assert json.loads(meta["dfwalk.mean"]) == pytest.approx(CUSTOM_MEAN)
    assert json.loads(meta["dfwalk.std"]) == pytest.approx(CUSTOM_STD)
    assert json.loads(meta["dfwalk.arch"]) == "Unet"
    assert json.loads(meta["dfwalk.encoder"]) == "resnet18"
    assert json.loads(meta["dfwalk.num_classes"]) == 2
    assert json.loads(meta["dfwalk.class_names"]) == list(CLASS_NAMES)
    assert json.loads(meta["dfwalk.version"]) == __version__


# ---------------------------------------------------------------- onnxruntime 数值一致性


@pytest.mark.parametrize("batch", [1, 3])
def test_onnxruntime_parity_dynamic_batch(ckpt, dynamic_onnx, batch):
    sess = ort_session(dynamic_onnx)
    x = np.random.default_rng(batch).standard_normal((batch, 3, *IMAGE_SIZE)).astype(np.float32)
    (out,) = sess.run(["logits"], {"image": x})
    assert out.shape == (batch, 2, *IMAGE_SIZE)
    np.testing.assert_allclose(out, torch_logits(ckpt, x), rtol=1e-4, atol=1e-4)


def test_onnxruntime_parity_static_batch(ckpt, static_onnx):
    sess = ort_session(static_onnx)
    x = np.random.default_rng(5).standard_normal((1, 3, *IMAGE_SIZE)).astype(np.float32)
    (out,) = sess.run(None, {"image": x})
    np.testing.assert_allclose(out, torch_logits(ckpt, x), rtol=1e-4, atol=1e-4)
    with pytest.raises(Exception, match="(?i)dimension"):  # 固定 batch=1，不接受其他 batch
        sess.run(None, {"image": np.zeros((2, 3, *IMAGE_SIZE), np.float32)})


def test_onnx_matches_predictor_with_checkpoint_normalization(ckpt, dynamic_onnx):
    """ONNX 输入是按 checkpoint mean/std 归一化后的张量：softmax 后应与 Predictor.predict_prob 一致。"""
    sess = ort_session(dynamic_onnx)
    img = np.random.default_rng(9).integers(0, 256, size=(*IMAGE_SIZE, 3), dtype=np.uint8)
    # 部署端从 onnxruntime 读出 metadata 中的归一化参数
    meta = sess.get_modelmeta().custom_metadata_map
    mean, std = json.loads(meta["dfwalk.mean"]), json.loads(meta["dfwalk.std"])
    assert mean == pytest.approx(CUSTOM_MEAN) and std == pytest.approx(CUSTOM_STD)
    x = to_tensor_normalized(img, mean, std).unsqueeze(0).numpy()
    (logits,) = sess.run(["logits"], {"image": x})
    prob = torch.softmax(torch.from_numpy(logits), dim=1)[0, WALKABLE].numpy()
    expected = Predictor.from_checkpoint(ckpt, device="cpu").predict_prob(img)
    np.testing.assert_allclose(prob, expected, atol=1e-4)


def _readme_onnx_snippet() -> str:
    """README “部署 ONNX 模型” 一节中的 onnxruntime 示例代码。"""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    section = readme.split("## 部署 ONNX 模型", 1)[1]
    return section.split("```python", 1)[1].split("```", 1)[0]


@pytest.mark.parametrize("src_hw", [(128, 192), (48, 80), (80, 60), (40, 128)])
def test_readme_onnx_snippet_matches_predictor(ckpt, dynamic_onnx, tmp_path, monkeypatch, src_hw):
    """照抄 README 示例的用户应得到与 dfwalk infer 相同的概率：包括截图比模型输入小（需要放大）的情况，
    此时必须像 EvalTransform 一样用 INTER_LINEAR，而不是总用 INTER_AREA。"""
    pytest.importorskip("onnxruntime")
    import shutil

    from dfwalk.utils import read_image_rgb, write_image_rgb

    work = tmp_path / "work"
    (work / "runs" / "exp").mkdir(parents=True)
    shutil.copy(dynamic_onnx, work / "runs" / "exp" / "model.onnx")
    rng = np.random.default_rng(sum(src_hw))
    write_image_rgb(work / "shot.jpg", rng.integers(0, 256, size=(*src_hw, 3), dtype=np.uint8))
    monkeypatch.chdir(work)
    namespace: dict = {}
    exec(compile(_readme_onnx_snippet(), "README.md", "exec"), namespace)
    expected = Predictor.from_checkpoint(ckpt, device="cpu").predict_prob(read_image_rgb(work / "shot.jpg"))
    assert namespace["prob"].shape == src_hw
    np.testing.assert_allclose(namespace["prob"], expected, atol=1e-4)


# ---------------------------------------------------------------- 路径与覆盖


def test_creates_parent_dirs_unicode_path_and_overwrites(ckpt, tmp_path):
    out = tmp_path / "导出" / "子目录" / "可走模型.onnx"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"old file")
    # str 路径也可以；返回 Path；已存在的文件被覆盖
    result = export_onnx(str(ckpt), str(out), dynamic_batch=False)
    assert isinstance(result, Path) and result == out
    onnx.checker.check_model(str(out))
    assert dims(onnx.load(str(out)).graph.input[0]) == [1, 3, 64, 96]
    assert_no_leftovers(out.parent, {"可走模型.onnx"})
    # 父目录不存在时自动创建
    out2 = tmp_path / "新目录" / "m.onnx"
    assert export_onnx(ckpt, out2, dynamic_batch=False) == out2 and out2.is_file()


# ---------------------------------------------------------------- 错误路径


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_onnx(tmp_path / "missing.pt", tmp_path / "m.onnx")
    assert not (tmp_path / "m.onnx").exists()


def test_not_a_dfwalk_checkpoint(tmp_path):
    bogus = tmp_path / "bogus.pt"
    torch.save({"model_state": {}}, bogus)
    with pytest.raises(ValueError):
        export_onnx(bogus, tmp_path / "m.onnx")
    assert not (tmp_path / "m.onnx").exists()


@pytest.mark.parametrize("opset", [0, -1, True, 17.0, "17", None])
def test_invalid_opset(ckpt, tmp_path, opset):
    with pytest.raises(ValueError):
        export_onnx(ckpt, tmp_path / "m.onnx", opset=opset)
    assert not (tmp_path / "m.onnx").exists()


def test_legacy_failure_without_dynamo_raises_and_cleans_up(ckpt, tmp_path, monkeypatch):
    def boom(model, dummy, path, opset, dynamic_batch):
        Path(path).write_bytes(b"half written")  # 模拟写了一半的临时文件
        raise RuntimeError("legacy exporter exploded")

    monkeypatch.setattr(export_mod, "_export_legacy", boom)
    monkeypatch.setattr(export_mod, "_dynamo_available", lambda: False)
    with pytest.raises(RuntimeError, match="legacy exporter exploded"):
        export_onnx(ckpt, tmp_path / "out" / "m.onnx")
    assert_no_leftovers(tmp_path / "out", set())


def test_missing_onnx_package_gives_install_hint(ckpt, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("Module onnx is not installed!")

    monkeypatch.setattr(export_mod, "_export_legacy", boom)
    monkeypatch.setattr(export_mod, "_onnx_available", lambda: False)
    with pytest.raises(ImportError, match="pip install onnx"):
        export_onnx(ckpt, tmp_path / "m.onnx")
    assert not (tmp_path / "m.onnx").exists()


def test_without_onnx_skips_postprocessing(ckpt, tmp_path, monkeypatch):
    # 模拟没有 onnx 包（但导出器仍可用的旧版 torch）：只导出，不改写形状/metadata，也不运行 checker
    calls = []
    monkeypatch.setattr(export_mod, "_onnx_available", lambda: False)
    monkeypatch.setattr(export_mod, "_finalize", lambda *a, **k: calls.append(a))
    out = export_onnx(ckpt, tmp_path / "m.onnx")
    assert out.is_file() and calls == []
    model = onnx.load(str(out))
    assert [i.name for i in model.graph.input] == ["image"]
    assert "dfwalk.image_size" not in metadata_of(model)


def test_warns_when_fallback_exporter_ignores_requested_opset(ckpt, tmp_path, monkeypatch):
    """torch.export（dynamo）导出器降不到要求的 opset 时会保留 18（例如 UPerNet 在默认尺寸下）：
    不能悄悄写出与 --opset 不同的文件，要发出写明实际 opset 的警告。"""
    real_legacy = export_mod._export_legacy

    def legacy_fails(*args, **kwargs):
        raise RuntimeError("adaptive_avg_pool2d output size not a factor of input size")

    def dynamo_keeps_18(model, dummy, path, opset, dynamic_batch):
        real_legacy(model, dummy, path, 18, dynamic_batch)

    monkeypatch.setattr(export_mod, "_export_legacy", legacy_fails)
    monkeypatch.setattr(export_mod, "_dynamo_available", lambda: True)
    monkeypatch.setattr(export_mod, "_export_dynamo", dynamo_keeps_18)
    out = tmp_path / "m.onnx"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_onnx(ckpt, out, opset=13)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("opset 为 18" in m and "13" in m for m in messages), messages
    assert opset_of(onnx.load(str(out))) == 18
    assert export_mod.onnx_opset(out) == 18


def test_onnx_opset_reads_file(dynamic_onnx, static_onnx):
    assert export_mod.onnx_opset(dynamic_onnx) == 17
    assert export_mod.onnx_opset(static_onnx) == 13


def test_fallback_to_dynamo_exporter(ckpt, tmp_path, monkeypatch):
    pytest.importorskip("onnxscript")

    def boom(*args, **kwargs):
        raise RuntimeError("legacy exporter removed")

    monkeypatch.setattr(export_mod, "_export_legacy", boom)
    out = tmp_path / "dyn.onnx"
    with pytest.warns(UserWarning, match="dynamo"):
        export_onnx(ckpt, out)
    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    assert dims(model.graph.input[0]) == ["batch", 3, 64, 96]
    assert dims(model.graph.output[0]) == ["batch", 2, 64, 96]
    assert json.loads(metadata_of(model)["dfwalk.image_size"]) == list(IMAGE_SIZE)
    assert_no_leftovers(tmp_path, {"dyn.onnx"})
    try:
        sess = ort_session(out)
    except pytest.skip.Exception:
        return
    x = np.random.default_rng(3).standard_normal((2, 3, *IMAGE_SIZE)).astype(np.float32)
    (logits,) = sess.run(["logits"], {"image": x})
    np.testing.assert_allclose(logits, torch_logits(ckpt, x), rtol=1e-4, atol=1e-4)
