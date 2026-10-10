"""dfwalk.model 的单元测试：模型构建、输出形状、参数校验、model_config 与 checkpoint 往返。

全部使用 encoder_weights=None + resnet18 小编码器，只用 CPU；测试期间禁止任何网络连接。
"""

from __future__ import annotations

import pickle
import socket

import pytest
import segmentation_models_pytorch as smp
import torch

from dfwalk import NUM_CLASSES
from dfwalk.checkpoint import (
    load_checkpoint,
    model_from_checkpoint,
    save_checkpoint,
)
from dfwalk.model import available_archs, build_model, model_config

H, W = 64, 96  # 32 的倍数的小尺寸
ENCODER = "resnet18"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """任何尝试联网（例如误下载预训练权重）都直接失败。"""

    def guard(*args, **kwargs):
        raise RuntimeError("测试中禁止联网")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _forward(model: torch.nn.Module, n: int = 2, h: int = H, w: int = W) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(torch.randn(n, 3, h, w))


# ---------------------------------------------------------------- 构建与输出形状


def test_default_unet_output_shape_and_dtype():
    model = build_model("Unet", ENCODER, encoder_weights=None)
    assert isinstance(model, torch.nn.Module)
    out = _forward(model)
    assert out.shape == (2, NUM_CLASSES, H, W)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_default_num_classes_is_project_constant():
    model = build_model("Unet", ENCODER, encoder_weights=None)
    assert _forward(model, n=1).shape[1] == NUM_CLASSES == 2


@pytest.mark.parametrize("arch", ["Unet", "FPN", "DeepLabV3Plus", "Segformer", "UnetPlusPlus"])
def test_other_archs_output_shape(arch):
    model = build_model(arch, ENCODER, encoder_weights=None, num_classes=2)
    assert _forward(model, n=1).shape == (1, 2, H, W)


@pytest.mark.parametrize("num_classes", [1, 3])
def test_num_classes_controls_channels(num_classes):
    model = build_model("Unet", ENCODER, encoder_weights=None, num_classes=num_classes)
    assert _forward(model, n=1).shape == (1, num_classes, H, W)


def test_model_is_trainable_and_in_train_mode():
    model = build_model("Unet", ENCODER, encoder_weights=None)
    assert model.training
    params = [p for p in model.parameters() if p.requires_grad]
    assert params
    out = model(torch.randn(2, 3, H, W))
    out.mean().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in params)


def test_build_is_on_cpu():
    model = build_model("Unet", ENCODER, encoder_weights=None)
    assert {p.device.type for p in model.parameters()} == {"cpu"}


# ---------------------------------------------------------------- arch 名处理


@pytest.mark.parametrize("arch", ["unet", "UNET", "fpn", "deeplabv3plus"])
def test_arch_is_case_insensitive(arch):
    model = build_model(arch, ENCODER, encoder_weights=None)
    assert _forward(model, n=1).shape == (1, 2, H, W)


def test_available_archs_matches_smp():
    names = available_archs()
    assert names == [cls.__name__ for cls in smp.MODEL_ARCHITECTURES_MAPPING.values()]
    for expected in ("Unet", "UnetPlusPlus", "FPN", "DeepLabV3Plus", "Segformer"):
        assert expected in names
    # 每个名字都能被 smp 识别
    assert all(n.lower() in smp.MODEL_ARCHITECTURES_MAPPING for n in names)


@pytest.mark.parametrize("bad", ["NotAnArch", "", "Unet3D", "resnet34"])
def test_invalid_arch_raises_value_error_listing_archs(bad):
    with pytest.raises(ValueError) as exc_info:
        build_model(bad, ENCODER, encoder_weights=None)
    msg = str(exc_info.value)
    for name in available_archs():
        assert name in msg
    assert repr(bad) in msg


@pytest.mark.parametrize("bad", [None, 123, ["Unet"]])
def test_non_string_arch_raises_value_error(bad):
    with pytest.raises(ValueError, match="Unet"):
        build_model(bad, ENCODER, encoder_weights=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------- encoder / weights / num_classes 校验


def test_invalid_encoder_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        build_model("Unet", "definitely_not_an_encoder", encoder_weights=None)
    msg = str(exc_info.value)
    assert "definitely_not_an_encoder" in msg
    assert "resnet18" in msg  # 保留 smp 给出的可选编码器列表
    assert isinstance(exc_info.value.__cause__, KeyError)


@pytest.mark.parametrize("bad", ["", None])
def test_empty_encoder_raises_value_error(bad):
    with pytest.raises(ValueError):
        build_model("Unet", bad, encoder_weights=None)  # type: ignore[arg-type]


def test_invalid_encoder_weights_raise_value_error():
    with pytest.raises(ValueError, match="imagenet"):
        build_model("Unet", ENCODER, encoder_weights="no_such_weights")


def test_non_string_encoder_weights_raise_type_error():
    with pytest.raises(TypeError):
        build_model("Unet", ENCODER, encoder_weights=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("weights", ["none", "None", "NULL", "null", "", "  none "])
def test_none_like_weight_strings_mean_random_init(weights):
    # 若没有被归一化为 None，smp 会抛 KeyError（未知权重）或尝试联网（被 _no_network 拦截）
    model = build_model("Unet", ENCODER, encoder_weights=weights)
    assert _forward(model, n=1).shape == (1, 2, H, W)


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "2"])
def test_invalid_num_classes_raise_value_error(bad):
    with pytest.raises(ValueError, match="num_classes"):
        build_model("Unet", ENCODER, encoder_weights=None, num_classes=bad)  # type: ignore[arg-type]


def test_create_model_called_with_contract_arguments(monkeypatch):
    calls = []
    sentinel = torch.nn.Identity()

    def fake_create_model(arch, **kwargs):
        calls.append((arch, kwargs))
        return sentinel

    monkeypatch.setattr(smp, "create_model", fake_create_model)
    result = build_model("FPN", "resnet50", encoder_weights="imagenet", num_classes=2)
    assert result is sentinel
    assert calls == [("FPN", {"encoder_name": "resnet50", "encoder_weights": "imagenet",
                              "in_channels": 3, "classes": 2})]


def test_create_model_receives_none_for_none_like_string(monkeypatch):
    seen = {}

    def fake_create_model(arch, **kwargs):
        seen.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(smp, "create_model", fake_create_model)
    build_model("Unet", ENCODER, encoder_weights="None")
    assert seen["encoder_weights"] is None


def test_unrelated_errors_from_smp_are_not_swallowed(monkeypatch):
    def fake_create_model(arch, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(smp, "create_model", fake_create_model)
    with pytest.raises(RuntimeError, match="boom"):
        build_model("Unet", ENCODER, encoder_weights=None)


@pytest.mark.parametrize(
    "error",
    [
        OSError("Cannot reach https://huggingface.co"),  # 网络不通 / HF 离线（huggingface_hub 的错误都是 OSError）
        ConnectionError("connection reset"),
        EOFError("Ran out of input"),  # 下载中断留下的空文件
        pickle.UnpicklingError("unpickling stack underflow"),  # 缓存文件损坏或旧格式（实测 smp 0.5 + torch 2.14）
        RuntimeError("PytorchStreamReader failed reading zip archive"),
    ],
)
def test_pretrained_weight_failures_become_actionable_runtime_error(monkeypatch, error):
    def fake_create_model(arch, **kwargs):
        raise error

    monkeypatch.setattr(smp, "create_model", fake_create_model)
    with pytest.raises(RuntimeError) as info:
        build_model("Unet", "resnet34", encoder_weights="imagenet")
    msg = str(info.value)
    assert "resnet34" in msg and "imagenet" in msg
    assert "HF_ENDPOINT" in msg and "encoder_weights=null" in msg and "hub/checkpoints" in msg
    assert type(error).__name__ in msg
    assert info.value.__cause__ is error


def test_weight_failures_without_pretrained_weights_are_not_wrapped(monkeypatch):
    def fake_create_model(arch, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(smp, "create_model", fake_create_model)
    with pytest.raises(OSError, match="disk full"):
        build_model("Unet", ENCODER, encoder_weights=None)


# ---------------------------------------------------------------- model_config 与 checkpoint 往返


def test_model_config_exact_dict():
    cfg = model_config("Unet", "resnet34", 2)
    assert cfg == {"arch": "Unet", "encoder": "resnet34", "num_classes": 2}
    assert type(cfg["arch"]) is str and type(cfg["encoder"]) is str and type(cfg["num_classes"]) is int


def test_model_config_default_num_classes_and_basic_types():
    import numpy as np

    cfg = model_config("FPN", ENCODER, num_classes=np.int64(3))  # type: ignore[arg-type]
    assert cfg == {"arch": "FPN", "encoder": ENCODER, "num_classes": 3}
    assert type(cfg["num_classes"]) is int
    assert model_config("FPN", ENCODER)["num_classes"] == NUM_CLASSES


def test_model_config_feeds_build_model():
    cfg = model_config("Unet", ENCODER)
    model = build_model(cfg["arch"], cfg["encoder"], encoder_weights=None, num_classes=cfg["num_classes"])
    assert _forward(model, n=1).shape == (1, cfg["num_classes"], H, W)


def test_checkpoint_round_trip(tmp_path):
    model =build_model("Unet", ENCODER, encoder_weights=None)
    model.eval()
    x = torch.randn(1, 3, H, W)
    with torch.no_grad():
        expected = model(x)

    path = save_checkpoint(tmp_path / "中文目录" / "best.pt", model, model_config("Unet", ENCODER),
                           image_size=[H, W], epoch=3, metrics={"iou_walkable": 0.5})
    state = load_checkpoint(path)
    assert state["model_cfg"] == {"arch": "Unet", "encoder": ENCODER, "num_classes": 2}
    restored = model_from_checkpoint(state)
    assert not restored.training
    with torch.no_grad():
        got = restored(x)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize(
    "name, content",
    [
        ("model.onnx", b"\x08\x07\x12\x07pytorch:\x8a\x01\n"),  # 误把导出的 ONNX 当成 checkpoint
        ("config.yaml", "data:\n  root: data\n".encode("utf-8")),
        ("history.csv", b"epoch,lr\n1,0.1\n"),
        ("notes.txt", b"hello world"),
        ("empty.pt", b""),
        ("shot.jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"),
    ],
)
def test_load_checkpoint_rejects_non_checkpoint_files_with_value_error(tmp_path, name, content):
    """非 torch 文件不能把 UnpicklingError / IndexError 等原始异常（及“改用 weights_only=False”的建议）抛给用户。"""
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(ValueError, match="不是 dfwalk checkpoint") as info:
        load_checkpoint(path)
    assert "--ckpt" in str(info.value) and "weights_only" not in str(info.value)
    assert info.value.__cause__ is not None
    if name.endswith(".onnx"):
        assert "onnxruntime" in str(info.value)


def test_load_checkpoint_truncated_file_and_missing_file(tmp_path):
    model = build_model("Unet", ENCODER, encoder_weights=None)
    path = save_checkpoint(tmp_path / "best.pt", model, model_config("Unet", ENCODER), [H, W], 1, {})
    data = path.read_bytes()
    broken = tmp_path / "broken.pt"
    broken.write_bytes(data[: len(data) // 2])  # 复制中断的 checkpoint
    with pytest.raises(ValueError, match="已损坏"):
        load_checkpoint(broken)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")
