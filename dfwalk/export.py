"""导出 ONNX，便于在其他推理框架（onnxruntime、TensorRT 等）中部署做离线分析。"""

from __future__ import annotations

from pathlib import Path


def export_onnx(checkpoint_path: str | Path, out_path: str | Path, opset: int = 17,
                dynamic_batch: bool = True) -> Path:
    """加载 checkpoint，导出 ONNX：
    输入名 "image"：float32 (N, 3, H, W)，H/W 为 checkpoint 的 image_size，已按 checkpoint 的 mean/std 归一化；
    输出名 "logits"：(N, 2, H, W)。dynamic_batch 时 N 为动态维度。
    若安装了 onnx，导出后运行 onnx.checker.check_model。返回 out_path。"""
    raise NotImplementedError
