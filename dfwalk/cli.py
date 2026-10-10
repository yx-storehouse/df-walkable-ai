"""命令行入口：dfwalk <子命令> [-c CONFIG] [--set section.key=value ...]

子命令：
    extract-frames  从录像抽帧            hud-preview    在截图上预览 HUD 矩形
    autolabel       生成伪标签            to-labelme     伪标签 → LabelMe JSON
    from-labelme    LabelMe JSON → 掩码   split          划分训练/验证集
    check-data      检查数据并可视化标签   train          训练
    eval            评估 checkpoint       infer          图片/目录/视频推理
    export-onnx     导出 ONNX
"""

from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    raise NotImplementedError
