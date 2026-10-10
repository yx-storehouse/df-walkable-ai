"""dfwalk：《三角洲行动》第一人称画面可走区域语义分割。

只做离线的图像/录像分析：不读取游戏进程，不注入输入，不控制角色。

全项目统一的标签约定（掩码 PNG 为单通道 uint8）：
    0   = NON_WALKABLE  不可走（墙体、障碍、天空、水面……）
    1   = WALKABLE      可走地面
    255 = IGNORE_INDEX  忽略（HUD、自动标注不确定区域、人工标记的无法判断区域），不参与训练和评估
"""

__version__ = "0.1.0"

NON_WALKABLE = 0
WALKABLE = 1
IGNORE_INDEX = 255
NUM_CLASSES = 2
CLASS_NAMES = ("non_walkable", "walkable")

# ImageNet 归一化参数（smp 预训练编码器与 SegFormer 均使用）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm")
