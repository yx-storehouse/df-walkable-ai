# dfwalk：《三角洲行动》识路模型（可走区域分割）

dfwalk 训练一个**语义分割模型**。给它一张《三角洲行动》第一人称画面（截图，或录像里的一帧），它会判断每个像素是不是**可以走的地面**，并输出：

- 掩码 PNG：`0` = 不可走，`1` = 可走，`255` = 忽略（HUD 等）；
- 叠加可视化：可走区域涂成半透明绿色并描边，HUD 区域压暗；录像会输出整段叠加视频。

完整流程是：用 OBS 录制自己的对局 → 抽帧 → SegFormer 零样本自动生成伪标签 → 在 LabelMe 里人工修正 → 训练 U-Net 等分割模型 → 评估 → 对新截图/录像推理 → 导出 ONNX。每一步都有对应的 `dfwalk` 命令。

适合用来：复盘自己的录像、研究地图里哪些地方能走，或者把它当作一个完整的语义分割实战项目来学习。

---

## 项目边界（请先阅读）

**dfwalk 是离线分析工具，不做外挂。**

- **只**处理你自己录制的视频和截图文件，全部在本地离线完成。
- **不**读取游戏内存或进程，**不**注入键盘/鼠标输入，**不**控制角色，不提供任何自动寻路、自动移动、自瞄或其他自动操作功能。
- **不要**把本项目接到实时游戏画面上去驱动角色或辅助操作。这类用法属于外挂，**违反《三角洲行动》的用户协议**，会被游戏的**反作弊系统**检测并导致封号，在一些地区还可能违法。本项目不提供、也不会接受这类功能的代码。
- 本项目与《三角洲行动》的开发商、运营商没有任何关联。游戏画面的版权归其权利人所有。如果要公开分享数据集，请先确认相关授权。

---

## 目录

- [环境要求与安装](#环境要求与安装)
- [完整流程（可直接复制的命令）](#完整流程可直接复制的命令)
- [标注规范](#标注规范)
- [数据建议](#数据建议)
- [训练建议与指标说明](#训练建议与指标说明)
- [部署 ONNX 模型](#部署-onnx-模型)
- [配置参考](#配置参考)
- [命令速查](#命令速查)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [开发与测试](#开发与测试)
- [许可证](#许可证)

---

## 环境要求与安装

| 项目 | 要求 |
|---|---|
| 系统 | Windows 10/11、Linux、macOS |
| Python | 3.10 及以上（推荐 3.11 / 3.12） |
| 显卡 | 可选。训练强烈建议使用 NVIDIA 显卡（6 GB 显存以上）+ CUDA 版 PyTorch。CPU 也能跑通全流程，只是训练慢 |
| 磁盘 | 每 1000 张 1280x720 的抽帧约占 200–400 MB |

### 安装步骤

```bash
# 1. 获取代码
git clone <仓库地址> df-walkable-ai
cd df-walkable-ai

# 2. 创建并激活虚拟环境
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 3. （有 NVIDIA 显卡时）先安装 CUDA 版 PyTorch。
#    如果之前已经装过 CPU 版（例如先做了第 4 步），要先运行 pip uninstall -y torch torchvision，
#    否则下面的命令只会提示 “Requirement already satisfied”，不会换成 CUDA 版。
#    具体命令以 https://pytorch.org/get-started/locally/ 为准，按你的显卡驱动选择 CUDA 版本，例如：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. 安装本项目（含自动标注、ONNX 导出和测试依赖）
pip install -e ".[dev]"

# 5. 验证
dfwalk --help
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

可选依赖分组：

| 安装方式 | 内容 |
|---|---|
| `pip install -e .` | 核心功能：抽帧、标注转换、训练、推理 |
| `pip install -e ".[autolabel]"` | 加上 `transformers`，用于 `dfwalk autolabel` |
| `pip install -e ".[onnx]"` | 加上 `onnx`、`onnxscript`（导出器后备）和 `onnxruntime`（校验/部署导出的模型），用于 `dfwalk export-onnx` |
| `pip install -e ".[dev]"` | 以上全部，再加 `pytest` |

> 国内网络安装慢时可以使用 PyPI 镜像，例如：`pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple`

### Windows 注意事项

- PowerShell 提示“禁止运行脚本”时，先执行一次 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，再激活虚拟环境。
- 如果提示找不到 `dfwalk` 命令，可以用 `python -m dfwalk ...` 代替，两者完全等价。
- `--set` 的值里有方括号、逗号或空格时请加**双引号**，例如 `--set "data.image_size=[384,672]"`。这种写法在 PowerShell、cmd 和 bash 中都能用。
- 下文命令都写成单行，直接复制即可，不用处理不同终端的换行符。
- 图片读写支持中文路径。少数 Windows 上的 OpenCV 版本无法打开中文路径下的**视频**：程序会自动改用 8.3 短路径重试，仍然失败时请把视频放到纯英文路径下。
- 训练时 DataLoader 报错（多进程相关）时，加上 `--set train.num_workers=0`。

---

## 完整流程（可直接复制的命令）

### 目录约定

所有数据都放在 `data.root`（默认 `data/`）下，各命令的默认输入输出路径都由它推出：

```
data/
├── images/         抽帧得到的图片  {视频名}_{帧序号:06d}.jpg
├── pseudo_masks/   autolabel 生成的伪标签（未经人工修正）
├── labelme/        LabelMe JSON（人工修正的标注）
├── masks/          最终训练用掩码 {stem}.png，值域 {0, 1, 255}
└── splits/         train.txt / val.txt，每行一个 stem
runs/
├── exp/            训练输出：best.pt、last.pt、history.csv、config.yaml
└── infer/          推理输出
```

> 帧文件名的前缀就是“视频分组”。划分训练/验证集时，同一段录像的帧只会落在同一边，避免数据泄漏。**不要手动重命名抽出的帧。**

### 0. 准备配置文件

```bash
# Linux / macOS
cp configs/default.yaml configs/my.yaml
# Windows
copy configs\default.yaml configs\my.yaml
```

下文的命令都带 `-c configs/my.yaml`。不写 `-c` 时，如果当前目录下有 `configs/default.yaml`，程序会使用它并打印一行提示；否则使用内置默认值。任何配置项都可以临时用 `--set section.key=value` 覆盖，子命令自己的参数（例如 `train --epochs`）优先级最高。

### 1. 用 OBS 录制对局

- 录制格式选 **mkv**（OBS 崩溃时文件也不会损坏），本项目直接支持 `.mp4 .mkv .avi .mov .flv .webm .ts .m2ts .mts .m4v .wmv`。
- 分辨率和游戏一致即可（1080p 或 1440p 都行），帧率 30 就够用，抽帧时会缩放。
- **录制期间不要改变 UI 缩放和分辨率**，否则 HUD 位置会变。关闭 OBS 的摄像头/弹幕叠加，以及显卡驱动或平台的帧数显示；如果关不掉，就把它们也写进 `hud.rects`。
- 多录几段：不同地图、白天/夜晚、不同天气、室内/室外都要有（见[数据建议](#数据建议)）。
- 文件名会成为帧的前缀，建议写成 `零号大坝_白天_01.mkv` 这样的格式，每段录像用不同的名字。

把录像放到一个目录，例如 `recordings/`。

### 2. 抽帧

```bash
dfwalk extract-frames recordings/ -c configs/my.yaml --fps 1
```

- `INPUT` 可以是视频文件，也可以是目录（处理目录第一层中的视频文件）。可以同时给多个。
- `--fps`：每秒抽几帧，默认 2。走路画面变化慢，1–2 就够。
- `--max-side`：最长边缩放到多少像素，默认 1280；设为 0 表示不缩放。
- `--dedup`：去重阈值，默认 6。和上一张保留帧的平均灰度差小于该值时丢弃（例如站着不动的片段）；设为 0 关闭去重。
- 输出默认是 `data/images/`，每个视频会打印写出的帧数。
- 两个视频的文件名清洗后得到相同前缀时（例如 `a b.mp4` 和 `a_b.mp4`），命令会在写入任何文件之前报错，请先重命名。

### 3. 校准 HUD 区域

小地图、血条、弹药、准星、击杀提示等 HUD 在训练和评估中一律视为“忽略”。先在一张截图上预览：

```bash
dfwalk hud-preview data/images/零号大坝_白天_01_000000.jpg -c configs/my.yaml -o hud_preview.jpg
```

打开 `hud_preview.jpg` 对照画面，修改 `configs/my.yaml` 中的 `hud.rects`。坐标是归一化矩形 `[x0, y0, x1, y1]`（0~1，左上角为原点，x 向右，y 向下）：

```yaml
hud:
  rects:
    - [0.00, 0.00, 0.20, 0.30]   # 左上角小地图（示意，请按自己的画面调整）
    - [0.80, 0.85, 1.00, 1.00]   # 右下角弹药
    - [0.48, 0.47, 0.52, 0.53]   # 准星
```

反复运行 `hud-preview`，直到每个半透明矩形都完整盖住对应的 HUD。命令还会打印每个矩形对应的像素范围。`hud.rects` 为空时会给出警告。

> 不同分辨率、UI 缩放比例下 HUD 的位置不同。混用多种分辨率录像时，要在每种分辨率的截图上都检查一遍。

### 4. 自动生成伪标签

```bash
dfwalk autolabel -c configs/my.yaml
```

- 使用在 ADE20K 上预训练的 SegFormer（默认 `nvidia/segformer-b2-finetuned-ade-512-512`）。它会把 floor、road、grass、stairs 等类别的概率相加，得到“可走概率”：`>= 0.7` 标为可走，`<= 0.3` 标为不可走，介于两者之间标为忽略。
- 首次运行会从 Hugging Face 下载约 100 MB 的模型。国内网络请先设置镜像（见[常见问题](#常见问题)）。
- 速度参考（1280x720 截图，4 核 CPU）：默认的 b2 约 2 秒/张；只有 CPU 时可以换更小的 `--model nvidia/segformer-b0-finetuned-ade-512-512`，约 0.4 秒/张，结果略粗糙（反正都要人工修正）。有 NVIDIA 显卡时都很快。
- 已存在的伪标签会被跳过，加 `--overwrite` 重新生成。`--model` 可以指定其他模型名或本地目录，`--device` 可以指定 `cpu` / `cuda`。
- **伪标签只是起点**：游戏画面和真实照片差别很大，结果必然有错，一定要人工检查修正。

### 5. 导出为 LabelMe 标注

```bash
dfwalk to-labelme -c configs/my.yaml
```

把 `data/pseudo_masks/` 转换成 `data/labelme/*.json`，多边形标签为 `walkable` / `non_walkable` / `ignore`。

- `--epsilon`：多边形简化程度（像素），默认 1.0；设为 0 时不做简化，可以精确还原掩码。
- `--min-area`：丢弃面积小于该值的碎片，默认 16 像素。
- **已存在的 JSON 永远不会被覆盖**（以免冲掉人工修正的结果），除非加 `--overwrite`。

### 6. 在 LabelMe 中人工修正

LabelMe 是独立的标注工具，建议装在单独的环境里，以免和本项目的依赖冲突：

```bash
pipx install "labelme>=6"
# 或者：另建一个虚拟环境，然后 pip install "labelme>=6"
```

打开图片目录，让 LabelMe 从 `data/labelme/` 读取和保存标注：

```bash
labelme data/images --output data/labelme --labels walkable,non_walkable,ignore
```

LabelMe 6 及以上默认自动保存，并且不会把图片数据写进 JSON，不需要额外参数（7.x 已经删除了 `--nodata` / `--autosave`，加上会直接报错退出）。如果你用的是旧版 LabelMe 5.x，请在上面的命令末尾加上 `--nodata --autosave`。路径中有空格时，请给路径加上双引号。

修正技巧：

- **后画的形状覆盖先画的形状**。修正错误区域最快的办法是：直接在错的地方画一个标签正确的新多边形（Ctrl+N），不必去拖动原来多边形的顶点。
- 常用快捷键：`D` 下一张，`A` 上一张，`Ctrl+N` 画多边形，`Ctrl+J` 编辑多边形，`Delete` 删除选中形状，`Ctrl+Z` 撤销。
- 只支持 `polygon` 和 `rectangle` 两种形状。圆、线、点等形状会让下一步报错。
- 标签名也可以写中文：`可走` / `不可走` / `忽略`（大小写和首尾空格不影响）。
- 没有被任何形状覆盖的像素会被当作**不可走**。如果不用伪标签、从零开始标注，至少要把可走区域和拿不准的区域都画出来。
- 拿不准的地方标 `ignore`，不要硬猜。

### 7. 转回训练用掩码

```bash
dfwalk from-labelme -c configs/my.yaml
```

把 `data/labelme/*.json` 转换为 `data/masks/*.png`，并把 `hud.rects` 区域统一置为忽略。JSON 有错误（未知标签、不支持的形状等）时，报错信息里会带上文件名。

### 8. 划分训练集和验证集

```bash
dfwalk split -c configs/my.yaml
```

- 只使用同时有图片和掩码的样本，**按视频分组**划分：同一段录像的帧只会出现在训练集或验证集中的一边。
- 会打印每个划分的帧数和视频段数。`--val-ratio`、`--seed` 可以覆盖配置。
- 划分文件已存在时默认**拒绝覆盖**（重新划分后，前后的验证指标就没法比较了）。新增标注后确实要重新划分时，加 `--force`。
- 只有一段录像时会退化为按时间切分（最后一部分帧作为验证集），并给出警告。建议至少录 3–5 段。

### 9. 检查数据

```bash
dfwalk check-data -c configs/my.yaml --vis-dir runs/check
```

会统计并检查以下内容：

- `images/`、`masks/`、`pseudo_masks/`、`labelme/` 中的文件数量；
- 有掩码但没有图片的样本（**错误**）；有图片但还没有掩码的样本（只是提示）；
- 掩码中出现 0/1/255 以外的值、掩码与图片尺寸不一致、文件无法读取（**错误**）；
- 划分文件引用了不存在的样本、训练集和验证集有重复样本（**错误**）；同一段视频跨两个划分（提示）；
- 每个划分（还没划分时是全部数据）的可走/不可走/忽略像素占比。

`--vis-dir` 会在全部样本中均匀抽取 `--vis-limit` 张（默认 50 张，0 表示全部），输出标签叠加图：绿色为可走，压暗的部分为忽略。训练前快速翻一遍，很容易发现标反了的标注。**发现错误时退出码为 1。**

### 10. 训练

```bash
dfwalk train -c configs/my.yaml --epochs 50 --batch-size 8
```

- 输出到 `runs/exp/`（`--out-dir` 可以修改）：`best.pt`（验证集 `iou_walkable` 最高的一轮）、`last.pt`、`history.csv`（每轮的 loss 和指标）、`config.yaml`（本次训练用的完整配置）。
- 输出目录中已经有 `best.pt` / `last.pt` 时默认**拒绝训练**：重新训练会在第 1 轮结束时就替换 `best.pt`，上一版模型无法找回。补充数据后再训练时，用新的目录保留上一版，例如 `--out-dir runs/exp2`（后面的 `eval` / `infer` / `export-onnx` 命令里的 `runs/exp/best.pt` 也要跟着改）；确认要覆盖时加 `--force`。
- 如果还没有划分文件，训练前会自动划分。
- `--device cpu` / `--device cuda` 可以指定设备；在 CUDA 上默认开启混合精度（AMP）。
- 其他超参数用 `--set` 修改，例如 `--set train.lr=0.0005 --set train.encoder=resnet50`。

### 11. 评估

```bash
dfwalk eval -c configs/my.yaml --ckpt runs/exp/best.pt
```

在 `splits/val.txt` 上评估，打印对齐的指标表（各指标的含义见[指标说明](#指标说明)）。`--split train` 可以评估训练集，用来判断是否过拟合。

### 12. 推理

```bash
# 单张截图、整个目录、录像都可以，可以混着给多个
dfwalk infer -c configs/my.yaml --ckpt runs/exp/best.pt data/images/零号大坝_白天_01_000120.jpg recordings/新录像.mp4 -o runs/infer
```

- 图片会输出 `{stem}_overlay.jpg` 叠加图和 `{stem}_mask.png` 掩码（`--no-mask` 只输出叠加图）。
- 视频会输出 `{stem}_overlay.mp4`。用 `--every-n 3` 可以每 3 帧推理一次（中间帧复用结果），速度约为原来的 3 倍；`--max-frames 600` 只处理前 600 帧。
- 目录会处理其第一层中的图片和视频。
- `--threshold`：可走概率阈值，默认 0.5。调高会更保守，只把很确定的地方标为可走。
- `hud.rects` 中的区域会被压暗，并标为忽略（255）。

### 13. 导出 ONNX

```bash
dfwalk export-onnx --ckpt runs/exp/best.pt -o runs/exp/model.onnx
```

导出的 ONNX 模型可以用 onnxruntime、TensorRT 等推理引擎做离线分析，用法见[部署 ONNX 模型](#部署-onnx-模型)。`--static-batch` 会把 batch 维固定为 1，`--opset` 可以指定 opset 版本（默认 17）。命令会打印文件实际的 opset：少数结构（例如默认尺寸下的 `UPerNet`）只能改用 torch.export（dynamo）导出器，实际 opset 会是 18，此时会给出警告。

---

## 标注规范

所有人、所有帧都要使用**同一套规则**。标注不一致是分割模型效果差最常见的原因。

| 标签 | 掩码值 | 包括 |
|---|---|---|
| `walkable` 可走 | 1 | 地面、道路、草地、土路、沙地、室内地板、地毯、楼梯和台阶的踏面、可以走上去的缓坡、桥面、人物脚下和阴影里的地面 |
| `non_walkable` 不可走 | 0 | 墙体、掩体和障碍物的侧面、栏杆、车辆、水面、陡坡和悬崖、无法跨越的箱子和集装箱、树干、天空、建筑外立面、人物（敌人、队友）本身 |
| `ignore` 忽略 | 255 | HUD（小地图、血条、弹药、准星、击杀提示、交互提示）、第一人称的手臂和枪械、看不清或拿不准的区域（烟雾、闪光、强烈运动模糊、过暗）、远处很小的区域 |

细则：

1. **只标“能踩上去”的表面**。箱子顶面、矮墙顶面这类“理论上翻上去能站”的表面，拿不准时标 `ignore`。
2. **边界宁可保守**：地面和墙的交界处画在墙根上即可，不需要精确到像素。离视角很远、只有几个像素宽的地面，标 `ignore`。
3. **楼梯**：踏面标可走。从侧面看过去的楼梯侧板标不可走。
4. **门和窗**：开着的门里能看到的地板标可走，门板本身标不可走；窗户和窗外的远景一般标不可走或忽略。
5. **水**：水面统一标不可走。如果你决定把能涉水通过的浅滩标为可走，就要在所有数据里都这样标。
6. **车辆、载具**：一律标不可走，包括车顶。
7. **HUD 已经由 `hud.rects` 自动处理**。会移动的提示（例如屏幕中间弹出的交互提示）需要手动标 `ignore`。
8. 一张图里**拿不准的地方标 `ignore`**，比硬猜好得多：忽略区域不参与训练，也不参与评估。

---

## 数据建议

- **覆盖面比数量重要**。尽量覆盖多张地图，例如**零号大坝、长弓溪谷、航天基地、巴克什、潮汐监狱**，以及不同时段（白天/夜晚）、不同天气（晴、雨、雾、沙尘）、室内/室外、不同画质设置。
- **从 200–300 张人工修正过的帧开始**。先训练一版，再对新录像推理，找出效果差的场景，有针对性地补录和补标。循环几次，效果提升最快。
- **按视频划分**：`split` 会自动按视频分组，这是验证指标可信的前提。相邻帧几乎一模一样，如果随机打乱划分，验证指标会虚高。
- 至少录 **3–5 段**不同的视频，验证集里最好有一整张训练集没见过的地图，这样才能真实反映泛化能力。
- 抽帧不要太密：`--fps 1` 到 `--fps 2` 配合默认去重即可。同一场景的帧太多，只会让数据集“看起来很大”。
- 截图也可以用：放进 `data/images/`，命名为 `{分组名}_{数字}.png`（例如 `截图_巴克什_000001.png`），以便正确分组。
- 新增标注后，运行 `dfwalk split --force` 重新划分，再运行 `check-data`。重新训练时用新的 `--out-dir`（例如 `runs/exp2`），保留上一版模型以便对比。

---

## 训练建议与指标说明

### CPU 还是 GPU

| 设备 | 建议配置 |
|---|---|
| NVIDIA GPU（≥6 GB） | 默认配置即可：`Unet` + `resnet34`，`image_size [384, 672]`，`batch_size 8`，自动开启 AMP |
| 小显存 GPU（4 GB） | `--batch-size 4`，或 `--set "data.image_size=[288,512]"` |
| 只有 CPU | `--set train.encoder=resnet18 --set "data.image_size=[256,448]" --batch-size 4 --epochs 20`，能跑通，但会慢很多 |
| Apple Silicon | `--device mps`（PyTorch 对 MPS 的支持程度取决于版本） |

### 主要超参数

- **image_size**：`[高, 宽]`，必须都是 32 的倍数。16:9 画面推荐 `[384, 672]`（默认）、`[288, 512]`（更快）或 `[576, 1024]`（细节更好，显存占用约为默认的 2.3 倍）。
- **batch_size**：显存不足（`CUDA out of memory`）时，先减小它，再减小 `image_size`。`DeepLabV3` / `DeepLabV3Plus` / `PAN` / `UPerNet` 要求 `batch_size` 至少为 2。
- **arch / encoder**（来自 segmentation_models_pytorch）：

  | 组合 | 特点 |
  |---|---|
  | `Unet` + `resnet18` | 最快，适合 CPU 或快速试验 |
  | `Unet` + `resnet34` | 默认，速度和效果比较均衡 |
  | `Unet` + `resnet50` / `efficientnet-b3` | 效果更好，也更慢 |
  | `FPN` / `DeepLabV3Plus` + `resnet34` | 对大块区域的边界更平滑 |
  | `Segformer` + `mit_b2` | Transformer 编码器，数据较多时值得一试 |

- **encoder_weights**：默认 `imagenet`，首次训练时会下载预训练权重（segmentation_models_pytorch 0.5 起从 Hugging Face 的 `smp-hub/...` 仓库下载，国内网络同样可以用 `HF_ENDPOINT` 镜像，见[常见问题](#常见问题)），效果明显好于随机初始化。完全离线时用 `--set train.encoder_weights=null`。
- **epochs**：默认 50。观察 `runs/exp/history.csv`：如果 `train_loss` 一直在下降而 `val_loss` 开始上升，说明过拟合了，需要更多（更多样的）数据。
- **学习率**：默认 3e-4（AdamW），先线性 warmup，再余弦退火到 0。一般不用改；要改时 `--set train.lr=1e-3` 或 `--set train.lr=0.001` 都可以。

### 指标说明

指标只统计非忽略像素。

| 指标 | 含义 |
|---|---|
| `iou_walkable` | 可走区域的 IoU（交并比）。**主要指标**，`best.pt` 按它挑选 |
| `iou_non_walkable` | 不可走区域的 IoU |
| `miou` | 两类 IoU 的平均值 |
| `pixel_acc` | 像素准确率。画面里不可走的像素占多数时，这个值天然偏高，参考意义有限 |
| `precision_walkable` | 预测为可走的像素中，真正可走的比例。偏低说明把墙根、障碍物误判成了路 |
| `recall_walkable` | 真正可走的像素中被找出来的比例。偏低说明漏掉了地面（例如暗处、远处） |
| `f1_walkable` | 精确率和召回率的调和平均 |
| `loss` | 验证集 loss（交叉熵 + Dice） |

推理时可以用 `--threshold` 在精确率和召回率之间取舍：阈值越高，精确率越高，召回率越低。

---

## 部署 ONNX 模型

导出的模型：

- 输入 `image`：float32 `(N, 3, H, W)`，RGB，先 `/255`，再按 `(x - mean) / std` 归一化；
- 输出 `logits`：`(N, 2, H, W)`，未经 softmax，第 1 通道为可走。

`H`、`W`、`mean`、`std` 都写在模型的 metadata 里（`dfwalk.image_size`、`dfwalk.mean`、`dfwalk.std` 等，值为 JSON 字符串）。下面是一个 onnxruntime 示例：

```python
import json

import cv2
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("runs/exp/model.onnx")
meta = sess.get_modelmeta().custom_metadata_map
h, w = json.loads(meta["dfwalk.image_size"])
mean = np.array(json.loads(meta["dfwalk.mean"]), np.float32)
std = np.array(json.loads(meta["dfwalk.std"]), np.float32)

img = cv2.cvtColor(cv2.imread("shot.jpg"), cv2.COLOR_BGR2RGB)
# 与训练一致：宽、高都缩小时用 INTER_AREA，只要有一个方向放大就用 INTER_LINEAR
interp = cv2.INTER_AREA if (w <= img.shape[1] and h <= img.shape[0]) else cv2.INTER_LINEAR
x = cv2.resize(img, (w, h), interpolation=interp).astype(np.float32) / 255.0
x = ((x - mean) / std).transpose(2, 0, 1)[None]

logits = sess.run(["logits"], {"image": x})[0]
e = np.exp(logits - logits.max(axis=1, keepdims=True))
prob = e[0, 1] / e[0].sum(axis=0)                  # (h, w) 可走概率
prob = cv2.resize(prob, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
walkable = prob >= 0.5
```

再次强调：导出的模型同样只用于离线分析截图和录像。

---

## 配置参考

配置分三层，后面的覆盖前面的：内置默认值 → YAML 配置文件（`-c`）→ `--set section.key=value`（可重复，值按 YAML 解析）。子命令自己的参数优先级最高。`configs/default.yaml` 与内置默认值完全一致。

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `data.root` | `data` | 数据根目录（见[目录约定](#目录约定)） |
| `data.image_size` | `[384, 672]` | 训练/推理输入尺寸 `[高, 宽]`，必须是 32 的倍数 |
| `data.val_ratio` | `0.2` | `split` 时验证集占比，取值在 (0, 1) 之间 |
| `data.seed` | `42` | 划分用的随机种子 |
| `hud.rects` | `[]` | HUD 忽略区域，归一化矩形 `[x0, y0, x1, y1]` 的列表 |
| `frames.fps` | `2.0` | 每秒抽帧数；<= 0 表示每帧都取 |
| `frames.max_side` | `1280` | 抽帧后最长边；0 表示不缩放 |
| `frames.dedup_threshold` | `6.0` | 去重阈值（平均灰度差，0~255）；0 表示关闭 |
| `frames.jpeg_quality` | `95` | 抽帧 JPEG 质量 |
| `autolabel.model` | `nvidia/segformer-b2-finetuned-ade-512-512` | Hugging Face 模型名或本地目录 |
| `autolabel.short_side` | `512` | 自动标注推理时短边缩放到的尺寸 |
| `autolabel.hi_threshold` | `0.7` | 可走概率 >= hi 标为可走 |
| `autolabel.lo_threshold` | `0.3` | 可走概率 <= lo 标为不可走，介于两者之间标为忽略 |
| `autolabel.device` | `auto` | `auto`（cuda > mps > cpu）/ `cpu` / `cuda` / `cuda:0` / `mps` |
| `autolabel.walkable_labels` | `floor, road, grass, sidewalk, earth, rug, field, sand, path, stairs, runway, stairway, bridge, dirt track, land, step` | 被视为“可走地面”的 ADE20K 类别名；写错名字会报错并给出拼写建议 |
| `train.arch` | `Unet` | segmentation_models_pytorch 架构名：`Unet` / `UnetPlusPlus` / `FPN` / `DeepLabV3Plus` / `Segformer` 等 |
| `train.encoder` | `resnet34` | 编码器名，如 `resnet18` / `resnet50` / `efficientnet-b3` / `mit_b2` |
| `train.encoder_weights` | `imagenet` | 预训练权重；`null` 表示随机初始化（离线时使用） |
| `train.epochs` | `50` | 训练轮数 |
| `train.batch_size` | `8` | batch 大小 |
| `train.lr` | `0.0003` | AdamW 学习率 |
| `train.weight_decay` | `0.0001` | AdamW 权重衰减 |
| `train.num_workers` | `4` | DataLoader 进程数；Windows 出问题时设为 0 |
| `train.ce_weight` | `1.0` | 交叉熵 loss 权重 |
| `train.dice_weight` | `1.0` | Dice loss 权重 |
| `train.amp` | `true` | 混合精度，仅在 CUDA 上生效 |
| `train.out_dir` | `runs/exp` | 训练输出目录 |
| `train.device` | `auto` | 训练和 `eval` 使用的设备 |
| `train.seed` | `42` | 训练随机种子 |
| `infer.threshold` | `0.5` | 推理时的可走概率阈值，取值在 [0, 1] 之间 |
| `infer.alpha` | `0.45` | 叠加图透明度（`check-data` 的可视化也使用它） |
| `infer.color` | `[0, 255, 0]` | 可走区域叠加颜色（RGB） |
| `infer.device` | `auto` | 推理设备 |

示例：

```bash
dfwalk train -c configs/my.yaml --set train.encoder=resnet18 --set "data.image_size=[288,512]" --set train.encoder_weights=null
```

---

## 命令速查

| 命令 | 作用 | 常用参数 |
|---|---|---|
| `dfwalk extract-frames INPUT...` | 从录像抽帧 | `-o` `--fps` `--max-side` `--dedup` |
| `dfwalk hud-preview IMAGE` | 在截图上预览 HUD 矩形 | `-o`（默认 `hud_preview.jpg`） |
| `dfwalk autolabel` | 生成伪标签 | `-i` `-o` `--model` `--device` `--overwrite` |
| `dfwalk to-labelme` | 伪标签 → LabelMe JSON | `--masks` `--images` `-o` `--epsilon` `--min-area` `--overwrite` |
| `dfwalk from-labelme` | LabelMe JSON → 掩码 | `-i` `-o` |
| `dfwalk split` | 按视频划分训练/验证集 | `--val-ratio` `--seed` `--force` |
| `dfwalk check-data` | 检查数据、统计标签、输出可视化 | `--vis-dir` `--vis-limit` |
| `dfwalk train` | 训练 | `--epochs` `--batch-size` `--out-dir` `--device` `--force` |
| `dfwalk eval --ckpt PATH` | 评估 | `--split` `--device` |
| `dfwalk infer --ckpt PATH INPUT...` | 推理图片/目录/视频 | `-o` `--threshold` `--every-n` `--max-frames` `--no-mask` `--device` |
| `dfwalk export-onnx --ckpt PATH` | 导出 ONNX | `-o` `--opset` `--static-batch` |

所有子命令都支持 `-c/--config` 和 `--set`，这两个选项写在子命令前后都可以。`dfwalk <子命令> -h` 可以查看完整说明。

退出码：`0` 成功；`1` 运行出错，或 `check-data` 发现了错误；`2` 命令行参数错误；`130` 被 Ctrl+C 中断。出错时打印 `错误: …` 而不是一大段堆栈。要查看完整的错误堆栈，可以设置环境变量 `DFWALK_DEBUG=1`。

---

## 项目结构

```
df-walkable-ai/
├── configs/
│   └── default.yaml      默认配置（与 dfwalk/config.py 中的默认值一致）
├── dfwalk/
│   ├── __init__.py       标签约定（0/1/255）与全局常量
│   ├── __main__.py       python -m dfwalk
│   ├── cli.py            命令行入口
│   ├── config.py         配置：dataclass 默认值 ← YAML ← --set
│   ├── utils.py          设备选择、随机种子、支持中文路径的图片/掩码读写、打开视频
│   ├── frames.py         录像抽帧与去重
│   ├── hud.py            HUD 矩形 → 忽略区域、预览图
│   ├── autolabel.py      SegFormer 零样本伪标签
│   ├── labelme_io.py     掩码 ⇄ LabelMe JSON
│   ├── dataset.py        数据集、按视频分组划分、标签统计
│   ├── transforms.py     预处理与数据增强
│   ├── model.py          segmentation_models_pytorch 模型构建
│   ├── losses.py         交叉熵 + Dice loss（忽略 255）
│   ├── metrics.py        混淆矩阵与 IoU 等指标
│   ├── checkpoint.py     checkpoint 保存/加载（weights_only）
│   ├── train.py          训练与评估
│   ├── infer.py          推理与叠加可视化（图片/视频）
│   └── export.py         ONNX 导出
├── tests/                pytest 测试（离线、CPU、合成数据；test_e2e.py 走一遍完整命令行流程）
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## 常见问题

**自动标注的效果很差？**
这是正常的。SegFormer 是在真实照片（ADE20K）上训练的，而游戏画面的光照、材质和后处理都和真实照片不同，伪标签一定会有错，**必须人工检查修正**。可以尝试以下方法：调整 `autolabel.hi_threshold` / `lo_threshold`（把两者拉开，拿不准的区域会更多地标为忽略）；在 `autolabel.walkable_labels` 中增删类别；换用更大的模型，例如 `--model nvidia/segformer-b5-finetuned-ade-640-640`。等有了几百张修正过的数据并训练出第一版模型后，用它对新数据推理，通常比 SegFormer 更准。

**从 Hugging Face 下载模型很慢或失败（国内网络）？**
设置镜像后再运行：

```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
# Windows cmd
set HF_ENDPOINT=https://hf-mirror.com
```

也可以先在能联网的机器上下载模型目录，再用 `--model 本地目录路径` 加载。训练时默认下载的 ImageNet 预训练编码器权重同样来自 Hugging Face（`smp-hub/resnet34.imagenet` 等），设置同一个镜像即可；Hugging Face 下载失败时 segmentation_models_pytorch 会改从原始网址下载，若报“无法获取预训练编码器权重”，可删除 `~/.cache/torch/hub/checkpoints` 中下载不完整的文件后重试，完全离线时使用 `--set train.encoder_weights=null`。

**Windows 下的中文路径？**
图片和掩码的读写都支持中文路径。读取视频时，如果 OpenCV 打不开中文路径，程序会自动改用 8.3 短路径重试，仍然失败时请把视频移到纯英文路径下。`infer` 输出视频时，如果遇到“无法创建输出视频”，请把 `-o` 指向纯英文目录。

**训练没有用上显卡？**
运行 `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`，按输出处理（给训练命令加上 `--device cuda` 再运行一次，程序也会直接说明是哪种情况）：

- 输出 `None False`：装的是 CPU 版 PyTorch。请先运行 `pip uninstall -y torch torchvision` 卸载 CPU 版，再按[安装步骤](#安装步骤)第 3 步安装 CUDA 版（不先卸载的话，第 3 步的命令只会提示 “Requirement already satisfied”，装的仍是 CPU 版）。
- 输出 CUDA 版本号和 `False`（例如 `12.8 False`）：PyTorch 已经是 CUDA 版，**不要**卸载重装，重装后结果还是一样。问题出在显卡或驱动：运行 `nvidia-smi`，确认能看到显卡，并且右上角的 “CUDA Version” 不低于上面输出的版本号，否则请更新 NVIDIA 显卡驱动；笔记本还要确认独立显卡没有被关掉（例如切到了“集显模式”或“节能模式”）。实在不能更新驱动时，才需要先 `pip uninstall -y torch torchvision`，再用 pytorch.org 上 CUDA 版本更低的命令重新安装。

**`CUDA out of memory`？**
减小 `--batch-size`，或者减小 `data.image_size`，或者换用更小的编码器（`resnet18`）。

**验证指标很高，但在新录像上效果差？**
通常是验证集和训练集太像了。确认划分是按视频进行的（`check-data` 会提示同一段视频跨划分的情况），并加入训练集里没有的地图和场景。

**换了分辨率或 UI 缩放？**
模型的输入会自动缩放，不受影响；但 `hud.rects` 需要用 `hud-preview` 重新核对。

**推理生成的 mp4 无法播放？**
输出使用 `mp4v` 编码，少数播放器不支持。可以用 VLC 播放，或者用 ffmpeg 转码：`ffmpeg -i clip_overlay.mp4 -c:v libx264 clip_h264.mp4`。

**能不能让它实时识别游戏画面，或者帮我自动走路？**
不能，本项目也永远不会提供这类功能，见[项目边界](#项目边界请先阅读)。

**报错信息太简单，想看完整堆栈？**
设置环境变量 `DFWALK_DEBUG=1` 后重新运行（PowerShell：`$env:DFWALK_DEBUG=1`）。

---

## 开发与测试

```bash
pip install -e ".[dev]"
pytest
```

测试全部离线运行，只用 CPU，使用测试中现场合成的图片和视频，不会下载任何模型。

---

## 许可证

MIT License，见 [LICENSE](LICENSE)。
