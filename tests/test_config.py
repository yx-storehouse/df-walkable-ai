"""dfwalk.config 的单元测试：默认值与 configs/default.yaml 一致、--set 覆盖项解析、未知配置项、校验规则。

全部离线、纯 Python，不依赖 torch。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from dfwalk.config import (
    DEFAULT_WALKABLE_LABELS,
    AutolabelConfig,
    Config,
    apply_dict,
    load_config,
    parse_override,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"


def write_yaml(path: Path, data) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- default.yaml 与 dataclass 一致


def test_default_yaml_exists():
    assert DEFAULT_YAML.is_file()


def test_default_yaml_equals_dataclass_defaults():
    """configs/default.yaml 必须与 Config() 完全一致（键集合、值、类型都一致）。"""
    with open(DEFAULT_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    expected = Config().to_dict()
    assert raw == expected
    # 逐项比较类型（例如 lr 写成 3e-4 会被 YAML 当作字符串）
    for section, values in expected.items():
        for key, value in values.items():
            got = raw[section][key]
            assert type(got) is type(value), f"{section}.{key}: {type(got).__name__} != {type(value).__name__}"


def test_default_yaml_sections_and_keys_complete():
    with open(DEFAULT_YAML, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config()
    assert set(raw) == {f.name for f in dataclasses.fields(cfg)}
    for f in dataclasses.fields(cfg):
        section = getattr(cfg, f.name)
        assert set(raw[f.name]) == {sf.name for sf in dataclasses.fields(section)}, f.name


def test_load_default_yaml_equals_config():
    assert load_config(DEFAULT_YAML) == Config()


def test_load_config_without_path_is_defaults():
    assert load_config() == Config()
    assert load_config(None, []) == Config()


def test_default_walkable_labels_are_copied():
    """default_factory 每次生成新列表，修改一个实例不影响其他实例和模块常量。"""
    a, b = Config(), Config()
    a.autolabel.walkable_labels.append("water")
    a.data.image_size[0] = 32
    a.infer.color[1] = 0
    assert "water" not in b.autolabel.walkable_labels
    assert "water" not in DEFAULT_WALKABLE_LABELS
    assert b.data.image_size == [384, 672]
    assert b.infer.color == [0, 255, 0]
    assert AutolabelConfig().walkable_labels == DEFAULT_WALKABLE_LABELS


def test_default_values_sanity():
    cfg = Config()
    assert cfg.data.root == "data"
    assert cfg.data.image_size == [384, 672]
    assert cfg.hud.rects == []
    assert cfg.train.encoder_weights == "imagenet"
    assert cfg.train.lr == pytest.approx(3e-4)
    validate(cfg)  # 默认配置必须合法


def test_to_dict_yaml_roundtrip(tmp_path):
    cfg = load_config(overrides=["data.root=数据/自定义", "hud.rects=[[0, 0, 0.2, 0.3]]", "train.encoder_weights=null"])
    path = write_yaml(tmp_path / "cfg.yaml", cfg.to_dict())
    assert load_config(path) == cfg


# ---------------------------------------------------------------- parse_override


@pytest.mark.parametrize(
    "text, expected",
    [
        ("train.epochs=5", ("train", "epochs", 5)),
        ("train.lr=0.001", ("train", "lr", 0.001)),
        ("train.lr=1.0e-3", ("train", "lr", 0.001)),
        ("data.image_size=[64, 96]", ("data", "image_size", [64, 96])),
        ("data.image_size=[64,96]", ("data", "image_size", [64, 96])),
        ("hud.rects=[[0, 0, 0.2, 0.3], [0.8, 0.85, 1, 1]]", ("hud", "rects", [[0, 0, 0.2, 0.3], [0.8, 0.85, 1, 1]])),
        ("hud.rects=[]", ("hud", "rects", [])),
        ("train.encoder_weights=null", ("train", "encoder_weights", None)),
        ("train.encoder_weights=~", ("train", "encoder_weights", None)),
        ("train.encoder_weights=", ("train", "encoder_weights", None)),
        ("train.amp=false", ("train", "amp", False)),
        ("train.amp=true", ("train", "amp", True)),
        ("train.amp=False", ("train", "amp", False)),
        ("train.encoder=resnet18", ("train", "encoder", "resnet18")),
        ("data.root=D:/数据/三角洲", ("data", "root", "D:/数据/三角洲")),
        ("autolabel.walkable_labels=[floor, dirt track]", ("autolabel", "walkable_labels", ["floor", "dirt track"])),
        ("autolabel.model=a=b", ("autolabel", "model", "a=b")),  # 只按第一个 '=' 切分
        (" train.epochs =3", ("train", "epochs", 3)),  # 键两侧空白被去掉
    ],
)
def test_parse_override_values(text, expected):
    assert parse_override(text) == expected


@pytest.mark.parametrize("text", ["train.epochs", "epochs=5", "a.b.c=1", ".epochs=1", "train.=1", "=1", ""])
def test_parse_override_bad_format(text):
    with pytest.raises(ValueError):
        parse_override(text)


def test_overrides_applied_in_order():
    cfg = load_config(overrides=["train.epochs=5", "train.epochs=7", "train.batch_size=2"])
    assert cfg.train.epochs == 7
    assert cfg.train.batch_size == 2
    # 其余保持默认
    assert cfg.train.lr == Config().train.lr


def test_overrides_take_precedence_over_file(tmp_path):
    path = write_yaml(tmp_path / "c.yaml", {"train": {"epochs": 3, "batch_size": 4}, "data": {"root": "x"}})
    cfg = load_config(path, ["train.epochs=9"])
    assert cfg.train.epochs == 9
    assert cfg.train.batch_size == 4
    assert cfg.data.root == "x"


def test_partial_yaml_keeps_other_defaults(tmp_path):
    path = write_yaml(tmp_path / "c.yaml", {"hud": {"rects": [[0.0, 0.0, 0.25, 0.3]]}})
    cfg = load_config(path)
    assert cfg.hud.rects == [[0.0, 0.0, 0.25, 0.3]]
    assert cfg.train == Config().train


def test_empty_yaml_and_null_section(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_config(empty) == Config()
    nulls = tmp_path / "nulls.yaml"
    nulls.write_text("train:\nhud:\n", encoding="utf-8")
    assert load_config(nulls) == Config()


def test_yaml_with_chinese_comments_and_bom_free_utf8(tmp_path):
    path = tmp_path / "中文配置.yaml"
    path.write_text("# 中文注释\ndata:\n  root: 数据目录  # 根目录\n", encoding="utf-8")
    assert load_config(path).data.root == "数据目录"


# ---------------------------------------------------------------- 未知配置段 / 配置项


@pytest.mark.parametrize("override", ["nosuch.key=1", "train.nosuch=1", "data.Root=x", "to_dict.x=1"])
def test_unknown_override_raises_keyerror(override):
    with pytest.raises(KeyError):
        load_config(overrides=[override])


def test_unknown_yaml_key_raises_keyerror(tmp_path):
    with pytest.raises(KeyError, match="train.epoch"):
        load_config(write_yaml(tmp_path / "a.yaml", {"train": {"epoch": 3}}))
    with pytest.raises(KeyError, match="trian"):
        load_config(write_yaml(tmp_path / "b.yaml", {"trian": {"epochs": 3}}))


def test_section_must_be_mapping(tmp_path):
    with pytest.raises(TypeError):
        load_config(write_yaml(tmp_path / "a.yaml", {"train": [1, 2]}))
    with pytest.raises(TypeError):
        apply_dict(Config(), {"data": "root"})


def test_apply_dict_in_place():
    cfg = Config()
    out = apply_dict(cfg, {"infer": {"threshold": 0.7}})
    assert out is cfg
    assert cfg.infer.threshold == 0.7
    assert apply_dict(cfg, None) is cfg


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# ---------------------------------------------------------------- 校验


@pytest.mark.parametrize(
    "size",
    [[100, 672], [384, 670], [0, 64], [-32, 64], [384, 0], [33, 64]],
)
def test_image_size_must_be_positive_multiple_of_32(size):
    with pytest.raises(ValueError, match="image_size"):
        load_config(overrides=[f"data.image_size={size}"])


@pytest.mark.parametrize("size", [[32, 32], [64, 96], [384, 672], [736, 1280]])
def test_image_size_valid(size):
    assert load_config(overrides=[f"data.image_size={size}"]).data.image_size == size


@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.5, 0.0, 1.0])
def test_val_ratio_bounds(ratio):
    with pytest.raises(ValueError, match="val_ratio"):
        load_config(overrides=[f"data.val_ratio={ratio}"])


@pytest.mark.parametrize(
    "lo, hi",
    [(0.7, 0.3), (0.5, 0.5), (-0.1, 0.7), (0.3, 1.1), (0.3, 0.3)],
)
def test_bad_autolabel_thresholds(lo, hi):
    with pytest.raises(ValueError, match="threshold"):
        load_config(overrides=[f"autolabel.lo_threshold={lo}", f"autolabel.hi_threshold={hi}"])


def test_autolabel_threshold_edges_valid():
    cfg = load_config(overrides=["autolabel.lo_threshold=0", "autolabel.hi_threshold=1"])
    assert (cfg.autolabel.lo_threshold, cfg.autolabel.hi_threshold) == (0, 1)


@pytest.mark.parametrize(
    "rects",
    [
        "[[0, 0, 0.2]]",  # 少一个数
        "[[0, 0, 0.2, 0.3, 0.4]]",  # 多一个数
        "[[0.3, 0, 0.2, 0.3]]",  # x0 > x1
        "[[0, 0.3, 0.2, 0.3]]",  # y0 == y1
        "[[-0.1, 0, 0.2, 0.3]]",  # 越界
        "[[0, 0, 1.2, 0.3]]",  # 越界
        "[[0, 0, 20, 30]]",  # 像素坐标而非归一化
        "[[0, 0, 0.2, 0.3], [0.5, 0.5, 0.4, 0.6]]",  # 第二个矩形非法
    ],
)
def test_bad_hud_rects(rects):
    with pytest.raises(ValueError, match="hud.rects"):
        load_config(overrides=[f"hud.rects={rects}"])


def test_hud_rects_valid():
    cfg = load_config(overrides=["hud.rects=[[0, 0, 1, 1], [0.48, 0.47, 0.52, 0.53]]"])
    assert len(cfg.hud.rects) == 2


@pytest.mark.parametrize("t", [-0.01, 1.01, 2])
def test_bad_infer_threshold(t):
    with pytest.raises(ValueError, match="infer.threshold"):
        load_config(overrides=[f"infer.threshold={t}"])


@pytest.mark.parametrize("t", [0, 0.5, 1])
def test_infer_threshold_edges_valid(t):
    assert load_config(overrides=[f"infer.threshold={t}"]).infer.threshold == t


def test_validate_directly_on_modified_config():
    cfg = Config()
    cfg.data.image_size = [64, 90]
    with pytest.raises(ValueError):
        validate(cfg)
    cfg.data.image_size = [64, 96]
    validate(cfg)


def test_validation_also_runs_for_yaml_values(tmp_path):
    path = write_yaml(tmp_path / "bad.yaml", {"data": {"image_size": [100, 100]}})
    with pytest.raises(ValueError, match="image_size"):
        load_config(path)


# ---------------------------------------------------------------- 类型与范围校验（validate 统一抛 ValueError）


@pytest.mark.parametrize(
    "override, field",
    [
        ("data.image_size=5", "data.image_size"),
        ("data.image_size=[384]", "data.image_size"),
        ("data.image_size=[384.0, 672.0]", "data.image_size"),
        ("data.image_size=['384', '672']", "data.image_size"),
        ("data.root=''", "data.root"),
        ("data.root=3", "data.root"),
        ("data.val_ratio=true", "data.val_ratio"),
        ("data.val_ratio=abc", "data.val_ratio"),
        ("data.seed=1.5", "data.seed"),
        ("hud.rects=[[0, 0, 'a', 1]]", "hud.rects"),
        ("hud.rects=[0, 0, 1, 1]", "hud.rects"),
        ("hud.rects=[[0, 0, .nan, 1]]", "hud.rects"),
        ("hud.rects=5", "hud.rects"),
        ("frames.fps=.nan", "frames.fps"),
        ("frames.fps=fast", "frames.fps"),
        ("frames.max_side=-1", "frames.max_side"),
        ("frames.max_side=1280.5", "frames.max_side"),
        ("frames.dedup_threshold=-1", "frames.dedup_threshold"),
        ("frames.jpeg_quality=101", "frames.jpeg_quality"),
        ("autolabel.model=''", "autolabel.model"),
        ("autolabel.short_side=0", "autolabel.short_side"),
        ("autolabel.hi_threshold=high", "autolabel.hi_threshold"),
        ("autolabel.walkable_labels=floor", "autolabel.walkable_labels"),
        ("autolabel.walkable_labels=[floor, 3]", "autolabel.walkable_labels"),
        ("train.arch=''", "train.arch"),
        ("train.encoder=null", "train.encoder"),
        ("train.encoder_weights=3", "train.encoder_weights"),
        ("train.epochs=0", "train.epochs"),
        ("train.epochs=2.5", "train.epochs"),
        ("train.batch_size=0", "train.batch_size"),
        ("train.lr=0", "train.lr"),
        ("train.lr=-1e-3", "train.lr"),
        ("train.lr=abc", "train.lr"),
        ("train.lr=.inf", "train.lr"),
        ("train.weight_decay=-0.1", "train.weight_decay"),
        ("train.num_workers=-1", "train.num_workers"),
        ("train.ce_weight=-1", "train.ce_weight"),
        ("train.dice_weight=.nan", "train.dice_weight"),
        ("train.amp=maybe", "train.amp"),
        ("train.amp=1", "train.amp"),
        ("train.out_dir=''", "train.out_dir"),
        ("train.device=null", "train.device"),
        ("train.seed=abc", "train.seed"),
        ("infer.threshold=abc", "infer.threshold"),
        ("infer.alpha=1.5", "infer.alpha"),
        ("infer.alpha=-0.1", "infer.alpha"),
        ("infer.color=[0, 255]", "infer.color"),
        ("infer.color=[0, 256, 0]", "infer.color"),
        ("infer.color=[0, 255.0, 0]", "infer.color"),
        ("infer.device=''", "infer.device"),
    ],
)
def test_type_and_range_errors_are_value_errors(override, field):
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        load_config(overrides=[override])


def test_both_loss_weights_zero_rejected():
    with pytest.raises(ValueError, match="ce_weight"):
        load_config(overrides=["train.ce_weight=0", "train.dice_weight=0"])
    load_config(overrides=["train.ce_weight=0"])  # 只关掉一项是允许的
    load_config(overrides=["train.dice_weight=0"])


@pytest.mark.parametrize(
    "override, attr, expected",
    [
        ("frames.fps=0", ("frames", "fps"), 0),  # <= 0 表示每帧都取
        ("frames.fps=.inf", ("frames", "fps"), float("inf")),
        ("frames.max_side=0", ("frames", "max_side"), 0),  # 0 表示不缩放
        ("frames.dedup_threshold=0", ("frames", "dedup_threshold"), 0),  # 0 关闭去重
        ("frames.jpeg_quality=100", ("frames", "jpeg_quality"), 100),
        ("train.weight_decay=0", ("train", "weight_decay"), 0),
        ("train.num_workers=0", ("train", "num_workers"), 0),
        ("train.encoder_weights=null", ("train", "encoder_weights"), None),
        ("train.amp=false", ("train", "amp"), False),
        ("infer.alpha=0", ("infer", "alpha"), 0),
        ("infer.alpha=1", ("infer", "alpha"), 1),
        ("infer.color=[255, 0, 0]", ("infer", "color"), [255, 0, 0]),
        ("autolabel.walkable_labels=[]", ("autolabel", "walkable_labels"), []),
    ],
)
def test_edge_values_accepted(override, attr, expected):
    cfg = load_config(overrides=[override])
    assert getattr(getattr(cfg, attr[0]), attr[1]) == expected


def test_pyyaml_parses_bare_exponent_as_string():
    # 前提：这正是需要 _coerce 的原因（若将来 PyYAML 改为 YAML 1.2 语义，本测试提醒可以去掉转换）
    assert yaml.safe_load("1e-3") == "1e-3"


@pytest.mark.parametrize("text, value", [("1e-3", 1e-3), ("3E-4", 3e-4), ("2e0", 2.0), ("1.5e-2", 1.5e-2)])
def test_float_fields_accept_scientific_notation_strings(text, value):
    # PyYAML 把不带小数点的 "1e-3" 解析成字符串；float 类型的配置项要能接受这种常见写法
    cfg = load_config(overrides=[f"train.lr={text}"])
    assert isinstance(cfg.train.lr, float) and cfg.train.lr == pytest.approx(value)


def test_scientific_notation_in_yaml_file(tmp_path):
    path = tmp_path / "sci.yaml"
    path.write_text("train:\n  lr: 1e-4\n  weight_decay: 5e-5\nframes:\n  fps: 2\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.train.lr == pytest.approx(1e-4) and isinstance(cfg.train.lr, float)
    assert cfg.train.weight_decay == pytest.approx(5e-5)
    assert cfg.frames.fps == 2  # int 本来就是合法数字，保持原样


def test_numeric_strings_not_coerced_for_non_float_fields():
    # 只有 float 字段做转换：int / str 字段收到字符串时不擅自改写
    with pytest.raises(ValueError, match=r"train\.epochs"):
        load_config(overrides=["train.epochs='5'"])
    assert load_config(overrides=["train.encoder='1e3'"]).train.encoder == "1e3"


def test_validate_accepts_paths_and_tuples():
    cfg = Config()
    cfg.data.root = Path("some") / "数据"
    cfg.train.out_dir = Path("runs") / "exp"
    cfg.data.image_size = (64, 96)
    cfg.infer.color = (0, 255, 0)
    cfg.hud.rects = [(0.0, 0.0, 0.5, 0.5)]
    validate(cfg)


def test_validate_rejects_bool_as_number():
    cfg = Config()
    cfg.train.epochs = True
    with pytest.raises(ValueError, match=r"train\.epochs"):
        validate(cfg)


def test_validate_accepts_numpy_scalars():
    import numpy as np

    cfg = Config()
    cfg.train.lr = np.float32(1e-3)
    cfg.train.epochs = np.int64(3)
    cfg.data.image_size = [np.int64(64), np.int64(96)]
    validate(cfg)
