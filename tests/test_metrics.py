"""dfwalk.metrics.ConfusionMatrix 的单元测试：手算示例、忽略像素、NaN 语义、越界校验、输入类型、累计与重置。

所有数据都在测试内合成，只用 CPU。
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pytest
import torch

from dfwalk import IGNORE_INDEX, NON_WALKABLE, NUM_CLASSES, WALKABLE
from dfwalk.metrics import ConfusionMatrix

I = IGNORE_INDEX
EXPECTED_KEYS = {
    "iou_non_walkable", "iou_walkable", "miou", "pixel_acc",
    "precision_walkable", "recall_walkable", "f1_walkable",
}

# 手算示例：
#   有效像素 (t, p)：(0,0) (0,1) (1,1) (1,1) / (1,0) (1,1) [忽略] (0,0)
#   matrix = [[2, 1],
#             [1, 3]]
HAND_TARGET = np.array([[0, 0, 1, 1], [1, 1, I, 0]], dtype=np.uint8)
HAND_PRED = np.array([[0, 1, 1, 1], [0, 1, 0, 0]], dtype=np.int64)
HAND_MATRIX = np.array([[2, 1], [1, 3]], dtype=np.int64)


def _naive_matrix(pred: np.ndarray, target: np.ndarray, c: int = NUM_CLASSES, ignore: int = I) -> np.ndarray:
    m = np.zeros((c, c), dtype=np.int64)
    for t, p in zip(np.asarray(target).ravel().tolist(), np.asarray(pred).ravel().tolist()):
        if t == ignore:
            continue
        m[t, p] += 1
    return m


def _assert_metrics_equal(got: dict, expected: dict) -> None:
    assert set(got) == set(expected)
    for k, v in expected.items():
        if math.isnan(v):
            assert math.isnan(got[k]), (k, got[k])
        else:
            assert got[k] == pytest.approx(v, abs=1e-12), (k, got[k], v)


# ---------------------------------------------------------------- 手算示例


def test_hand_computed_matrix_and_metrics():
    cm = ConfusionMatrix()
    cm.update(HAND_PRED, HAND_TARGET)
    np.testing.assert_array_equal(cm.matrix, HAND_MATRIX)
    assert cm.matrix.dtype == np.int64
    _assert_metrics_equal(cm.compute(), {
        "iou_non_walkable": 2 / 4,   # TP=2, FP=1, FN=1
        "iou_walkable": 3 / 5,       # TP=3, FP=1, FN=1
        "miou": (0.5 + 0.6) / 2,
        "pixel_acc": 5 / 7,
        "precision_walkable": 3 / 4,
        "recall_walkable": 3 / 4,
        "f1_walkable": 3 / 4,
    })


def test_asymmetric_hand_example_orientation():
    # matrix[t, p]：真实 1 被预测成 0 记在 [1, 0]
    cm = ConfusionMatrix()
    cm.update(np.array([0, 0, 0, 1]), np.array([1, 1, 0, 1]))
    np.testing.assert_array_equal(cm.matrix, [[1, 0], [2, 1]])
    r = cm.compute()
    assert r["precision_walkable"] == pytest.approx(1.0)       # 1 / (1 + 0)
    assert r["recall_walkable"] == pytest.approx(1 / 3)        # 1 / (1 + 2)
    assert r["f1_walkable"] == pytest.approx(2 * 1 / (2 * 1 + 0 + 2))
    assert r["iou_walkable"] == pytest.approx(1 / 3)
    assert r["iou_non_walkable"] == pytest.approx(1 / 3)       # 1 / (1 + 2 + 0)
    assert r["pixel_acc"] == pytest.approx(2 / 4)


def test_f1_equals_harmonic_mean_when_tp_positive():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, size=500)
    p = rng.integers(0, 2, size=500)
    cm = ConfusionMatrix()
    cm.update(p, t)
    r = cm.compute()
    prec, rec = r["precision_walkable"], r["recall_walkable"]
    assert r["f1_walkable"] == pytest.approx(2 * prec * rec / (prec + rec), rel=1e-12)


def test_compute_returns_plain_python_floats_with_expected_keys():
    cm = ConfusionMatrix()
    cm.update(HAND_PRED, HAND_TARGET)
    r = cm.compute()
    assert set(r) == EXPECTED_KEYS
    assert all(type(v) is float for v in r.values())
    empty = ConfusionMatrix().compute()
    assert set(empty) == EXPECTED_KEYS
    assert all(type(v) is float for v in empty.values())


# ---------------------------------------------------------------- 忽略像素


def test_ignored_pixels_skipped_regardless_of_pred():
    target = HAND_TARGET.copy()
    for p_at_ignore in (0, 1, I):
        pred = HAND_PRED.copy()
        pred[1, 2] = p_at_ignore
        cm = ConfusionMatrix()
        cm.update(pred, target)
        np.testing.assert_array_equal(cm.matrix, HAND_MATRIX)


def test_all_ignored_update_changes_nothing():
    cm = ConfusionMatrix()
    cm.update(np.zeros((3, 4), np.int64), np.full((3, 4), I, np.uint8))
    assert cm.matrix.sum() == 0
    assert all(math.isnan(v) for v in cm.compute().values())


def test_custom_ignore_index_and_more_classes():
    cm = ConfusionMatrix(num_classes=3, ignore_index=-1)
    cm.update(np.array([0, 1, 2, 2, 0]), np.array([0, 1, 2, -1, 1]))
    np.testing.assert_array_equal(cm.matrix, [[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    r = cm.compute()
    assert set(r) == EXPECTED_KEYS | {"iou_class2"}
    assert r["iou_class2"] == pytest.approx(1.0)
    assert r["iou_non_walkable"] == pytest.approx(1 / 2)
    assert r["iou_walkable"] == pytest.approx(1 / 2)
    assert r["miou"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    # 255 在这里不是忽略值，而是越界
    with pytest.raises(ValueError):
        cm.update(np.array([0]), np.array([255]))


# ---------------------------------------------------------------- NaN 语义


def test_empty_matrix_all_nan_without_warnings():
    cm = ConfusionMatrix()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = cm.compute()
    assert set(r) == EXPECTED_KEYS
    assert all(math.isnan(v) for v in r.values())


def test_only_non_walkable_everywhere_correct():
    cm = ConfusionMatrix()
    cm.update(np.zeros(10, np.int64), np.zeros(10, np.int64))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = cm.compute()
    _assert_metrics_equal(r, {
        "iou_non_walkable": 1.0,
        "iou_walkable": math.nan,          # TP+FP+FN = 0
        "miou": 1.0,                        # nanmean 跳过 NaN
        "pixel_acc": 1.0,
        "precision_walkable": math.nan,    # 没有预测为可走的像素
        "recall_walkable": math.nan,       # 没有真实可走像素
        "f1_walkable": math.nan,
    })


def test_only_walkable_everywhere_correct():
    cm = ConfusionMatrix()
    cm.update(np.ones(6, np.int64), np.ones(6, np.int64))
    _assert_metrics_equal(cm.compute(), {
        "iou_non_walkable": math.nan,
        "iou_walkable": 1.0,
        "miou": 1.0,
        "pixel_acc": 1.0,
        "precision_walkable": 1.0,
        "recall_walkable": 1.0,
        "f1_walkable": 1.0,
    })


def test_predicts_walkable_where_none_exists():
    cm = ConfusionMatrix()
    cm.update(np.ones(5, np.int64), np.zeros(5, np.int64))
    _assert_metrics_equal(cm.compute(), {
        "iou_non_walkable": 0.0,           # 0 / (0 + 0 + 5)
        "iou_walkable": 0.0,               # 0 / (0 + 5 + 0)
        "miou": 0.0,
        "pixel_acc": 0.0,
        "precision_walkable": 0.0,         # 0 / 5
        "recall_walkable": math.nan,       # 0 / 0
        "f1_walkable": 0.0,                # 0 / (0 + 5 + 0)
    })


def test_never_predicts_walkable():
    cm = ConfusionMatrix()
    cm.update(np.zeros(4, np.int64), np.ones(4, np.int64))
    _assert_metrics_equal(cm.compute(), {
        "iou_non_walkable": 0.0,
        "iou_walkable": 0.0,
        "miou": 0.0,
        "pixel_acc": 0.0,
        "precision_walkable": math.nan,    # 没有预测为可走的像素
        "recall_walkable": 0.0,
        "f1_walkable": 0.0,                # 完全失败记 0 而非 NaN
    })


# ---------------------------------------------------------------- 越界与类型校验


@pytest.mark.parametrize("bad", [2, -1, 254, 3])
def test_target_out_of_range_raises(bad):
    cm = ConfusionMatrix()
    cm.update(HAND_PRED, HAND_TARGET)
    target = HAND_TARGET.astype(np.int64)
    target[0, 0] = bad
    with pytest.raises(ValueError, match="target"):
        cm.update(HAND_PRED, target)
    np.testing.assert_array_equal(cm.matrix, HAND_MATRIX)  # 失败时不修改累计结果


@pytest.mark.parametrize("bad", [2, -1, 7, I])
def test_pred_out_of_range_at_valid_pixel_raises(bad):
    cm = ConfusionMatrix()
    pred = HAND_PRED.copy()
    pred[0, 0] = bad
    with pytest.raises(ValueError, match="pred"):
        cm.update(pred, HAND_TARGET)
    assert cm.matrix.sum() == 0


@pytest.mark.parametrize("bad", [2, -1, 7])
def test_pred_out_of_range_at_ignored_pixel_raises(bad):
    pred = HAND_PRED.copy()
    pred[1, 2] = bad  # HAND_TARGET[1, 2] 是忽略像素
    cm = ConfusionMatrix()
    with pytest.raises(ValueError, match="pred"):
        cm.update(pred, HAND_TARGET)
    assert cm.matrix.sum() == 0


def test_out_of_range_detected_in_all_ignored_batch():
    cm = ConfusionMatrix()
    with pytest.raises(ValueError):
        cm.update(np.full(4, 9), np.full(4, I))


def test_shape_mismatch_raises():
    cm = ConfusionMatrix()
    with pytest.raises(ValueError, match="形状"):
        cm.update(np.zeros((2, 3), np.int64), np.zeros((3, 2), np.int64))
    with pytest.raises(ValueError):
        cm.update(np.zeros((1, 2, 3), np.int64), np.zeros((2, 3), np.int64))


@pytest.mark.parametrize("which", ["pred", "target"])
def test_float_inputs_rejected(which):
    ints = np.zeros((2, 2), np.int64)
    floats = np.zeros((2, 2), np.float32)
    cm = ConfusionMatrix()
    with pytest.raises(TypeError, match=which):
        if which == "pred":
            cm.update(floats, ints)
        else:
            cm.update(ints, floats)
    with pytest.raises(TypeError):
        cm.update(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.long))


@pytest.mark.parametrize("kwargs", [{"num_classes": 1}, {"num_classes": 0}, {"ignore_index": 0},
                                    {"ignore_index": 1}, {"num_classes": 3, "ignore_index": 2}])
def test_invalid_constructor_args(kwargs):
    with pytest.raises(ValueError):
        ConfusionMatrix(**kwargs)


# ---------------------------------------------------------------- 输入类型与形状


def test_torch_int64_tensors_accepted():
    cm = ConfusionMatrix()
    cm.update(torch.from_numpy(HAND_PRED), torch.from_numpy(HAND_TARGET.astype(np.int64)))
    np.testing.assert_array_equal(cm.matrix, HAND_MATRIX)


def test_torch_argmax_pipeline():
    # 与 train.evaluate 的用法一致：pred = logits.argmax(1)，target 为 int64 张量（含 255）
    torch.manual_seed(0)
    logits = torch.randn(2, 2, 6, 7)
    target = torch.randint(0, 2, (2, 6, 7))
    target[:, 0] = IGNORE_INDEX
    pred = logits.argmax(1)
    cm = ConfusionMatrix()
    cm.update(pred, target)
    np.testing.assert_array_equal(cm.matrix, _naive_matrix(pred.numpy(), target.numpy()))


def test_mixed_torch_and_numpy_inputs():
    cm = ConfusionMatrix()
    cm.update(torch.from_numpy(HAND_PRED), HAND_TARGET)
    np.testing.assert_array_equal(cm.matrix, HAND_MATRIX)


def test_non_contiguous_tensor_accepted():
    pred = torch.from_numpy(HAND_PRED).t()
    target = torch.from_numpy(HAND_TARGET.astype(np.int64)).t()
    assert not pred.is_contiguous()
    cm = ConfusionMatrix()
    cm.update(pred, target)
    np.testing.assert_array_equal(cm.matrix, HAND_MATRIX)


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.int16, np.int32, np.int64, np.uint16])
def test_numpy_integer_dtypes(dtype):
    pred = HAND_PRED.astype(dtype)
    target = HAND_TARGET.astype(np.int64)
    target[target == I] = 0 if np.iinfo(dtype).max < I else I
    cm = ConfusionMatrix()
    cm.update(pred, target.astype(dtype))
    np.testing.assert_array_equal(cm.matrix, _naive_matrix(pred, target))


def test_bool_inputs_treated_as_zero_one():
    cm = ConfusionMatrix()
    cm.update(np.array([True, False, True]), np.array([1, 0, 0]))
    np.testing.assert_array_equal(cm.matrix, [[1, 1], [0, 1]])
    cm.update(torch.tensor([True]), torch.tensor([True]))
    np.testing.assert_array_equal(cm.matrix, [[1, 1], [0, 2]])


@pytest.mark.parametrize("shape", [(7,), (3, 5), (2, 3, 4), (2, 1, 3, 4), ()])
def test_any_matching_shape(shape):
    rng = np.random.default_rng(0)
    target = rng.choice([0, 1, I], size=shape)
    pred = rng.integers(0, 2, size=shape)
    cm = ConfusionMatrix()
    cm.update(pred, target)
    np.testing.assert_array_equal(cm.matrix, _naive_matrix(pred, target))


def test_python_lists_accepted():
    cm = ConfusionMatrix()
    cm.update([[0, 1], [1, 1]], [[0, 1], [I, 0]])
    np.testing.assert_array_equal(cm.matrix, [[1, 1], [0, 1]])


def test_empty_arrays_are_noop():
    cm = ConfusionMatrix()
    cm.update(np.zeros((0, 4), np.int64), np.zeros((0, 4), np.uint8))
    cm.update(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    assert cm.matrix.sum() == 0


# ---------------------------------------------------------------- 累计、重置、matrix 副本


def test_accumulation_equals_single_update():
    rng = np.random.default_rng(42)
    batches = [(rng.integers(0, 2, size=(2, 9, 11)), rng.choice([0, 1, I], size=(2, 9, 11))) for _ in range(5)]
    cm_acc = ConfusionMatrix()
    for p, t in batches:
        cm_acc.update(p, t)
    cm_all = ConfusionMatrix()
    cm_all.update(np.concatenate([b[0] for b in batches]), np.concatenate([b[1] for b in batches]))
    np.testing.assert_array_equal(cm_acc.matrix, cm_all.matrix)
    assert cm_acc.compute() == cm_all.compute()


def test_matches_naive_counting_on_random_data():
    rng = np.random.default_rng(7)
    target = rng.choice([0, 1, I], size=(4, 33, 47), p=[0.45, 0.45, 0.1])
    pred = rng.integers(0, 2, size=target.shape)
    cm = ConfusionMatrix()
    cm.update(pred, target)
    np.testing.assert_array_equal(cm.matrix, _naive_matrix(pred, target))
    assert cm.matrix.sum() == int((target != I).sum())


def test_reset_clears_counts():
    cm = ConfusionMatrix()
    cm.update(HAND_PRED, HAND_TARGET)
    cm.reset()
    np.testing.assert_array_equal(cm.matrix, np.zeros((2, 2)))
    assert cm.matrix.dtype == np.int64
    cm.update(HAND_PRED, HAND_TARGET)
    np.testing.assert_array_equal(cm.matrix, HAND_MATRIX)


def test_matrix_property_returns_copy():
    cm = ConfusionMatrix()
    cm.update(HAND_PRED, HAND_TARGET)
    m = cm.matrix
    m[:] = 999
    np.testing.assert_array_equal(cm.matrix, HAND_MATRIX)


def test_default_constructor_matches_project_constants():
    cm = ConfusionMatrix()
    assert cm.num_classes == NUM_CLASSES
    assert cm.ignore_index == IGNORE_INDEX
    assert cm.matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert (NON_WALKABLE, WALKABLE) == (0, 1)
    assert "ConfusionMatrix" in repr(cm)


def test_counts_do_not_overflow_int32_range():
    cm = ConfusionMatrix()
    big = np.ones(1 << 20, np.uint8)
    for _ in range(3):
        cm.update(big, big)
    cm._matrix[1, 1] += np.int64(1) << 33  # 模拟长时间累计后的大计数
    cm.update(big, big)
    assert cm.matrix[1, 1] == 4 * (1 << 20) + (1 << 33)
    assert cm.matrix.dtype == np.int64


# ---------------------------------------------------------------- 性能：np.bincount


def test_uses_bincount(monkeypatch):
    calls = []
    real = np.bincount

    def spy(*args, **kwargs):
        calls.append(kwargs.get("minlength"))
        return real(*args, **kwargs)

    monkeypatch.setattr(np, "bincount", spy)
    cm = ConfusionMatrix()
    cm.update(HAND_PRED, HAND_TARGET)
    assert calls == [NUM_CLASSES * NUM_CLASSES]
    np.testing.assert_array_equal(cm.matrix, HAND_MATRIX)


def test_large_batch_is_fast():
    # 一个训练 batch 大小（8 x 384 x 672 ≈ 206 万像素）的 update 应远小于 1 秒
    rng = np.random.default_rng(0)
    target = torch.from_numpy(rng.choice(np.array([0, 1, I], np.int64), size=(8, 384, 672)))
    pred = torch.from_numpy(rng.integers(0, 2, size=(8, 384, 672)))
    cm = ConfusionMatrix()
    start = time.perf_counter()
    cm.update(pred, target)
    elapsed = time.perf_counter() - start
    assert cm.matrix.sum() == int((target != I).sum())
    assert elapsed < 1.0, elapsed
