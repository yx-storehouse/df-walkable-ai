"""dfwalk.losses.SegLoss 的单元测试：CE / Dice 数值、忽略像素、全忽略 batch、梯度、参数与输入校验。

所有张量都在测试内合成，只用 CPU。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dfwalk import IGNORE_INDEX, NON_WALKABLE, WALKABLE
from dfwalk.losses import SegLoss

N, C, H, W = 2, 2, 8, 12


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _random_batch(ignore_frac: float = 0.0, n: int = N, h: int = H, w: int = W, scale: float = 2.0):
    logits = (torch.randn(n, C, h, w) * scale).requires_grad_(True)
    target = torch.randint(0, C, (n, h, w), dtype=torch.long)
    if ignore_frac > 0:
        target[torch.rand(n, h, w) < ignore_frac] = IGNORE_INDEX
    return logits, target


def _manual_dice_term(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """按文档公式逐项手算整个 batch 的 1 - Dice_walkable。"""
    valid = target != IGNORE_INDEX
    p = torch.softmax(logits.float(), dim=1)[:, WALKABLE][valid]
    g = (target[valid] == WALKABLE).float()
    dice = (2 * (p * g).sum() + smooth) / (p.sum() + g.sum() + smooth)
    return 1 - dice


# ---------------------------------------------------------------- 与 F.cross_entropy 一致


def test_matches_cross_entropy_when_dice_disabled_no_ignore():
    logits, target = _random_batch()
    loss = SegLoss(ce_weight=1.0, dice_weight=0.0)(logits, target)
    expected = F.cross_entropy(logits, target)
    torch.testing.assert_close(loss, expected)


def test_matches_cross_entropy_with_ignored_pixels():
    logits, target = _random_batch(ignore_frac=0.4)
    assert (target == IGNORE_INDEX).any() and (target != IGNORE_INDEX).any()
    loss = SegLoss(ce_weight=1.0, dice_weight=0.0)(logits, target)
    expected = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
    torch.testing.assert_close(loss, expected)


def test_cross_entropy_gradient_matches_reference():
    logits, target = _random_batch(ignore_frac=0.3)
    SegLoss(1.0, 0.0)(logits, target).backward()
    ours = logits.grad.clone()
    ref_logits = logits.detach().clone().requires_grad_(True)
    F.cross_entropy(ref_logits, target, ignore_index=IGNORE_INDEX).backward()
    torch.testing.assert_close(ours, ref_logits.grad)


def test_ce_weight_scales_ce():
    logits, target = _random_batch(ignore_frac=0.2)
    base = SegLoss(1.0, 0.0)(logits, target)
    scaled = SegLoss(2.5, 0.0)(logits, target)
    torch.testing.assert_close(scaled, 2.5 * base)


# ---------------------------------------------------------------- Dice 数值


def test_dice_matches_manual_formula():
    logits, target = _random_batch(ignore_frac=0.3)
    loss = SegLoss(ce_weight=0.0, dice_weight=1.0)(logits, target)
    torch.testing.assert_close(loss, _manual_dice_term(logits, target))


@pytest.mark.parametrize("smooth", [0.0, 1e-3, 1.0, 10.0])
def test_dice_respects_smooth(smooth):
    logits, target = _random_batch(ignore_frac=0.2)
    loss = SegLoss(ce_weight=0.0, dice_weight=1.0, smooth=smooth)(logits, target)
    torch.testing.assert_close(loss, _manual_dice_term(logits, target, smooth))


def test_combined_loss_is_weighted_sum():
    logits, target = _random_batch(ignore_frac=0.25)
    loss = SegLoss(ce_weight=0.3, dice_weight=2.0)(logits, target)
    expected = 0.3 * F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX) + 2.0 * _manual_dice_term(
        logits, target
    )
    torch.testing.assert_close(loss, expected)


def test_default_weights_are_one_each():
    logits, target = _random_batch(ignore_frac=0.1)
    loss = SegLoss()(logits, target)
    expected = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX) + _manual_dice_term(logits, target)
    torch.testing.assert_close(loss, expected)


def test_dice_is_computed_over_whole_batch_not_per_sample():
    # 样本 0：全部为可走且预测完美；样本 1：只有 1 个有效像素，且预测错误
    logits = torch.zeros(2, C, 4, 4)
    target = torch.full((2, 4, 4), IGNORE_INDEX, dtype=torch.long)
    target[0] = WALKABLE
    logits[0, WALKABLE] = 30.0
    target[1, 0, 0] = WALKABLE
    logits[1, NON_WALKABLE, 0, 0] = 30.0
    loss = SegLoss(ce_weight=0.0, dice_weight=1.0, smooth=0.0)(logits, target)
    # 整个 batch：sum(p*g)≈16, sum(p)≈16, sum(g)=17 → 1 - 32/33
    torch.testing.assert_close(loss, torch.tensor(1 - 32 / 33), atol=1e-5, rtol=0)
    # 逐样本平均会得到 (0 + 1) / 2 = 0.5，明显不同
    assert abs(loss.item() - 0.5) > 0.4


def test_near_perfect_logits_give_near_zero_loss():
    target = torch.randint(0, C, (N, H, W))
    target[:, :2] = IGNORE_INDEX
    logits = (F.one_hot(target.clamp(max=C - 1), C).permute(0, 3, 1, 2).float() * 2 - 1) * 20.0
    logits.requires_grad_(True)
    dice_only = SegLoss(ce_weight=0.0, dice_weight=1.0)(logits, target)
    assert 0.0 <= dice_only.item() < 1e-6
    total = SegLoss()(logits, target)
    assert 0.0 <= total.item() < 1e-6


def test_wrong_logits_give_large_loss():
    target = torch.randint(0, C, (N, H, W))
    logits = (F.one_hot(1 - target, C).permute(0, 3, 1, 2).float() * 2 - 1) * 20.0
    dice_term = SegLoss(ce_weight=0.0, dice_weight=1.0)(logits, target)
    assert dice_term.item() > 0.9
    assert SegLoss(1.0, 0.0)(logits, target).item() > 30.0


def test_loss_is_scalar_float32():
    logits, target = _random_batch()
    loss = SegLoss()(logits, target)
    assert loss.ndim == 0 and loss.dtype == torch.float32
    assert loss.item() >= 0.0


# ---------------------------------------------------------------- 全部忽略


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"ce_weight": 1.0, "dice_weight": 0.0}, {"ce_weight": 0.0, "dice_weight": 1.0},
     {"smooth": 0.0}, {"ce_weight": 0.0, "dice_weight": 0.0}, {"ce_weight": 3.0, "dice_weight": 5.0}],
)
def test_all_ignored_returns_exact_zero_and_backward_works(kwargs):
    logits = (torch.randn(N, C, H, W) * 5).requires_grad_(True)
    target = torch.full((N, H, W), IGNORE_INDEX, dtype=torch.long)
    loss = SegLoss(**kwargs)(logits, target)
    assert loss.ndim == 0
    assert loss.item() == 0.0
    assert not torch.isnan(loss)
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 0


def test_all_ignored_with_nonfinite_logits_still_zero():
    logits = torch.randn(N, C, H, W)
    logits[0, 0, 0, 0] = float("inf")
    logits[1, 1, 3, 3] = float("-inf")
    logits.requires_grad_(True)
    target = torch.full((N, H, W), IGNORE_INDEX, dtype=torch.long)
    loss = SegLoss()(logits, target)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_all_ignored_with_custom_ignore_index():
    logits = torch.randn(N, C, H, W, requires_grad=True)
    target = torch.full((N, H, W), -1, dtype=torch.long)
    loss = SegLoss(ignore_index=-1)(logits, target)
    assert loss.item() == 0.0
    loss.backward()


def test_partially_ignored_sample_in_batch_is_fine():
    logits, target = _random_batch()
    target[0] = IGNORE_INDEX  # 第一个样本全部忽略，第二个正常
    loss = SegLoss()(logits, target)
    assert torch.isfinite(loss)
    expected = SegLoss()(logits[1:], target[1:])
    torch.testing.assert_close(loss, expected)


# ---------------------------------------------------------------- 忽略像素不影响 loss


def test_ignored_pixel_logits_do_not_change_loss():
    logits, target = _random_batch(ignore_frac=0.4)
    ignored = (target == IGNORE_INDEX).unsqueeze(1).expand_as(logits)
    loss_fn = SegLoss(ce_weight=0.7, dice_weight=1.3)
    before = loss_fn(logits, target)

    perturbed = logits.detach().clone()
    perturbed[ignored] = torch.randn(int(ignored.sum())) * 1000.0
    after = loss_fn(perturbed, target)
    assert torch.equal(before.detach(), after.detach())


def test_nonfinite_logits_at_ignored_pixels_do_not_change_forward():
    logits, target = _random_batch(ignore_frac=0.4)
    loss_fn = SegLoss()
    before = loss_fn(logits, target)
    perturbed = logits.detach().clone()
    idx = (target == IGNORE_INDEX).nonzero()
    n, h, w = idx[0].tolist()
    perturbed[n, :, h, w] = torch.tensor([float("inf"), float("nan")])
    perturbed.requires_grad_(True)
    after = loss_fn(perturbed, target)
    assert torch.isfinite(after)
    torch.testing.assert_close(after, before.detach())
    # 反向同样不受影响：梯度处处有限，被忽略像素梯度恰为 0，有效像素梯度与原来一致
    after.backward()
    before.backward()
    assert torch.isfinite(perturbed.grad).all()
    ignored = (target == IGNORE_INDEX).unsqueeze(1).expand_as(logits)
    assert torch.count_nonzero(perturbed.grad[ignored]) == 0
    torch.testing.assert_close(perturbed.grad, logits.grad)


def test_ignored_pixels_get_zero_gradient():
    logits, target = _random_batch(ignore_frac=0.4)
    SegLoss()(logits, target).backward()
    ignored = (target == IGNORE_INDEX).unsqueeze(1).expand_as(logits)
    assert torch.count_nonzero(logits.grad[ignored]) == 0
    assert torch.count_nonzero(logits.grad[~ignored]) > 0


# ---------------------------------------------------------------- 梯度


@pytest.mark.parametrize("scale", [1e-3, 1.0, 50.0, 1e4])
def test_gradients_finite_for_various_logit_scales(scale):
    logits, target = _random_batch(ignore_frac=0.2, scale=scale)
    loss = SegLoss()(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("smooth", [0.0, 1.0])
def test_gradients_finite_when_no_walkable_pixels(smooth):
    logits = torch.randn(N, C, H, W, requires_grad=True)
    target = torch.zeros(N, H, W, dtype=torch.long)  # 全部不可走
    loss = SegLoss(smooth=smooth)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_optimization_decreases_loss():
    target = torch.randint(0, C, (1, H, W))
    target[:, :, :3] = IGNORE_INDEX
    logits = torch.zeros(1, C, H, W, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=5.0)
    loss_fn = SegLoss()
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(logits, target)
        first = loss.item() if first is None else first
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first


# ---------------------------------------------------------------- dtype 与 AMP


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.int16])
def test_integer_target_dtypes_accepted(dtype):
    logits, target = _random_batch(ignore_frac=0.2)
    expected = SegLoss()(logits, target)
    got = SegLoss()(logits, target.to(dtype))
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_logits_computed_in_float32(dtype):
    logits, target = _random_batch(ignore_frac=0.2)
    low = logits.detach().to(dtype).requires_grad_(True)
    loss = SegLoss()(low, target)
    assert loss.dtype == torch.float32
    expected = SegLoss()(low.detach().float(), target)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert low.grad is not None and low.grad.dtype == dtype
    assert torch.isfinite(low.grad.float()).all()


def test_large_batch_sum_does_not_overflow_half():
    # 有效像素数远超 float16 上限 65504；输入为 float16 时也不能溢出
    logits = torch.zeros(1, C, 300, 300, dtype=torch.float16)
    logits[:, WALKABLE] = 10.0
    target = torch.ones(1, 300, 300, dtype=torch.long)
    loss = SegLoss()(logits, target)
    assert torch.isfinite(loss) and loss.item() < 1e-3


def test_more_than_two_classes_supported():
    logits = torch.randn(N, 3, H, W, requires_grad=True)
    target = torch.randint(0, 3, (N, H, W))
    target[0, 0] = IGNORE_INDEX
    loss = SegLoss()(logits, target)
    valid = target != IGNORE_INDEX
    p = torch.softmax(logits, 1)[:, WALKABLE][valid]
    g = (target[valid] == WALKABLE).float()
    expected = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX) + 1 - (2 * (p * g).sum() + 1) / (
        p.sum() + g.sum() + 1
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()


# ---------------------------------------------------------------- 参数与输入校验


@pytest.mark.parametrize("kwargs", [{"ce_weight": -1.0}, {"dice_weight": -0.1}, {"smooth": -1.0},
                                    {"ce_weight": float("nan")}])
def test_negative_or_nan_hyperparams_rejected(kwargs):
    with pytest.raises(ValueError):
        SegLoss(**kwargs)


def test_attributes_and_repr():
    loss_fn = SegLoss(ce_weight=0.5, dice_weight=2, ignore_index=7, smooth=0.25)
    assert (loss_fn.ce_weight, loss_fn.dice_weight, loss_fn.ignore_index, loss_fn.smooth) == (0.5, 2.0, 7, 0.25)
    assert "dice_weight=2.0" in repr(loss_fn)
    assert isinstance(loss_fn, torch.nn.Module)
    assert list(loss_fn.parameters()) == []


def test_bad_logits_rank_rejected():
    with pytest.raises(ValueError, match="logits"):
        SegLoss()(torch.randn(C, H, W), torch.zeros(H, W, dtype=torch.long).unsqueeze(0))


def test_bad_target_rank_rejected():
    with pytest.raises(ValueError, match="target"):
        SegLoss()(torch.randn(N, C, H, W), torch.zeros(N, 1, H, W, dtype=torch.long))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="不匹配"):
        SegLoss()(torch.randn(N, C, H, W), torch.zeros(N, H, W + 1, dtype=torch.long))


def test_single_channel_logits_rejected():
    with pytest.raises(ValueError, match="WALKABLE"):
        SegLoss()(torch.randn(N, 1, H, W), torch.zeros(N, H, W, dtype=torch.long))


def test_float_target_rejected():
    with pytest.raises(TypeError):
        SegLoss()(torch.randn(N, C, H, W), torch.zeros(N, H, W))


def test_integer_logits_rejected():
    with pytest.raises(TypeError):
        SegLoss()(torch.zeros(N, C, H, W, dtype=torch.long), torch.zeros(N, H, W, dtype=torch.long))
