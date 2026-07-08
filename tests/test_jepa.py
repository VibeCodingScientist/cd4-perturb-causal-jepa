"""Unit tests for the Cell-JEPA recipe (§7e), on Developer 1's real GeneTokenEncoder
with tiny synthetic (values, esm2, ctx) tensors."""
import pytest
import torch

from core.models.jepa import (
    CellJEPA,
    JEPACollapse,
    JEPAConfig,
    JEPAReport,
    MASK_VALUE_SENTINEL,
    cosine_alignment_loss,
    lr_schedule,
    mask_values,
    momentum_schedule,
    pretrain_jepa,
    vicreg_penalty,
)
from tests.fixtures import random_window_batch, tiny_jepa


# --------------------------------------------------------------------------- masking
def test_mask_values_touches_only_values():
    values, _, _ = random_window_batch(batch=4, window=20, seed=1)
    g = torch.Generator().manual_seed(0)
    masked, mask = mask_values(values, frac=0.5, generator=g)
    assert torch.all(masked[mask] == MASK_VALUE_SENTINEL)
    assert torch.allclose(masked[~mask], values[~mask])
    assert 0.2 < mask.float().mean().item() < 0.8


def test_mask_fraction_extremes():
    values, _, _ = random_window_batch(seed=2)
    none_masked, m0 = mask_values(values, frac=0.0)
    assert not m0.any() and torch.allclose(none_masked, values)
    _, m1 = mask_values(values, frac=1.0)
    assert m1.all()


# --------------------------------------------------------------------------- losses
def test_cosine_alignment_loss_bounds():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert cosine_alignment_loss(a, a.clone()).item() == pytest.approx(0.0, abs=1e-6)
    assert cosine_alignment_loss(a, -a).item() == pytest.approx(2.0, abs=1e-6)


def test_cosine_loss_stops_gradient_on_target():
    pred = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8, requires_grad=True)
    cosine_alignment_loss(pred, target).backward()
    assert pred.grad is not None
    assert target.grad is None, "target must be detached (stop-gradient)"


def test_vicreg_penalty_flags_collapse():
    collapsed = torch.zeros(16, 8) + 0.01 * torch.randn(1, 8)
    diverse = torch.randn(16, 8)
    assert vicreg_penalty(collapsed) > vicreg_penalty(diverse)


# --------------------------------------------------------------------------- module
def test_forward_returns_finite_losses_with_grad():
    jepa = tiny_jepa(seed=3).train()
    values, esm2, ctx = random_window_batch(batch=8, window=16, seed=3)
    out = jepa(values, esm2, ctx)
    assert out["loss"].requires_grad
    for key in ("loss", "loss_jepa", "loss_rec", "teacher_std"):
        assert torch.isfinite(out[key]).all()
    assert out["teacher_std"].item() >= 0.0


def test_forward_survives_batch_size_one():
    """Regression: the predictor must not crash on B=1 (LayerNorm, not BatchNorm)."""
    jepa = tiny_jepa(seed=3).train()
    values, esm2, ctx = random_window_batch(batch=1, window=16, seed=3)
    out = jepa(values, esm2, ctx)
    assert torch.isfinite(out["loss"]).all()


def test_teacher_is_frozen_and_gets_no_gradient():
    jepa = tiny_jepa(seed=4).train()
    values, esm2, ctx = random_window_batch(batch=8, window=16, seed=4)
    jepa(values, esm2, ctx)["loss"].backward()
    assert all(not p.requires_grad for p in jepa.teacher.parameters())
    assert all(p.grad is None for p in jepa.teacher.parameters())
    assert any(p.grad is not None for p in jepa.student.parameters())


def test_ema_update_is_a_convex_combination():
    jepa = tiny_jepa(seed=5)
    t_old = [p.clone() for p in jepa.teacher.parameters()]
    with torch.no_grad():
        for p in jepa.student.parameters():
            p.add_(0.5)
    s_new = [p.clone() for p in jepa.student.parameters()]
    jepa.ema_update(0.9)
    for pt, to_, sn in zip(jepa.teacher.parameters(), t_old, s_new):
        assert torch.allclose(pt, 0.9 * to_ + 0.1 * sn, atol=1e-6)


def test_teacher_independent_of_student():
    jepa = tiny_jepa(seed=6)
    t_before = next(jepa.teacher.parameters()).clone()
    with torch.no_grad():
        next(jepa.student.parameters()).add_(1.0)
    assert torch.allclose(next(jepa.teacher.parameters()), t_before)


def test_value_head_required():
    import torch.nn as nn

    class NoValueHead(nn.Module):
        d_model = 8

    with pytest.raises(TypeError):
        CellJEPA(NoValueHead(), 8)


# --------------------------------------------------------------------------- schedules
def test_momentum_schedule_ramps_base_to_final():
    assert momentum_schedule(0, 100, 0.996, 0.999) == pytest.approx(0.996, abs=1e-6)
    assert momentum_schedule(100, 100, 0.996, 0.999) == pytest.approx(0.999, abs=1e-6)
    mids = [momentum_schedule(s, 100, 0.996, 0.999) for s in range(0, 101, 10)]
    assert all(a <= b + 1e-9 for a, b in zip(mids, mids[1:])), "momentum must be non-decreasing"


def test_lr_schedule_warmup_then_decay():
    total, base = 100, 2e-4
    assert lr_schedule(0, total, base, warmup_frac=0.1) < base
    assert lr_schedule(10, total, base, warmup_frac=0.1) == pytest.approx(base, rel=0.2)
    assert lr_schedule(99, total, base, warmup_frac=0.1) < base
    assert lr_schedule(99, total, base, warmup_frac=0.1) >= 0.0


# --------------------------------------------------------------------------- training loop
def _synthetic_loader(cfg, hvg_n=64):
    from core.models.jepa_data import SyntheticHVGCells, build_cell_loader, synthetic_priors

    cells = SyntheticHVGCells(n_cells=256, hvg_n=hvg_n, seed=1).matrix()
    esm2, ctx = synthetic_priors(hvg_n, cfg.esm2_dim, cfg.ctx_dim, seed=1)
    return build_cell_loader(cfg, cells=cells, esm2=esm2, ctx=ctx)


def _tiny_cfg(**kw):
    base = dict(hvg_n=64, window=16, d_model=32, esm2_dim=16, ctx_dim=8, n_proxy=4,
                n_heads=2, steps=8, gate_steps=2, batch_size=16, budget_seconds=1e9,
                device="cpu", log_every=2, ckpt_every=0, seed=1)
    base.update(kw)
    return JEPAConfig(**base)


def test_pretrain_runs_and_checkpoints(tmp_path):
    cfg = _tiny_cfg(ckpt_path=str(tmp_path / "jepa.pt"))
    report = pretrain_jepa(_synthetic_loader(cfg), cfg, log_fn=lambda *_: None)
    assert report.steps_run == 8
    assert (tmp_path / "jepa.pt").exists()
    assert report.teacher_std_history and report.teacher_std_history[-1][1] > 0.0
    payload = torch.load(tmp_path / "jepa.pt", map_location="cpu", weights_only=False)
    assert "student_state_dict" in payload and payload["step"] == 8


def test_gate_applies_fallback_when_over_budget(tmp_path):
    cfg = _tiny_cfg(steps=10, budget_seconds=1e-9, auto_fallback=True, ckpt_path=str(tmp_path / "jepa.pt"))
    report = pretrain_jepa(_synthetic_loader(cfg), cfg, log_fn=lambda *_: None)
    assert report.fallback_applied
    assert report.config["d_model"] < 32


def test_pretrain_rejects_zero_steps(tmp_path):
    cfg = _tiny_cfg(steps=0, ckpt_path=str(tmp_path / "jepa.pt"))
    with pytest.raises(ValueError):
        pretrain_jepa(_synthetic_loader(cfg), cfg, log_fn=lambda *_: None)


def test_collapse_guard_escalates_then_raises():
    from core.models.jepa import _collapse_guard

    jepa = tiny_jepa(seed=7)
    report = JEPAReport(steps_run=0, final_loss=0.0)
    cfg = JEPAConfig(collapse_std_floor=1e-3, collapse_warn_ratio=0.1)
    _init, _floor = _collapse_guard({"teacher_std": torch.tensor(0.05)}, 1.0, cfg, jepa, report, lambda *_: None, 0.0)
    assert report.escalations == 1 and jepa.w_vicreg > 0.0
    with pytest.raises(JEPACollapse):
        _collapse_guard({"teacher_std": torch.tensor(1e-6)}, 1.0, cfg, jepa, report, lambda *_: None, 0.0)
