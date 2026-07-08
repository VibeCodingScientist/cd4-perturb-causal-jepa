"""Unit tests for the Cell-JEPA recipe (§7e). No real cells: tiny reference encoders
on random (gene_id, value) tensors, exactly as the role brief prescribes."""
import math

import pytest
import torch

from core.models.encoder_api import MASK_VALUE_SENTINEL
from core.models.jepa import (
    CellJEPA,
    JEPACollapse,
    JEPAConfig,
    JEPAReport,
    cosine_alignment_loss,
    lr_schedule,
    mask_values,
    momentum_schedule,
    pretrain_jepa,
    vicreg_penalty,
)
from tests.fixtures import random_cell_batch, tiny_jepa


# --------------------------------------------------------------------------- masking
def test_mask_values_touches_only_values_not_ids():
    _, values, _ = random_cell_batch(batch=4, length=20, seed=1)
    g = torch.Generator().manual_seed(0)
    masked, mask = mask_values(values, frac=0.5, generator=g)
    # masked positions hold the sentinel; unmasked positions are unchanged
    assert torch.all(masked[mask] == MASK_VALUE_SENTINEL)
    assert torch.allclose(masked[~mask], values[~mask])
    # roughly the requested fraction is masked
    assert 0.2 < mask.float().mean().item() < 0.8


def test_mask_never_selects_padding():
    _, values, kpm = random_cell_batch(batch=4, length=20, seed=2, pad=True)
    g = torch.Generator().manual_seed(0)
    _, mask = mask_values(values, frac=0.9, key_padding_mask=kpm, generator=g)
    assert not (mask & kpm).any(), "padding positions must never be masked"


# --------------------------------------------------------------------------- losses
def test_cosine_alignment_loss_bounds():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert cosine_alignment_loss(a, a.clone()).item() == pytest.approx(0.0, abs=1e-6)
    assert cosine_alignment_loss(a, -a).item() == pytest.approx(2.0, abs=1e-6)


def test_cosine_loss_stops_gradient_on_target():
    pred = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8, requires_grad=True)
    loss = cosine_alignment_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert target.grad is None, "target must be detached (stop-gradient)"


def test_vicreg_penalty_flags_collapse():
    collapsed = torch.zeros(16, 8) + 0.01 * torch.randn(1, 8)  # ~no variance across batch
    diverse = torch.randn(16, 8)
    assert vicreg_penalty(collapsed) > vicreg_penalty(diverse)


# --------------------------------------------------------------------------- module
def test_forward_returns_finite_losses_with_grad():
    jepa = tiny_jepa(seed=3).train()
    gene_ids, values, _ = random_cell_batch(batch=8, length=16, seed=3)
    out = jepa(gene_ids, values)
    assert out["loss"].requires_grad
    for key in ("loss", "loss_jepa", "loss_rec", "teacher_std"):
        assert torch.isfinite(out[key]).all()
    assert out["teacher_std"].item() >= 0.0


def test_teacher_is_frozen_and_gets_no_gradient():
    jepa = tiny_jepa(seed=4).train()
    gene_ids, values, _ = random_cell_batch(batch=8, length=16, seed=4)
    jepa(gene_ids, values)["loss"].backward()
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


def test_teacher_not_deepcopy_shared_with_student():
    jepa = tiny_jepa(seed=6)
    # perturbing the student must not change the teacher (independent tensors)
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
    assert lr_schedule(0, total, base, warmup_frac=0.1) < base       # warming up
    assert lr_schedule(10, total, base, warmup_frac=0.1) == pytest.approx(base, rel=0.2)
    assert lr_schedule(99, total, base, warmup_frac=0.1) < base      # decayed
    assert lr_schedule(99, total, base, warmup_frac=0.1) >= 0.0


# --------------------------------------------------------------------------- training loop
def test_pretrain_runs_and_checkpoints(tmp_path):
    from core.models.jepa_data import SyntheticCellDataset, build_cell_loader

    ckpt = tmp_path / "jepa.pt"
    cfg = JEPAConfig(
        n_genes=64, d_model=32, n_heads=2, n_layers=2, n_proxy=4,
        steps=8, gate_steps=2, batch_size=16, budget_seconds=1e9,
        device="cpu", log_every=2, ckpt_every=0, ckpt_path=str(ckpt), seed=1,
    )
    ds = SyntheticCellDataset(n_cells=256, n_genes=64, max_genes=16, seed=1)
    loader = build_cell_loader(cfg, dataset=ds)
    report = pretrain_jepa(loader, cfg, log_fn=lambda *_: None)
    assert report.steps_run == 8
    assert ckpt.exists()
    assert report.teacher_std_history, "collapse monitor recorded no teacher_std"
    assert report.teacher_std_history[-1][1] > 0.0, "teacher collapsed on a benign run"
    # the checkpoint holds a loadable student state dict
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "student_state_dict" in payload and payload["step"] == 8


def test_gate_applies_fallback_when_over_budget(tmp_path):
    from core.models.jepa_data import SyntheticCellDataset, build_cell_loader

    cfg = JEPAConfig(
        n_genes=64, d_model=32, n_heads=2, n_layers=2, n_proxy=4,
        steps=10, gate_steps=2, batch_size=16, budget_seconds=1e-9,  # impossible budget
        auto_fallback=True, device="cpu", ckpt_path=str(tmp_path / "jepa.pt"), seed=1,
    )
    ds = SyntheticCellDataset(n_cells=128, n_genes=64, max_genes=12, seed=1)
    loader = build_cell_loader(cfg, dataset=ds)
    report = pretrain_jepa(loader, cfg, log_fn=lambda *_: None)
    assert report.fallback_applied
    assert report.config["d_model"] < 32, "fallback should shrink d_model"


def test_collapse_guard_escalates_then_raises():
    from core.models.jepa import _collapse_guard

    jepa = tiny_jepa(seed=7)
    report = JEPAReport(steps_run=0, final_loss=0.0)
    cfg = JEPAConfig(collapse_std_floor=1e-3, collapse_warn_ratio=0.1)
    initial_std = 1.0
    # a low std relative to initial triggers an escalation (VICReg turns on)
    out = {"teacher_std": torch.tensor(0.05)}
    _init, _floor = _collapse_guard(out, initial_std, cfg, jepa, report, lambda *_: None, 0.0)
    assert report.escalations == 1
    assert jepa.w_vicreg > 0.0
    # a truly collapsed std after an escalation is fatal
    collapsed = {"teacher_std": torch.tensor(1e-6)}
    with pytest.raises(JEPACollapse):
        _collapse_guard(collapsed, initial_std, cfg, jepa, report, lambda *_: None, 0.0)


def test_pretrain_handles_padded_cells(tmp_path):
    """Padding must not break attention/pooling/reconstruction."""
    from core.models.jepa import build_jepa
    from tests.fixtures import random_cell_batch

    cfg = JEPAConfig(n_genes=64, d_model=32, n_heads=2, n_layers=2, n_proxy=4, device="cpu")
    jepa = build_jepa(cfg).train()
    gene_ids, values, kpm = random_cell_batch(batch=8, length=20, seed=8, pad=True)
    out = jepa(gene_ids, values, key_padding_mask=kpm)
    assert torch.isfinite(out["loss"]).all()
