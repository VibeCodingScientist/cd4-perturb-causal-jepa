"""Tests for JEPA -> causal weight transfer (§7f) against Developer 1's real
GeneTokenEncoder. The student and the causal encoder are the SAME class, so the JEPA
student's state dict maps key-for-key into the causal model's encoder."""
import pytest
import torch
import torch.nn as nn

from core.models.gene_tokens import GeneTokenEncoder
from core.models.jepa import JEPAConfig, JEPAReport, _save_checkpoint, build_jepa
from core.models.jepa_integration import (
    LoadReport,
    initialize_causal_from_jepa,
    load_jepa_into_encoder,
)


def _real_encoder(d_model=32, esm2_dim=16, ctx_dim=8, n_proxy=4):
    return GeneTokenEncoder(d_model=d_model, esm2_dim=esm2_dim, ctx_dim=ctx_dim, n_proxy=n_proxy, n_heads=2)


def _write_ckpt(tmp_path, d_model=32):
    cfg = JEPAConfig(d_model=d_model, esm2_dim=16, ctx_dim=8, n_proxy=4, n_heads=2,
                     ckpt_path=str(tmp_path / "jepa.pt"))
    jepa = build_jepa(cfg)
    return _save_checkpoint(jepa, cfg, step=123, report=JEPAReport(steps_run=0, final_loss=0.0)), jepa


def test_load_transfers_all_weights(tmp_path):
    ckpt, jepa = _write_ckpt(tmp_path)
    causal_encoder = _real_encoder()
    assert not torch.allclose(causal_encoder.phi_expr.weight, jepa.student.phi_expr.weight)
    report = load_jepa_into_encoder(causal_encoder, ckpt)
    assert isinstance(report, LoadReport)
    assert report.fully_initialized, f"missing={report.missing_keys} unexpected={report.unexpected_keys}"
    assert report.n_loaded == report.n_target_params and report.n_loaded > 0
    assert report.ckpt_step == 123 and report.ckpt_d_model == 32
    assert torch.allclose(causal_encoder.phi_expr.weight, jepa.student.phi_expr.weight)


def test_initialize_via_model_attribute(tmp_path):
    ckpt, jepa = _write_ckpt(tmp_path)

    class DummyCausal(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _real_encoder()

    model = DummyCausal()
    report = initialize_causal_from_jepa(model, ckpt, encoder_attr="encoder")
    assert report.fully_initialized
    assert torch.allclose(model.encoder.phi_expr.weight, jepa.student.phi_expr.weight)


def test_missing_attribute_raises(tmp_path):
    ckpt, _ = _write_ckpt(tmp_path)
    with pytest.raises(AttributeError):
        initialize_causal_from_jepa(nn.Linear(4, 4), ckpt, encoder_attr="encoder")


def test_causal_specific_params_are_reported_not_fatal(tmp_path):
    # the causal encoder may carry params beyond the JEPA-shared backbone -> reported missing
    ckpt, _ = _write_ckpt(tmp_path)
    target = _real_encoder()
    target.register_parameter("causal_only_head", nn.Parameter(torch.zeros(3)))
    report = load_jepa_into_encoder(target, ckpt, strict=False)
    assert not report.fully_initialized
    assert any("causal_only_head" in k for k in report.missing_keys)
    assert report.n_loaded < report.n_target_params


def test_not_a_jepa_checkpoint_raises(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"not_student": 1}, bad)
    with pytest.raises(KeyError):
        load_jepa_into_encoder(_real_encoder(), bad)


def test_transfer_into_real_causal_model(tmp_path):
    """The definitive §7f proof: a JEPA student loads into Developer 1's actual
    CausalCisTransFormer.encoder, while the do-masked blocks/heads stay fresh."""
    from core.models import causal_cistransformer as cc

    ccfg = cc.CausalConfig(d_model=32, n_proxy=4, n_heads=2, n_layers=1)
    model = cc._build_model(ccfg, use_causal_mask=True)   # real causal model

    # JEPA student must mirror the causal encoder's construction (esm2/ctx dims are
    # fixed to the contract in _build_model).
    from core import contract

    jcfg = JEPAConfig(d_model=32, esm2_dim=contract.ESM2_DIM, ctx_dim=contract.CONTEXT_PRIOR_DIM,
                      n_proxy=4, n_proxy_rounds=2, n_heads=2, ckpt_path=str(tmp_path / "jepa.pt"))
    jepa = build_jepa(jcfg)
    ckpt = _save_checkpoint(jepa, jcfg, step=7, report=JEPAReport(steps_run=0, final_loss=0.0))

    blocks_before = [p.clone() for p in model.blocks.parameters()]
    report = load_jepa_into_encoder(model.encoder, ckpt)
    assert report.fully_initialized, f"missing={report.missing_keys[:5]} unexpected={report.unexpected_keys[:5]}"
    # encoder now equals the JEPA student
    assert torch.allclose(model.encoder.phi_expr.weight, jepa.student.phi_expr.weight)
    assert torch.allclose(model.encoder.cls_query, jepa.student.cls_query)
    # the causal-only do-masked blocks are untouched (fine-tuned fresh)
    for pb, pa in zip(blocks_before, model.blocks.parameters()):
        assert torch.allclose(pb, pa)


def test_finetune_detects_hook_absence(monkeypatch, tmp_path):
    """finetune falls back (with a warning) when Dev1's _run has no encoder_init_ckpt hook."""
    from core.models import jepa_integration as ji

    ckpt, _ = _write_ckpt(tmp_path)
    calls = {"replica": 0}

    def fake_replicate(cc, model_name, use_mask, splits, cfg, jc, record):
        calls["replica"] += 1
        return {s: f"/runs/{model_name}_{s}.parquet" for s in splits}

    monkeypatch.setattr(ji, "_replicate_run_with_init", fake_replicate)
    logs = []
    out = ji.finetune_jepa_models(jepa_ckpt=ckpt, splits=["condition"], log_fn=logs.append)
    # both jepa cells routed through the fallback replica (no hook in the current cc._run)
    from core import contract

    assert set(out) == {contract.MODEL_JEPA_CAUSAL, contract.MODEL_JEPA_ONLY}
    assert calls["replica"] == 2
    assert any("encoder_init_ckpt" in m for m in logs)
