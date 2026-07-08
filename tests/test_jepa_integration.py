"""Tests for JEPA -> causal weight transfer (§7f). The load-bearing integration step:
the student and the causal encoder are the SAME class, so weights map key-for-key."""
import pytest
import torch

from core.models._reference_gene_tokens import ReferenceGeneTokenEncoder
from core.models.jepa import JEPAConfig, JEPAReport, _save_checkpoint, build_jepa
from core.models.jepa_integration import (
    LoadReport,
    initialize_causal_from_jepa,
    load_jepa_into_encoder,
)


def _write_ckpt(tmp_path, d_model=32, n_layers=2):
    cfg = JEPAConfig(
        n_genes=64, d_model=d_model, n_heads=2, n_layers=n_layers, n_proxy=4,
        ckpt_path=str(tmp_path / "jepa.pt"),
    )
    jepa = build_jepa(cfg)
    return _save_checkpoint(jepa, cfg, step=123, report=JEPAReport(steps_run=0, final_loss=0.0)), jepa


def test_load_transfers_all_weights(tmp_path):
    ckpt, jepa = _write_ckpt(tmp_path)
    # a fresh "causal encoder" of the same class/dims
    causal_encoder = ReferenceGeneTokenEncoder(n_genes=64, d_model=32, n_heads=2, n_layers=2, n_proxy=4)
    # it starts different from the student
    assert not torch.allclose(causal_encoder.gene_embed.weight, jepa.student.gene_embed.weight)
    report = load_jepa_into_encoder(causal_encoder, ckpt)
    assert isinstance(report, LoadReport)
    assert report.fully_initialized, f"missing={report.missing_keys} unexpected={report.unexpected_keys}"
    assert report.n_loaded == report.n_target_params and report.n_loaded > 0
    assert report.ckpt_step == 123 and report.ckpt_d_model == 32
    # after loading, a representative tensor matches the student exactly
    assert torch.allclose(causal_encoder.gene_embed.weight, jepa.student.gene_embed.weight)


def test_initialize_via_model_attribute(tmp_path):
    ckpt, jepa = _write_ckpt(tmp_path)

    class DummyCausal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ReferenceGeneTokenEncoder(
                n_genes=64, d_model=32, n_heads=2, n_layers=2, n_proxy=4
            )

    model = DummyCausal()
    report = initialize_causal_from_jepa(model, ckpt, encoder_attr="encoder")
    assert report.fully_initialized
    assert torch.allclose(model.encoder.gene_embed.weight, jepa.student.gene_embed.weight)


def test_missing_attribute_raises(tmp_path):
    ckpt, _ = _write_ckpt(tmp_path)
    model = torch.nn.Linear(4, 4)
    with pytest.raises(AttributeError):
        initialize_causal_from_jepa(model, ckpt, encoder_attr="encoder")


def test_arch_mismatch_reported_not_fatal(tmp_path):
    # student has 2 layers; target has 3 -> the extra layer's keys are "missing"
    ckpt, _ = _write_ckpt(tmp_path, n_layers=2)
    target = ReferenceGeneTokenEncoder(n_genes=64, d_model=32, n_heads=2, n_layers=3, n_proxy=4)
    report = load_jepa_into_encoder(target, ckpt, strict=False)
    assert not report.fully_initialized
    assert report.missing_keys, "extra target layer should surface as missing keys"
    assert report.n_loaded < report.n_target_params


def test_not_a_jepa_checkpoint_raises(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"not_student": 1}, bad)
    with pytest.raises(KeyError):
        load_jepa_into_encoder(ReferenceGeneTokenEncoder(n_genes=64, d_model=32, n_heads=2, n_layers=2, n_proxy=4), bad)
