"""core.models.jepa_integration — wire a JEPA checkpoint into the causal model (§7f).

  1. ``load_jepa_into_encoder`` — copy the JEPA student's weights into a causal model's
     ``GeneTokenEncoder``. Load-bearing and fully tested: the student and the causal
     encoder are the SAME class (§7e), so the state dict maps key-for-key.

  2. ``finetune_jepa_models`` (G5) — produce the two JEPA cells of the 2x2
     (``jepa_causal`` mask on, ``jepa_only`` mask off), each initialized from ``jepa.pt``
     and fine-tuned with **Developer 1's causal trainer**, so the only difference from
     the random-init ``causal``/``noncausal`` runs is the encoder initialization (the
     whole point of the ablation).

     Preferred path: Developer 1 adds an ``encoder_init_ckpt`` param to
     ``causal_cistransformer._run`` (a ~2-line hook); ``finetune_jepa_models`` detects
     and uses it. Until then it falls back to a faithful replica of ``_run`` with the
     encoder init inserted, and logs a warning — see DEV2_NOTES / the handshake flag.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import nn


@dataclass
class LoadReport:
    n_loaded: int
    n_target_params: int
    missing_keys: list = field(default_factory=list)
    unexpected_keys: list = field(default_factory=list)
    ckpt_step: Optional[int] = None
    ckpt_d_model: Optional[int] = None

    @property
    def fully_initialized(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys


def _load_ckpt(ckpt_path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "student_state_dict" not in ckpt:
        raise KeyError(
            f"{ckpt_path} is not a JEPA checkpoint (no 'student_state_dict'); expected the "
            "payload written by core.models.jepa._save_checkpoint"
        )
    return ckpt


def load_jepa_into_encoder(encoder: nn.Module, ckpt_path, strict: bool = False) -> LoadReport:
    """Load the JEPA student's weights into ``encoder`` (the causal model's
    GeneTokenEncoder) in place. Returns a LoadReport for logging/asserts."""
    ckpt = _load_ckpt(ckpt_path)
    result = encoder.load_state_dict(ckpt["student_state_dict"], strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    n_target = sum(1 for _ in encoder.state_dict())
    return LoadReport(
        n_loaded=n_target - len(missing),
        n_target_params=n_target,
        missing_keys=missing,
        unexpected_keys=unexpected,
        ckpt_step=ckpt.get("step"),
        ckpt_d_model=ckpt.get("d_model"),
    )


def initialize_causal_from_jepa(causal_model: nn.Module, ckpt_path, encoder_attr: str = "encoder",
                                strict: bool = False) -> LoadReport:
    """Locate ``causal_model.<encoder_attr>`` and initialize it from a JEPA ckpt."""
    if not hasattr(causal_model, encoder_attr):
        raise AttributeError(
            f"causal model {type(causal_model).__name__} has no attribute '{encoder_attr}'"
        )
    return load_jepa_into_encoder(getattr(causal_model, encoder_attr), ckpt_path, strict=strict)


# ---------------------------------------------------------------------------
# G5 — fine-tune the JEPA-init causal models (jepa_causal, jepa_only).
# ---------------------------------------------------------------------------
def finetune_jepa_models(jepa_ckpt=None, splits: Optional[Sequence[str]] = None,
                         cfg=None, record: bool = True, log_fn=print, **kw) -> dict:
    """Produce ``jepa_causal`` + ``jepa_only`` runs from a JEPA checkpoint (§7f, G5).

    Returns {model_name: {split: run_path}}.
    """
    from core import contract
    from core.models import causal_cistransformer as cc

    jepa_ckpt = jepa_ckpt or str(contract.checkpoint_path("jepa"))
    splits = list(splits) if splits is not None else list(contract.SPLITS)
    out = {}
    for model_name, use_mask in ((contract.MODEL_JEPA_CAUSAL, True), (contract.MODEL_JEPA_ONLY, False)):
        out[model_name] = _run_causal_with_jepa_init(
            cc, model_name, use_mask, splits, cfg, jepa_ckpt, record, log_fn
        )
    return out


def _run_causal_with_jepa_init(cc, model_name, use_causal_mask, splits, cfg, jepa_ckpt, record, log_fn):
    # Preferred: Developer 1's runner exposes an encoder-init hook.
    if "encoder_init_ckpt" in inspect.signature(cc._run).parameters:
        cc._run(model_name, use_causal_mask, splits, cfg, record=record, encoder_init_ckpt=jepa_ckpt)
        return {s: str(_run_path(model_name, s)) for s in splits}
    # Fallback: faithful replica of cc._run with the encoder init inserted.
    log_fn(
        f"[jepa_integration] cc._run has no 'encoder_init_ckpt' hook; using a private-internals "
        f"replica for {model_name}. Ask Developer 1 to add the hook so the 2x2 shares one trainer."
    )
    return _replicate_run_with_init(cc, model_name, use_causal_mask, splits, cfg, jepa_ckpt, record)


def _run_path(model_name, split):
    from core import contract
    return contract.run_path(model_name, split)


def _replicate_run_with_init(cc, model_name, use_causal_mask, splits, cfg, jepa_ckpt, record):
    """Mirror of ``causal_cistransformer._run`` with ``load_jepa_into_encoder`` inserted
    right after model construction. Depends on Developer 1's causal internals; kept in
    lockstep with ``_run`` (flagged for replacement by the clean hook)."""
    import pandas as pd

    from core import contract, split as split_mod
    from core import eval as ev

    cfg = cfg or cc.CausalConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hvg = split_mod.load_hvg()
    train_pb = pd.read_parquet(contract.PSEUDOBULK_TRAIN)
    hvg = [g for g in hvg if g in contract.pseudobulk_expr(train_pb).columns]

    samples = cc._build_samples(train_pb, hvg, cfg)
    esm2_t, ctx_t = cc._feature_tensors(hvg, device)
    model = cc._build_model(cfg, use_causal_mask).to(device)

    report = load_jepa_into_encoder(model.encoder, jepa_ckpt)   # <-- the only added step
    if not report.n_loaded:
        raise RuntimeError(f"JEPA init transferred 0 params into {model_name}'s encoder: {report}")

    cc._train(model, samples, esm2_t, ctx_t, cfg, device)
    out = {}
    for split in splits:
        pred = cc._predict_split(model, split, hvg, esm2_t, ctx_t, cfg, device)
        path = contract.run_path(model_name, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(path)
        if record:
            ev.evaluate_and_record(pred, split, model_name)
        out[split] = str(path)
    return out


__all__ = [
    "LoadReport", "load_jepa_into_encoder", "initialize_causal_from_jepa",
    "finetune_jepa_models",
]
