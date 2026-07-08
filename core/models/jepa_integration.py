"""core.models.jepa_integration — wire a JEPA checkpoint into the causal model (§7f).

Two responsibilities:

  1. ``load_jepa_into_encoder`` — copy the JEPA student's weights into a causal
     model's gene-token encoder. This is the load-bearing, fully-tested step: the
     student and the causal encoder are the SAME class (§7e), so the state dict maps
     key-for-key. Verified now against the reference encoder.

  2. ``build_jepa_causal_model`` / ``build_jepa_only_model`` + ``finetune_and_predict``
     — the G5 seams that produce the two JEPA cells of the 2x2 (``jepa_causal`` with
     the do-mask ON, ``jepa_only`` with it OFF), each initialized from ``jepa.pt``
     and fine-tuned on the pseudobulk delta task. These lazily import Developer 1's
     ``CausalCisTransFormer`` + trainer and are exercised for real at integration;
     until then they raise a clear, actionable error.

The random-init cells of the 2x2 (``causal``, ``noncausal``) come straight from
Developer 1's G2/G3 runs — Developer 2 does not retrain them (see ``core.ablation``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import nn


@dataclass
class LoadReport:
    """Outcome of copying JEPA weights into a target encoder."""

    n_loaded: int
    n_target_params: int
    missing_keys: list = field(default_factory=list)      # in target, absent from ckpt
    unexpected_keys: list = field(default_factory=list)   # in ckpt, absent from target
    ckpt_step: Optional[int] = None
    ckpt_d_model: Optional[int] = None

    @property
    def fully_initialized(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys


def _load_ckpt(ckpt_path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "student_state_dict" not in ckpt:
        raise KeyError(
            f"{ckpt_path} is not a JEPA checkpoint (no 'student_state_dict'); "
            "expected the payload written by core.models.jepa._save_checkpoint"
        )
    return ckpt


def load_jepa_into_encoder(
    encoder: nn.Module,
    ckpt_path,
    strict: bool = False,
) -> LoadReport:
    """Load the JEPA student's weights into ``encoder`` in place.

    ``encoder`` is the causal model's gene-token encoder (same class as the JEPA
    student). With ``strict=False`` any keys the causal encoder adds beyond the shared
    backbone are reported, not fatal. Returns a ``LoadReport`` for logging/asserts.
    """
    ckpt = _load_ckpt(ckpt_path)
    student_sd = ckpt["student_state_dict"]
    result = encoder.load_state_dict(student_sd, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    n_target = sum(1 for _ in encoder.state_dict())
    n_loaded = n_target - len(missing)
    return LoadReport(
        n_loaded=n_loaded,
        n_target_params=n_target,
        missing_keys=missing,
        unexpected_keys=unexpected,
        ckpt_step=ckpt.get("step"),
        ckpt_d_model=ckpt.get("d_model"),
    )


def initialize_causal_from_jepa(
    causal_model: nn.Module,
    ckpt_path,
    encoder_attr: str = "encoder",
    strict: bool = False,
) -> LoadReport:
    """Locate ``causal_model.<encoder_attr>`` and initialize it from a JEPA ckpt."""
    if not hasattr(causal_model, encoder_attr):
        raise AttributeError(
            f"causal model {type(causal_model).__name__} has no attribute "
            f"'{encoder_attr}'; pass the correct encoder_attr for weight transfer"
        )
    encoder = getattr(causal_model, encoder_attr)
    return load_jepa_into_encoder(encoder, ckpt_path, strict=strict)


# ---------------------------------------------------------------------------
# G5 seams — realized once Developer 1 ships CausalCisTransFormer + trainer.
# ---------------------------------------------------------------------------
def _causal_cls():
    try:
        from .causal_cistransformer import CausalCisTransFormer  # Developer 1 (G2/G3)
        return CausalCisTransFormer
    except Exception as e:  # pragma: no cover - integration-time path
        raise ImportError(
            "core.models.causal_cistransformer.CausalCisTransFormer is not available "
            "yet (Developer 1 / core-frozen). build_jepa_causal_model is a G5 seam and "
            "runs only after the causal model lands."
        ) from e


def build_jepa_causal_model(ckpt_path, use_causal_mask: bool, **model_kwargs) -> nn.Module:
    """Construct a CausalCisTransFormer (mask on=jepa_causal / off=jepa_only) and
    initialize its encoder from ``jepa.pt``.  (§7f)"""
    cls = _causal_cls()
    model = cls(use_causal_mask=use_causal_mask, **model_kwargs)
    report = initialize_causal_from_jepa(model, ckpt_path)
    if not report.n_loaded:
        raise RuntimeError(f"JEPA init transferred 0 params into the causal encoder: {report}")
    return model


def build_jepa_only_model(ckpt_path, **model_kwargs) -> nn.Module:
    """jepa_only = JEPA-init + causal mask OFF (§7f)."""
    return build_jepa_causal_model(ckpt_path, use_causal_mask=False, **model_kwargs)


def finetune_and_predict(
    model_name: str,
    jepa_ckpt: str,
    splits,
    use_causal_mask: bool,
    **kwargs,
) -> dict:
    """G5 job body: init from JEPA, fine-tune on the pseudobulk delta task, and write
    ``runs/<model_name>_<split>.parquet`` for each split via the contract path.

    Delegates the actual optimization to Developer 1's causal trainer
    (``core.models.causal_cistransformer.train_causal_model`` or equivalent), which is
    the same code that produced the random-init ``causal``/``noncausal`` runs — so the
    only JEPA-specific change is the encoder initialization. Returns {split: run_path}.
    """
    from core import contract

    if model_name not in (contract.MODEL_JEPA_CAUSAL, contract.MODEL_JEPA_ONLY):
        raise ValueError(f"finetune_and_predict is for jepa_causal / jepa_only, got {model_name!r}")

    try:  # pragma: no cover - integration-time path
        from .causal_cistransformer import train_causal_model
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Developer 1's causal trainer (core.models.causal_cistransformer."
            "train_causal_model) is required to fine-tune the JEPA-init model. This "
            "runs at core-frozen (G5)."
        ) from e

    model = build_jepa_causal_model(jepa_ckpt, use_causal_mask=use_causal_mask, **kwargs)
    out = {}
    for split in splits:  # pragma: no cover - integration-time path
        pred_delta_df = train_causal_model(model, split=split, model_name=model_name, **kwargs)
        path = contract.run_path(model_name, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        pred_delta_df.to_parquet(path)
        out[split] = str(path)
    return out


__all__ = [
    "LoadReport", "load_jepa_into_encoder", "initialize_causal_from_jepa",
    "build_jepa_causal_model", "build_jepa_only_model", "finetune_and_predict",
]
