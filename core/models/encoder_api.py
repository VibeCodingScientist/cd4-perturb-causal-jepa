"""core.models.encoder_api — the gene-token encoder interface shared by the causal
model (Developer 1) and JEPA pretraining (Developer 2).

Why this file exists
--------------------
UNIFIED_BUILD_PLAN.md §7e requires that the JEPA *student* be **the same
``GeneTokenEncoder`` class the causal model uses**, so that a JEPA checkpoint can
initialize ``CausalCisTransFormer.encoder`` and its weights transfer. Until
Developer 1 commits ``core/models/gene_tokens.py``, Developer 2 codes and unit-tests
against *this* interface plus a faithful reference implementation
(``core.models._reference_gene_tokens.ReferenceGeneTokenEncoder``).

The contract an encoder must satisfy so JEPA + weight-transfer work:

    * attribute ``d_model: int``               — pooled/token embedding width
    * attribute ``value_head: nn.Module``      — maps token embeddings [.., d_model]
                                                  to a scalar predicted expression
                                                  value (the JEPA reconstruction term)
    * ``forward(gene_ids, values, key_padding_mask=None, attn_mask=None)``
          gene_ids          : LongTensor  [B, L]  (Ensembl-id integer codes)
          values            : FloatTensor [B, L]  (log1p-CP10k expression; -1 == masked)
          key_padding_mask  : BoolTensor  [B, L] or None  (True == pad, ignored)
          attn_mask         : FloatTensor [L, L] / [B, L, L] or None
                              additive gene->gene attention bias. JEPA passes None;
                              the causal model passes the DoAttention do-mask (§7d).
                              **Both share the same weights** — this is the whole
                              point of making the encoder mask-aware rather than the
                              causal wrapper owning a separate attention stack.
      returns: EncoderOutput(tokens=[B, L, d_model], pooled=[B, d_model])
               ``pooled`` is the cell-level (CLS-like) embedding JEPA predicts.

Anything that satisfies the ``GeneEncoder`` Protocol below can be dropped into
``core.models.jepa`` unchanged. At core-frozen we swap the reference for the real
class; if Developer 1's signature differs, only the thin adapter in
``core.models.jepa_integration`` changes — never the JEPA training logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@dataclass
class EncoderOutput:
    """Return type of a gene-token encoder's ``forward``.

    tokens : [B, L, d_model]  per-gene contextual embeddings (used by value_head +
             the causal delta head).
    pooled : [B, d_model]     one cell-level embedding (CLS-like). JEPA's student and
             teacher both produce this; the predictor maps student.pooled toward
             teacher.pooled.
    """

    tokens: Tensor
    pooled: Tensor


@runtime_checkable
class GeneEncoder(Protocol):
    """Structural type both ``ReferenceGeneTokenEncoder`` and Developer 1's real
    ``GeneTokenEncoder`` must satisfy. ``runtime_checkable`` so tests can assert
    ``isinstance(enc, GeneEncoder)`` on the attribute surface."""

    d_model: int
    value_head: nn.Module

    def forward(
        self,
        gene_ids: Tensor,
        values: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> EncoderOutput:
        ...


# Sentinel written into `values` at masked positions (§7e: "mask EXPRESSION VALUES
# only, sentinel = -1"). Defined here so encoder, masker, and reconstruction agree.
MASK_VALUE_SENTINEL: float = -1.0
