"""
core.models.do_attention — the corrected do-operator attention (UNIFIED_BUILD_PLAN.md
§1 + §7d).

do(g_k = c) removes only the edges *into* the perturbed gene (DoFormer, bioRxiv
2026.05.02.722054):
  * the perturbed gene must NOT attend to others  -> mask its QUERY ROW
  * other genes MUST still attend to it           -> DO NOT mask its KEY COLUMN (this is
    what lets the intervention propagate downstream — deleting it was the bug)
  * keep the perturbed gene's self-attention

The prior plans added `M[:, perturbed] = -inf`, severing the perturbed gene's outgoing
edges and deleting the very signal being predicted. That line is intentionally absent and
a regression test (`tests/test_do_mask.py`) asserts it never comes back.

`build_do_mask` returns an ADDITIVE (B, S, S) float mask (0 / -inf) suitable for
`F.scaled_dot_product_attention`. A CLS token at sequence position 0 is never masked.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_do_mask(seq_len: int, perturbed_seq_idx: torch.Tensor) -> torch.Tensor:
    """Additive do-mask, one per batch element.

    Args:
        seq_len: S (sequence length, including any CLS token).
        perturbed_seq_idx: (B,) long tensor — the perturbed gene's position in the
            sequence (already offset for CLS by the caller). Use -1 for "no intervention"
            (that row is left unmasked, e.g. control samples).

    Returns:
        (B, S, S) float mask: row = perturbed position is -inf except its diagonal; the
        perturbed *column* is left finite so downstream genes keep attending to it.
    """
    B = perturbed_seq_idx.shape[0]
    device = perturbed_seq_idx.device
    M = torch.zeros(B, seq_len, seq_len, device=device)
    has_pert = perturbed_seq_idx >= 0
    if has_pert.any():
        b_idx = torch.arange(B, device=device)[has_pert]
        rows = perturbed_seq_idx[has_pert]
        # cut ALL incoming edges to the perturbed gene (its query row) ...
        M[b_idx, rows, :] = float("-inf")
        # ... but keep self-attention so the row is not all -inf (no NaN in softmax).
        M[b_idx, rows, rows] = 0.0
        # NOTE: column `rows` is deliberately NOT masked -> intervention propagates.
    return M


class MultiHeadDoAttention(nn.Module):
    """Standard multi-head self-attention that accepts the additive do-mask above."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, do_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, S, d). do_mask: (B, S, S) additive mask or None (non-causal twin)."""
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, S, H, d_head)
        q = q.transpose(1, 2)  # (B, H, S, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if do_mask is not None:
            # broadcast (B, S, S) over heads -> (B, 1, S, S)
            attn_mask = do_mask.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.proj(out)


class DoTransformerBlock(nn.Module):
    """Pre-norm transformer block using MultiHeadDoAttention + gradient-checkpointing-friendly."""

    def __init__(self, d_model: int, n_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadDoAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor, do_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), do_mask=do_mask)
        x = x + self.mlp(self.norm2(x))
        return x
