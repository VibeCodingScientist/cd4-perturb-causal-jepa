"""
core.models.gene_tokens — CisTransCell-style gene-token encoder (UNIFIED_BUILD_PLAN.md §7d).

Gene token:  Z = phi_expr(expr) + phi_fuse(cat[phi_reg(context_prior), phi_cod(esm2)])
plus K=32 learned regulatory-proxy tokens that mediate gene->gene effects through a
bottleneck (proxy round: proxy = g2p(proxy, genes); genes = p2g(genes, proxy); 2 rounds,
each gene's contribution to the proxies gated by a learned per-gene regulator score).

This is the SAME class the causal model's encoder uses and the SAME class Cell-JEPA
pretrains (§7e), so a JEPA checkpoint initializes the causal encoder directly — the whole
point of the 2x2's JEPA cells. It therefore exposes both a per-gene token stack (consumed
by the do-masked transformer blocks) and a pooled CLS embedding + a `value_head` (consumed
by JEPA's cosine + reconstruction losses).

The intervention is applied at the INPUT (the caller clamps the perturbed gene's value to
its knockdown level, §7d); directional propagation is enforced later by the do-mask in the
transformer blocks. The proxy rounds here build shared context on that intervened input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    tokens: torch.Tensor  # (B, L, d) per-gene embeddings
    cls: torch.Tensor     # (B, d) pooled cell/sample embedding (JEPA target)


class GeneTokenEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        esm2_dim: int = 1280,
        ctx_dim: int = 512,
        n_proxy: int = 32,
        n_proxy_rounds: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_proxy_rounds = n_proxy_rounds

        # token construction
        self.phi_expr = nn.Linear(1, d_model)
        self.phi_cod = nn.Linear(esm2_dim, d_model)
        self.phi_reg = nn.Linear(ctx_dim, d_model)
        self.phi_fuse = nn.Linear(2 * d_model, d_model)

        # per-gene regulator score gates a gene's contribution to the proxy bottleneck
        self.regulator = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        # K learned regulatory-proxy tokens + the two cross-attentions of a proxy round
        self.proxy = nn.Parameter(torch.randn(1, n_proxy, d_model) * 0.02)
        self.g2p = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.p2g = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_p = nn.LayerNorm(d_model)
        self.norm_g = nn.LayerNorm(d_model)

        # CLS = learned query attention-pool over the final gene tokens
        self.cls_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # JEPA reconstruction head (token embedding -> expression value)
        self.value_head = nn.Linear(d_model, 1)

    def build_gene_prior(self, esm2: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """phi_fuse(cat[phi_reg(ctx), phi_cod(esm2)]) -> (L, d) static per-gene prior."""
        fused = self.phi_fuse(torch.cat([self.phi_reg(ctx), self.phi_cod(esm2)], dim=-1))
        return fused  # (L, d)

    def forward(
        self,
        values: torch.Tensor,        # (B, L) expression values (perturbed gene pre-clamped)
        esm2: torch.Tensor,          # (L, esm2_dim) static gene prior (shared over batch)
        ctx: torch.Tensor,           # (L, ctx_dim)
    ) -> EncoderOutput:
        B, L = values.shape
        gene_prior = self.build_gene_prior(esm2, ctx)          # (L, d)
        Z = self.phi_expr(values.unsqueeze(-1)) + gene_prior.unsqueeze(0)  # (B, L, d)

        proxy = self.proxy.expand(B, -1, -1)                   # (B, K, d)
        gate = self.regulator(gene_prior).squeeze(-1).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
        for _ in range(self.n_proxy_rounds):
            # genes -> proxies: weight each gene's contribution by its regulator score
            genes_weighted = Z * gate                          # (B, L, d)
            p_upd, _ = self.g2p(proxy, genes_weighted, genes_weighted)
            proxy = self.norm_p(proxy + p_upd)
            # proxies -> genes
            g_upd, _ = self.p2g(Z, proxy, proxy)
            Z = self.norm_g(Z + g_upd)

        cls_q = self.cls_query.expand(B, -1, -1)               # (B, 1, d)
        cls, _ = self.cls_attn(cls_q, Z, Z)                    # (B, 1, d)
        return EncoderOutput(tokens=Z, cls=cls.squeeze(1))
