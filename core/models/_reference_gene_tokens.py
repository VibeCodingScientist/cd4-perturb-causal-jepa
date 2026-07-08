"""core.models._reference_gene_tokens — a faithful reference ``GeneTokenEncoder``.

STAND-IN, NOT the production encoder. Developer 1 owns ``core/models/gene_tokens.py``
(§7d). This module lets Developer 2 build and unit-test JEPA pretraining
(``core.models.jepa``) and the JEPA->causal weight transfer *before* that file
exists. At core-frozen, ``core.models.jepa`` imports the real class if present and
falls back to this one otherwise (see ``jepa._default_encoder_cls``).

It is deliberately a REAL per-gene transformer, never a pseudobulk MLP (§1 / §7e
forbid the latter):

    gene token   Z_i = phi_fuse(gene_embed[id_i]) + phi_expr(value_i)      (§7d)
    proxy round  K=32 learned regulatory-proxy tokens mediate gene<->gene
                 over 2 rounds (g2p then p2g), gated by a per-gene regulator score
    gene block   N mask-aware self-attention layers  <-- the causal model's
                 DoAttention do-mask (§7d) is injected HERE, on the SAME weights
    pool         a learned CLS query attends over gene tokens -> cell embedding
    value_head   Linear(d_model -> 1) reconstructs masked expression (JEPA recon)

Because pooling is a separate attention head (not a prepended CLS inside the masked
self-attention), the ``attn_mask`` interface stays a clean gene-only [L, L] bias —
exactly the shape ``DoAttention.build_mask`` (§7d) produces.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from .encoder_api import MASK_VALUE_SENTINEL, EncoderOutput


def _safe_masked_softmax(scores: Tensor, dim: int = -1) -> Tensor:
    """Softmax that returns 0 (not NaN) for rows that are entirely -inf.

    A do-masked query row always keeps its self entry (§7d), and every cell has >=1
    real gene, so full -inf rows should not occur in practice — but padding plus an
    aggressive mask could produce one, and a single NaN silently poisons the whole
    batch. This guard makes the reference robust for adversarial unit tests.
    """
    all_masked = torch.isneginf(scores).all(dim=dim, keepdim=True)
    # Replace -inf-only rows with zeros so softmax is finite; then zero them out.
    safe = torch.where(all_masked, torch.zeros_like(scores), scores)
    out = torch.softmax(safe, dim=dim)
    out = torch.where(all_masked, torch.zeros_like(out), out)
    return out


class _MaskAwareMHA(nn.Module):
    """Multi-head attention accepting an *additive* attn_mask and a key-padding mask.

    attn_mask : None | [L, L] | [B, L, L]  additive bias added to the score matrix
                (do-mask uses 0 / -inf). Broadcast over heads.
    key_padding_mask : None | [B, Lk] bool, True == pad (its columns are killed).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: Tensor) -> Tensor:  # [B, L, d] -> [B, H, L, dh]
        b, l, _ = x.shape
        return x.view(b, l, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self._split(self.q_proj(query))  # [B, H, Lq, dh]
        k = self._split(self.k_proj(key))    # [B, H, Lk, dh]
        v = self._split(self.v_proj(value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            if attn_mask.dim() == 2:            # [Lq, Lk] -> [1, 1, Lq, Lk]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:          # [B, Lq, Lk] -> [B, 1, Lq, Lk]
                attn_mask = attn_mask.unsqueeze(1)
            scores = scores + attn_mask
        if key_padding_mask is not None:        # [B, Lk] -> kill pad columns
            pad = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, Lk]
            scores = scores.masked_fill(pad, float("-inf"))
        attn = _safe_masked_softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                      # [B, H, Lq, dh]
        b, h, lq, dh = out.shape
        out = out.transpose(1, 2).contiguous().view(b, lq, h * dh)
        return self.out_proj(out)


class _Block(nn.Module):
    """Pre-norm transformer block over gene tokens; forwards the do-mask straight
    into attention so pretrained and causal-finetuned encoders share weights."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _MaskAwareMHA(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ReferenceGeneTokenEncoder(nn.Module):
    """Reference implementation of the §7d gene-token encoder / §7e JEPA student.

    Parameters mirror what the production ``GeneTokenEncoder`` is expected to expose,
    so weight transfer (``load_state_dict``) works once the real class lands with the
    same architecture. All submodule *names* here are part of the transfer contract;
    keep them stable.
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_proxy: int = 32,
        n_proxy_rounds: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model
        self.n_proxy = n_proxy
        self.n_proxy_rounds = n_proxy_rounds

        # --- gene token construction (§7d) -------------------------------------
        # gene_embed: learned per-gene vector. In production it is *initialized*
        # from the ESM-2 (coding) ⊕ context (regulatory) priors; here it is learned
        # from scratch. phi_fuse mixes it; phi_expr embeds the scalar value.
        self.gene_embed = nn.Embedding(n_genes, d_model)
        self.phi_fuse = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.phi_expr = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # --- regulatory proxy tokens + mediation (§7d) -------------------------
        self.proxy = nn.Parameter(torch.zeros(n_proxy, d_model))
        nn.init.normal_(self.proxy, std=0.02)
        self.g2p = _MaskAwareMHA(d_model, n_heads, dropout)   # proxy <- genes
        self.p2g = _MaskAwareMHA(d_model, n_heads, dropout)   # genes <- proxy
        self.regulator_gate = nn.Linear(d_model, 1)           # per-gene "regulator score"
        self.proxy_norm = nn.LayerNorm(d_model)
        self.gene_norm = nn.LayerNorm(d_model)

        # --- masked gene self-attention (DoAttention lives here) ---------------
        self.blocks = nn.ModuleList(
            [_Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # --- pooling: a learned query attends over gene tokens -> cell embed ----
        self.pool_query = nn.Parameter(torch.zeros(1, d_model))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool_attn = _MaskAwareMHA(d_model, n_heads, dropout)
        self.pool_norm = nn.LayerNorm(d_model)

        # --- reconstruction head (JEPA recon term, §7e) ------------------------
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    # ------------------------------------------------------------------ pieces
    def build_tokens(self, gene_ids: Tensor, values: Tensor) -> Tensor:
        """Z_i = phi_fuse(gene_embed[id_i]) + phi_expr(value_i).  [B, L, d_model]

        Masked positions carry value == MASK_VALUE_SENTINEL; phi_expr still runs on
        the sentinel so the model learns a distinct 'masked' expression code.
        """
        g = self.phi_fuse(self.gene_embed(gene_ids))          # [B, L, d]
        e = self.phi_expr(values.unsqueeze(-1))               # [B, L, d]
        return g + e

    def _proxy_mediation(self, genes: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        b = genes.shape[0]
        proxy = self.proxy.unsqueeze(0).expand(b, -1, -1)     # [B, K, d]
        for _ in range(self.n_proxy_rounds):
            # g2p: proxies read from genes (respect padding on the gene/key side)
            proxy = proxy + self.g2p(proxy, genes, genes, key_padding_mask=key_padding_mask)
            proxy = self.proxy_norm(proxy)
            # p2g: genes read from proxies, gated by each gene's regulator score
            gate = torch.sigmoid(self.regulator_gate(genes))  # [B, L, 1]
            genes = genes + gate * self.p2g(genes, proxy, proxy)
            genes = self.gene_norm(genes)
        return genes

    def _pool(self, tokens: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        b = tokens.shape[0]
        q = self.pool_query.unsqueeze(0).expand(b, -1, -1)    # [B, 1, d]
        pooled = self.pool_attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
        return self.pool_norm(pooled.squeeze(1))              # [B, d]

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        gene_ids: Tensor,
        values: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> EncoderOutput:
        x = self.build_tokens(gene_ids, values)
        x = self._proxy_mediation(x, key_padding_mask)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        tokens = self.final_norm(x)                           # [B, L, d]
        pooled = self._pool(tokens, key_padding_mask)         # [B, d]
        return EncoderOutput(tokens=tokens, pooled=pooled)


__all__ = ["ReferenceGeneTokenEncoder"]
