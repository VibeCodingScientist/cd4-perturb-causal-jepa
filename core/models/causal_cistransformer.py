"""
core.models.causal_cistransformer — CausalCisTransFormer + its non-causal twin
(UNIFIED_BUILD_PLAN.md §7d).

The model = GeneTokenEncoder (shared with JEPA) -> a stack of do-masked transformer blocks
-> a delta head + a DE-classification head. The intervention enters two ways:
  1. the perturbed gene's INPUT value is clamped to a knockdown level (CRISPRi != full KO);
  2. the do-mask cuts the perturbed gene's incoming edges (query row) while keeping its key
     column, so the knockdown propagates downstream (the corrected mask, do_attention.py).

The non-causal twin is the SAME architecture with `use_causal_mask=False` — the single most
important ablation (C2). Loss = MSE_full + 0.5*MSE_delta(DE genes) + 0.1*BCE_DE (§7d).

Scope: the perturbed gene must be a token in the sequence, so training/prediction cover
HVG-panel perturbations (core.eval.evaluable_perts) — the same set the baselines are scored
on. Prediction conditions on the split's observed control profile (the baseline state is
observable even in the zero-shot condition), then clamps + propagates the intervention.

Needs torch; authored to run on the L4 via `gpu_queue.py` (BF16 autocast + gradient
checkpointing on CUDA), and unit-tested on CPU with the synthetic core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import contract as C
from .. import eval as ev
from .. import features as feat
from .. import split as split_mod


@dataclass
class CausalConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_proxy: int = 32
    n_proxy_rounds: int = 2
    knockdown_frac: float = 0.9      # CRISPRi input clamp: value -> control*(1-kd)
    de_threshold: float = 0.25       # |delta| above this = a DE gene (BCE_DE target)
    gene_window: int = 1000          # HVG per forward; >= n_genes -> full sequence
    batch_size: int = 32
    epochs: int = 60
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.05
    seed: int = C.SPLIT_SEED
    bf16: bool = True                # autocast on CUDA
    grad_checkpoint: bool = True
    # CP2-v2 (Developer 2, challenger): perturbation-discrimination InfoNCE term.
    # DEFAULT 0.0 -> term disabled -> training is byte-identical to CP2. Only the v2
    # runner (core.models.discrim_v2) sets discrim_weight > 0.
    discrim_weight: float = 0.0      # weight of the in-batch discrimination loss
    discrim_tau: float = 0.1         # InfoNCE temperature


def _build_model(cfg: CausalConfig, use_causal_mask: bool):
    import torch.nn as nn
    from .gene_tokens import GeneTokenEncoder
    from .do_attention import DoTransformerBlock

    class CausalCisTransFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_causal_mask = use_causal_mask
            self.cfg = cfg
            self.encoder = GeneTokenEncoder(
                d_model=cfg.d_model, esm2_dim=C.ESM2_DIM, ctx_dim=C.CONTEXT_PRIOR_DIM,
                n_proxy=cfg.n_proxy, n_proxy_rounds=cfg.n_proxy_rounds, n_heads=cfg.n_heads,
            )
            self.blocks = nn.ModuleList(
                [DoTransformerBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)]
            )
            self.delta_head = nn.Linear(cfg.d_model, 1)
            self.de_head = nn.Linear(cfg.d_model, 1)

        def forward(self, values, esm2, ctx, perturbed_idx):
            import torch
            from torch.utils.checkpoint import checkpoint
            from .do_attention import build_do_mask
            enc = self.encoder(values, esm2, ctx)     # tokens (B, L, d)
            Z = enc.tokens
            L = Z.shape[1]
            do_mask = build_do_mask(L, perturbed_idx) if self.use_causal_mask else None
            for blk in self.blocks:
                if self.cfg.grad_checkpoint and self.training and Z.requires_grad:
                    Z = checkpoint(blk, Z, do_mask, use_reentrant=False)
                else:
                    Z = blk(Z, do_mask=do_mask)
            delta = self.delta_head(Z).squeeze(-1)    # (B, L)
            de_logit = self.de_head(Z).squeeze(-1)    # (B, L)
            return delta, de_logit

    return CausalCisTransFormer()


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
@dataclass
class _Samples:
    values: np.ndarray      # (N, G) control profile with perturbed gene clamped
    target: np.ndarray      # (N, G) true delta
    is_de: np.ndarray       # (N, G) bool DE mask
    pidx: np.ndarray        # (N,) perturbed gene index in the HVG order
    genes: List[str]        # HVG order (length G)


def _build_samples(train_pb: pd.DataFrame, hvg: List[str], cfg: CausalConfig) -> _Samples:
    expr = C.pseudobulk_expr(train_pb)[hvg]
    delta = C.pseudobulk_delta(train_pb)[hvg]
    controls = expr.xs(C.CONTROL_PERT_ID, level="pert_id")  # (cond, donor) x genes
    gpos = {g: i for i, g in enumerate(hvg)}

    V, T, D, P = [], [], [], []
    for (pert, cond, donor) in delta.index:
        if pert == C.CONTROL_PERT_ID or pert not in gpos:
            continue
        if (cond, donor) not in controls.index:
            continue
        ctrl = controls.loc[(cond, donor)].to_numpy(dtype=float)
        pidx = gpos[pert]
        vals = ctrl.copy()
        vals[pidx] = ctrl[pidx] * (1.0 - cfg.knockdown_frac)  # clamp intervention input
        tgt = delta.loc[(pert, cond, donor)].to_numpy(dtype=float)
        V.append(vals); T.append(tgt); D.append(np.abs(tgt) > cfg.de_threshold); P.append(pidx)
    if not V:
        raise ValueError("no HVG-panel training perturbations found")
    return _Samples(np.vstack(V), np.vstack(T), np.vstack(D), np.asarray(P, dtype=int), hvg)


def _control_for_split(split: str, hvg: List[str]) -> np.ndarray:
    """Mean control profile to condition prediction on (the observable baseline state)."""
    if split == C.SPLIT_CONDITION:
        test = pd.read_parquet(C.PSEUDOBULK_TEST)
        expr = C.pseudobulk_expr(test)[hvg]
        idx = expr.index
        mask = (idx.get_level_values("pert_id") == C.CONTROL_PERT_ID) & \
               (idx.get_level_values("condition") == C.CONDITION_HOLDOUT)
        sub = expr[mask]
        if len(sub):
            return sub.mean(axis=0).to_numpy(dtype=float)
    if split == C.SPLIT_DONOR:
        test = pd.read_parquet(C.PSEUDOBULK_TEST)
        expr = C.pseudobulk_expr(test)[hvg]
        idx = expr.index
        mask = (idx.get_level_values("pert_id") == C.CONTROL_PERT_ID) & \
               (idx.get_level_values("donor") == C.DONOR_PROBE)
        sub = expr[mask]
        if len(sub):
            return sub.mean(axis=0).to_numpy(dtype=float)
    # gene hold-out (train conditions) or fallback: train control mean
    train = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    expr = C.pseudobulk_expr(train)[hvg]
    ctrl = expr.xs(C.CONTROL_PERT_ID, level="pert_id")
    return ctrl.mean(axis=0).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------
def _feature_tensors(hvg: List[str], device):
    import torch
    esm2 = feat.load_esm2().reindex(hvg).fillna(0.0).to_numpy()
    ctx = feat.load_context_prior().reindex(hvg).fillna(0.0).to_numpy()
    return (torch.tensor(esm2, dtype=torch.float32, device=device),
            torch.tensor(ctx, dtype=torch.float32, device=device))


def _train(model, s: _Samples, esm2_t, ctx_t, cfg: CausalConfig, device):
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(cfg.seed)
    G = len(s.genes)
    window = min(cfg.gene_window, G)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    N = len(s.values)
    steps_per_epoch = max(1, N // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min((t + 1) / warmup, 0.5 * (1 + np.cos(np.pi * (t - warmup) /
                                                              max(1, total_steps - warmup)))
                           if t >= warmup else (t + 1) / warmup))
    use_amp = cfg.bf16 and device.type == "cuda"
    model.train()
    step = 0
    for _ in range(cfg.epochs):
        order = rng.permutation(N)
        for b in range(steps_per_epoch):
            bi = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            if len(bi) == 0:
                continue
            # gene window that must contain every perturbed gene in the batch
            if window >= G:
                cols = np.arange(G)
            else:
                must = np.unique(s.pidx[bi])
                extra = rng.choice(np.setdiff1d(np.arange(G), must),
                                   size=max(0, window - len(must)), replace=False)
                cols = np.sort(np.concatenate([must, extra]))
            colpos = {c: i for i, c in enumerate(cols)}
            vals = torch.tensor(s.values[np.ix_(bi, cols)], dtype=torch.float32, device=device)
            tgt = torch.tensor(s.target[np.ix_(bi, cols)], dtype=torch.float32, device=device)
            de = torch.tensor(s.is_de[np.ix_(bi, cols)], dtype=torch.float32, device=device)
            pidx = torch.tensor([colpos[p] for p in s.pidx[bi]], dtype=torch.long, device=device)
            e_sub, c_sub = esm2_t[cols], ctx_t[cols]

            opt.zero_grad()
            ctx_mgr = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp \
                else _nullctx()
            with ctx_mgr:
                dhat, de_logit = model(vals, e_sub, c_sub, pidx)
                mse_full = nn.functional.mse_loss(dhat, tgt)
                de_mask = de > 0.5
                mse_delta = (nn.functional.mse_loss(dhat[de_mask], tgt[de_mask])
                             if de_mask.any() else dhat.new_zeros(()))
                bce = nn.functional.binary_cross_entropy_with_logits(de_logit, de)
                loss = mse_full + 0.5 * mse_delta + 0.1 * bce
                if cfg.discrim_weight > 0 and len(pidx) > 2:
                    # Perturbation-discrimination InfoNCE (CP2-v2). Center by the batch
                    # mean to target the perturbation-SPECIFIC residual (the exact axis
                    # the mode-collapse diagnostic showed was failing), then cosine
                    # in-batch contrastive: each predicted residual must match its own
                    # true residual and not another perturbation's. Same-gene rows are
                    # masked out of the negatives so they are never treated as wrong.
                    zc = nn.functional.normalize(dhat - dhat.mean(0, keepdim=True), dim=1)
                    tc = nn.functional.normalize(tgt - tgt.mean(0, keepdim=True), dim=1)
                    sim = (zc @ tc.T) / cfg.discrim_tau
                    same = pidx[:, None] == pidx[None, :]
                    same.fill_diagonal_(False)
                    sim = sim.masked_fill(same, -1e4)
                    labels = torch.arange(len(pidx), device=device)
                    loss = loss + cfg.discrim_weight * nn.functional.cross_entropy(sim, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step(); step += 1
    return model


def _predict_split(model, split: str, hvg: List[str], esm2_t, ctx_t, cfg: CausalConfig, device):
    import torch
    perts = ev.evaluable_perts(split)
    ctrl = _control_for_split(split, hvg)
    gpos = {g: i for i, g in enumerate(hvg)}
    G = len(hvg)
    model.eval()
    rows, kept = [], []
    with torch.no_grad():
        for i in range(0, len(perts), cfg.batch_size):
            chunk = [p for p in perts[i:i + cfg.batch_size] if p in gpos]
            if not chunk:
                continue
            vals = np.tile(ctrl, (len(chunk), 1))
            pcols = []
            for j, p in enumerate(chunk):
                pi = gpos[p]
                vals[j, pi] = ctrl[pi] * (1.0 - cfg.knockdown_frac)
                pcols.append(pi)
            vt = torch.tensor(vals, dtype=torch.float32, device=device)
            pidx = torch.tensor(pcols, dtype=torch.long, device=device)
            dhat, _ = model(vt, esm2_t, ctx_t, pidx)
            rows.append(dhat.float().cpu().numpy())
            kept.extend(chunk)
    pred = pd.DataFrame(np.vstack(rows), index=kept, columns=hvg)
    pred.index.name = "pert_id"
    return pred


def _run(model_name: str, use_causal_mask: bool, splits: Sequence[str],
         cfg: Optional[CausalConfig] = None, record: bool = True):
    import torch
    cfg = cfg or CausalConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hvg = split_mod.load_hvg()
    train_pb = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    # keep HVG order consistent with the pseudobulk columns
    hvg = [g for g in hvg if g in C.pseudobulk_expr(train_pb).columns]

    samples = _build_samples(train_pb, hvg, cfg)
    esm2_t, ctx_t = _feature_tensors(hvg, device)
    model = _build_model(cfg, use_causal_mask).to(device)
    _train(model, samples, esm2_t, ctx_t, cfg, device)

    C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for split in splits:
        pred = _predict_split(model, split, hvg, esm2_t, ctx_t, cfg, device)
        pred.to_parquet(C.run_path(model_name, split))
        if record:
            ev.evaluate_and_record(pred, split, model_name)
    return model


# §6/§10 fallback config, applied by the epoch-1 gate when a job would overrun its slot.
FALLBACK_CONFIG = CausalConfig(d_model=128, n_layers=2, gene_window=1000)


def epoch1_gate(cfg: CausalConfig, use_causal_mask: bool, slot_hours: float):
    """§6 measure-then-extrapolate gate: train 1 epoch, extrapolate to cfg.epochs, and return
    (fits, cfg_to_use). If the projected wall-time exceeds the slot, return FALLBACK_CONFIG so
    the job cannot silently eat the GPU queue. Invoked by gpu_queue before full training."""
    import time
    from dataclasses import replace
    import torch

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hvg = split_mod.load_hvg()
    train_pb = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    hvg = [g for g in hvg if g in C.pseudobulk_expr(train_pb).columns]
    samples = _build_samples(train_pb, hvg, cfg)
    esm2_t, ctx_t = _feature_tensors(hvg, device)
    model = _build_model(cfg, use_causal_mask).to(device)

    t0 = time.time()
    _train(model, samples, esm2_t, ctx_t, replace(cfg, epochs=1), device)
    projected_s = (time.time() - t0) * cfg.epochs
    fits = projected_s <= slot_hours * 3600
    label = "causal" if use_causal_mask else "noncausal"
    print(f"[epoch-1 gate] {label}: 1 epoch measured -> projected {projected_s/3600:.2f}h "
          f"vs slot {slot_hours:.1f}h ({'OK' if fits else 'OVER BUDGET -> §6 fallback'})",
          flush=True)
    return fits, (cfg if fits else FALLBACK_CONFIG)


def run_causal(splits: Sequence[str] = C.SPLITS, cfg: Optional[CausalConfig] = None,
               record: bool = True):
    """G2: CausalCisTransFormer with the corrected do-mask -> runs/causal_<split>.parquet."""
    return _run(C.MODEL_CAUSAL, True, splits, cfg, record)


def run_noncausal(splits: Sequence[str] = C.SPLITS, cfg: Optional[CausalConfig] = None,
                  record: bool = True):
    """G3: the non-causal twin (mask disabled) -> runs/noncausal_<split>.parquet."""
    return _run(C.MODEL_NONCAUSAL, False, splits, cfg, record)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
