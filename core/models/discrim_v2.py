"""core.models.discrim_v2 — CP2-v2 challenger (Developer 2).

Retrains the four 2x2 cells with a perturbation-discrimination InfoNCE term
(core.models.causal_cistransformer with `discrim_weight > 0`), tunes that weight on a
DISJOINT dev split (a random 15% of TRAIN perturbations — never the reported hold-outs),
and writes namespaced `runs/*_v2.parquet` + `results/benchmark_table_v2.csv`. It NEVER
touches CP2: same trainer, same frozen split, same JEPA checkpoint, same epochs=40 — only
the loss gains the (default-off) discrimination term.

Accept/reject (§4) is applied by the caller after reading benchmark_table_v2.csv.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from core import contract as C
from core import eval as ev
from core.models import causal_cistransformer as cc
from core.models.jepa_integration import load_jepa_into_encoder

CELLS = [
    (C.MODEL_NONCAUSAL, False),
    (C.MODEL_CAUSAL, True),
    (C.MODEL_JEPA_ONLY, False),
    (C.MODEL_JEPA_CAUSAL, True),
]
BENCH_V2 = C.RESULTS_DIR / "benchmark_table_v2.csv"


def v2_run_path(model: str, split: str) -> Path:
    return C.RUNS_DIR / f"{model}_{split}_v2.parquet"


def _subset(s, mask):
    from core.models.causal_cistransformer import _Samples
    return _Samples(s.values[mask], s.target[mask], s.is_de[mask], s.pidx[mask], s.genes)


def _residual_rank(P, T):
    return ev._perturbench_rank(P - P.mean(0, keepdims=True), T - T.mean(0, keepdims=True))


def _dev_eval(model, s, dev_rows, esm2_t, ctx_t, cfg, device):
    """Predict dev perturbations from their clamped-control sample rows (full HVG panel,
    exactly as _predict_split does), aggregate per perturbation, and return
    (raw_rank, residual_rank, pearson_delta_top50)."""
    import torch

    model.eval()
    idx = np.flatnonzero(dev_rows)
    preds = np.empty((len(idx), len(s.genes)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), cfg.batch_size):
            bi = idx[i:i + cfg.batch_size]
            vals = torch.tensor(s.values[bi], dtype=torch.float32, device=device)
            pidx = torch.tensor(s.pidx[bi], dtype=torch.long, device=device)
            dhat, _ = model(vals, esm2_t, ctx_t, pidx)
            preds[i:i + len(bi)] = dhat.float().cpu().numpy()
    dev_pidx = s.pidx[idx]
    tgt = s.target[idx]
    perts = np.unique(dev_pidx)
    P = np.vstack([preds[dev_pidx == p].mean(0) for p in perts])
    T = np.vstack([tgt[dev_pidx == p].mean(0) for p in perts])
    return ev._perturbench_rank(P, T), _residual_rank(P, T), ev._pearson_delta_topk(P, T, C.TOP_DEG_N)


def tune_weight(weights, samples, esm2_t, ctx_t, base_cfg, device, *,
                dev_frac=0.15, tune_epochs=20, seed=42, tol=0.90, log=print):
    """Train the CAUSAL cell (the headline) at each weight on train-minus-dev, evaluate
    discrimination + accuracy on the held-out dev perturbations, and pick the weight with
    the lowest dev residual-rank whose dev Pearson-δ stays >= tol × the w=0 baseline."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(samples.pidx)
    n_dev = max(2, int(dev_frac * len(uniq)))
    dev_genes = rng.choice(uniq, size=n_dev, replace=False)
    dev_rows = np.isin(samples.pidx, dev_genes)
    train_s = _subset(samples, ~dev_rows)
    log(f"[tune] dev split: {n_dev}/{len(uniq)} genes held out ({int(dev_rows.sum())} rows); "
        f"train on {int((~dev_rows).sum())} rows")

    results = []
    for w in weights:
        cfg = replace(base_cfg, epochs=tune_epochs, discrim_weight=w)
        import torch
        torch.manual_seed(cfg.seed)
        model = cc._build_model(cfg, use_causal_mask=True).to(device)
        cc._train(model, train_s, esm2_t, ctx_t, cfg, device)
        raw, res, pear = _dev_eval(model, samples, dev_rows, esm2_t, ctx_t, cfg, device)
        results.append({"w": w, "dev_raw_rank": raw, "dev_resid_rank": res, "dev_pearson": pear})
        log(f"[tune] w={w:<4}: dev raw_rank={raw:.4f}  resid_rank={res:.4f}  pearson_delta={pear:.4f}")

    base = next((r for r in results if r["w"] == 0), None)
    base_pear = base["dev_pearson"] if base else max(r["dev_pearson"] for r in results)
    cand = [r for r in results if r["w"] > 0 and r["dev_pearson"] >= tol * base_pear]
    if cand:
        best = min(cand, key=lambda r: r["dev_resid_rank"])["w"]
    else:
        best = min((r for r in results if r["w"] > 0), key=lambda r: r["dev_resid_rank"])["w"]
        log("[tune] no weight kept Pearson within tolerance; picking best-rank anyway (verify §4)")
    log(f"[tune] chosen discrim_weight = {best}")
    return best, results


def _bench_row(model_name, split, metrics):
    row = {"model": model_name, "split": split}
    for k in C.METRICS_HEADLINE:
        row[k] = metrics.get(k)
    rank = metrics.get(C.METRIC_PERTURBENCH_RANK)
    row[C.MODE_COLLAPSE_FLAG] = bool(rank is not None and rank > C.MODE_COLLAPSE_THRESHOLD)
    return row


def run_v2(discrim_weight, jepa_ckpt, splits, samples, esm2_t, ctx_t, hvg, base_cfg, device, log=print):
    """Train all four cells at ``discrim_weight`` (epochs=40) and write *_v2 runs + the v2
    benchmark table. jepa cells load the JEPA checkpoint into their encoder first."""
    import torch

    cfg = replace(base_cfg, epochs=40, discrim_weight=discrim_weight)
    rows = []
    for model_name, use_mask in CELLS:
        torch.manual_seed(cfg.seed)
        model = cc._build_model(cfg, use_mask).to(device)
        if model_name in (C.MODEL_JEPA_ONLY, C.MODEL_JEPA_CAUSAL):
            rep = load_jepa_into_encoder(model.encoder, jepa_ckpt)
            if not rep.n_loaded:
                raise RuntimeError(f"JEPA init loaded 0 params into {model_name}")
        log(f"[v2] training {model_name} (mask={use_mask}, w={discrim_weight}) ...")
        cc._train(model, samples, esm2_t, ctx_t, cfg, device)
        for split in splits:
            pred = cc._predict_split(model, split, hvg, esm2_t, ctx_t, cfg, device)
            pred.to_parquet(v2_run_path(model_name, split))
            rows.append(_bench_row(model_name, split, ev.evaluate(pred, split)))
            log(f"[v2] {model_name}/{split}: "
                f"pearson_delta={rows[-1][C.METRIC_PEARSON_DELTA]:.4f} rank={rows[-1][C.METRIC_PERTURBENCH_RANK]:.4f}")
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(C.BENCHMARK_COLUMNS))
    df.to_csv(BENCH_V2, index=False)
    log(f"[v2] wrote {BENCH_V2}")
    return df


def main(weights=(0.0, 0.15, 0.3), tune_epochs=20):
    import torch

    from core import features as feat, split as split_mod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v2] device={device}")
    hvg = split_mod.load_hvg()
    train_pb = pd.read_parquet(C.PSEUDOBULK_TRAIN)
    hvg = [g for g in hvg if g in C.pseudobulk_expr(train_pb).columns]
    base_cfg = cc.CausalConfig(epochs=40)   # match CP2 exactly (only the loss differs)
    samples = cc._build_samples(train_pb, hvg, base_cfg)
    esm2_t, ctx_t = cc._feature_tensors(hvg, device)
    jepa_ckpt = str(C.checkpoint_path("jepa"))

    # serial-GPU discipline (honor the queue's lock even though this runs standalone)
    try:
        import gpu_queue
        got = gpu_queue.acquire_lock(timeout=120)
    except Exception:
        got = True
    try:
        best_w, tune = tune_weight(list(weights), samples, esm2_t, ctx_t, base_cfg, device,
                                   tune_epochs=tune_epochs)
        print("[v2] tune results:", tune)
        df = run_v2(best_w, jepa_ckpt, list(C.SPLITS), samples, esm2_t, ctx_t, hvg, base_cfg, device)
        print("[v2] benchmark_table_v2:\n" + df.to_string(index=False))
        print(f"[v2] chosen_weight={best_w}")
    finally:
        try:
            if got:
                import gpu_queue
                gpu_queue.release_lock()
        except Exception:
            pass


if __name__ == "__main__":
    main()
