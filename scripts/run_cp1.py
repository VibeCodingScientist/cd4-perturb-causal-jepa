"""
scripts/run_cp1.py — run the full CP1 benchmark once data + priors are staged on the box.

Baselines (CPU) + CausalCisTransFormer + non-causal twin (GPU) on the gene and condition
hold-outs, scored through the frozen eval harness into results/benchmark_table.csv. This is a
solo controlled run, so the transformers are called directly (the gpu_queue serial lock is for
coordinating two worktrees; here nothing else contends for the GPU).

Run on the box:  ./.venv/bin/python scripts/run_cp1.py
"""
import pandas as pd

from core import contract as C
from core import eval as ev
from core.models import baselines as b
from core.models import causal_cistransformer as cc

SPLITS = (C.SPLIT_GENE, C.SPLIT_CONDITION)


def main():
    print("=== baselines ===", flush=True)
    b.run_ridge(SPLITS)
    print("ridge done", flush=True)
    try:
        b.run_fcn(SPLITS)
        print("fcn done", flush=True)
    except Exception as e:
        print(f"fcn skipped: {e}", flush=True)
    try:
        b.run_tabpfn(SPLITS)
        print("tabpfn done", flush=True)
    except Exception as e:
        print(f"tabpfn skipped: {e}", flush=True)

    print("=== causal (corrected do-mask) ===", flush=True)
    cfg = cc.CausalConfig(epochs=40)          # L4; gene_window 1000 of 3000 HVG
    cc.run_causal(SPLITS, cfg=cfg)
    print("causal done", flush=True)

    print("=== non-causal twin ===", flush=True)
    cc.run_noncausal(SPLITS, cfg=cfg)
    print("noncausal done", flush=True)

    print("\n=== results/benchmark_table.csv ===", flush=True)
    bt = pd.read_csv(C.BENCHMARK_TABLE).sort_values(["split", "model"])
    print(bt.to_string(index=False))
    # highlight the condition hold-out (the primary C1/C2 test)
    print("\n--- condition hold-out (primary; higher pearson better, rank<0.4 = not collapsed) ---")
    cond = bt[bt["split"] == "condition"][
        ["model", C.METRIC_PEARSON_DELTA, C.METRIC_PERTURBENCH_RANK, C.METRIC_DES, C.MODE_COLLAPSE_FLAG]
    ]
    print(cond.to_string(index=False))


if __name__ == "__main__":
    main()
