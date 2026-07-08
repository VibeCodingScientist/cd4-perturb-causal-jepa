"""
Snakefile — one-command reproduction of the CD4+ perturbation benchmark
(UNIFIED_BUILD_PLAN.md §6, §9, CP1/CP2).

    snakemake --cores all                 # full CP1: baselines + causal + non-causal
    snakemake --cores all runs_2x2        # CP2 add-ons owned by Developer 2 (jepa cells)

Lane C (CPU) steps run under Snakemake directly. Lane G (GPU) steps are dispatched through
the SERIAL single-GPU queue (`python gpu_queue.py submit <job>`) so two worktrees never train
at once — Snakemake schedules the DAG, gpu_queue serializes the GPU. Large artifacts live in
DATA_ROOT (contract.py), outside the repo; only code + split_manifest.json + hypotheses.md +
results/benchmark_table.csv + figures/ are committed.

The raw dataset (GSE278572, tens of GB) is NOT downloaded automatically — that is a flagged
step (see rule `data`). Point RAW_H5AD at the downloaded file, or set it in config.
"""
import os
from pathlib import Path

# import the frozen contract for canonical paths
import sys
sys.path.insert(0, os.path.dirname(workflow.snakefile))
from core import contract as C

# CP1 data source. Default = the CZI pre-computed pseudobulk (44.6 GB), which carries the
# condition/donor/guide obs and is all CP1 (baselines + causal) needs. Set data_source=cells
# to instead build pseudobulk by streaming the ~1.7 TB of single cells (the JEPA lane's input).
DATA_SOURCE = config.get("data_source", "czi_pseudobulk")
RAW_H5AD = config.get("raw_h5ad", str(C.RAW_DIR / "GWCD4i.pseudobulk_merged.h5ad"))

CP1_MODELS = list(C.CP1_MODELS)           # ridge, tabpfn, fcn, causal, noncausal
SPLITS = list(C.SPLITS)                    # gene, condition

def run_files(models):
    return [str(C.run_path(m, s)) for m in models for s in SPLITS]


# ---------------------------------------------------------------------------
rule all:
    """CP1: the full benchmark table over the baselines + causal + non-causal."""
    input:
        str(C.BENCHMARK_TABLE),
        str(C.BENCHMARK_TABLE_FULL),


rule data:
    """Freeze the split + build pseudobulk + DEG-frequency (Lane C).

    FLAGGED: the download is a manual, out-of-band step (44.6 GB CZI pseudobulk, or the
    ~1.7 TB of cells). Default `czi_pseudobulk` runs core.data.build_from_czi_pseudobulk
    (normalize per-guide profiles -> average guides -> per (gene,condition,donor) pseudobulk,
    HVG, freeze split against the file SHA256). `cells` runs core.data.prepare_core to stream
    the backed cell object instead.
    """
    input:
        h5ad=RAW_H5AD,
    output:
        manifest=str(C.SPLIT_MANIFEST),
        hvg=str(C.HVG_LIST_PATH),
        train=str(C.PSEUDOBULK_TRAIN),
        test=str(C.PSEUDOBULK_TEST),
        deg=str(C.DEG_FREQ_CACHE),
    run:
        from core import data
        if DATA_SOURCE == "cells":
            data.prepare_core(input.h5ad)
        else:
            data.build_from_czi_pseudobulk(input.h5ad)


rule esm2:
    """G1: ESM-2 650M gene-token embeddings via the serial GPU queue."""
    input:
        manifest=str(C.SPLIT_MANIFEST),
    output:
        str(C.ESM2_CACHE),
    shell:
        "python gpu_queue.py submit esm2"


rule context_prior:
    """Regulatory/context prior (node2vec over STRING/GRN); CPU."""
    input:
        manifest=str(C.SPLIT_MANIFEST),
    output:
        str(C.CONTEXT_PRIOR_CACHE),
    run:
        raise WorkflowError(
            "context_prior needs a downloaded STRING/GRN edge list; run "
            "core.features.build_context_prior_node2vec and cache it, then re-run."
        )


FEATURES = [str(C.ESM2_CACHE), str(C.CONTEXT_PRIOR_CACHE), str(C.DEG_FREQ_CACHE)]


rule ridge:
    input: FEATURES, train=str(C.PSEUDOBULK_TRAIN)
    output: run_files([C.MODEL_RIDGE])
    run:
        from core.models import baselines
        baselines.run_ridge(SPLITS)


rule tabpfn:
    input: FEATURES, train=str(C.PSEUDOBULK_TRAIN)
    output: run_files([C.MODEL_TABPFN])
    run:
        from core.models import baselines
        baselines.run_tabpfn(SPLITS)


rule fcn:
    input: FEATURES, train=str(C.PSEUDOBULK_TRAIN)
    output: run_files([C.MODEL_FCN])
    run:
        from core.models import baselines
        baselines.run_fcn(SPLITS)


rule causal:
    """G2: CausalCisTransFormer (corrected do-mask) via the serial GPU queue."""
    input: FEATURES, train=str(C.PSEUDOBULK_TRAIN)
    output: run_files([C.MODEL_CAUSAL])
    shell: "python gpu_queue.py submit causal"


rule noncausal:
    """G3: the non-causal twin via the serial GPU queue."""
    input: FEATURES, train=str(C.PSEUDOBULK_TRAIN)
    output: run_files([C.MODEL_NONCAUSAL])
    shell: "python gpu_queue.py submit noncausal"


rule benchmark_table:
    """Score every CP1 run through the frozen eval harness into the benchmark tables."""
    input:
        run_files(CP1_MODELS),
    output:
        str(C.BENCHMARK_TABLE),
        str(C.BENCHMARK_TABLE_FULL),
    run:
        from core import eval as ev
        for m in CP1_MODELS:
            for s in SPLITS:
                ev.score_run_file(m, s)


rule runs_2x2:
    """CP2 hook (Developer 2): the JEPA cells of the 2x2 (jepa_only, jepa_causal)."""
    input:
        run_files([C.MODEL_JEPA_ONLY, C.MODEL_JEPA_CAUSAL]),
