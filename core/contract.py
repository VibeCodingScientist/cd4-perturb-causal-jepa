"""
core.contract — the single frozen interface both worktrees code against.

UNIFIED_BUILD_PLAN.md §4 (layout) + §5 (data contract). This module is committed
FIRST, before any implementation, so a second worktree (JEPA + analysis) can code
and unit-test against these paths, schemas, and signatures before the
implementations exist.

Design rules for this file:
  * Top-level imports are STDLIB ONLY, so `import core.contract` never fails in a
    minimal environment (the analysis worktree imports it constantly).
  * Everything shared is a named constant or a small pure helper here. If two
    modules need to agree on a path, a dimension, a column name, or a metric name,
    it is defined HERE and imported — never re-declared.
  * Nothing here reads data, touches the GPU, or has side effects at import time
    (except computing paths). Directory creation is explicit via `ensure_dirs()`.

Canonical gene identifier
--------------------------
One id space is used everywhere (pert_id, HVG columns, gene_holdout, feature-cache
index): **Ensembl gene IDs (ENSG...)**. `core.data` maps the raw AnnData to this
space once; every downstream module then joins cleanly. Non-targeting controls use
the sentinel `CONTROL_PERT_ID`.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 0. Versioning
# ---------------------------------------------------------------------------
CONTRACT_VERSION = "1.0.0"
SPLIT_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# 1. Roots
# ---------------------------------------------------------------------------
# The git repo root (this file lives at <repo>/core/contract.py).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Large/shared artifacts live OUTSIDE git so both worktrees and the eval harness
# see them instantly. Override with env var CD4_DATA_ROOT (e.g. on the GPU box).
DATA_ROOT: Path = Path(
    os.environ.get("CD4_DATA_ROOT", str(Path.home() / "cd4-perturb-data"))
).expanduser()

# --- Artifact directories (outside git; see .gitignore) --------------------
RAW_DIR: Path = DATA_ROOT / "raw"                 # downloaded .h5ad lives here
EMBEDDINGS_DIR: Path = DATA_ROOT / "embeddings"   # gene -> vector caches
PSEUDOBULK_DIR: Path = DATA_ROOT / "pseudobulk"   # (pert,cond,donor) profiles + deltas
FEATURES_DIR: Path = DATA_ROOT / "features"       # DEG-frequency etc.
CELLS_DIR: Path = DATA_ROOT / "cells"             # JEPA single-cell subsample
RUNS_DIR: Path = DATA_ROOT / "runs"               # <model>_<split>.parquet predictions
CHECKPOINTS_DIR: Path = DATA_ROOT / "checkpoints" # model weights
GPU_LOCK: Path = DATA_ROOT / "GPU_LOCK"           # single-GPU exclusive lock (gpu_queue.py)

# --- Committed-to-git paths (code + small reproducibility artifacts) -------
SPLIT_MANIFEST: Path = REPO_ROOT / "split_manifest.json"
HVG_LIST_PATH: Path = REPO_ROOT / "split" / "hvg_3000.txt"  # committed with the split
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = REPO_ROOT / "figures"
HYPOTHESES: Path = REPO_ROOT / "hypotheses.md"

BENCHMARK_TABLE: Path = RESULTS_DIR / "benchmark_table.csv"        # headline metrics (demo)
BENCHMARK_TABLE_FULL: Path = RESULTS_DIR / "benchmark_table_full.csv"  # 8-metric appendix

# --- Named artifact files --------------------------------------------------
ESM2_CACHE: Path = EMBEDDINGS_DIR / "esm2.parquet"            # gene -> ESM-2 650M vector
CONTEXT_PRIOR_CACHE: Path = EMBEDDINGS_DIR / "context_prior.parquet"  # gene -> network/GO vector
PSEUDOBULK_TRAIN: Path = PSEUDOBULK_DIR / "train.parquet"
PSEUDOBULK_TEST: Path = PSEUDOBULK_DIR / "test.parquet"
DEG_FREQ_CACHE: Path = FEATURES_DIR / "deg_freq.parquet"     # 50-dim BioMap feature


def ensure_dirs() -> None:
    """Create every DATA_ROOT artifact directory + committed results/figures dirs.

    Idempotent. Call once at the start of any job that writes artifacts. Deliberately
    NOT run at import time so that merely importing the contract has no side effects.
    """
    for d in (
        DATA_ROOT, RAW_DIR, EMBEDDINGS_DIR, PSEUDOBULK_DIR, FEATURES_DIR,
        CELLS_DIR, RUNS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, FIGURES_DIR,
        HVG_LIST_PATH.parent,
    ):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Biology namespace (dataset structure — GSE278572)
# ---------------------------------------------------------------------------
GEO_ACCESSION = "GSE278572"

CONDITIONS: tuple[str, ...] = ("Rest", "Stim8hr", "Stim48hr")
CONDITION_HOLDOUT = "Stim48hr"                       # §3 primary zero-shot test
TRAIN_CONDITIONS: tuple[str, ...] = ("Rest", "Stim8hr")

DONORS: tuple[str, ...] = ("donor_1", "donor_2", "donor_3", "donor_4")
DONOR_PROBE = "donor_4"                               # §3 cross-donor sanity check

CONTROL_PERT_ID = "non-targeting"                    # sentinel pert_id for NTC guides

# Canonical gene-identifier convention (see module docstring).
GENE_ID = "ensembl"                                  # pert_id / HVG columns / gene_holdout

# ---------------------------------------------------------------------------
# 3. Fixed dimensions (every module honors these; features.py truncates/PCAs to them)
# ---------------------------------------------------------------------------
HVG_N = 3000            # §7a highly-variable genes
TOP_DEG_N = 50          # §7i headline metric is over the top-50 DEGs
DEG_FREQ_DIM = 50       # §7a BioMap DEG-frequency feature dimensionality
ESM2_DIM = 1280         # ESM-2 650M mean-pooled embedding width
CONTEXT_PRIOR_DIM = 512 # node2vec / GenePT-PCA regulatory-context vector width
TABPFN_MAX_FEATURES = 200   # §7c TabPFN-3 ceiling (use 500 only if pinning v2)
TABPFN_MAX_ROWS = 10_000    # §7c conservative row ceiling (v2); v3 allows up to 1M

# ---------------------------------------------------------------------------
# 4. Split namespace (§3)
# ---------------------------------------------------------------------------
SPLIT_GENE = "gene"            # 15% of perturbed genes withheld (interpolation, secondary)
SPLIT_CONDITION = "condition"  # Stim48hr held out (zero-shot; PRIMARY test for C1-C3)
SPLIT_DONOR = "donor"          # donor_4 reserved (sanity probe)

# The two splits CP1 requires on the benchmark table (§11). SPLIT_DONOR is optional.
SPLITS: tuple[str, ...] = (SPLIT_GENE, SPLIT_CONDITION)
ALL_SPLITS: tuple[str, ...] = (SPLIT_GENE, SPLIT_CONDITION, SPLIT_DONOR)

GENE_HOLDOUT_FRACTION = 0.15   # §3
SPLIT_SEED = 42                # §3


@dataclass
class SplitManifest:
    """Schema for split_manifest.json (§3). Immutable once frozen + committed.

    `sha256_h5ad` binds the split to a specific dataset file; every module calls
    `core.split.verify()` at startup, which recomputes the hash and asserts equality.

    Two-phase freeze. The split *policy* (seed, held-out condition, donor probe, HVG
    count, gene-holdout fraction) is immutable from Day 0 and committed now. The
    data-*derived* fields (`gene_holdout` list, `sha256_h5ad`, `hvg` list file,
    `created_utc`) can only be materialized once the .h5ad is downloaded; when
    `core.split.freeze()` runs on the real data it fills them deterministically
    (seed 42) and sets `data_frozen = True`. `gene_holdout == [] and not data_frozen`
    therefore means "policy frozen, awaiting data", NOT an empty hold-out.
    """
    seed: int = SPLIT_SEED
    geo: str = GEO_ACCESSION
    doi: str = ""                                  # bioRxiv DOI, filled at freeze
    sha256_h5ad: str = ""                          # hash of the frozen .h5ad
    hvg_n: int = HVG_N
    hvg_list_path: str = "split/hvg_3000.txt"      # repo-relative, committed
    gene_holdout_fraction: float = GENE_HOLDOUT_FRACTION
    gene_holdout: List[str] = field(default_factory=list)  # ENSG ids (materialized at data freeze)
    condition_holdout: str = CONDITION_HOLDOUT
    donor_probe: str = DONOR_PROBE
    created_utc: str = ""
    data_frozen: bool = False                      # True once gene_holdout + sha are materialized
    schema_version: int = SPLIT_SCHEMA_VERSION
    contract_version: str = CONTRACT_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SplitManifest":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


# ---------------------------------------------------------------------------
# 5. Model namespace (§5) — fixed strings; filenames derive from them so parallel
#    writers never collide.
# ---------------------------------------------------------------------------
MODEL_RIDGE = "ridge"
MODEL_TABPFN = "tabpfn"
MODEL_FCN = "fcn"
MODEL_NONCAUSAL = "noncausal"       # random-init, mask off  (2x2)
MODEL_CAUSAL = "causal"             # random-init, mask on   (2x2, = C2 treatment)
MODEL_JEPA_ONLY = "jepa_only"       # JEPA-init,  mask off   (2x2)
MODEL_JEPA_CAUSAL = "jepa_causal"   # JEPA-init,  mask on    (2x2 main model)
MODEL_ARC_STATE = "arc_state"       # gated external reference

MODELS: tuple[str, ...] = (
    MODEL_RIDGE, MODEL_TABPFN, MODEL_FCN,
    MODEL_NONCAUSAL, MODEL_CAUSAL,
    MODEL_JEPA_ONLY, MODEL_JEPA_CAUSAL,
    MODEL_ARC_STATE,
)

# Developer 1 (this worktree) delivers these for CP1 (§11):
CP1_MODELS: tuple[str, ...] = (
    MODEL_RIDGE, MODEL_TABPFN, MODEL_FCN, MODEL_CAUSAL, MODEL_NONCAUSAL,
)
# The 2x2 experimental core (§2):
GRID_2X2: tuple[str, ...] = (
    MODEL_NONCAUSAL, MODEL_CAUSAL, MODEL_JEPA_ONLY, MODEL_JEPA_CAUSAL,
)


def run_path(model_name: str, split: str) -> Path:
    """Canonical prediction-file path: DATA_ROOT/runs/<model>_<split>.parquet (§5).

    The file's index is `pert_id`, columns are the HVG gene ids, values are the
    predicted DELTA (post - control). No model writes anywhere else.
    """
    if model_name not in MODELS:
        raise ValueError(f"unknown model '{model_name}'; expected one of {MODELS}")
    if split not in ALL_SPLITS:
        raise ValueError(f"unknown split '{split}'; expected one of {ALL_SPLITS}")
    return RUNS_DIR / f"{model_name}_{split}.parquet"


def checkpoint_path(model_name: str, tag: str = "final") -> Path:
    """Canonical checkpoint path: DATA_ROOT/checkpoints/<model>_<tag>.pt."""
    return CHECKPOINTS_DIR / f"{model_name}_{tag}.pt"


# ---------------------------------------------------------------------------
# 6. Pseudobulk parquet schema (§5): index = (pert_id, condition, donor);
#    columns are a 2-level MultiIndex (block, gene) with block in {"expr","delta"}.
#    Use the accessors below rather than reaching into the layout directly.
# ---------------------------------------------------------------------------
PSEUDOBULK_INDEX_NAMES: tuple[str, str, str] = ("pert_id", "condition", "donor")
PSEUDOBULK_BLOCK_EXPR = "expr"     # absolute mean profile
PSEUDOBULK_BLOCK_DELTA = "delta"   # pert - matched control (same condition, donor)
PSEUDOBULK_COL_LEVELS: tuple[str, str] = ("block", "gene")


def pseudobulk_block(df, block: str):
    """Return the `expr` or `delta` sub-frame (columns = genes) of a pseudobulk table."""
    import pandas as pd  # lazy: keep top-level imports stdlib-only
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("pseudobulk frame must have a (block, gene) MultiIndex on columns")
    if block not in (PSEUDOBULK_BLOCK_EXPR, PSEUDOBULK_BLOCK_DELTA):
        raise ValueError(f"block must be one of expr/delta, got {block!r}")
    sub = df.xs(block, axis=1, level=PSEUDOBULK_COL_LEVELS[0])
    return sub


def pseudobulk_expr(df):
    """The absolute mean-expression block (columns = HVG genes)."""
    return pseudobulk_block(df, PSEUDOBULK_BLOCK_EXPR)


def pseudobulk_delta(df):
    """The delta block: pert pseudobulk - matched control (columns = HVG genes)."""
    return pseudobulk_block(df, PSEUDOBULK_BLOCK_DELTA)


def build_pseudobulk_frame(expr_df, delta_df):
    """Assemble the frozen 2-block pseudobulk parquet from two aligned gene frames.

    `expr_df` and `delta_df` share the (pert_id, condition, donor) MultiIndex and the
    same gene columns. Returns a frame with the canonical (block, gene) column
    MultiIndex ready to write to PSEUDOBULK_TRAIN / PSEUDOBULK_TEST.
    """
    import pandas as pd
    expr = expr_df.copy()
    delta = delta_df.copy()
    expr.columns = pd.MultiIndex.from_product(
        [[PSEUDOBULK_BLOCK_EXPR], expr.columns], names=PSEUDOBULK_COL_LEVELS
    )
    delta.columns = pd.MultiIndex.from_product(
        [[PSEUDOBULK_BLOCK_DELTA], delta.columns], names=PSEUDOBULK_COL_LEVELS
    )
    out = pd.concat([expr, delta], axis=1)
    out.index.names = PSEUDOBULK_INDEX_NAMES
    return out


# ---------------------------------------------------------------------------
# 7. Eval interface (§5, §7i) — implemented in core.eval; declared here so all
#    callers agree on metric names and the benchmark-table column order.
#
#    Signature (frozen):
#        core.eval.evaluate(pred_delta_df: pd.DataFrame, split: str) -> dict
#
#    `pred_delta_df`: index = pert_id, columns = HVG gene ids, values = predicted
#        delta. (Exactly the schema written to run_path(...).)
#    `split`: one of ALL_SPLITS; selects the frozen ground-truth test set.
#    returns: dict keyed by the metric names below (+ "model"/"split" filled by the
#        caller before appending a row to the benchmark table).
# ---------------------------------------------------------------------------
# Headline metrics shown in the demo (§7i).
METRIC_PEARSON_DELTA = "pearson_delta_top50"   # accuracy over top-50 DEGs (higher better)
METRIC_PERTURBENCH_RANK = "perturbench_rank"   # mode-collapse detector (LOWER better; >0.4 = red)
METRIC_DES = "des"                             # sign-correct DEG overlap (higher better)
METRICS_HEADLINE: tuple[str, ...] = (
    METRIC_PEARSON_DELTA, METRIC_PERTURBENCH_RANK, METRIC_DES,
)

# Full battery (appendix, benchmark_table_full.csv; never shown in demo).
METRIC_MAE = "mae"
METRIC_SPEARMAN_LFC = "spearman_lfc"
METRIC_SPEARMAN_EFFECT = "spearman_effect"
METRIC_AUPRC = "auprc"
METRIC_EDISTANCE = "edistance"
METRICS_FULL: tuple[str, ...] = METRICS_HEADLINE + (
    METRIC_MAE, METRIC_SPEARMAN_LFC, METRIC_SPEARMAN_EFFECT,
    METRIC_AUPRC, METRIC_EDISTANCE,
)

# Metrics where a LOWER value is better (for correct highlighting / sorting).
METRICS_LOWER_IS_BETTER: frozenset[str] = frozenset({
    METRIC_PERTURBENCH_RANK, METRIC_MAE, METRIC_EDISTANCE,
})

# Mode-collapse detector (§7i): perturbench_rank above this is flagged red.
MODE_COLLAPSE_THRESHOLD = 0.4
MODE_COLLAPSE_FLAG = "mode_collapse"   # bool column appended to the benchmark table

# Benchmark-table column order (both files share the leading columns).
BENCHMARK_ID_COLUMNS: tuple[str, ...] = ("model", "split")
BENCHMARK_COLUMNS: tuple[str, ...] = (
    BENCHMARK_ID_COLUMNS + METRICS_HEADLINE + (MODE_COLLAPSE_FLAG,)
)
BENCHMARK_COLUMNS_FULL: tuple[str, ...] = (
    BENCHMARK_ID_COLUMNS + METRICS_FULL + (MODE_COLLAPSE_FLAG,)
)


# ---------------------------------------------------------------------------
# 8. GPU queue namespace (§6) — job name -> priority. gpu_queue.py imports this so
#    both worktrees submit consistent names and the serial queue runs in §6 order.
# ---------------------------------------------------------------------------
GPU_JOB_ORDER: tuple[str, ...] = (
    "esm2",           # G1  ESM-2 650M embeddings
    "causal",         # G2  CausalCisTransFormer (corrected mask)
    "noncausal",      # G3  non-causal twin
    "jepa",           # G4  JEPA pretraining (overnight; Developer 2)
    "jepa_finetune",  # G5  JEPA+causal fine-tune + JEPA-only (Developer 2)
    "arc_state",      # G6  Arc State (gated)
    "iftime",         # G7  GEARS / scVI / Geneformer (remaining time only)
)


def gpu_job_priority(job_name: str) -> int:
    """Lower number = higher priority (runs earlier). Unknown jobs sort last."""
    try:
        return GPU_JOB_ORDER.index(job_name)
    except ValueError:
        return len(GPU_JOB_ORDER)


# ---------------------------------------------------------------------------
# 9. Small shared utilities (stdlib only)
# ---------------------------------------------------------------------------
def sha256_file(path: os.PathLike | str, chunk: int = 1 << 20) -> str:
    """Streaming SHA256 of a (possibly large) file. Used to bind the split to a .h5ad."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    # versions
    "CONTRACT_VERSION", "SPLIT_SCHEMA_VERSION",
    # roots + dirs
    "REPO_ROOT", "DATA_ROOT", "RAW_DIR", "EMBEDDINGS_DIR", "PSEUDOBULK_DIR",
    "FEATURES_DIR", "CELLS_DIR", "RUNS_DIR", "CHECKPOINTS_DIR", "GPU_LOCK",
    "SPLIT_MANIFEST", "HVG_LIST_PATH", "RESULTS_DIR", "FIGURES_DIR", "HYPOTHESES",
    "BENCHMARK_TABLE", "BENCHMARK_TABLE_FULL",
    "ESM2_CACHE", "CONTEXT_PRIOR_CACHE", "PSEUDOBULK_TRAIN", "PSEUDOBULK_TEST",
    "DEG_FREQ_CACHE", "ensure_dirs",
    # biology
    "GEO_ACCESSION", "CONDITIONS", "CONDITION_HOLDOUT", "TRAIN_CONDITIONS",
    "DONORS", "DONOR_PROBE", "CONTROL_PERT_ID", "GENE_ID",
    # dims
    "HVG_N", "TOP_DEG_N", "DEG_FREQ_DIM", "ESM2_DIM", "CONTEXT_PRIOR_DIM",
    "TABPFN_MAX_FEATURES", "TABPFN_MAX_ROWS",
    # splits
    "SPLIT_GENE", "SPLIT_CONDITION", "SPLIT_DONOR", "SPLITS", "ALL_SPLITS",
    "GENE_HOLDOUT_FRACTION", "SPLIT_SEED", "SplitManifest",
    # models
    "MODEL_RIDGE", "MODEL_TABPFN", "MODEL_FCN", "MODEL_NONCAUSAL", "MODEL_CAUSAL",
    "MODEL_JEPA_ONLY", "MODEL_JEPA_CAUSAL", "MODEL_ARC_STATE", "MODELS",
    "CP1_MODELS", "GRID_2X2", "run_path", "checkpoint_path",
    # pseudobulk schema
    "PSEUDOBULK_INDEX_NAMES", "PSEUDOBULK_BLOCK_EXPR", "PSEUDOBULK_BLOCK_DELTA",
    "PSEUDOBULK_COL_LEVELS", "pseudobulk_block", "pseudobulk_expr",
    "pseudobulk_delta", "build_pseudobulk_frame",
    # eval
    "METRIC_PEARSON_DELTA", "METRIC_PERTURBENCH_RANK", "METRIC_DES",
    "METRICS_HEADLINE", "METRIC_MAE", "METRIC_SPEARMAN_LFC",
    "METRIC_SPEARMAN_EFFECT", "METRIC_AUPRC", "METRIC_EDISTANCE", "METRICS_FULL",
    "METRICS_LOWER_IS_BETTER", "MODE_COLLAPSE_THRESHOLD", "MODE_COLLAPSE_FLAG",
    "BENCHMARK_ID_COLUMNS", "BENCHMARK_COLUMNS", "BENCHMARK_COLUMNS_FULL",
    # gpu queue
    "GPU_JOB_ORDER", "gpu_job_priority",
    # utils
    "sha256_file", "sha256_text",
]
