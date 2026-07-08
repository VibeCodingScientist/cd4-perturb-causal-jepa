"""
core.split — write / load / verify the immutable split (UNIFIED_BUILD_PLAN.md §3).

Two-phase freeze (see contract.SplitManifest):
  * The split *policy* is committed on Day 0 (seed, held-out condition, donor probe,
    HVG count, gene-holdout fraction).
  * `freeze()` materializes the data-*derived* fields once the real .h5ad exists:
    the deterministic gene-holdout list (seed 42), the SHA256 that binds the split
    to that exact file, the committed HVG list, and `data_frozen = True`.

Every module that touches data calls `verify()` at startup: it reloads the manifest,
recomputes the .h5ad SHA256, and asserts equality — so no model can silently train on
a different dataset than the one the split was frozen against.

`sample_gene_holdout` is a pure, deterministic function used by BOTH `freeze()` and the
synthetic-data generator, so the synthetic split is drawn exactly like the real one.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import contract as C


def sample_gene_holdout(
    perturbed_genes: Sequence[str],
    fraction: float = C.GENE_HOLDOUT_FRACTION,
    seed: int = C.SPLIT_SEED,
) -> List[str]:
    """Deterministically withhold `fraction` of the perturbed genes (sorted for stability).

    Sorting first makes the draw independent of the input ordering, so re-freezing the
    same gene universe always yields the same hold-out.
    """
    genes = sorted(set(str(g) for g in perturbed_genes if g != C.CONTROL_PERT_ID))
    n_hold = max(1, int(round(len(genes) * fraction)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(genes), size=n_hold, replace=False)
    return sorted(genes[i] for i in idx)


def freeze(
    *,
    perturbed_genes: Sequence[str],
    hvg_genes: Sequence[str],
    h5ad_path: Optional[Path] = None,
    doi: str = "",
    write: bool = True,
) -> C.SplitManifest:
    """Materialize the data-derived split fields and (optionally) write the manifest.

    `perturbed_genes` — every gene that was silenced (defines the gene-holdout universe).
    `hvg_genes`       — the 3,000 selected HVGs (persisted to the committed HVG list).
    `h5ad_path`       — the frozen dataset; its SHA256 binds the split. May be None when
                        freezing against synthetic data (sha256_h5ad left "").
    """
    if len(hvg_genes) != C.HVG_N:
        # not fatal for synthetic tests, but loud so a real freeze can't drift silently
        import warnings
        warnings.warn(f"hvg_genes has {len(hvg_genes)} entries, expected {C.HVG_N}")

    man = C.SplitManifest(
        doi=doi,
        sha256_h5ad=C.sha256_file(h5ad_path) if h5ad_path else "",
        gene_holdout=sample_gene_holdout(perturbed_genes),
        created_utc=datetime.now(timezone.utc).isoformat(),
        data_frozen=True,
    )
    if write:
        C.SPLIT_MANIFEST.write_text(man.to_json() + "\n")
        C.HVG_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        C.HVG_LIST_PATH.write_text("\n".join(str(g) for g in hvg_genes) + "\n")
    return man


def load() -> C.SplitManifest:
    """Load the manifest from disk."""
    return C.SplitManifest.from_dict(json.loads(C.SPLIT_MANIFEST.read_text()))


def load_hvg() -> List[str]:
    """Load the committed HVG list (order-preserving)."""
    return [ln for ln in C.HVG_LIST_PATH.read_text().splitlines() if ln.strip()]


def verify(h5ad_path: Optional[Path] = None, *, require_data_frozen: bool = True) -> C.SplitManifest:
    """Assert the on-disk split is frozen and (if a path is given) matches the .h5ad.

    Raises RuntimeError on any mismatch. Modules call this at startup before reading data.
    """
    man = load()
    if require_data_frozen and not man.data_frozen:
        raise RuntimeError(
            "split_manifest.json is policy-frozen but the data-derived fields are not "
            "materialized yet — run core.split.freeze() against the downloaded .h5ad first."
        )
    if man.schema_version != C.SPLIT_SCHEMA_VERSION:
        raise RuntimeError(
            f"split schema_version {man.schema_version} != contract {C.SPLIT_SCHEMA_VERSION}"
        )
    if h5ad_path is not None and man.sha256_h5ad:
        actual = C.sha256_file(h5ad_path)
        if actual != man.sha256_h5ad:
            raise RuntimeError(
                f"SHA256 mismatch: {h5ad_path} has {actual[:12]}… but the split was frozen "
                f"against {man.sha256_h5ad[:12]}…. Refusing to run on a different dataset."
            )
    return man


# --- Split routing (used by pseudobulk to place a (pert,cond,donor) row) ---------
def is_train_row(pert_id: str, condition: str, donor: str, man: C.SplitManifest) -> bool:
    """A row is TRAIN iff it is a train condition, a non-held-out gene, and not the
    donor probe. Everything else is held out (routed to pseudobulk/test.parquet)."""
    if condition not in C.TRAIN_CONDITIONS:
        return False
    if pert_id in set(man.gene_holdout):
        return False
    if donor == C.DONOR_PROBE:
        return False
    return True


if __name__ == "__main__":  # pragma: no cover
    m = load()
    status = "DATA-FROZEN" if m.data_frozen else "POLICY-FROZEN (awaiting data)"
    print(f"split: {status}")
    print(f"  seed={m.seed}  condition_holdout={m.condition_holdout}  donor_probe={m.donor_probe}")
    print(f"  gene_holdout: {len(m.gene_holdout)} genes  hvg_n={m.hvg_n}")
    print(f"  sha256_h5ad={m.sha256_h5ad or '(pending)'}")
