#!/usr/bin/env python
"""Build the JEPA single-cell cache from the public per-(donor,condition) h5ads.

Runs ON THE BOX (needs aws CLI + disk). The single-cell files are 110-161 GB EACH and
the box holds only ~one at a time, so this does **download -> ingest -> delete** per
file, appending a compact log1p-CP10k HVG subsample to CELLS_DIR each time.

Holdout-clean defaults (UNIFIED_BUILD_PLAN §2/§3):
  * conditions = Rest, Stim8hr   (Stim48hr is the held-out zero-shot test — JEPA must
    NOT see it, or the C3 condition-transfer claim leaks)
  * donors     = D1, D2, D3      (D4 is the reserved donor probe)

Usage (on the box):
  python scripts/fetch_jepa_cells.py --donors D1 D2 --cells-per-file 300000
  python scripts/fetch_jepa_cells.py --dry-run          # plan + sizes + disk check only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# bootstrap repo root onto sys.path so this runs from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import contract  # noqa: E402

BUCKET = "genome-scale-tcell-perturb-seq"
PREFIX = "marson2025_data"
GB = 1 << 30


def _aws(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["aws", "--no-sign-request", *args], capture_output=True, text=True)


def object_size_bytes(key: str) -> int:
    r = _aws("s3api", "head-object", "--bucket", BUCKET, "--key", key, "--query", "ContentLength", "--output", "text")
    if r.returncode != 0:
        raise RuntimeError(f"head-object failed for {key}: {r.stderr.strip()}")
    return int(r.stdout.strip())


def download(key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # --only-show-errors: no per-part progress spam (we track progress by polling size)
    r = subprocess.run(
        ["aws", "--no-sign-request", "s3", "cp", "--only-show-errors", f"s3://{BUCKET}/{key}", str(dest)]
    )
    if r.returncode != 0:
        raise RuntimeError(f"download failed for {key}")


def load_hvg(hvg_path: str | None = None) -> list:
    """Frozen HVG gene order. Prefer an explicit --hvg-path (e.g. Dev1's box-local
    split/hvg_3000.txt, which is untracked so not in this worktree); else core.split;
    else the contract path."""
    if hvg_path:
        return [g.strip() for g in Path(hvg_path).read_text().splitlines() if g.strip()]
    try:
        from core import split as split_mod
        if hasattr(split_mod, "load_hvg"):
            return list(split_mod.load_hvg())
    except Exception:
        pass
    return [g.strip() for g in contract.HVG_LIST_PATH.read_text().splitlines() if g.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch + ingest JEPA single cells (box run)")
    p.add_argument("--donors", nargs="+", default=["D1", "D2", "D3"])
    p.add_argument("--conditions", nargs="+", default=["Rest", "Stim8hr"])
    p.add_argument("--cells-per-file", type=int, default=300_000)
    p.add_argument("--tmp-dir", default=str(contract.RAW_DIR / "_jepa_cells_tmp"))
    p.add_argument("--min-headroom-gb", type=float, default=10.0)
    p.add_argument("--keep-raw", action="store_true", help="do not delete each h5ad after ingest")
    p.add_argument("--hvg-path", default=None, help="explicit frozen HVG list (Dev1's box-local split/hvg_3000.txt)")
    p.add_argument("--seed", type=int, default=contract.SPLIT_SEED)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    contract.ensure_dirs()
    tmp = Path(args.tmp_dir)
    files = [(d, c, f"{PREFIX}/{d}_{c}.assigned_guide.h5ad") for d in args.donors for c in args.conditions]

    print(f"[fetch] plan: {len(files)} files x {args.cells_per_file:,} cells -> {contract.CELLS_DIR}")
    total_bytes = 0
    for d, c, key in files:
        try:
            size = object_size_bytes(key)
        except Exception as e:
            print(f"  ! {key}: {e}")
            continue
        total_bytes += size
        print(f"  {d} {c}: {size/GB:.1f} GiB")
    free = shutil.disk_usage(tmp.parent if tmp.parent.exists() else Path.home()).free
    print(f"[fetch] total transfer ~{total_bytes/GB:.0f} GiB (sequential); free disk {free/GB:.0f} GiB")

    if args.dry_run:
        biggest = 0
        for d, c, key in files:
            try:
                biggest = max(biggest, object_size_bytes(key))
            except Exception:
                pass
        ok = free > biggest + args.min_headroom_gb * GB
        print(f"[fetch] largest single file {biggest/GB:.1f} GiB; fits one-at-a-time: {ok}")
        return 0

    hvg = load_hvg(args.hvg_path)
    print(f"[fetch] HVG panel: {len(hvg)} genes")
    # Gene-hold-out cells must NOT enter the JEPA cache (else the gene-split C3 claim leaks).
    import json as _json
    manifest = _json.loads(contract.SPLIT_MANIFEST.read_text())
    holdout_genes = manifest.get("gene_holdout", [])
    print(f"[fetch] gene-holdout: excluding {len(holdout_genes)} held-out genes' cells from the JEPA cache")
    from core.models.jepa_data import ingest_file_to_cache

    added_total = 0
    for i, (d, c, key) in enumerate(files, 1):
        size = object_size_bytes(key)
        free = shutil.disk_usage(tmp.parent).free
        if free < size + args.min_headroom_gb * GB:
            print(f"[fetch] STOP: {key} needs {size/GB:.0f} GiB but only {free/GB:.0f} GiB free "
                  f"(+{args.min_headroom_gb} GiB headroom). Free space or expand the disk.")
            return 2
        dest = tmp / Path(key).name
        t0 = time.time()
        print(f"[fetch] ({i}/{len(files)}) downloading {key} ({size/GB:.0f} GiB) ...")
        download(key, dest)
        print(f"[fetch] ingesting {d} {c} ({args.cells_per_file:,} cells) ...")
        added = ingest_file_to_cache(dest, hvg, args.cells_per_file, donor=d, condition=c, seed=args.seed,
                                     holdout_genes=holdout_genes)
        added_total += added
        if not args.keep_raw:
            dest.unlink(missing_ok=True)
        print(f"[fetch] +{added:,} cells ({time.time()-t0:.0f}s); cache total {added_total:,}")

    print(f"[fetch] DONE: {added_total:,} cells across {len(files)} files -> {contract.CELLS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
