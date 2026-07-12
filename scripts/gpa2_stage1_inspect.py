#!/usr/bin/env python
"""G-PA.2 Stage 1 — download Schmidt 2022 (GSE190604) to an ISOLATED dir and inspect its structure.

Never writes to CD4_DATA_ROOT. Determines what annotations are recoverable from the cheap GEO supplement
(perturbation guide-calls certainly; donor + condition/trajectory = unknown until inspected) — which decides
the portable probe set. Prints a structured report; downloads nothing into the repo.
"""
import os, sys, gzip, subprocess, urllib.request

DEST = os.path.expanduser("~/gpa2-data/raw")
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE190nnn/GSE190604/suppl/"
FILES = ["GSE190604_features.tsv.gz", "GSE190604_barcodes.tsv.gz",
         "GSE190604_cellranger-guidecalls-aggregated-unfiltered.txt.gz", "GSE190604_matrix.mtx.gz"]


def dl(name):
    out = os.path.join(DEST, name)
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print(f"[dl] {name} already present ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)
        return out
    print(f"[dl] fetching {name} ...", flush=True)
    subprocess.run(["curl", "-sL", "-o", out, BASE + name], check=True)
    print(f"[dl] {name} -> {os.path.getsize(out)/1e6:.1f} MB", flush=True)
    return out


def head_gz(path, n=6):
    with gzip.open(path, "rt") as f:
        return [next(f).rstrip("\n") for _ in range(n)]


def main():
    os.makedirs(DEST, exist_ok=True)
    paths = {f: dl(f) for f in FILES}

    print("\n==== FEATURES ====", flush=True)
    ft = head_gz(paths["GSE190604_features.tsv.gz"], 4)
    for l in ft:
        print("  ", l)
    nfeat = sum(1 for _ in gzip.open(paths["GSE190604_features.tsv.gz"], "rt"))
    print(f"  n_features = {nfeat}")

    print("\n==== BARCODES ====", flush=True)
    bc = head_gz(paths["GSE190604_barcodes.tsv.gz"], 4)
    for l in bc:
        print("  ", l)
    # aggregation lane suffix distribution (10x aggr appends -1/-2/... = lanes/samples/donors)
    from collections import Counter
    suf = Counter()
    nbc = 0
    for l in gzip.open(paths["GSE190604_barcodes.tsv.gz"], "rt"):
        nbc += 1
        s = l.strip().split("-")[-1] if "-" in l else "none"
        suf[s] += 1
    print(f"  n_barcodes = {nbc}")
    print(f"  barcode suffix (aggr lane) distribution: {dict(sorted(suf.items()))}")

    print("\n==== GUIDE CALLS (the annotation that decides probe set) ====", flush=True)
    gc = paths["GSE190604_cellranger-guidecalls-aggregated-unfiltered.txt.gz"]
    lines = head_gz(gc, 8)
    for l in lines:
        print("  ", l[:200])
    ngc = sum(1 for _ in gzip.open(gc, "rt"))
    print(f"  n_guidecall_rows = {ngc}")
    # try to read as a table and report columns + how many target genes
    import csv
    with gzip.open(gc, "rt") as f:
        sample = f.readline()
    delim = "\t" if "\t" in sample else ("," if "," in sample else None)
    print(f"  header delim guess = {repr(delim)}")

    print("\n==== MATRIX header (dims, no full load) ====", flush=True)
    with gzip.open(paths["GSE190604_matrix.mtx.gz"], "rt") as f:
        for i in range(4):
            line = f.readline().rstrip("\n")
            print("  ", line)
            if not line.startswith("%") and line.strip():
                break

    print("\nSTAGE1_INSPECT_DONE", flush=True)


if __name__ == "__main__":
    main()
