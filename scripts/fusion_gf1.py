#!/usr/bin/env python
"""C-FUSE G-F.1 — measurable cross-dataset causal overlap (CPU, committed + external data).

External causal edge sets (obtained this run, verified against supplements):
  Weinstock 2024 Cell Genomics (10.1016/j.xgen.2024.100671, Table S1 mmc2.xlsx): LLCB posterior over
    6972 parent->child pairs among 84 CD4 TFs; 211 edges at |beta|>=0.025 (VERIFIED: 350/211/151 at
    0.020/0.025/0.030). beta sign = regulator->target direction.
  Freimer 2022 Nat Genet (10.1038/s41588-022-01106-y, Suppl Data 1 RNA_Seq_results.csv): per-regulator
    KO RNA-seq DE; edge = adj.P.Val<0.05, logFC sign = direction. sample col = regulator KO.

Both are Marson/Pritchard-consortium (same lab as our GSE278572) -> cross-PLATFORM/MODALITY validation
(our CRISPRi Perturb-seq vs their CRISPR-KO + LLCB / bulk RNA-seq), not fully independent-lab.

MEASURABLE shared edge = external edge (regulator R -> target T) where R is in our ASSAYED perturbations
AND T is in our HVG readout genes (so we can test the do-operator's prediction of T's response to R).
Held-out = additionally R in the frozen gene_holdout (for G-F.2). PASS G-F.1: >=50 combined measurable.
Read-only on committed + external artifacts; frozen tag/CP2 untouched.
"""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import anndata as ad
from core import contract as C
from core import split as split_mod

EXT = Path(__file__).resolve().parent.parent / "data" / "external"
RES = Path(__file__).resolve().parent.parent / "results"
PSEUDO = "/home/ubuntu/cd4-perturb-data/raw/GWCD4i.pseudobulk_merged.h5ad"
BETA_THR = 0.025      # Weinstock primary network (211 edges)
FDR_THR = 0.05        # Freimer DE significance


def build_gene_space():
    a = ad.read_h5ad(PSEUDO, backed="r")
    o = a.obs
    tg = o[o["guide_type"].astype(str) == "targeting"]
    def strip(s): return str(s).split(".")[0]
    assayed_ens = set(strip(x) for x in tg["perturbed_gene_id"].astype(str).unique())
    sym2ens = {}
    for s, e in zip(tg["perturbed_gene_name"].astype(str), tg["perturbed_gene_id"].astype(str)):
        sym2ens[str(s)] = strip(e)
    # var: gene_name <-> gene_ids for HVG symbol map
    vid = a.var["gene_ids"].astype(str).to_numpy() if "gene_ids" in a.var.columns else np.array(a.var_names)
    vn = a.var["gene_name"].astype(str).to_numpy()
    for s, e in zip(vn, vid):
        sym2ens.setdefault(str(s), strip(e))
    hvg = set(strip(g) for g in split_mod.load_hvg())
    man = json.loads((Path(__file__).resolve().parent.parent / "split_manifest.json").read_text())
    holdout = set(strip(g) for g in man["gene_holdout"])
    return assayed_ens, hvg, holdout, sym2ens


def load_weinstock(sym2ens):
    edges = []
    with open(EXT / "weinstock_edges.csv") as f:
        for row in csv.DictReader(f):
            beta = float(row["beta"])
            if abs(beta) < BETA_THR:
                continue
            r, t = row["parent_symbol"], row["child_symbol"]
            edges.append({"reg_sym": r, "tgt_sym": t, "reg_ens": sym2ens.get(r), "tgt_ens": sym2ens.get(t),
                          "sign": int(np.sign(beta)), "src": "weinstock"})
    return edges


def load_freimer(sym2ens):
    edges = []
    p = EXT / "Supplementary_Data_1_RNA_Seq_results.csv"
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                fdr = float(row["adj.P.Val"]); lfc = float(row["logFC"])
            except (ValueError, KeyError):
                continue
            if fdr >= FDR_THR:
                continue
            reg = row["sample"].replace(" KO", "").replace("KO", "").strip()
            edges.append({"reg_sym": reg, "tgt_sym": row.get("gene_name"),
                          "reg_ens": sym2ens.get(reg), "tgt_ens": row["ens_id"].split(".")[0],
                          "sign": int(np.sign(lfc)), "src": "freimer"})
    return edges


def summarize(edges, assayed, hvg, holdout, name):
    n_total = len(edges)
    mapped = [e for e in edges if e["reg_ens"] and e["tgt_ens"]]
    measurable = [e for e in mapped if e["reg_ens"] in assayed and e["tgt_ens"] in hvg]
    heldout = [e for e in measurable if e["reg_ens"] in holdout]
    regs = set(e["reg_ens"] for e in measurable)
    regs_ho = set(e["reg_ens"] for e in heldout)
    print(f"[{name}] edges={n_total}  id-mapped={len(mapped)}  MEASURABLE={len(measurable)} "
          f"(regs={len(regs)})  held-out-measurable={len(heldout)} (regs={len(regs_ho)})")
    return {"set": name, "edges": n_total, "mapped": len(mapped), "measurable": len(measurable),
            "measurable_regs": len(regs), "heldout_measurable": len(heldout), "heldout_regs": len(regs_ho)}


def main():
    RES.mkdir(exist_ok=True)
    assayed, hvg, holdout, sym2ens = build_gene_space()
    print(f"[space] assayed perts={len(assayed)}  HVG readout={len(hvg)}  gene_holdout={len(holdout)}")
    w = load_weinstock(sym2ens)
    fr = load_freimer(sym2ens)
    rows = [summarize(w, assayed, hvg, holdout, "weinstock"),
            summarize(fr, assayed, hvg, holdout, "freimer"),
            summarize(w + fr, assayed, hvg, holdout, "combined")]
    combined_measurable = rows[2]["measurable"]
    combined_heldout = rows[2]["heldout_measurable"]
    import pandas as pd
    pd.DataFrame(rows).to_csv(RES / "fusion_gf1_overlap.csv", index=False)
    # persist the measurable+heldout edge lists for G-F.2
    allm = [e for e in (w + fr) if e["reg_ens"] and e["tgt_ens"]
            and e["reg_ens"] in assayed and e["tgt_ens"] in hvg]
    pd.DataFrame(allm).to_csv(RES / "fusion_measurable_edges.csv", index=False)
    print("\n########## G-F.1 ROUTING ##########")
    print(f"  combined MEASURABLE shared edges = {combined_measurable}  (bar: >=50)")
    print(f"  of which HELD-OUT regulators = {combined_heldout}  (usable for the non-circular G-F.2)")
    gf1 = combined_measurable >= 50
    print(f"  G-F.1 VERDICT: {'PASS -> proceed to G-F.2' if gf1 else 'FAIL -> overlap too thin to fuse; clean negative; STOP'}")
    if gf1 and combined_heldout < 10:
        print(f"  CAUTION: only {combined_heldout} held-out measurable edges — G-F.2 headline power is limited; report as such")


if __name__ == "__main__":
    main()
