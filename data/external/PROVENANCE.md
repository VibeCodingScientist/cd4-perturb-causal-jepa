# External causal-edge datasets — provenance (C-FUSE Phase 1, retrieved 2026-07-10)

All numbers were verified against the actual supplements during this run. The raw supplement files are not
committed (one is 50 MB); the direct URLs below re-obtain them. The following are committed here: the parsed
edge tables (`weinstock_edges.csv`) and the harmonized measurable-edge list
(`../../results/fusion_measurable_edges.csv`).

## ⚠️ Source-of-truth corrections vs the brief
- The brief referred to the Cell Genomics paper as "emdann 2024"; the actual paper is Weinstock, Arce,
  Freimer, Ota, Marson, Battle, Pritchard (2024), Cell Genomics (LLCB network inference). This has been
  corrected.
- The brief's "~211 edges over 84 TFs" is [VERIFIED] correct: 350 / 211 / 151 edges at
  |β| ≥ 0.020 / 0.025 / 0.030, which matches the paper's Results text exactly. 211 corresponds to the primary
  network.
- Same-consortium caveat: both external papers originate from the Marson/Pritchard lab, the same consortium
  as GSE278572. This therefore constitutes cross-platform/modality validation (CRISPRi Perturb-seq versus
  CRISPR-KO + LLCB / bulk RNA-seq), not fully independent-lab replication. This is stated in FUSION_GATES.md.

## Weinstock 2024 — Cell Genomics [VERIFIED]
- DOI 10.1016/j.xgen.2024.100671 (PII S2666979X24002908); PMC11605694.
- Supplement Table S1 (`mmc2.xlsx`), single sheet "Causal gene network S1":
  `https://ars.els-cdn.com/content/image/1-s2.0-S2666979X24002908-mmc2.xlsx` (815 KB, HTTP 200, 2026-07-10).
- Rows: 6972 parent→child pairs among 84 genes; columns `parent gene, child gene, posterior mean
  (β, signed), 2.5%, 97.5%, PIP, LFSR, rhat, ess, posterior sd`. Edge direction is given by sign(β).
- Verified edge counts at |β| thresholds: 0.020→350, 0.025→211, 0.030→151 (matching the paper).
- Code: github.com/weinstockj/LLCB; Zenodo 10.5281/zenodo.12807946 (figures), 10.5281/zenodo.12807979 (LLCB).
- Read-tier: [VERIFIED] (supplement table parsed directly).

## Freimer 2022 — Nature Genetics [VERIFIED]
- DOI 10.1038/s41588-022-01106-y; PMC10035359; GEO GSE171737; code Zenodo 10.5281/zenodo.6371646.
- Supplementary Data 1 (`MOESM4_ESM.zip` → `Supplementary_Data_1_RNA_Seq_results.csv`):
  `https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-022-01106-y/MediaObjects/41588_2022_1106_MOESM4_ESM.zip`
  (22 MB zip → 50 MB CSV, HTTP 200, 2026-07-10). Also MOESM3 (tables) and MOESM5 (235 MB).
- Rows: 269,080 regulator-KO × gene DE results; columns `ens_id (target ENSG), gene_name, logFC
  (direction), P.Value, adj.P.Val, sample (= regulator KO)`. An edge is defined by adj.P.Val < 0.05; the
  direction is given by sign(logFC).
- Edges at FDR<0.05: 50,654 across 24 regulators of IL2RA/IL2/CTLA4. Note that this DE analysis includes
  indirect downstream effects, unlike Weinstock's direct LLCB edges; these represent different edge semantics
  and are kept separate.
- Read-tier: [VERIFIED] (supplement CSV parsed directly).

## ID harmonization
- Weinstock uses gene symbols, which were mapped to ENSG via the pseudobulk
  `perturbed_gene_name↔perturbed_gene_id` and `var.gene_name↔var.gene_ids` mappings. All 84 symbols were
  mapped (0 unmapped in the 211-edge set).
- Freimer targets are ENSG (`ens_id`) and were used directly; regulators (from `sample`) were mapped via the
  symbol map.
- Gene space: 12,730 assayed perturbations · 3,000 HVG readout genes · 1,729-gene frozen `gene_holdout`.
