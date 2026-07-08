# RUNBOOK — producing the real CP1 benchmark on a GPU box

Everything here runs on a single GPU box (A10G / L4, 24 GB) with ~150 GB free disk. CP1
(baselines + causal + non-causal on both hold-outs) needs the **44.6 GB CZI pseudobulk file**,
not the 1.7 TB of single cells — those are the JEPA lane's input (Developer 2 / CP2).

## 0. Box requirements
- GPU: NVIDIA A10G or L4, 24 GB (matches the plan's L4 sizing).
- Disk: ~150 GB free (44.6 GB pseudobulk + 16.8 GB DE-stats optional + caches + checkpoints).
- ~32–64 GB RAM. CUDA 12.x. conda/mamba (or a venv).

## 1. Clone + environment
```bash
git clone https://github.com/VibeCodingScientist/cd4-perturb-causal-jepa.git
cd cd4-perturb-causal-jepa
conda env create -f environment.yml && conda activate cd4-perturb
# shared artifacts live outside git; put them on the big disk:
export CD4_DATA_ROOT=/data/cd4-perturb-data      # any path with ~150 GB free
python -c "from core import contract as C; C.ensure_dirs(); print('DATA_ROOT', C.DATA_ROOT)"
```

## 2. Download the data (public S3, no credentials) — FLAGGED >5 GB
```bash
# 44.6 GB pseudobulk (all CP1 needs) + tiny supplementary tables
aws s3 cp --no-sign-request \
  s3://genome-scale-tcell-perturb-seq/marson2025_data/GWCD4i.pseudobulk_merged.h5ad \
  "$CD4_DATA_ROOT/raw/GWCD4i.pseudobulk_merged.h5ad"
aws s3 cp --no-sign-request --recursive \
  s3://genome-scale-tcell-perturb-seq/marson2025_data/suppl_tables/ \
  "$CD4_DATA_ROOT/raw/suppl_tables/"
# optional: DE-stats for cross-checking DES/DEG-frequency (16.8 GB)
# aws s3 cp --no-sign-request \
#   s3://genome-scale-tcell-perturb-seq/marson2025_data/GWCD4i.DE_stats.h5ad \
#   "$CD4_DATA_ROOT/raw/GWCD4i.DE_stats.h5ad"
```

## 3. Build the frozen core (Lane C, CPU)
Normalizes per-guide profiles, averages guides, builds per-(gene,condition,donor) pseudobulk +
deltas, freezes `split_manifest.json` against the file SHA256, and writes the DEG-frequency cache.
```bash
python -c "from core import data; data.build_from_czi_pseudobulk('$CD4_DATA_ROOT/raw/GWCD4i.pseudobulk_merged.h5ad', doi='10.64898/2025.12.23.696273')"
python -m core.split            # prints the frozen split summary; commit split_manifest.json + split/hvg_3000.txt
```

## 4. Gene-token priors (features)
Two priors feed every model (ESM-2 function prior + a regulatory/context prior). Pick one path:

**A. ESM-2 (job G1, GPU) + node2vec context prior.** Requires a gene→protein-sequence map and a
STRING graph:
```bash
# prepare esm2.parquet (mean-pooled ESM-2 650M over each HVG gene's protein) via the queue
python gpu_queue.py submit esm2       # reads $CD4_DATA_ROOT/raw/gene_sequences.parquet
# context_prior.parquet from node2vec over a downloaded human STRING edge list
```
**B. Fast fallback (recommended to unblock CP1).** Use GenePT precomputed gene embeddings for
both priors (PCA to ESM2_DIM / CONTEXT_PRIOR_DIM). Instant, no sequence fetching. See
`core.features` — wire `build_esm2` fallback / `build_context_prior`.

> The gene-token prior is not the headline claim (the do-operator is); fallback B is fine for CP1.
> The exact prep script is finalized interactively on the box against the real gene panel.

## 5. Run CP1 (baselines on CPU, transformers via the serial GPU queue)
```bash
# baselines (CPU; TabPFN/FCN use GPU if present)
python -c "from core.models import baselines as b; b.run_ridge(); b.run_tabpfn(); b.run_fcn()"
# causal + non-causal through the single-GPU serial queue (epoch-1 gate runs first)
python gpu_queue.py submit causal
python gpu_queue.py submit noncausal
# score everything into results/benchmark_table.csv (+ full 8-metric appendix)
python -c "
from core import eval as ev, contract as C
for m in C.CP1_MODELS:
    for s in C.SPLITS:
        ev.score_run_file(m, s)
"
cat results/benchmark_table.csv
```

Or the whole thing via Snakemake:
```bash
snakemake --cores all --config raw_h5ad="$CD4_DATA_ROOT/raw/GWCD4i.pseudobulk_merged.h5ad"
```

## 6. CP1 done
`results/benchmark_table.csv` has ridge / tabpfn / fcn / causal / noncausal on the gene and
condition hold-outs; the split SHA is frozen; the mode-collapse detector flags any collapsed
model. Commit `results/benchmark_table.csv` + `split_manifest.json` + `split/hvg_3000.txt`.

CP2 (Developer 2): the JEPA cells of the 2×2 need the per-donor cell h5ads
(`s3://genome-scale-tcell-perturb-seq/marson2025_data/D*_*.assigned_guide.h5ad`, ~118–173 GB
each) — subsample per §7e; submit `jepa` / `jepa_finetune` to the same queue.
