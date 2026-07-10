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

## 4. Gene-token priors (features, job G1, GPU)
```bash
./.venv/bin/python scripts/build_priors.py
```
This downloads the bulk Ensembl human peptide FASTA once (~15 MB), maps each HVG gene to its
longest protein (≈99.8% coverage on the real panel), embeds with ESM-2 650M on the GPU
(~14 min on an L4), and writes `esm2.parquet` (1280-d) + `context_prior.parquet` (a 512-d PCA
projection). The plan's node2vec-over-STRING network prior is wired
(`core.features.build_context_prior_node2vec`) as a follow-up; the ESM-2 function prior is the
one that matters for zero-shot generalization to unseen genes.

> Do NOT use `gget.seq` for the whole panel — 3000 per-gene Ensembl queries stall. The bulk
> FASTA is one request.

## 5. Run CP1 (baselines + causal + non-causal)
```bash
./.venv/bin/python scripts/run_cp1.py     # Ridge + FCN + causal + non-causal -> benchmark_table.csv
```
Or the whole thing via Snakemake:
```bash
snakemake --cores all --config raw_h5ad="$CD4_DATA_ROOT/raw/GWCD4i.pseudobulk_merged.h5ad"
```

> **TabPFN gate.** `tabpfn>=8` downloads license-gated Prior-Labs models (a one-time browser
> license acceptance on Hugging Face); older ungated versions break the sklearn/scipy stack.
> On a fresh headless box TabPFN is therefore marked N/A — accept the Prior-Labs license with
> your HF account (one click) + a token, then `python -c "from core.models import baselines as b;
> b.run_tabpfn()"` to add its rows. CP1 stands on Ridge (the strong baseline) + FCN + causal +
> non-causal without it.

## 6. CP1 done
`results/benchmark_table.csv` has ridge / tabpfn / fcn / causal / noncausal on the gene and
condition hold-outs; the split SHA is frozen; the mode-collapse detector flags any collapsed
model. Commit `results/benchmark_table.csv` + `split_manifest.json` + `split/hvg_3000.txt`.

CP2 (Developer 2): the JEPA cells of the 2×2 need the per-donor cell h5ads
(`s3://genome-scale-tcell-perturb-seq/marson2025_data/D*_*.assigned_guide.h5ad`, ~118–173 GB
each) — subsample per §7e; submit `jepa` / `jepa_finetune` to the same queue.

## 7. Box infra notes (supplementary CPU gates)

- **Run CZI-reading gates in the FOREGROUND** over SSH with `-o ServerAliveInterval=15` (and a
  `timeout`), e.g. `ssh -o ServerAliveInterval=15 box 'cd ~/gate && CD4_DATA_ROOT=... timeout 400
  python -u scripts/<gate>.py'`. Detached `setsid nohup ... & disown` launches **died silently**
  (empty logs, no traceback) during the C-DON gate — do not rely on them for the multi-minute
  CZI read. The 44 GB pseudobulk is OS-page-cached after the first read, so re-runs are fast.
- **Large per-guide aggregation** (mean over conditions across ~90k guide×donor groups on 3,000
  HVG): use `np.argsort` + `np.add.reduceat` on a `float32` array. `np.add.at` (unbuffered, slow)
  and `pd.DataFrame(s).groupby(...).mean()` (4 GB copy) both stalled/OOM'd G-D.2. Science was
  unaffected — G-D.1 reproduced identically across three runs.
