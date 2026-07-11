#!/usr/bin/env bash
# C-FUSE 1b — full AFK-safe pipeline: restore clobbered DATA_ROOT (pinned to the frozen panel/split),
# rebuild ESM2 priors, re-train do-operator + twin with 10 external regulators held out, run powered
# G-F.2. Checkpointed (each stage skips if its real output already exists) so a re-launch resumes.
# Launch detached:  tmux new-session -d -s fusion 'bash ~/cd4-fusion/run_fusion_pipeline.sh > ~/cd4-fusion/fusion_run.log 2>&1'
set -uo pipefail

ROOT=/home/ubuntu/cd4-fusion
DR=/home/ubuntu/cd4-perturb-data
PY=/home/ubuntu/cd4-perturb-causal-jepa/.venv/bin/python
export CD4_DATA_ROOT="$DR"
export HF_HUB_DISABLE_TELEMETRY=1
STATUS="$ROOT/FUSION_STATUS.txt"
cd "$ROOT" || { echo "no ROOT"; exit 1; }

say(){ echo "=== $(date -u '+%F %T')  $* ==="; }
fail(){ echo "FUSION_STATUS=FAILED_$1"; echo "FAILED_$1 $(date -u '+%F %T')" > "$STATUS"; exit 1; }

echo "RUNNING $(date -u '+%F %T')" > "$STATUS"
say "PIPELINE START  data_root=$DR"

# ---- STAGE 1: pseudobulk restore (pinned to frozen HVG/split) --------------------------------
if $PY - <<'PY'
import os,sys,pandas as pd
try:
    d=pd.read_parquet(os.environ["CD4_DATA_ROOT"]+"/pseudobulk/train.parquet")
    n=d.index.get_level_values("pert_id").nunique()
    sys.exit(0 if n>1000 else 1)   # >1000 real perts = already restored
except Exception: sys.exit(1)
PY
then say "STAGE1 pseudobulk already real — skip"
else
  say "STAGE1 rebuild pseudobulk from CZI (pinned)"
  $PY -u scripts/rebuild_pseudobulk_frozen.py || fail STAGE1_PSEUDOBULK
  $PY - <<'PY' || exit 1
import os,pandas as pd
d=pd.read_parquet(os.environ["CD4_DATA_ROOT"]+"/pseudobulk/train.parquet")
n=d.index.get_level_values("pert_id").nunique()
print(f"[verify] train perts={n} cols={d.shape[1]}"); assert n>1000, "pseudobulk still small"
PY
  [ $? -eq 0 ] || fail STAGE1_VERIFY
fi

# ---- STAGE 2: ESM2 priors -------------------------------------------------------------------
if $PY - <<'PY'
import os,sys,pandas as pd
try:
    e=pd.read_parquet(os.environ["CD4_DATA_ROOT"]+"/embeddings/esm2.parquet")
    sys.exit(0 if (e.shape[0]==3000 and float(abs(e.to_numpy()).sum())>0) else 1)
except Exception: sys.exit(1)
PY
then say "STAGE2 esm2 priors already real — skip"
else
  say "STAGE2 build ESM2 priors (downloads ESM2-650M + peptide FASTA, GPU inference ~15min)"
  $PY -u scripts/build_priors.py || fail STAGE2_PRIORS
  $PY - <<'PY' || exit 1
import os,pandas as pd
e=pd.read_parquet(os.environ["CD4_DATA_ROOT"]+"/embeddings/esm2.parquet")
print(f"[verify] esm2 shape={e.shape} nonzero-rows={(e.to_numpy()!=0).any(1).sum()}")
assert e.shape[0]==3000, "esm2 not 3000 rows"
PY
  [ $? -eq 0 ] || fail STAGE2_VERIFY
fi

# ---- STAGE 3: re-train do-operator + twin (external regulators held out) + predict -----------
if [ -s "$ROOT/results/fusion_pred_causal.parquet" ] && [ -s "$ROOT/results/fusion_pred_noncausal.parquet" ]; then
  say "STAGE3 retrain predictions already exist — skip"
else
  say "STAGE3 re-train causal+twin (regulators held out) + predict (GPU, 40 epochs x2)"
  $PY -u scripts/fusion_retrain.py 40 || fail STAGE3_RETRAIN
  [ -s "$ROOT/results/fusion_pred_causal.parquet" ] || fail STAGE3_NOOUTPUT
fi

# ---- STAGE 4: powered G-F.2 -----------------------------------------------------------------
say "STAGE4 G-F.2 powered edge-recovery"
$PY -u scripts/fusion_gf2.py || fail STAGE4_GF2

say "PIPELINE DONE"
echo "SUCCESS $(date -u '+%F %T')" > "$STATUS"
echo "FUSION_PIPELINE_DONE"
