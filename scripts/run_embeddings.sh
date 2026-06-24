#!/usr/bin/env bash
# Standalone wrapper for the 100K embedding precompute.
# Run with: nohup bash scripts/run_embeddings.sh &
# Progress: tail -f /home/user/Documents/redrob-ranker/data/embeddings_nohup.log

set -euo pipefail

REPO=/home/user/Documents/redrob-ranker
LOG=$REPO/data/embeddings_nohup.log
PYTHON=/home/user/anaconda3/bin/python3

mkdir -p "$REPO/data"
exec > "$LOG" 2>&1   # redirect all stdout/stderr to log

echo "[$(date)] Starting embedding run from $REPO"

$PYTHON - <<'PYEOF'
import sys, time
sys.path.insert(0, '/home/user/Documents/redrob-ranker')
from src.precompute_features import run_embeddings

t = time.perf_counter()
print('START', flush=True)
jd_emb, cand_embs = run_embeddings()
elapsed = time.perf_counter() - t
print(f'DONE  elapsed={elapsed:.1f}s  shape={cand_embs.shape}', flush=True)
PYEOF

echo "[$(date)] Script finished."
