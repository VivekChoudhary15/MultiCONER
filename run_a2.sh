#!/usr/bin/env bash
# Stage A2 run: filtered retrieval + leakage-free (K-fold) training context.
# Normally started via:  RUNNER=run_a2.sh ./start_all_bg.sh
# (survives closing the terminal). Foreground: ./run_a2.sh
#
# Steps (sequential, one GPU):
#   1_a2_en     EN-English
#   2_a2_multi  all 12 languages pooled
# Compare with: A0-EN 0.628 / A1-EN 0.581, A0-multi 0.704 / A1-multi 0.653 (pooled test F1)
#
# Each step's full output -> $RUN_DIR/<step>.log; summary -> stdout (master.log).
# Env: PY=.venv/bin/python  GPU=0  RUN_DIR=logs/run_a2_<ts>  SKIP="1_a2_en"
#      MIN_SCORE=1.0  MAX_ENTRIES=3  FOLDS=5
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PY="${PY:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-logs/run_a2_$(date +%Y%m%d_%H%M%S)}"
SKIP="${SKIP:-}"
MIN_SCORE="${MIN_SCORE:-1.0}"
MAX_ENTRIES="${MAX_ENTRIES:-3}"
FOLDS="${FOLDS:-5}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TQDM_MININTERVAL=30

A2_ARGS=(--use-retrieval --kb-index-dir ./kb_index
         --retrieval-min-score "$MIN_SCORE" --max-context-entries "$MAX_ENTRIES"
         --skip-low-confidence --kb-train-folds "$FOLDS"
         --epochs 5 --batch-size 16)

note() { echo "[$(date '+%F %T')] $*"; }

run_step() {   # run_step <name> <cmd...>
  local name=$1; shift
  local log="$RUN_DIR/$name.log"
  if [[ " $SKIP " == *" $name "* ]]; then note "SKIP  $name"; return 0; fi
  note "START $name  ->  $log"
  { echo "=== $(date) | host $(hostname) | GPU $CUDA_VISIBLE_DEVICES | git $(git rev-parse --short HEAD)$(git diff --quiet || echo '+local-changes')"
    echo "=== $*"; } > "$log"
  local t0=$SECONDS
  "$@" >> "$log" 2>&1
  local rc=$?
  local mins=$(( (SECONDS - t0) / 60 ))
  if [[ $rc -eq 0 ]]; then
    note "DONE  $name  (${mins} min)"
    grep -E "POOLED test|test -> precision" "$log" | sed 's/^/        /' | tail -n 13
  else
    note "FAIL  $name  exit=$rc  (${mins} min) -- last lines of $log:"
    tail -n 25 "$log" | sed 's/^/        /'
  fi
  return $rc
}

if [[ ! -f kb_index/entries.json ]]; then
  note "kb_index/entries.json missing -- build the KB first (scripts/build_kb.py)."; exit 1
fi
note "Run directory: $RUN_DIR   (PID $$)   min_score=$MIN_SCORE max_entries=$MAX_ENTRIES folds=$FOLDS"

run_step 1_a2_en "$PY" -m src.multiconer_ner.main --language EN-English \
  "${A2_ARGS[@]}" --experiment-name a2_en_kfold
EN_RC=$?

run_step 2_a2_multi "$PY" -m src.multiconer_ner.main --multilingual \
  "${A2_ARGS[@]}" --experiment-name a2_multi_kfold
MULTI_RC=$?

note "ALL FINISHED. exit codes: a2_en=$EN_RC a2_multi=$MULTI_RC"
note "Models: models/{a2_en_kfold,a2_multi_kfold}  Predictions: outputs/<same>  TensorBoard: runs/"
