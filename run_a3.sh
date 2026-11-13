#!/usr/bin/env bash
# Stage A3 run: A2 retrieval (filtered, K-fold train context) + noise-robust training.
# Normally started via:  RUNNER=run_a3.sh ./start_all_bg.sh
# (survives closing the terminal). Foreground: ./run_a3.sh
#
# Steps (sequential, one GPU):
#   1_a0_zh     ZH-Chinese only, no retrieval (baseline for the ZH-only check)
#   2_a2_zh     ZH-Chinese only, A2 retrieval with the context-truncation fix
#               (old A1/A2 runs scored only ~3% of ZH test sentences; other languages were unaffected)
#   3_a3_en     EN-English, A1+A2+A3
#   4_a3_multi  all 12 languages, A1+A2+A3
# Compare with: A2-EN 0.6316, A0-multi 0.7042, A2-multi (ZH re-scored: outputs/a2_multi_kfold/pooled_test_metrics_rescored.json)
#
# Each step's full output -> $RUN_DIR/<step>.log; summary -> stdout (master.log).
# Env: PY=.venv/bin/python  GPU=0  RUN_DIR=logs/run_a3_<ts>  SKIP="1_a0_zh 2_a2_zh"
#      MIN_SCORE=1.0  MAX_ENTRIES=3  FOLDS=5  NOISE_RATE=0.3  NOISE_TYPES=typo,swap,drop,insert
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PY="${PY:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-logs/run_a3_$(date +%Y%m%d_%H%M%S)}"
SKIP="${SKIP:-}"
MIN_SCORE="${MIN_SCORE:-1.0}"
MAX_ENTRIES="${MAX_ENTRIES:-3}"
FOLDS="${FOLDS:-5}"
NOISE_RATE="${NOISE_RATE:-0.3}"
NOISE_TYPES="${NOISE_TYPES:-typo,swap,drop,insert}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TQDM_MININTERVAL=30

COMMON=(--epochs 5 --batch-size 16 --eval-batch-size 128 --amp)
A2_ARGS=(--use-retrieval --kb-index-dir ./kb_index
         --retrieval-min-score "$MIN_SCORE" --max-context-entries "$MAX_ENTRIES"
         --skip-low-confidence --kb-train-folds "$FOLDS" "${COMMON[@]}")
A3_ARGS=(--noise-training --noise-rate "$NOISE_RATE" --noise-types "$NOISE_TYPES"
         --coarse-head)

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
note "Run directory: $RUN_DIR   (PID $$)   min_score=$MIN_SCORE max_entries=$MAX_ENTRIES folds=$FOLDS noise=$NOISE_RATE/$NOISE_TYPES"

run_step 1_a0_zh "$PY" -m src.multiconer_ner.main --language ZH-Chinese \
  "${COMMON[@]}" --experiment-name a0_zh
ZH0_RC=$?

run_step 2_a2_zh "$PY" -m src.multiconer_ner.main --language ZH-Chinese \
  "${A2_ARGS[@]}" --experiment-name a2_zh_fixed
ZH2_RC=$?

run_step 3_a3_en "$PY" -m src.multiconer_ner.main --language EN-English \
  "${A2_ARGS[@]}" "${A3_ARGS[@]}" --experiment-name a1a2a3_en
EN_RC=$?

run_step 4_a3_multi "$PY" -m src.multiconer_ner.main --multilingual \
  "${A2_ARGS[@]}" "${A3_ARGS[@]}" --experiment-name a1a2a3_multi
MULTI_RC=$?

note "ALL FINISHED. exit codes: a0_zh=$ZH0_RC a2_zh=$ZH2_RC a3_en=$EN_RC a3_multi=$MULTI_RC"
note "Models: models/{a0_zh,a2_zh_fixed,a1a2a3_en,a1a2a3_multi}  Predictions: outputs/<same>  TensorBoard: runs/"
