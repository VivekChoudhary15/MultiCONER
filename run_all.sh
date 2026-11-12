#!/usr/bin/env bash
# Full Stage A0/A1 run. Normally started via ./start_all_bg.sh (survives closing
# the terminal). Can also be run in the foreground: ./run_all.sh
#
# Order:
#   1. Build KB index (CPU/network)  ┐ run in parallel (A0 needs no KB)
#   2. A0 baseline, EN               ┘
#   3. A1 retrieval, EN              (after 1+2 finish)
#   4. A1 retrieval, multilingual
#
# Each step writes its full terminal output to $RUN_DIR/<step>.log;
# start/finish/failure lines go to stdout (master.log when backgrounded).
#
# Env: PY=.venv/bin/python  GPU=0  WIKI_CONTACT=<email or URL for Wikipedia User-Agent>
#      RUN_DIR=logs/run_all_<timestamp>  SKIP="a0_en"  (space-separated step names to skip)
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PY="${PY:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-logs/run_all_$(date +%Y%m%d_%H%M%S)}"
SKIP="${SKIP:-}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TQDM_MININTERVAL=30   # progress bars update every 30 s -> small log files
export WIKI_CONTACT="${WIKI_CONTACT:-}"

KB_DIR=./kb_index
LANGS=(EN-English BN-Bangla HI-Hindi DE-German ES-Spanish FR-French IT-Italian PT-Portuguese SV-Swedish UK-Ukrainian FA-Farsi ZH-Chinese)

note() { echo "[$(date '+%F %T')] $*"; }

run_step() {   # run_step <name> <cmd...>
  local name=$1; shift
  local log="$RUN_DIR/$name.log"
  if [[ " $SKIP " == *" $name "* ]]; then note "SKIP  $name"; return 0; fi
  note "START $name  ->  $log"
  { echo "=== $(date) | host $(hostname) | GPU $CUDA_VISIBLE_DEVICES"; echo "=== $*"; } > "$log"
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

note "Run directory: $RUN_DIR   (PID $$)"

# 1 + 2 in parallel
run_step 1_build_kb "$PY" scripts/build_kb.py --languages "${LANGS[@]}" --output "$KB_DIR" &
KB_PID=$!

run_step 2_a0_en "$PY" -m src.multiconer_ner.main --language EN-English \
  --epochs 5 --batch-size 16 --experiment-name a0_en_baseline
A0_RC=$?

wait "$KB_PID"
KB_RC=$?

if [[ $KB_RC -ne 0 || ! -f "$KB_DIR/entries.json" ]]; then
  note "KB build failed -> skipping A1 steps. Fix it, then re-run: SKIP=\"2_a0_en\" ./start_all_bg.sh  (already-fetched titles are cached and reused)"
  exit 1
fi

run_step 3_a1_en "$PY" -m src.multiconer_ner.main --language EN-English \
  --use-retrieval --kb-index-dir "$KB_DIR" --epochs 5 --batch-size 16 \
  --experiment-name a1_en_retrieval
A1_EN_RC=$?

run_step 4_a1_multi "$PY" -m src.multiconer_ner.main --multilingual \
  --use-retrieval --kb-index-dir "$KB_DIR" --epochs 5 --batch-size 16 \
  --experiment-name a1_multi_retrieval
A1_MULTI_RC=$?

note "ALL FINISHED. exit codes: build_kb=$KB_RC a0_en=$A0_RC a1_en=$A1_EN_RC a1_multi=$A1_MULTI_RC"
note "Models: models/{a0_en_baseline,a1_en_retrieval,a1_multi_retrieval}  Predictions: outputs/<same>  TensorBoard: runs/"
