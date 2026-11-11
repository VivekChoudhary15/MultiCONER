#!/usr/bin/env bash
# Launch MultiCoNER training in the background (survives closing the terminal / SSH).
#
# Usage:
#   ./train_bg.sh en         # EN-English, xlm-roberta-base           (A0 baseline)
#   ./train_bg.sh multi      # all 12 languages pooled, xlm-roberta-base
#   ./train_bg.sh large      # EN-English, xlm-roberta-large
#   ./train_bg.sh mdeberta   # EN-English, microsoft/mdeberta-v3-base
#   ./train_bg.sh custom --language HI-Hindi --epochs 8 ...   # any main.py args
#
# Extra args after a preset are appended and override it (argparse: last wins), e.g.
#   ./train_bg.sh en --epochs 10 --experiment-name en_10ep
#
# Env vars:  GPU=0 (CUDA device)   PY=.venv/bin/python
#
# Monitor:   tail -f logs/<experiment>.log
# Stop:      kill $(cat logs/<experiment>.pid)
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PY="${PY:-.venv/bin/python}"
GPU="${GPU:-0}"
MODE="${1:-en}"
[[ $# -gt 0 ]] && shift

case "$MODE" in
  en)       EXP=a0_en_xlmr_base
            ARGS=(--language EN-English --epochs 5 --batch-size 16 --lr 3e-5 --max-length 256 --amp) ;;
  multi)    EXP=a0_multi_xlmr_base
            ARGS=(--multilingual --epochs 5 --batch-size 16 --lr 3e-5 --max-length 256 --amp --eval-every 2000) ;;
  large)    EXP=a0_en_xlmr_large
            ARGS=(--language EN-English --model xlm-roberta-large --epochs 5 --batch-size 16 --lr 1e-5 --max-length 256 --amp) ;;
  mdeberta) EXP=a0_en_mdeberta_base
            ARGS=(--language EN-English --model microsoft/mdeberta-v3-base --epochs 5 --batch-size 16 --lr 3e-5 --max-length 256 --amp) ;;
  custom)   EXP=custom ; ARGS=() ;;
  *) echo "Unknown mode '$MODE' (use: en | multi | large | mdeberta | custom)"; exit 1 ;;
esac
ARGS+=("$@")

# If the user passed --experiment-name, use it for the log/pid file names too.
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  [[ "${ARGS[$i]}" == "--experiment-name" ]] && EXP="${ARGS[$((i + 1))]}"
done
if [[ " ${ARGS[*]} " != *" --experiment-name "* ]]; then
  ARGS+=(--experiment-name "$EXP")
fi

mkdir -p logs
LOG="logs/${EXP}.log"
PIDF="logs/${EXP}.pid"

if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "Experiment '$EXP' is already running (PID $(cat "$PIDF")). Stop it first or use --experiment-name."
  exit 1
fi
[[ -f "$LOG" ]] && mv "$LOG" "logs/${EXP}.$(date +%Y%m%d_%H%M%S).log.bak"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TQDM_MININTERVAL=30   # keep the log file small (progress bar updates every 30 s)

{
  echo "=== $(date) | host $(hostname) | GPU $GPU"
  echo "=== $PY -m src.multiconer_ner.main ${ARGS[*]}"
} > "$LOG"

setsid nohup "$PY" -m src.multiconer_ner.main "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"

echo "Started '$EXP' (PID $(cat "$PIDF"))"
echo "  log : tail -f $LOG"
echo "  stop: kill \$(cat $PIDF)"
echo "  tb  : $PY -m tensorboard.main --logdir runs --port 6006"
