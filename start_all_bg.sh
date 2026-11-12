#!/usr/bin/env bash
# Start a runner script in the background, detached from this terminal: it keeps
# running if you close the terminal, VS Code, or the SSH session.
#
# Usage:
#   RUNNER=run_a2.sh ./start_all_bg.sh                   # A2: filtered + K-fold retrieval (EN, then multi)
#   WIKI_CONTACT="you@example.org" ./start_all_bg.sh     # default RUNNER=run_all.sh (KB build + A0 + A1)
#   SKIP="1_a2_en" RUNNER=run_a2.sh ./start_all_bg.sh    # skip steps by name
#
# Logs go to logs/<runner>_<timestamp>/ and logs/<runner>_latest points to the newest:
#   tail -f logs/run_a2_latest/master.log      (overall progress + F1 of each step)
#   tail -f logs/run_a2_latest/<step>.log      (full output of one step)
# Stop:  kill -- -$(cat logs/<runner>.pid)     (stops the whole run incl. python)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

RUNNER="${RUNNER:-run_all.sh}"
[[ -f "$RUNNER" ]] || { echo "No such runner: $RUNNER"; exit 1; }
NAME="$(basename "$RUNNER" .sh)"

mkdir -p logs
for pidf in logs/*.pid; do   # one GPU job at a time
  [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null && {
    echo "A run is already active: $pidf (PID $(cat "$pidf")). Stop it first: kill -- -\$(cat $pidf)"
    exit 1; }
done
PIDF="logs/${NAME}.pid"

export RUN_DIR="logs/${NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
ln -sfn "$(basename "$RUN_DIR")" "logs/${NAME}_latest"

setsid nohup bash "./$RUNNER" > "$RUN_DIR/master.log" 2>&1 < /dev/null &
echo $! > "$PIDF"

echo "Started $RUNNER (PID $(cat "$PIDF")), logs in $RUN_DIR/"
echo "  overall : tail -f logs/${NAME}_latest/master.log"
echo "  steps   : ls logs/${NAME}_latest/"
echo "  stop    : kill -- -\$(cat $PIDF)"
if [[ "$NAME" == "run_all" && -z "${WIKI_CONTACT:-}" ]]; then
  echo "  note    : WIKI_CONTACT not set -> Wikipedia allows only ~10 req/min; KB build will take ~20 h"
fi
exit 0
