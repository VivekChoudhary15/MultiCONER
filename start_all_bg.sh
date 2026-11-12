#!/usr/bin/env bash
# Start run_all.sh in the background, detached from this terminal: it keeps
# running if you close the terminal, VS Code, or the SSH session.
#
# Usage:
#   WIKI_CONTACT="you@example.org" ./start_all_bg.sh     # recommended (200 req/min to Wikipedia)
#   ./start_all_bg.sh                                    # works, but KB build is ~20x slower
#   SKIP="2_a0_en" WIKI_CONTACT=... ./start_all_bg.sh    # skip steps (names: 1_build_kb 2_a0_en 3_a1_en 4_a1_multi)
#
# Monitor:  tail -f logs/run_all_latest/master.log         (overall progress)
#           tail -f logs/run_all_latest/<step>.log         (full output of one step)
# Stop:     kill -- -$(cat logs/run_all.pid)               (stops the whole run incl. python)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PIDF=logs/run_all.pid
mkdir -p logs
if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "A run is already active (PID $(cat "$PIDF")). Stop it first: kill -- -\$(cat $PIDF)"
  exit 1
fi

export RUN_DIR="logs/run_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
ln -sfn "$(basename "$RUN_DIR")" logs/run_all_latest

setsid nohup bash ./run_all.sh > "$RUN_DIR/master.log" 2>&1 < /dev/null &
echo $! > "$PIDF"

echo "Started full run (PID $(cat "$PIDF")), logs in $RUN_DIR/"
echo "  overall : tail -f logs/run_all_latest/master.log"
echo "  steps   : ls logs/run_all_latest/   (1_build_kb.log 2_a0_en.log 3_a1_en.log 4_a1_multi.log)"
echo "  stop    : kill -- -\$(cat $PIDF)"
[[ -z "${WIKI_CONTACT:-}" ]] && echo "  note    : WIKI_CONTACT not set -> Wikipedia allows only ~10 req/min; KB build will take ~20 h"
exit 0
