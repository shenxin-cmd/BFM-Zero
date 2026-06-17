#!/usr/bin/env bash
# Start inference + z-viz inside tmux (session stays open on success or failure).
#
# Run from anywhere:
#   bash scripts/tmux_track_viz.sh
#   bash scripts/tmux_track_viz.sh results/main-combined-splitz-mse-324-64
#
# Optional env:
#   UV_BIN=../../uv_binary
#   TMUX_SESSION=track-viz

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FOLDER="${1:-results/main-combined-splitz-mse-324-64}"
SESSION="${TMUX_SESSION:-track-viz}"
mkdir -p "$REPO_ROOT/logs"
LOG="$REPO_ROOT/logs/track_viz_$(date +%Y%m%d_%H%M%S).log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found in PATH" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Attach with:"
  echo "  tmux attach -t $SESSION"
  exit 1
fi

# NOTE: do NOT use placeholder paths like /path/to/BFM-Zero — this script resolves REPO_ROOT automatically.
RUN="
set -e
cd '$REPO_ROOT'
export UV_BIN=\"\${UV_BIN:-../../uv_binary}\"
echo '=== repo:' \$(pwd)
echo '=== log:  $LOG'
echo
bash scripts/run_tracking_inference_and_viz.sh '$MODEL_FOLDER' 2>&1 | tee '$LOG'
ec=\${PIPESTATUS[0]}
echo
echo '=== finished exit='\$ec' ==='
echo 'log: $LOG'
exec bash
"

tmux new-session -s "$SESSION" bash -lc "$RUN"

echo "Started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    $LOG"
