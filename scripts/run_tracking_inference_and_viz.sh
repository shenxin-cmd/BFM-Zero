#!/usr/bin/env bash
# Run tracking_inference_split, then batch z-sphere plots for every processed clip.
#
# Usage (repo root):
#   bash scripts/run_tracking_inference_and_viz.sh
#   bash scripts/run_tracking_inference_and_viz.sh results/my-checkpoint
#
# tmux one-liner:
#   tmux new -s track-viz 'cd /path/to/BFM-Zero && bash scripts/run_tracking_inference_and_viz.sh'
#
# Environment overrides:
#   UV_BIN=../../uv_binary          # default: uv run
#   TRAJ_OBS_DIR=...                # default: shapes_v3
#   Z_BODY_DIM=324                  # must match checkpoint
#   SAVE_VIZ_MP4=1                  # also export sphere MP4 per clip (slow)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_FOLDER="${1:-results/main-combined-splitz-mse-324-64}"
TRAJ_OBS_DIR="${TRAJ_OBS_DIR:-humanoidverse/data/inference_clips/shapes_v3}"
Z_BODY_DIM="${Z_BODY_DIM:-324}"

if [[ -n "${UV_BIN:-}" ]]; then
  UV=( "$UV_BIN" run )
else
  UV=( uv run )
fi

OUT_ROOT="$MODEL_FOLDER/tracking_inference_split"
VIZ_ARGS=( --labels expert actual actual_smoothed --z-body-dim "$Z_BODY_DIM" --prefix z_compare )
if [[ "${SAVE_VIZ_MP4:-0}" == "1" ]]; then
  VIZ_ARGS+=( --save-mp4 )
fi

echo "=== [1/2] tracking_inference_split ==="
echo "  model:  $MODEL_FOLDER"
echo "  traj:   $TRAJ_OBS_DIR"
echo "  output: $OUT_ROOT"
echo

"${UV[@]}" -m humanoidverse.tracking_inference_split \
  --model-folder "$MODEL_FOLDER" \
  --traj-obs-dir "$TRAJ_OBS_DIR" \
  --traj-glob "**/*_obs.npz" \
  --one-per-shape-plane \
  --z-window 8 --z-ema-alpha 0.6 \
  --disable-dr --disable-obs-noise \
  --episode-len 300 --save-mp4

echo
echo "=== [2/2] z sphere visualization (all clips) ==="

mapfile -t CLIP_DIRS < <(
  find "$OUT_ROOT/clips" -type f -name 'z_expert.npz' 2>/dev/null \
    | sed 's|/z_expert\.npz$||' | sort -u
)

if [[ ${#CLIP_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: no clip dirs with z_expert.npz under $OUT_ROOT/clips" >&2
  exit 1
fi

n_ok=0
n_skip=0
for clip_dir in "${CLIP_DIRS[@]}"; do
  z_exp="$clip_dir/z_expert.npz"
  z_act="$clip_dir/z_actual.npz"
  z_smooth="$clip_dir/z_actual_smoothed.npz"
  if [[ ! -f "$z_act" || ! -f "$z_smooth" ]]; then
    echo "SKIP (missing z files): $clip_dir"
    n_skip=$((n_skip + 1))
    continue
  fi
  rel="${clip_dir#"$OUT_ROOT"/}"
  echo "[$((n_ok + 1))/${#CLIP_DIRS[@]}] viz $rel"
  "${UV[@]}" python visualize_z_sphere_multi.py \
    --pkl-files "$z_exp" "$z_act" "$z_smooth" \
    --out-dir "$clip_dir" \
    "${VIZ_ARGS[@]}"
  n_ok=$((n_ok + 1))
done

echo
echo "Done. inference → $OUT_ROOT"
echo "  z sphere PNG:  clips/*/*/*/z_compare_sphere.png  (${n_ok} clips)"
echo "  skipped:       ${n_skip}"
