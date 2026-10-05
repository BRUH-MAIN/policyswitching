#!/bin/bash
# Add the sensing-matched generalist (arm 1) to the switching comparison.
#
# Pulls go2_generalist/model_<ITER>.pt from the private HF repo into eval_ckpts/ (the
# cluster pushes every checkpoint there when HF_TOKEN is set at submission), then runs
# it as a fixed policy next to the on-time switch and the best fixed specialist on the
# confirmation seeds, with observation noise off (as pre-registered) and on.
#
# Usage (from anywhere):
#   unitree_rl_mjlab/scripts/switch_follow_generalist.sh          # final checkpoint, 9999
#   ITER=200 SEEDS="9" NUM_ENVS=16 TAG=codecheck unitree_rl_mjlab/scripts/switch_follow_generalist.sh
#   EXPERIMENT=go2_generalist_v2 TAG=generalistv2 STAIRS_CKPT=eval_ckpts/go2_spec_stairs_v2/model_9999.pt \
#       unitree_rl_mjlab/scripts/switch_follow_generalist.sh   # another generalist, another stairs slot
#
# Only model_9999 is a result. An earlier iteration is a code check of the loading path,
# or a look at a half-trained policy; TAG keeps its output apart from the real files.
set -euo pipefail

ITER="${ITER:-9999}"
SEEDS="${SEEDS:-500 501 502}"
NUM_ENVS="${NUM_ENVS:-256}"
TAG="${TAG:-generalist}"
HF_REPO="${HF_REPO:-RohanRamesh/go2-specialists}"
EXPERIMENT="${EXPERIMENT:-go2_generalist}"   # HF folder / eval_ckpts folder of the generalist
STAIRS_CKPT="${STAIRS_CKPT:-}"               # optional: checkpoint for the stairs slot of the bank

[ -n "$STAIRS_CKPT" ] && STAIRS_CKPT="$(realpath "$STAIRS_CKPT")"   # before the cd below: it may be relative
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$W/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/unitree_rl_mjlab/bin/python}"
CK="${CKPT_ROOT:-$W/eval_ckpts}"
CKPT="$CK/$EXPERIMENT/model_${ITER}.pt"
OUT="$W/eval_results/switch_follow"

# Same .env handling as coordination/scripts/laptop_pull_and_eval.sh.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi

if [ ! -f "$CKPT" ]; then
  echo "==> fetching $EXPERIMENT/model_${ITER}.pt from hf.co/$HF_REPO"
  "$PY" -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('$HF_REPO', '$EXPERIMENT/model_${ITER}.pt', local_dir='$CK'))"
fi

cd "$W"
export PYTHONPATH="$W" MUJOCO_GL=egl
STAIRS_ARGS=()
[ -n "$STAIRS_CKPT" ] && STAIRS_ARGS=(--extra-policy "stairs=$STAIRS_CKPT")
ARMS="fixed:generalist hard:0.3:label fixed:stairs"
for cond in clean noisy; do
  flag=""; [ "$cond" = noisy ] && flag="--obs-noise"
  for s in $SEEDS; do
    "$PY" scripts/switch_follow.py --seed "$s" --num-envs "$NUM_ENVS" --arms $ARMS $flag --resume \
      --extra-policy "generalist=$CKPT" "${STAIRS_ARGS[@]}" --ckpt-root "$CK" \
      --json-out "$OUT/${TAG}_${cond}_s$s.json" 2>&1 | grep -a 'RESULT\|SKIP\|Error\|Traceback\|  File' || true
  done
  echo "=========== $TAG, observation noise: $cond ==========="
  # shellcheck disable=SC2046
  "$PY" scripts/switch_follow_analyze.py confirm $(for s in $SEEDS; do echo "$OUT/${TAG}_${cond}_s$s.json"; done) \
    --pairs "hard:0.3:label>fixed:generalist" "fixed:generalist>fixed:stairs"
done
