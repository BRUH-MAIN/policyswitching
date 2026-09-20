#!/bin/bash
# Step 1 discriminating test for the straight-staircase failure: evaluate the existing
# Stairs specialist (model_9999.pt) on PINNED PYRAMID stairs at difficulty 0.5 / 0.7 / 0.9
# (~0.05 / 0.07 / 0.09 m risers). Fails on pyramids too -> the specialist is weak on
# stairs generally; fine on pyramids but bad on the straight course -> it overfit to
# pyramid geometry.
#
# Pinned conditions only: NO --keep-curricula (see CLAUDE.md), terrain curriculum off,
# command range = final training stage. Results land in
# unitree_rl_mjlab/eval_results/stairs_pyramid/ as one JSON per difficulty.
#
# Short walltime on purpose: this cluster schedules pure age-based FIFO with backfill,
# so a 30-minute request starts sooner than a long one. Runs on any GPU type.
#
#   sbatch a100/eval_stairs_pyramid_slurm.sh
#   CKPT=<other .pt> LABEL=<name> sbatch a100/eval_stairs_pyramid_slurm.sh

#SBATCH --job-name=go2-stairs-pyr-eval
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/go2-stairs-pyr-eval-%j.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/go2-stairs-pyr-eval-%j.err

set -euo pipefail

REPO=/dist_home/d_palmani/c-08/policyswitching
PYTHON=/dist_home/d_palmani/.venvs/policyswitching-pas/bin/python3
CKPT="${CKPT:-$REPO/unitree_rl_mjlab/logs/rsl_rl/go2_spec_stairs/2026-09-05_22-37-43/model_9999.pt}"
LABEL="${LABEL:-go2_spec_stairs}"
OUT_DIR="$REPO/unitree_rl_mjlab/eval_results/stairs_pyramid"

export MUJOCO_GL=egl
export PYTHONPATH="$REPO/unitree_rl_mjlab:${PYTHONPATH:-}"

cd "$REPO/unitree_rl_mjlab"
mkdir -p "$OUT_DIR"

# A failing difficulty must not stop the others from running.
for D in 0.5 0.7 0.9; do
  echo "================ difficulty $D ================"
  "$PYTHON" scripts/eval_checkpoint.py \
    --task Unitree-Go2-Spec-Stairs --checkpoint "$CKPT" \
    --terrain stairs --difficulty "$D" \
    --num-envs 1024 --steps 1200 \
    --label "${LABEL}_pyramid_d${D}" \
    --json-out "$OUT_DIR/${LABEL}_pyramid_d${D}.json" \
    || echo "[ERROR] difficulty $D failed (exit $?)"
done
