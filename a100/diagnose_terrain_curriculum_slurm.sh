#!/bin/bash
# Per-row promote/demote diagnostic of the terrain curriculum for one checkpoint
# (unitree_rl_mjlab/scripts/diagnose_terrain_curriculum.py). Read-only: no training, no
# upload. ~5-10 minutes on one GPU of any type.
#
#   sbatch a100/diagnose_terrain_curriculum_slurm.sh
#   TASK=Unitree-Go2-Spec-StairsV2 CKPT=<path> LABEL=<name> sbatch a100/diagnose_terrain_curriculum_slurm.sh

#SBATCH --job-name=go2-terrain-diag
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/go2-terrain-diag-%j.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/go2-terrain-diag-%j.err

set -euo pipefail

REPO=/dist_home/d_palmani/c-08/policyswitching
PYTHON=/dist_home/d_palmani/.venvs/policyswitching-pas/bin/python3
TASK="${TASK:-Unitree-Go2-Spec-StairsV2}"
LABEL="${LABEL:-stairs_v2}"
CKPT="${CKPT:-$(ls -d $REPO/unitree_rl_mjlab/logs/rsl_rl/go2_spec_stairs_v2/*/ | head -1)model_9999.pt}"

export MUJOCO_GL=egl
export PYTHONPATH="$REPO/unitree_rl_mjlab:${PYTHONPATH:-}"
cd "$REPO/unitree_rl_mjlab"
echo "[INFO] task=$TASK checkpoint=$CKPT label=$LABEL"
"$PYTHON" scripts/diagnose_terrain_curriculum.py \
  --task "$TASK" --checkpoint "$CKPT" \
  --num-envs "${NUM_ENVS:-2048}" --steps "${STEPS:-4000}" \
  --json-out "$REPO/unitree_rl_mjlab/eval_results/terrain_curriculum/${LABEL}.json"
