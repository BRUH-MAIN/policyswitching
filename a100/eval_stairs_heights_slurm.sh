#!/bin/bash
# Pinned eval of a stairs checkpoint at REAL riser heights (9 / 12 / 15 / 17 cm) and both
# treads (0.30 / 0.26 m), at the stage-0 command range (<= 1 m/s, what the follow task
# commands). pyramid_stairs spawns on the top platform and walks DOWN; pyramid_stairs_inv
# spawns in the pit and walks UP: read eval_checkpoint.py's by_terrain block, not the pooled
# number (a policy that cannot climb 12 cm still descends it).
#
#   CKPT=<path> LABEL=<name> sbatch a100/eval_stairs_heights_slurm.sh
#   (defaults: stairs v2's model_9999, label stairs_v2)  TASK=... selects the task cfg.
# Outputs: unitree_rl_mjlab/eval_results/stairs_heights/<LABEL>_h<cm>_w<cm>.json

#SBATCH --job-name=go2-stairs-heights
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/go2-stairs-heights-%j.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/go2-stairs-heights-%j.err

set -euo pipefail

REPO=/dist_home/d_palmani/c-08/policyswitching
PYTHON=/dist_home/d_palmani/.venvs/policyswitching-pas/bin/python3
TASK="${TASK:-Unitree-Go2-Spec-StairsV2}"
LABEL="${LABEL:-stairs_v2}"
CKPT="${CKPT:-$(ls -d $REPO/unitree_rl_mjlab/logs/rsl_rl/go2_spec_stairs_v2/*/ | head -1)model_9999.pt}"
OUT_DIR="$REPO/unitree_rl_mjlab/eval_results/stairs_heights"

export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/unitree_rl_mjlab:${PYTHONPATH:-}"
cd "$REPO/unitree_rl_mjlab"
mkdir -p "$OUT_DIR"
echo "[INFO] task=$TASK checkpoint=$CKPT label=$LABEL"

for W in 0.30 0.26; do
  for H in 0.09 0.12 0.15 0.17; do
    cm=$(python3 -c "print(round($H*100))"); wcm=$(python3 -c "print(round($W*100))")
    echo "================ riser ${cm} cm, tread ${wcm} cm ================"
    "$PYTHON" scripts/eval_checkpoint.py \
      --task "$TASK" --checkpoint "$CKPT" \
      --terrain stairs --step-height "$H" --step-width "$W" \
      --lin-vel-x -0.5 1.0 --lin-vel-y -0.5 0.5 \
      --num-envs 1024 --steps 1200 \
      --label "${LABEL}_h${cm}_w${wcm}" \
      --json-out "$OUT_DIR/${LABEL}_h${cm}_w${wcm}.json" \
      || echo "[ERROR] riser ${cm} tread ${wcm} failed (exit $?)"
  done
done
