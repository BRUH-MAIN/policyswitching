#!/bin/bash
# Round-2 qualitative clips (Addendum 9): the onboard switch with the round-2 classifier (labels
# 0.8 m ahead) and bank R at 15 and 17 cm, and the round-1 classifier + old bank at 17 cm (where it
# deadlocked in 45% of trials). Same course builder and seed as a100/present_videos_slurm.sh.
#   SEED=3 sbatch --nice=10000 a100/present_videos_r2_slurm.sh
#SBATCH --job-name=present-video
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/present-video-%j.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/present-video-%j.err
set -uo pipefail
REPO=/dist_home/d_palmani/c-08/policyswitching
W="$REPO/unitree_rl_mjlab"
PY=/dist_home/d_palmani/.venvs/policyswitching-pas/bin/python3
RL="$W/logs/rsl_rl"; SEED="${SEED:-3}"
OUT="$REPO/report_content/videos/presentation"; mkdir -p "$OUT"
STAIRS="$RL/go2_spec_stairs_v8b/2026-10-09_12-49-32/model_1999.pt"
NEW=(--flat "$RL/go2_spec_flat_v8b/2026-10-11_09-37-31/model_1999.pt" --rough "$RL/go2_spec_rough_v8b/2026-10-11_10-48-57/model_1999.pt" --stairs "$STAIRS")
OLD=(--flat "$RL/go2_spec_flat/2026-09-06_17-17-15/model_9999.pt" --rough "$RL/go2_spec_rough/2026-09-06_12-07-49/model_9999.pt" --stairs "$STAIRS")
CLF2="$W/logs/switch_follow/clf_r2_lead08.pt"; CLF1="$W/logs/switch_follow/clf_v8b.pt"
cd "$W"; export PYTHONPATH="$W" MUJOCO_GL=egl PYTHONUNBUFFERED=1
for h in 15 17; do
  "$PY" scripts/record_switching_demo.py --step-height "0.$h" --seed "$SEED" "${NEW[@]}" --classifier "$CLF2" --hold 10 \
    --out "$OUT/r2_switching_classifier_h${h}_s$SEED.mp4" || echo "FAILED r2 $h"
done
"$PY" scripts/record_switching_demo.py --step-height 0.17 --seed "$SEED" "${OLD[@]}" --classifier "$CLF1" --hold 5 \
  --out "$OUT/r1_switching_classifier_h17_s$SEED.mp4" || echo "FAILED r1 17"
echo "[DONE] r2 videos"
