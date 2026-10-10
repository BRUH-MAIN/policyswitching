#!/bin/bash
# Qualitative clips for the presentation: the same course (flat, rough, 10- and 5-step flights
# at RISER m) recorded with onboard-classifier switching, oracle switching, the matched
# generalist alone and stairs v8b alone. scripts/record_switching_demo.py does the rendering.
#   RISER=0.15 SEED=3 sbatch a100/present_videos_slurm.sh
#SBATCH --job-name=present-video
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/present-video-%j.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/present-video-%j.err
set -uo pipefail
REPO=/dist_home/d_palmani/c-08/policyswitching
W="$REPO/unitree_rl_mjlab"
PY=/dist_home/d_palmani/.venvs/policyswitching-pas/bin/python3
RL="$W/logs/rsl_rl"
RISER="${RISER:-0.15}"; SEED="${SEED:-3}"
OUT="$REPO/report_content/videos/presentation"; mkdir -p "$OUT"
BANK=(--flat "$RL/go2_spec_flat/2026-09-06_17-17-15/model_9999.pt" --rough "$RL/go2_spec_rough/2026-09-06_12-07-49/model_9999.pt"
      --stairs "$RL/go2_spec_stairs_v8b/2026-10-09_12-49-32/model_1999.pt")
GENV3="$RL/go2_generalist_v3/2026-10-10_20-36-39/model_3999.pt"
CLF="$W/logs/switch_follow/clf_v8b.pt"
cd "$W"; export PYTHONPATH="$W" MUJOCO_GL=egl PYTHONUNBUFFERED=1
tag="h$("$PY" -c "print(round($RISER*100))")_s$SEED"
common=(--step-height "$RISER" --seed "$SEED" "${BANK[@]}")
"$PY" scripts/record_switching_demo.py "${common[@]}" --classifier "$CLF" --out "$OUT/switching_classifier_$tag.mp4" || echo "FAILED classifier"
"$PY" scripts/record_switching_demo.py "${common[@]}" --out "$OUT/switching_oracle_$tag.mp4" || echo "FAILED oracle"
"$PY" scripts/record_switching_demo.py "${common[@]}" --single generalist --generalist "$GENV3" --out "$OUT/generalist_v3_$tag.mp4" || echo "FAILED generalist"
"$PY" scripts/record_switching_demo.py "${common[@]}" --single stairs --out "$OUT/stairs_v8b_alone_$tag.mp4" || echo "FAILED stairs"
echo "[DONE] videos $tag"
