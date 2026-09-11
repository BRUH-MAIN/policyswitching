#!/bin/bash
# Headless, deterministic numeric eval of a PAS checkpoint (see scripts/eval_checkpoint.py).
# sbatch version of the srun eval command -- queues instead of blocking/getting cancelled
# while pending on a busy cluster.
#
# By default this pulls the LATEST checkpoint for --hf-stage from the HF model repo rather
# than a path baked in here: a stale hardcoded checkpoint (stage1_model_31800.pt, saved
# mid-divergence before the clip_actions fix) is exactly how a policy that never walked
# got evaluated and reported as "80.5% survival, +39.5 return".
#
# Override the checkpoint by exporting EVAL_CHECKPOINT=<local .pt path> before sbatch,
# or the stage with EVAL_STAGE=stage1.

#SBATCH --job-name=go2-pas-eval-numeric
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/go2-pas-eval-numeric-%j.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/go2-pas-eval-numeric-%j.err

set -euo pipefail

REPO=/dist_home/d_palmani/c-08/policyswitching
PYTHON=/dist_home/d_palmani/.venvs/policyswitching-pas/bin/python3

export MUJOCO_GL=egl
export PYTHONPATH="$REPO/unitree_rl_mjlab:${PYTHONPATH:-}"

cd "$REPO/unitree_rl_mjlab"

if [ -n "${EVAL_CHECKPOINT:-}" ]; then
  CKPT_ARGS=(--checkpoint "$EVAL_CHECKPOINT")
else
  # The model repo currently reads anonymously, so HF_TOKEN is not required here --
  # only warn, so a missing token can't block an otherwise-working eval.
  [ -z "${HF_TOKEN:-}" ] && echo "[WARN] HF_TOKEN unset; relying on anonymous read access."
  CKPT_ARGS=(--hf-repo RohanRamesh/go2-pas-saro --hf-stage "${EVAL_STAGE:-stage2}")
fi

# EVAL_EXTRA_ARGS forwards flags to eval_checkpoint.py, e.g.
#   EVAL_EXTRA_ARGS="--anneal-prob 0.0"   -> estimator-only (proprioception) eval
read -r -a EXTRA_ARGS <<< "${EVAL_EXTRA_ARGS:-}"

"$PYTHON" scripts/eval_checkpoint.py \
  "${CKPT_ARGS[@]}" \
  --num-envs 1024 --steps 1200 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
