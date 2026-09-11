#!/bin/bash
# Train ONE terrain-specialist policy (Unitree-Go2-Spec-{Flat,Rough,Stairs,Gaps}).
#
# Usage:
#   SPEC=Stairs sbatch a100/train_specialist_slurm.sh
#   SPEC=Gaps BUDGET=15000 sbatch a100/train_specialist_slurm.sh
#
# Deliberately requests a 1-day walltime rather than the 3 days train_pas_slurm.sh uses.
# Scheduling on this cluster is pure age-based FIFO (the FAIRSHARE component of sprio is
# 0 for every user), and a long job can only start when a correspondingly long GPU window
# opens, so a 3-day request sits unschedulable far longer than the extra hours are worth.
# A specialist needs ~10k iterations (~18h at the ~550 it/h measured on the PAS runs), and
# resuming is free -- local_ckpt_resume.py tracks an ABSOLUTE budget, so just resubmit
# this script until it prints "budget reached".
#
# Unlike PAS this is stock single-stage PPO: no terrain encoder, no privileged state, no
# annealing. A specialist only ever sees one terrain, so it has nothing to disambiguate.
#
# Resumes from LOCAL disk, not HF (see local_ckpt_resume.py for why). This means, unlike
# train_pas_slurm.sh, no HF_TOKEN is needed and checkpoints have no off-cluster backup --
# fine on this persistent cluster filesystem, but note it if this is ever adapted for
# ephemeral compute (Kaggle etc).

#SBATCH --job-name=go2-spec
#SBATCH --partition=workq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/go2-spec-%j.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/go2-spec-%j.err

set -euo pipefail

: "${SPEC:?Set SPEC to one of Flat, Rough, Stairs, Gaps (e.g. SPEC=Stairs sbatch ...)}"
case "$SPEC" in
  Flat|Rough|Stairs|Gaps) ;;
  *) echo "[ERROR] SPEC must be one of Flat, Rough, Stairs, Gaps (got '$SPEC')" >&2; exit 1 ;;
esac

TASK="Unitree-Go2-Spec-${SPEC}"
SPEC_LOWER=$(echo "$SPEC" | tr '[:upper:]' '[:lower:]')
EXPERIMENT_NAME="go2_spec_${SPEC_LOWER}"

REPO_DIR="${REPO_DIR:-/dist_home/d_palmani/c-08/policyswitching}"
MJLAB_DIR="$REPO_DIR/unitree_rl_mjlab"
A100_DIR="$REPO_DIR/a100"

NUM_ENVS="${NUM_ENVS:-8192}"
BUDGET="${BUDGET:-10000}"          # absolute iteration budget (stock Go2 default is 10001)
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
GPU_IDS="${GPU_IDS:-[0]}"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/policyswitching-pas}"
PY="$VENV_DIR/bin/python3"
if [[ ! -x "$PY" ]]; then
  echo "[ERROR] venv missing at $VENV_DIR -- run train_pas_slurm.sh once to bootstrap it." >&2
  exit 1
fi
export PATH="$VENV_DIR/bin:$PATH"

# Same rationale as train_pas_slurm.sh: unbuffered so per-iteration logs land in the .out
# as they happen, and expandable_segments so PyTorch's allocator doesn't fragment the
# mempool that Warp's CUDA-graph replay draws from (this is what killed job 11767).
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
export PYTHONPATH="$MJLAB_DIR:${PYTHONPATH:-}"

nvidia-smi
"$PY" --version
echo "[INFO] task=$TASK experiment=$EXPERIMENT_NAME budget=$BUDGET envs=$NUM_ENVS"

EXTRA_ARGS=$("$PY" "$A100_DIR/local_ckpt_resume.py" \
  --budget "$BUDGET" \
  --experiment-name "$EXPERIMENT_NAME" \
  --mjlab-dir "$MJLAB_DIR")

if [ "$EXTRA_ARGS" = "SKIP" ]; then
  echo "[INFO] $TASK already reached its ${BUDGET}-iteration budget -- nothing to do."
  exit 0
fi

cd "$MJLAB_DIR"
# shellcheck disable=SC2086
"$PY" scripts/train.py "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.save-interval "$SAVE_INTERVAL" \
  --agent.logger tensorboard \
  --agent.experiment-name "$EXPERIMENT_NAME" \
  --gpu-ids "$GPU_IDS" \
  $EXTRA_ARGS

echo "[INFO] Done. Resubmit the same command to continue if the budget wasn't reached."
