#!/bin/bash
# Train ONE stock-PPO locomotion policy: a terrain specialist, or the matched generalist.
#
# Usage:
#   SPEC=Stairs sbatch a100/train_specialist_slurm.sh
#   SPEC=Generalist sbatch a100/train_specialist_slurm.sh
#   SPEC=GapsWarm sbatch a100/train_specialist_slurm.sh          # warm-starts from Rough
#   SPEC=Flat SEED=1 sbatch a100/train_specialist_slurm.sh       # extra seed -> go2_spec_flat_s1
#
# SPEC:
#   Flat | Rough | Stairs | Gaps   -> Unitree-Go2-Spec-<SPEC>, experiment go2_spec_<spec>
#   GapsWarm                       -> 100% stepping_stones, warm-started (see INIT_FROM)
#   Generalist                     -> Unitree-Go2-Generalist, experiment go2_generalist:
#                                     all four terrain classes, otherwise identical to a
#                                     specialist (obs, rewards, runner, budget)
#
# Optional env:
#   SEED       default 42 (mjlab's default, used by every run so far). Any other seed gets
#              its own experiment dir (_s<SEED>) so resume never mixes seeds.
#   INIT_FROM  checkpoint to warm-start from (a100/warm_start_ckpt.py). Defaults to the
#              Rough specialist for GapsWarm; unset for everything else. Only applied when
#              the experiment has no checkpoint yet.
#   HF_TOKEN   if set, every checkpoint is also pushed to hf.co/$HF_CHECKPOINT_REPO
#              (default RohanRamesh/go2-pas-saro) under <experiment>/ -- the off-cluster
#              backup specialists never had. Unset = local disk only.
#   BUDGET, NUM_ENVS, SAVE_INTERVAL, GPU_IDS, REPO_DIR, VENV_DIR
#
# Deliberately requests a 1-day walltime rather than the 3 days train_pas_slurm.sh uses.
# Scheduling on this cluster is pure age-based FIFO (the FAIRSHARE component of sprio is
# 0 for every user), and a long job can only start when a correspondingly long GPU window
# opens, so a 3-day request sits unschedulable far longer than the extra hours are worth.
# A run needs ~10k iterations (~18h at the ~550 it/h measured on the PAS runs), and
# resuming is free -- local_ckpt_resume.py tracks an ABSOLUTE budget, so just resubmit
# this script until it prints "budget reached".
#
# Resumes from LOCAL disk, not HF (see local_ckpt_resume.py for why) -- the HF push is a
# backup, never the resume source.

#SBATCH --job-name=go2-spec
#SBATCH --partition=workq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/go2-spec-%j.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/go2-spec-%j.err

set -euo pipefail

: "${SPEC:?Set SPEC to one of Flat, Rough, Stairs, Gaps, GapsWarm, Generalist (e.g. SPEC=Stairs sbatch ...)}"
case "$SPEC" in
  Flat|Rough|Stairs|Gaps|GapsWarm)
    TASK="Unitree-Go2-Spec-${SPEC}"
    EXPERIMENT_NAME="go2_spec_$(echo "$SPEC" | tr '[:upper:]' '[:lower:]')"
    ;;
  Generalist)
    TASK="Unitree-Go2-Generalist"
    EXPERIMENT_NAME="go2_generalist"
    ;;
  *) echo "[ERROR] SPEC must be one of Flat, Rough, Stairs, Gaps, GapsWarm, Generalist (got '$SPEC')" >&2; exit 1 ;;
esac

REPO_DIR="${REPO_DIR:-/dist_home/d_palmani/c-08/policyswitching}"
MJLAB_DIR="$REPO_DIR/unitree_rl_mjlab"
A100_DIR="$REPO_DIR/a100"

SEED="${SEED:-42}"
if [ "$SEED" != "42" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_s${SEED}"
fi

if [ "$SPEC" = "GapsWarm" ] && [ -z "${INIT_FROM:-}" ]; then
  INIT_FROM="$MJLAB_DIR/logs/rsl_rl/go2_spec_rough/2026-09-06_12-07-49/model_9999.pt"
fi

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

if [ -n "${HF_TOKEN:-}" ]; then
  export HF_CHECKPOINT_REPO="${HF_CHECKPOINT_REPO:-RohanRamesh/go2-pas-saro}"
  export HF_CHECKPOINT_STAGE="$EXPERIMENT_NAME"
  echo "[INFO] Checkpoints will also be pushed to hf.co/$HF_CHECKPOINT_REPO/$HF_CHECKPOINT_STAGE/"
else
  echo "[WARN] HF_TOKEN unset: checkpoints stay on local disk only (no off-cluster backup)."
fi

nvidia-smi
"$PY" --version
echo "[INFO] task=$TASK experiment=$EXPERIMENT_NAME seed=$SEED budget=$BUDGET envs=$NUM_ENVS"

if [ -n "${INIT_FROM:-}" ]; then
  "$PY" "$A100_DIR/warm_start_ckpt.py" \
    --source "$INIT_FROM" \
    --experiment-name "$EXPERIMENT_NAME" \
    --mjlab-dir "$MJLAB_DIR"
fi

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
  --agent.seed "$SEED" \
  --gpu-ids "$GPU_IDS" \
  $EXTRA_ARGS

echo "[INFO] Done. Resubmit the same command to continue if the budget wasn't reached."
