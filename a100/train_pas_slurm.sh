#!/bin/bash
# Train SARO's PAS policy for Go2 on a single-GPU SLURM node (college A100 cluster).
#
# Adapted from the college's sample `batch.sh` template + kaggle/train_pas.ipynb and
# a100/train_pas.ipynb (same install-version pins and single-GPU config), rebuilt for
# unattended, multi-job execution:
#
#   - This queue's walltime is capped (24h below) and a full 40000-iteration stage will
#     likely take longer than that, so this script is meant to be resubmitted with
#     `sbatch train_pas_slurm.sh` repeatedly until both stages are done. Each run checks
#     Hugging Face for the latest checkpoint and continues from there via a100/hf_sync.py
#     -- rerunning it after a stage is already complete is a safe no-op.
#   - hf_sync.py exists specifically because rsl_rl's `--agent.max-iterations N` is
#     RELATIVE to wherever a run resumes, not an absolute target (`total_it = start_it +
#     N` -- see rsl_rl/runners/on_policy_runner.py). Passing a fixed N on every resume
#     would make the target grow by N every single resubmission, and this is not
#     hypothetical: it's exactly what happened on the Kaggle run this was adapted from
#     (resumed at iteration 7800, next target silently became 47800). hf_sync.py tracks
#     a fixed absolute iteration BUDGET per stage instead and computes what's left.
#   - No local disk persistence is assumed across job resubmissions -- HF Hub is the
#     source of truth for resuming, same as the Kaggle/A100 notebooks.
#
# Requires HF_TOKEN (and GITHUB_TOKEN, if the repo is private) exported in the shell you
# run `sbatch` from -- SLURM propagates the submitting shell's environment by default.

#SBATCH --job-name=go2-pas
#SBATCH --partition=workq              # Partition to submit to (Do Not Change)
#SBATCH --time=24:00:00                # Walltime cap -- this script is designed to be resubmitted past this
#SBATCH --gres=gpu:1                   # Single GPU (this launcher runs single-GPU jobs directly, no multi-GPU launcher)
#SBATCH --nodelist=asaicomputenode02   # Specify node(s) by name
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                      # Total memory per node -- bump this if you see a host (not GPU) OOM
#SBATCH --output=go2-pas-%j.out
#SBATCH --error=go2-pas-%j.err

set -euo pipefail

# ==== CONFIG — edit as needed ====
GITHUB_REPO="BRUH-MAIN/policyswitching"
HF_REPO_NAME="go2-pas-saro"          # final repo id will be "<your HF username>/${HF_REPO_NAME}"

# Single A100 (40GB): this launcher does NOT split num_envs across GPUs -- with one GPU
# there's just one worker, so this is the full VRAM budget. 8192 is an untested
# extrapolation from a confirmed-working 2048 envs on an 8GB local GPU (this
# terrain+heightscan+LSTM-estimator env is heavier than a flat-terrain baseline) --
# watch `nvidia-smi` (e.g. `srun --jobid=$SLURM_JOB_ID --pty nvidia-smi` from a login
# node) during the first run and adjust up/down via the .out log.
NUM_ENVS=8192                          # reduce (e.g. 4096) if you hit OOM, raise if there's headroom to spare
GPU_IDS="[0]"                          # single GPU -- runs directly, no torchrunx multi-process launch
STAGE1_BUDGET=40000                    # paper default; absolute iteration budget for stage 1 (not "more each resume", see header)
STAGE2_BUDGET=40000                    # absolute iteration budget for stage 2, counted from stage 1's final iteration
SAVE_INTERVAL=200                      # PPO iterations between checkpoints (and HF uploads)

REPO_DIR="$HOME/policyswitching"
MJLAB_DIR="$REPO_DIR/unitree_rl_mjlab"
A100_DIR="$REPO_DIR/a100"

# ==== Environment activation — EDIT THIS SECTION for your cluster ====
# Load Conda environment setup if necessary (path to conda)
#source /path/to/miniconda3/etc/profile.d/conda.sh  # Adjust this path as needed
# Activate the Conda environment
#conda activate my_env_name

if ! command -v python3 &>/dev/null; then
  echo "[ERROR] python3 not found on PATH -- fix the conda/module activation section above." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[ERROR] HF_TOKEN is not set. Export it in the shell before running 'sbatch train_pas_slurm.sh'." >&2
  exit 1
fi

## 1. Environment check ##
nvidia-smi
nvcc --version || echo 'nvcc not found (fine if a matching CUDA runtime is still installed via pip)'
python3 --version

## 2. Clone the repo ##
if [ ! -d "$REPO_DIR" ]; then
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    CLONE_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
  else
    CLONE_URL="https://github.com/${GITHUB_REPO}.git"
  fi
  git clone "$CLONE_URL" "$REPO_DIR"
else
  echo "$REPO_DIR already exists, skipping clone."
  git -C "$REPO_DIR" pull --ff-only || echo "[WARN] git pull failed, continuing with existing checkout."
fi
test -f "$MJLAB_DIR/src/tasks/velocity/mdp/pas.py" && echo 'PAS implementation found.'

## 3. Install dependencies ##
cd "$MJLAB_DIR"
pip install -q -e .
# setup.py pins mjlab==1.2.0 / mujoco-warp==3.5.0 but not an exact mujoco (core) version, so
# a loose resolver can silently pull a newer, incompatible mujoco -- force the matching
# version explicitly (verified locally that mujoco==3.5.0 is what mujoco-warp==3.5.0
# actually needs; bump both together if you ever change MJLAB/mujoco-warp versions).
pip install -q --upgrade "mujoco==3.5.0"
# mjlab==1.2.0's sim.py reaches into warp's *internal* `wp.context.runtime.driver_version`
# rather than the public `wp.get_cuda_driver_version()` API added in later mjlab releases.
# mjlab's own pin (warp-lang>=1.12.0) is a loose floor, so an unconstrained install grabs
# the newest warp-lang -- but warp-lang>=1.13.0 deleted the top-level `warp/context.py`
# shim entirely, so `wp.context` no longer resolves and training crashes with
# `AttributeError: module 'warp' has no attribute 'context'` right as the sim is
# constructed. Pin to the last warp-lang release that still ships the (deprecated but
# functional) `warp.context` shim.
pip install -q --upgrade "warp-lang==1.12.1"
pip install -q huggingface_hub

python3 -c "import mjlab, mujoco, mujoco_warp; print('mujoco:', mujoco.__version__); print('mujoco_warp OK:', mujoco_warp.__file__); print('mjlab OK:', mjlab.__file__)"

export MUJOCO_GL=egl

## 4. Hugging Face repo id (for HF_CHECKPOINT_REPO) ##
HF_USERNAME=$(HF_TOKEN="$HF_TOKEN" python3 -c "from huggingface_hub import login, whoami; import os; login(token=os.environ['HF_TOKEN']); print(whoami()['name'])")
HF_REPO_ID="${HF_USERNAME}/${HF_REPO_NAME}"
echo "Using Hugging Face model repo: $HF_REPO_ID"
export HF_CHECKPOINT_REPO="$HF_REPO_ID"

## 5. Stage 1 — Oracle ##
export HF_CHECKPOINT_STAGE=stage1
STAGE1_EXTRA_ARGS=$(python3 "$A100_DIR/hf_sync.py" \
  --stage stage1 \
  --budget "$STAGE1_BUDGET" \
  --local-run-name slurm_stage1 \
  --mjlab-dir "$MJLAB_DIR" \
  --hf-repo-name "$HF_REPO_NAME")

if [ "$STAGE1_EXTRA_ARGS" = "SKIP" ]; then
  echo "Stage 1 already reached its ${STAGE1_BUDGET}-iteration budget -- skipping."
else
  cd "$MJLAB_DIR"
  # shellcheck disable=SC2086
  python3 scripts/train.py Unitree-Go2-PAS-Oracle \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.save-interval "$SAVE_INTERVAL" \
    --agent.logger tensorboard \
    --agent.experiment-name go2_pas \
    --gpu-ids "$GPU_IDS" \
    $STAGE1_EXTRA_ARGS
fi

## 6. Stage 2 — Anneal ##
export HF_CHECKPOINT_STAGE=stage2
STAGE2_EXTRA_ARGS=$(python3 "$A100_DIR/hf_sync.py" \
  --stage stage2 \
  --budget "$STAGE2_BUDGET" \
  --local-run-name slurm_stage2 \
  --stage1-local-run-name slurm_stage1 \
  --mjlab-dir "$MJLAB_DIR" \
  --hf-repo-name "$HF_REPO_NAME")

if [ "$STAGE2_EXTRA_ARGS" = "SKIP" ]; then
  echo "Stage 2 already reached its ${STAGE2_BUDGET}-iteration budget (or stage 1 isn't done yet) -- skipping."
else
  cd "$MJLAB_DIR"
  # shellcheck disable=SC2086
  python3 scripts/train.py Unitree-Go2-PAS-Anneal \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.save-interval "$SAVE_INTERVAL" \
    --agent.logger tensorboard \
    --agent.experiment-name go2_pas \
    --agent.run-name stage2 \
    --gpu-ids "$GPU_IDS" \
    $STAGE2_EXTRA_ARGS
fi

echo "Job done. If either stage printed 'skipping' due to walltime rather than a met budget, resubmit: sbatch train_pas_slurm.sh"
