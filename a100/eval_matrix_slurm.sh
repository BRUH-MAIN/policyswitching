#!/bin/bash
# Cross-terrain eval matrix (a100/eval_matrix.py): every trained policy x terrain class x
# difficulty under pinned eval conditions, plus the height-scan ablation. Writes one JSON
# per cell and a summary.md to unitree_rl_mjlab/eval_results/matrix/.
#
# Usage:
#   sbatch a100/eval_matrix_slurm.sh
#   MATRIX_ARGS="--only flat rough --difficulties 0.5" sbatch a100/eval_matrix_slurm.sh
#
# Resumable: finished cells are skipped, so if walltime cuts it off, resubmit as-is.
# Requests any GPU type (not a100 specifically): eval at 1024 envs fits comfortably on the
# 48 GB RTX 6000 Ada nodes too, and accepting either type schedules sooner.

#SBATCH --job-name=go2-eval-matrix
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/go2-eval-matrix-%j.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/go2-eval-matrix-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/dist_home/d_palmani/c-08/policyswitching}"
MJLAB_DIR="$REPO_DIR/unitree_rl_mjlab"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/policyswitching-pas}"
PY="$VENV_DIR/bin/python3"
if [[ ! -x "$PY" ]]; then
  echo "[ERROR] venv missing at $VENV_DIR -- run train_pas_slurm.sh once to bootstrap it." >&2
  exit 1
fi
export PATH="$VENV_DIR/bin:$PATH"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
# Explicit PYTHONPATH: the env's editable install can resolve `import src` to an
# unrelated project otherwise (see CLAUDE.md).
export PYTHONPATH="$MJLAB_DIR:${PYTHONPATH:-}"

nvidia-smi

read -r -a EXTRA_ARGS <<< "${MATRIX_ARGS:-}"
"$PY" "$REPO_DIR/a100/eval_matrix.py" --mjlab-dir "$MJLAB_DIR" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
