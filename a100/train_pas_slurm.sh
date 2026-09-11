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
#SBATCH --time=3-00:00:00              # Walltime cap (workq partition max) -- this script is designed to be resubmitted past this
#SBATCH --gres=gpu:a100:1              # Explicitly A100 -- NUM_ENVS=8192 below is tuned for 40GB VRAM and would
                                        # likely OOM on the master node's smaller rtx6000, which a bare `gpu:1`
                                        # could also match since workq spans all three nodes.
# No --nodelist pin: this cluster has two A100 nodes (asaicomputenode02/03), and pinning to one
# leaves the job PENDING if that specific node is full while the other has a free GPU. Override
# with `--nodelist=...` at submit time if you ever need a specific node.
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

# hf_sync.py already resumes each stage from its latest HF checkpoint by default (that's
# its whole purpose -- see the header above). Set FROM_SCRATCH=1 to discard that stage's
# progress and restart its budget from 0 instead:
#   sbatch --export=ALL,FROM_SCRATCH=1 train_pas_slurm.sh
: "${FROM_SCRATCH:=0}"
FROM_SCRATCH_FLAG=()
[[ "$FROM_SCRATCH" == "1" ]] && FROM_SCRATCH_FLAG=(--from-scratch)

# Canonical checkout. This is the ONLY copy of the repo -- the former $HOME/policyswitching
# clone was removed on 2026-09-05 (its logs/, eval_ckpts/ and stage1/ were moved in here).
# Note `clip_actions=6.0` in rl_cfg.py is still uncommitted, so a fresh `git clone` would
# NOT have it and training would re-diverge the way job 11769 did -- keep using this
# checkout rather than letting the clone step below recreate one from GitHub.
REPO_DIR="${REPO_DIR:-/dist_home/d_palmani/c-08/policyswitching}"
MJLAB_DIR="$REPO_DIR/unitree_rl_mjlab"
A100_DIR="$REPO_DIR/a100"

# ==== Environment activation ====
# The site's system python3 (Ubuntu) ships without pip, and Debian/Ubuntu disables
# `ensurepip` for the system interpreter, so a bare `pip install` below would just
# fail with "pip: command not found" / "No module named pip". The `conda` module's
# python has a working pip -- but installing straight into that shared base env
# would let this repo's exact pins (mujoco==3.5.0, warp-lang==1.12.1) fight any
# other job's pins in the same shared site-packages (e.g. codeswitching's
# transformers/torch stack), which matters here specifically because this script
# is meant to run concurrently with other GPU jobs. So bootstrap an isolated venv
# from conda's python instead, and reuse it across resubmissions.
CONDA_PY=/dist_home/common-apps/conda/bin/python3
BOOTSTRAP_PY=${PYTHON:-$CONDA_PY}
[[ -x "$BOOTSTRAP_PY" ]] || BOOTSTRAP_PY=python3

VENV_DIR="${VENV_DIR:-$HOME/.venvs/policyswitching-pas}"
if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
  echo "[venv] creating $VENV_DIR with $BOOTSTRAP_PY"
  "$BOOTSTRAP_PY" -m venv "$VENV_DIR"
fi
export PATH="$VENV_DIR/bin:$PATH"
PY="$VENV_DIR/bin/python3"

if ! "$PY" -m pip --version &>/dev/null; then
  echo "[ERROR] $PY has no working pip. Set VENV_DIR to a fresh path, or PYTHON= to a" >&2
  echo "        working interpreter, and resubmit." >&2
  exit 1
fi

# Under sbatch stdout is fully buffered (not line-buffered like a tty), so log
# lines -- including rsl_rl's per-iteration training summary -- pile up in
# memory instead of landing in the .out/.err files as they happen.
export PYTHONUNBUFFERED=1

# mjlab's Warp-backed sim.step() replays a captured CUDA graph, which draws
# scratch memory from CUDA's default stream-ordered mempool at *replay* time
# (see unitree_rl_mjlab's Simulation docstring in sim.py). PyTorch's own
# caching allocator reserves blocks from the same GPU without releasing them
# back to the driver or cooperating with that mempool, so over thousands of
# PPO iterations -- especially with PasPPO's variable-shaped, time-padded
# recurrent minibatches (see pas.py's PasPPO._estimator_aux_step) -- it can
# fragment/starve the mempool until a graph launch fails with a CUDA OOM even
# though nvidia-smi still shows headroom. This is exactly what killed job
# 11767 at iteration 9477 after 1477 healthy iterations at NUM_ENVS=8192.
# expandable_segments lets PyTorch grow one virtual segment instead of
# hoarding many separate blocks, cutting that fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[ERROR] HF_TOKEN is not set. Export it in the shell before running 'sbatch train_pas_slurm.sh'." >&2
  exit 1
fi

## 1. Environment check ##
nvidia-smi
nvcc --version || echo 'nvcc not found (fine if a matching CUDA runtime is still installed via pip)'
"$PY" --version

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
# No -q: mjlab pulls in torch + CUDA libs transitively, a multi-GB first-time
# install that can take many minutes. -q suppresses pip's own per-package
# progress entirely, so the .out file goes silent for the whole install even
# though it's actively working. Not tqdm-style spam either way -- pip already
# renders one clean line per package when stdout isn't a tty, no \r spam.
cd "$MJLAB_DIR"
"$PY" -m pip install -e .
# setup.py pins mjlab==1.2.0 / mujoco-warp==3.5.0 but not an exact mujoco (core) version, so
# a loose resolver can silently pull a newer, incompatible mujoco -- force the matching
# version explicitly (verified locally that mujoco==3.5.0 is what mujoco-warp==3.5.0
# actually needs; bump both together if you ever change MJLAB/mujoco-warp versions).
"$PY" -m pip install --upgrade "mujoco==3.5.0"
# mjlab==1.2.0's sim.py reaches into warp's *internal* `wp.context.runtime.driver_version`
# rather than the public `wp.get_cuda_driver_version()` API added in later mjlab releases.
# mjlab's own pin (warp-lang>=1.12.0) is a loose floor, so an unconstrained install grabs
# the newest warp-lang -- but warp-lang>=1.13.0 deleted the top-level `warp/context.py`
# shim entirely, so `wp.context` no longer resolves and training crashes with
# `AttributeError: module 'warp' has no attribute 'context'` right as the sim is
# constructed. Pin to the last warp-lang release that still ships the (deprecated but
# functional) `warp.context` shim.
"$PY" -m pip install --upgrade "warp-lang==1.12.1"
# mjlab==1.2.0 imports scipy.interpolate directly (mjlab/terrains/heightfield_terrains.py)
# but doesn't declare it as a dependency at all (`pip show mjlab` confirms: no scipy in
# Requires) -- crashes with ModuleNotFoundError the moment train.py imports mjlab.envs,
# ~17 minutes in, right after the install step and HF checkpoint download. Same story as
# the mujoco/warp-lang pins above: mjlab's own metadata is incomplete, so install explicitly.
"$PY" -m pip install scipy
"$PY" -m pip install huggingface_hub

"$PY" -c "import mjlab, mujoco, mujoco_warp; print('mujoco:', mujoco.__version__); print('mujoco_warp OK:', mujoco_warp.__file__); print('mjlab OK:', mjlab.__file__)"

export MUJOCO_GL=egl

## 4. Hugging Face repo id (for HF_CHECKPOINT_REPO) ##
HF_USERNAME=$(HF_TOKEN="$HF_TOKEN" "$PY" -c "from huggingface_hub import login, whoami; import os; login(token=os.environ['HF_TOKEN']); print(whoami()['name'])")
HF_REPO_ID="${HF_USERNAME}/${HF_REPO_NAME}"
echo "Using Hugging Face model repo: $HF_REPO_ID"
export HF_CHECKPOINT_REPO="$HF_REPO_ID"

## 5. Stage 1 — Oracle ##
export HF_CHECKPOINT_STAGE=stage1
STAGE1_EXTRA_ARGS=$("$PY" "$A100_DIR/hf_sync.py" \
  --stage stage1 \
  --budget "$STAGE1_BUDGET" \
  --local-run-name slurm_stage1 \
  --mjlab-dir "$MJLAB_DIR" \
  --hf-repo-name "$HF_REPO_NAME" \
  "${FROM_SCRATCH_FLAG[@]}")

if [ "$STAGE1_EXTRA_ARGS" = "SKIP" ]; then
  echo "Stage 1 already reached its ${STAGE1_BUDGET}-iteration budget -- skipping."
else
  cd "$MJLAB_DIR"
  # shellcheck disable=SC2086
  "$PY" scripts/train.py Unitree-Go2-PAS-Oracle \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.save-interval "$SAVE_INTERVAL" \
    --agent.logger tensorboard \
    --agent.experiment-name go2_pas \
    --gpu-ids "$GPU_IDS" \
    $STAGE1_EXTRA_ARGS
fi

## 6. Stage 2 — Anneal ##
export HF_CHECKPOINT_STAGE=stage2
STAGE2_EXTRA_ARGS=$("$PY" "$A100_DIR/hf_sync.py" \
  --stage stage2 \
  --budget "$STAGE2_BUDGET" \
  --local-run-name slurm_stage2 \
  --stage1-local-run-name slurm_stage1 \
  --mjlab-dir "$MJLAB_DIR" \
  --hf-repo-name "$HF_REPO_NAME" \
  "${FROM_SCRATCH_FLAG[@]}")

if [ "$STAGE2_EXTRA_ARGS" = "SKIP" ]; then
  echo "Stage 2 already reached its ${STAGE2_BUDGET}-iteration budget (or stage 1 isn't done yet) -- skipping."
else
  cd "$MJLAB_DIR"
  # shellcheck disable=SC2086
  "$PY" scripts/train.py Unitree-Go2-PAS-Anneal \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.save-interval "$SAVE_INTERVAL" \
    --agent.logger tensorboard \
    --agent.experiment-name go2_pas \
    --agent.run-name stage2 \
    --gpu-ids "$GPU_IDS" \
    $STAGE2_EXTRA_ARGS
fi

echo "Job done. If either stage printed 'skipping' due to walltime rather than a met budget, resubmit: sbatch train_pas_slurm.sh"
