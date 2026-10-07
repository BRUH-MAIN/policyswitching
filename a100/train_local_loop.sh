#!/bin/bash
# Run a100/train_specialist_slurm.sh on the laptop (no SLURM), restarting from the latest local
# checkpoint if it dies (the 8 GB GPU can run out of memory when an eval shares it).
# Usage: SPEC=StairsV6a EXPERIMENT_SUFFIX=_lap INIT_FROM=<ckpt> NUM_ENVS=1536 BUDGET=6000 a100/train_local_loop.sh
# 1536 envs is what fits: 2048 runs out of memory on the second iteration, 4096 at startup.
# Start it detached so it outlives the session:
#   systemd-run --user --unit=<name> --collect --setenv=SPEC=... a100/train_local_loop.sh
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR VENV_DIR="${VENV_DIR:-$HOME/miniconda3/envs/unitree_rl_mjlab}"
export NUM_ENVS="${NUM_ENVS:-1536}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_DIR/.env" ]; then set -a; . "$REPO_DIR/.env"; set +a; fi
LOG="${LOG:-$REPO_DIR/unitree_rl_mjlab/logs/train_${SPEC}${EXPERIMENT_SUFFIX:-}.log}"
for attempt in $(seq 1 30); do
  echo "=== attempt $attempt $(date -Is)" >> "$LOG"
  if bash "$REPO_DIR/a100/train_specialist_slurm.sh" >> "$LOG" 2>&1; then
    echo "=== finished $(date -Is)" >> "$LOG"; exit 0
  fi
  echo "=== died $(date -Is); restarting from the latest checkpoint in 30 s" >> "$LOG"
  sleep 30
done
echo "=== gave up after 30 attempts $(date -Is)" >> "$LOG"; exit 1
