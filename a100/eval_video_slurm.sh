#!/bin/bash
# Qualitative video rollout of a PAS checkpoint via scripts/play.py. sbatch version of the
# srun eval command. play.py's viewer loop (ViserPlayViewer, headless on a compute node)
# never exits on its own, so this wraps it in `timeout` -- the video is written well before
# the timeout fires, so the non-zero exit from `timeout` here is expected, not a failure.
#
# Like eval_numeric_slurm.sh, this defaults to the LATEST checkpoint on the HF model repo
# instead of a hardcoded path. Override with EVAL_CHECKPOINT=<local .pt path>, or pick the
# stage with EVAL_STAGE=stage1.
#
# NOTE: play.py's `--video` REQUIRES an explicit value -- tyro renders it as
# `--video {True,False}` (confirmed via `play.py <task> --help`). A bare `--video` fails
# with "Missing value for argument '--video'. Expected 1 values." before any simulation
# happens; that is what made job 11800 produce no video. Keep the `True`.

#SBATCH --job-name=go2-pas-eval-video
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/go2-pas-eval-video-%j.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/go2-pas-eval-video-%j.err

set -uo pipefail

REPO=/dist_home/d_palmani/c-08/policyswitching
PYTHON=/dist_home/d_palmani/.venvs/policyswitching-pas/bin/python3

export MUJOCO_GL=egl
export PYTHONPATH="$REPO/unitree_rl_mjlab:${PYTHONPATH:-}"

cd "$REPO/unitree_rl_mjlab"

if [ -n "${EVAL_CHECKPOINT:-}" ]; then
  CKPT="$EVAL_CHECKPOINT"
else
  # The model repo currently reads anonymously, so HF_TOKEN is not required here --
  # only warn, so a missing token can't block an otherwise-working eval.
  [ -z "${HF_TOKEN:-}" ] && echo "[WARN] HF_TOKEN unset; relying on anonymous read access."
  CKPT=$("$PYTHON" - "${EVAL_STAGE:-stage2}" <<'PY'
import sys
from huggingface_hub import HfApi, hf_hub_download

stage = sys.argv[1]
repo_id = "RohanRamesh/go2-pas-saro"
files = HfApi().list_repo_files(repo_id, repo_type="model")
candidates = []
for f in files:
    if f.startswith(f"{stage}/model_") and f.endswith(".pt"):
        try:
            candidates.append((int(f.split("model_")[-1].split(".pt")[0]), f))
        except ValueError:
            pass
if not candidates:
    raise SystemExit(f"No {stage}/model_*.pt in hf.co/{repo_id}")
_, remote = max(candidates, key=lambda x: x[0])
print(hf_hub_download(repo_id, remote, repo_type="model", local_dir="eval_ckpts"))
PY
  ) || { echo "[ERROR] Could not resolve a checkpoint from HF."; exit 1; }
fi

echo "[INFO] Using checkpoint: $CKPT"

timeout 180 "$PYTHON" scripts/play.py \
  Unitree-Go2-PAS-Oracle \
  --checkpoint_file "$CKPT" \
  --num_envs 4 --video True --video_length 400 --viewer viser
rc=$?

if [ "$rc" -eq 124 ]; then
  echo "[INFO] play.py was still running the viewer loop and was cut by the 180s timeout, as expected."
elif [ "$rc" -ne 0 ]; then
  echo "[ERROR] play.py exited with code $rc for a reason OTHER than the timeout wrapper -- check the .err log above, video was likely NOT written."
fi

# play.py writes to <checkpoint dir>/videos/play/, so derive it from the checkpoint path.
VIDEO_DIR="$(dirname "$CKPT")/videos/play"
echo "[INFO] Video should be under $VIDEO_DIR"
ls -la "$VIDEO_DIR" 2>&1 || true
