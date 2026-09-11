"""Hugging Face checkpoint sync for resumable terrain-specialist training.

The single-stage sibling of hf_sync.py. Same job: a SLURM walltime cap means a
specialist's iteration budget may span several `sbatch` resubmissions, and each
resubmission has to pick up exactly where the last stopped and stop once the
ABSOLUTE budget is met.

It exists separately because hf_sync.py is hard-wired to PAS's two-stage
curriculum -- its `--stage` only accepts {stage1, stage2}, stage2 bootstraps
from stage1's final checkpoint, and `download_checkpoint` hardcodes the
`logs/rsl_rl/go2_pas/` experiment directory. Specialists are single-stage, have
no cross-stage baseline, and each writes to its own experiment dir, so the
stage machinery would be wrong rather than merely unused.

The iteration-accounting gotcha is identical and is the whole reason both
scripts exist: rsl_rl's `OnPolicyRunner.learn(num_learning_iterations=N)`
treats N as *relative* to wherever training resumes (`total_it = start_it + N`),
not as an absolute target. Passing a fixed N on every resubmission silently
extends the target each time -- which already bit this project once, when a
Kaggle run resumed at iteration 7800 and its target quietly became 47800. So
this script emits `--agent.max-iterations <remaining>`, computed as
`budget - already_done`.

Prints either the extra `train.py` args to use, or the literal string SKIP when
the budget is already met.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def log(*args: object) -> None:
  print(*args, file=sys.stderr, flush=True)


def latest_checkpoint(api, hf_repo_id: str, stage: str) -> tuple[int, str] | None:
  """Return (iteration, remote_path) for the highest-iteration model_*.pt under {stage}/."""
  files = api.list_repo_files(hf_repo_id, repo_type="model")
  candidates = []
  for f in files:
    if f.startswith(f"{stage}/model_") and f.endswith(".pt"):
      try:
        candidates.append((int(f.split("model_")[-1].split(".pt")[0]), f))
      except ValueError:
        continue
  if not candidates:
    return None
  return max(candidates, key=lambda x: x[0])


def download_checkpoint(
  hf_repo_id: str,
  mjlab_dir: str,
  experiment_name: str,
  local_run_name: str,
  iteration: int,
  remote_path: str,
) -> None:
  """Fetch one checkpoint into logs/rsl_rl/<experiment_name>/<local_run_name>/."""
  from huggingface_hub import hf_hub_download

  local_dir = f"{mjlab_dir}/logs/rsl_rl/{experiment_name}/{local_run_name}"
  os.makedirs(local_dir, exist_ok=True)
  local_file = hf_hub_download(hf_repo_id, remote_path, repo_type="model", local_dir=mjlab_dir)
  target = f"{local_dir}/model_{iteration}.pt"
  if local_file != target:
    shutil.copy(local_file, target)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--hf-stage", required=True, help="HF subdir, e.g. spec_stairs.")
  parser.add_argument("--budget", required=True, type=int, help="Absolute iteration budget.")
  parser.add_argument("--experiment-name", required=True, help="rsl_rl experiment name, e.g. go2_spec_stairs.")
  parser.add_argument("--local-run-name", required=True, help="logs/rsl_rl/<experiment>/<name> dir.")
  parser.add_argument("--mjlab-dir", required=True)
  parser.add_argument("--hf-repo-name", required=True)
  parser.add_argument(
    "--from-scratch",
    action="store_true",
    help="Ignore any existing HF checkpoint for this specialist and restart its budget from 0.",
  )
  args = parser.parse_args()

  hf_token = os.environ.get("HF_TOKEN")
  if not hf_token:
    log("HF_TOKEN is not set.")
    sys.exit(1)

  from huggingface_hub import HfApi, create_repo, login, whoami

  login(token=hf_token)
  hf_repo_id = f"{whoami()['name']}/{args.hf_repo_name}"
  create_repo(hf_repo_id, repo_type="model", exist_ok=True)
  api = HfApi()

  resume_args: list[str] = []
  found = None if args.from_scratch else latest_checkpoint(api, hf_repo_id, args.hf_stage)

  if found is None:
    log(
      f"--from-scratch: ignoring any existing '{args.hf_stage}' checkpoint."
      if args.from_scratch
      else f"No existing '{args.hf_stage}' checkpoint on HF -- starting fresh."
    )
    elapsed = 0
  else:
    iteration, remote_path = found
    download_checkpoint(
      hf_repo_id, args.mjlab_dir, args.experiment_name, args.local_run_name, iteration, remote_path
    )
    elapsed = iteration
    resume_args = [
      "--agent.resume", "True",
      "--agent.load-run", args.local_run_name,
      "--agent.load-checkpoint", f"model_{iteration}.pt",
    ]
    log(f"Resuming '{args.hf_stage}' from iteration {iteration} ({elapsed}/{args.budget} done).")

  remaining = args.budget - elapsed
  if remaining <= 0:
    log(f"'{args.hf_stage}' already reached its {args.budget}-iteration budget ({elapsed} done) -- skipping.")
    print("SKIP")
    return

  print(" ".join(["--agent.max-iterations", str(remaining), *resume_args]))


if __name__ == "__main__":
  main()
