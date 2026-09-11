#!/usr/bin/env python3
"""Hugging Face checkpoint sync for resumable, multi-job PAS training.

Used by train_pas_slurm.sh (a SLURM job has a walltime cap, e.g. 24h, so a
full 40000-iteration stage may need several `sbatch` resubmissions -- this
script makes each resubmission pick up exactly where the last one left off
and stop once the stage's iteration *budget* is met, instead of restarting
or overshooting).

IMPORTANT iteration-accounting gotcha this script exists to handle:
rsl_rl's `OnPolicyRunner.learn(num_learning_iterations=N)` treats N as
*relative* to wherever training resumes (`total_it = start_it + N`), not an
absolute target -- see rsl_rl/runners/on_policy_runner.py. So naively passing
the same fixed `--agent.max-iterations 40000` on every resume makes the
target grow by 40000 each time (this is exactly what happened on the Kaggle
run this was built for: it resumed from iteration 7800 and the next target
became 7800+40000=47800). This script instead tracks a fixed absolute
per-stage *budget* and computes `remaining = budget - iterations_done_in_this_stage`
each time, so re-running always converges on the same total.

Stage 2 has an extra wrinkle: its first invocation loads Stage 1's final
checkpoint, and rsl_rl's `current_learning_iteration` carries over from
whatever checkpoint was loaded (it is not reset per-stage) -- so Stage 2's
absolute iteration count starts at Stage 1's final iteration, not 0.
"iterations_done_in_this_stage" for stage2 is computed relative to that
inherited baseline.

Prints exactly one line to stdout:
  "SKIP"                                    -- this stage's budget is already met
  "--agent.max-iterations N [resume args]"  -- extra args to append to train.py

All other output goes to stderr, so stdout stays safe to capture in bash.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def log(*args: object) -> None:
  print(*args, file=sys.stderr, flush=True)


def latest_checkpoint(api, hf_repo_id: str, stage: str) -> tuple[int, str] | None:
  """Return (iteration, remote_path) for the highest-iteration model_*.pt under {stage}/, or None."""
  files = api.list_repo_files(hf_repo_id, repo_type="model")
  candidates = []
  for f in files:
    if f.startswith(f"{stage}/model_") and f.endswith(".pt"):
      try:
        it = int(f.split("model_")[-1].split(".pt")[0])
        candidates.append((it, f))
      except ValueError:
        continue
  if not candidates:
    return None
  return max(candidates, key=lambda x: x[0])


def download_checkpoint(
  hf_repo_id: str, mjlab_dir: str, local_run_name: str, iteration: int, remote_path: str
) -> None:
  from huggingface_hub import hf_hub_download

  local_dir = f"{mjlab_dir}/logs/rsl_rl/go2_pas/{local_run_name}"
  os.makedirs(local_dir, exist_ok=True)
  local_file = hf_hub_download(hf_repo_id, remote_path, repo_type="model", local_dir=mjlab_dir)
  target = f"{local_dir}/model_{iteration}.pt"
  if local_file != target:
    shutil.copy(local_file, target)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage", required=True, choices=["stage1", "stage2"])
  parser.add_argument("--budget", required=True, type=int, help="Absolute iteration budget for this stage.")
  parser.add_argument("--local-run-name", required=True, help="Local logs/rsl_rl/go2_pas/<name> dir for this stage.")
  parser.add_argument("--mjlab-dir", required=True)
  parser.add_argument("--hf-repo-name", required=True)
  parser.add_argument(
    "--stage1-local-run-name",
    help="Only used for --stage stage2: local dir to download stage1's checkpoint into on stage2's first entry.",
  )
  parser.add_argument(
    "--from-scratch",
    action="store_true",
    help="Ignore any existing HF checkpoint for THIS stage and restart its budget from 0. "
    "Stage 2 still bootstraps from stage 1's final checkpoint either way -- that's the "
    "curriculum, not resumable progress -- so this only discards stage 2's own progress.",
  )
  args = parser.parse_args()

  if args.stage == "stage2" and not args.stage1_local_run_name:
    parser.error("--stage2 requires --stage1-local-run-name")

  hf_token = os.environ.get("HF_TOKEN")
  if not hf_token:
    log("HF_TOKEN is not set.")
    sys.exit(1)

  from huggingface_hub import HfApi, create_repo, login, whoami

  login(token=hf_token)
  hf_username = whoami()["name"]
  hf_repo_id = f"{hf_username}/{args.hf_repo_name}"
  create_repo(hf_repo_id, repo_type="model", exist_ok=True)
  api = HfApi()

  resume_args: list[str] = []

  if args.stage == "stage1":
    baseline = 0
    found = None if args.from_scratch else latest_checkpoint(api, hf_repo_id, "stage1")
    if found is None:
      log("--from-scratch: ignoring any existing 'stage1' checkpoint." if args.from_scratch
          else "No existing 'stage1' checkpoint on HF -- starting fresh.")
      elapsed = 0
    else:
      iteration, remote_path = found
      download_checkpoint(hf_repo_id, args.mjlab_dir, args.local_run_name, iteration, remote_path)
      elapsed = iteration - baseline
      resume_args = [
        "--agent.resume", "True",
        "--agent.load-run", args.local_run_name,
        "--agent.load-checkpoint", f"model_{iteration}.pt",
      ]
      log(f"Resuming 'stage1' from iteration {iteration} ({elapsed}/{args.budget} done).")
  else:
    stage1_found = latest_checkpoint(api, hf_repo_id, "stage1")
    if stage1_found is None:
      log("No 'stage1' checkpoint on HF yet -- stage2 can't start.")
      sys.exit(1)
    baseline = stage1_found[0]

    found = None if args.from_scratch else latest_checkpoint(api, hf_repo_id, "stage2")
    if found is None:
      # First entry into stage2: resume from stage1's final checkpoint.
      iteration, remote_path = stage1_found
      download_checkpoint(hf_repo_id, args.mjlab_dir, args.stage1_local_run_name, iteration, remote_path)
      elapsed = 0
      resume_args = [
        "--agent.resume", "True",
        "--agent.load-run", args.stage1_local_run_name,
        "--agent.load-checkpoint", f"model_{iteration}.pt",
      ]
      log(f"Starting 'stage2' from stage1's iteration {iteration} (baseline for this stage's budget).")
    else:
      iteration, remote_path = found
      download_checkpoint(hf_repo_id, args.mjlab_dir, args.local_run_name, iteration, remote_path)
      elapsed = iteration - baseline
      resume_args = [
        "--agent.resume", "True",
        "--agent.load-run", args.local_run_name,
        "--agent.load-checkpoint", f"model_{iteration}.pt",
      ]
      log(f"Resuming 'stage2' from iteration {iteration} ({elapsed}/{args.budget} done, baseline {baseline}).")

  remaining = args.budget - elapsed
  if remaining <= 0:
    log(f"'{args.stage}' already reached its {args.budget}-iteration budget ({elapsed} done) -- skipping.")
    print("SKIP")
    return

  print(" ".join(["--agent.max-iterations", str(remaining), *resume_args]))


if __name__ == "__main__":
  main()
