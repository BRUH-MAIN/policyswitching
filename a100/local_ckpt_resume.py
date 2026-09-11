"""Local-disk checkpoint resume for specialist training.

Replaced hf_sync_specialist.py (since deleted; see git history) for the
specialist tasks. That script assumed
checkpoints get pushed to Hugging Face the same way PAS's do -- but PAS's
upload happens in `PasOnPolicyRunner.save()` (see pas.py), an override that
only exists on that subclass. Specialists register with the STOCK
`VelocityOnPolicyRunner`, whose `save()` has no HF logic at all. So setting
HF_CHECKPOINT_REPO/HF_CHECKPOINT_STAGE for a specialist job does nothing --
the checkpoints only ever land on local disk.

This went unnoticed for three of four specialists (Flat/Stairs/Rough) purely
because each finished its whole budget inside one job's walltime, so no
resume was ever needed. It bit the fourth (Gaps): job 11850 hit its 1-day
walltime at iteration 7528 with zero checkpoints on HF, so the resubmission
(job 11873) found nothing to resume and silently restarted from iteration 0
-- a full day of GPU time spent re-discovering the same plateau instead of
continuing past it.

Local disk on this cluster DOES persist across job resubmissions (confirmed:
all four specialists' logs/rsl_rl/<experiment>/<run>/ directories survived
job boundaries intact) -- it's the SLURM walltime that's the constraint, not
storage -- so this resumes directly from there instead of relying on HF.

Same iteration-accounting logic as hf_sync.py:
`OnPolicyRunner.learn(num_learning_iterations=N)` treats N as relative to
the resume point, not absolute, so this computes and emits
`--agent.max-iterations <absolute_budget - already_done>`.

Prints either the extra `train.py` args to use, or SKIP if the budget is met.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def log(*args: object) -> None:
  print(*args, file=sys.stderr, flush=True)


def latest_local_checkpoint(mjlab_dir: str, experiment_name: str) -> tuple[int, str] | None:
  """Scan logs/rsl_rl/<experiment_name>/*/model_*.pt across ALL run dirs (not
  just one) and return (iteration, run_dir_name) for the highest iteration
  found. Scanning every run dir, not just the most recent, matters here
  specifically: a run that hit walltime and then got wrongly restarted from
  scratch (the exact bug this script fixes) leaves an earlier, further-along
  run dir on disk that a "most recent dir only" scan would miss.
  """
  base = Path(mjlab_dir) / "logs" / "rsl_rl" / experiment_name
  if not base.is_dir():
    return None
  best: tuple[int, str] | None = None
  for run_dir in base.iterdir():
    if not run_dir.is_dir():
      continue
    for ckpt in run_dir.glob("model_*.pt"):
      m = re.match(r"model_(\d+)\.pt$", ckpt.name)
      if not m:
        continue
      iteration = int(m.group(1))
      if best is None or iteration > best[0]:
        best = (iteration, run_dir.name)
  return best


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--budget", required=True, type=int, help="Absolute iteration budget.")
  parser.add_argument("--experiment-name", required=True, help="rsl_rl experiment name, e.g. go2_spec_stairs.")
  parser.add_argument("--mjlab-dir", required=True)
  args = parser.parse_args()

  found = latest_local_checkpoint(args.mjlab_dir, args.experiment_name)

  if found is None:
    log(f"No existing local checkpoint for '{args.experiment_name}' -- starting fresh.")
    elapsed = 0
    resume_args: list[str] = []
  else:
    iteration, run_dir = found
    elapsed = iteration
    resume_args = [
      "--agent.resume", "True",
      "--agent.load-run", run_dir,
      "--agent.load-checkpoint", f"model_{iteration}.pt",
    ]
    log(f"Resuming '{args.experiment_name}' from {run_dir}/model_{iteration}.pt ({elapsed}/{args.budget} done).")

  remaining = args.budget - elapsed
  if remaining <= 0:
    log(f"'{args.experiment_name}' already reached its {args.budget}-iteration budget ({elapsed} done) -- skipping.")
    print("SKIP")
    return

  print(" ".join(["--agent.max-iterations", str(remaining), *resume_args]))


if __name__ == "__main__":
  main()
