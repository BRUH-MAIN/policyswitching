"""Seed a new experiment with another policy's weights, as a fresh iteration-0 run.

Why not just `--agent.resume` from the source checkpoint: resuming restores
the source's iteration counter (a Flat/Rough specialist is at 9999), its Adam
moments, and its env `common_step_counter` -- so the new run would believe it
had already spent its whole iteration budget and would skip the command
curriculum. This writes a copy that keeps only what a warm start should keep
(actor + critic weights), resets the rest, and places it where
local_ckpt_resume.py finds it as iteration 0 of the new experiment:

  logs/rsl_rl/<experiment>/0000_warmstart/model_0.pt

Observation-normalizer statistics are reset by default (--keep-normalizer to
opt out). Every specialist trained so far has a height_scan normalizer std of
~0.012 -- about the size of the injected observation noise alone -- so on
stepping stones, whose gaps drop 2 m, the source's frozen statistics would put
height_scan inputs ~17 standard deviations out of distribution. With count
reset to 0, the first rollout batch re-estimates them on the new terrain.

No-op if the experiment already has any checkpoint, so resubmitting a training
job never re-warm-starts on top of real progress.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_ckpt_resume import latest_local_checkpoint  # noqa: E402

WARMSTART_RUN_DIR = "0000_warmstart"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", required=True, help="Checkpoint to take weights from.")
  parser.add_argument("--experiment-name", required=True)
  parser.add_argument("--mjlab-dir", required=True)
  parser.add_argument(
    "--keep-normalizer",
    action="store_true",
    help="Keep the source's observation-normalizer statistics instead of re-estimating.",
  )
  args = parser.parse_args()

  existing = latest_local_checkpoint(args.mjlab_dir, args.experiment_name)
  if existing is not None:
    iteration, run_dir = existing
    print(
      f"[INFO] '{args.experiment_name}' already has {run_dir}/model_{iteration}.pt "
      "-- not warm-starting over it."
    )
    return

  source = Path(args.source)
  if not source.is_file():
    sys.exit(f"[ERROR] warm-start source not found: {source}")

  ckpt = torch.load(source, map_location="cpu", weights_only=False)
  source_iter = ckpt.get("iter")
  ckpt["iter"] = 0
  # Empty per-parameter state = fresh Adam moments; param_groups keeps the
  # learning rate the adaptive schedule had settled on.
  ckpt["optimizer_state_dict"]["state"] = {}
  infos = dict(ckpt.get("infos") or {})
  infos["env_state"] = {"common_step_counter": 0}
  infos["warm_start"] = {
    "source": str(source.resolve()),
    "source_iter": source_iter,
    "normalizer_reset": not args.keep_normalizer,
  }
  ckpt["infos"] = infos

  if not args.keep_normalizer:
    for key in ("actor_state_dict", "critic_state_dict"):
      state = ckpt[key]
      if "obs_normalizer.count" not in state:
        sys.exit(f"[ERROR] {key} has no obs_normalizer; pass --keep-normalizer.")
      state["obs_normalizer.count"] = torch.zeros_like(state["obs_normalizer.count"])

  dest_dir = Path(args.mjlab_dir) / "logs" / "rsl_rl" / args.experiment_name / WARMSTART_RUN_DIR
  dest_dir.mkdir(parents=True, exist_ok=True)
  dest = dest_dir / "model_0.pt"
  torch.save(ckpt, dest)
  print(
    f"[INFO] Warm start: {source} (iter {source_iter}) -> {dest} "
    f"(iter 0, fresh optimizer moments, "
    f"normalizer {'kept' if args.keep_normalizer else 'reset'})"
  )


if __name__ == "__main__":
  main()
