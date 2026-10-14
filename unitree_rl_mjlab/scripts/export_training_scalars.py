"""Export PPO training scalars from RSL-RL TensorBoard logs to small CSV files.

For each experiment directory given, every run directory under it that holds an
`events.out.tfevents*` file is written to <out>/<experiment>__<run>.csv, one row per iteration,
with the columns scripts/plot_training_losses.py reads (losses, action std, reward, episode
length, learning rate). The CSVs are a few hundred kB, small enough to commit, unlike the event
files.

Usage (from unitree_rl_mjlab/):
  python scripts/export_training_scalars.py logs/rsl_rl/go2_spec_stairs_v8b logs/rsl_rl/go2_generalist_v3 \
      --out eval_results/training_scalars
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ("Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate", "Policy/mean_std",
        "Train/mean_reward", "Train/mean_episode_length")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("experiments", nargs="+", help="logs/rsl_rl/<experiment> directories")
  ap.add_argument("--out", default="eval_results/training_scalars")
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  for exp in args.experiments:
    runs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(exp, "*", "events.out.tfevents*"))})
    if not runs:
      print("no event files under %s" % exp)
    for run in runs:
      ea = EventAccumulator(run, size_guidance={"scalars": 0})
      ea.Reload()
      have = [t for t in TAGS if t in ea.Tags()["scalars"]]
      rows: dict[int, dict[str, float]] = {}
      for t in have:
        for e in ea.Scalars(t):
          rows.setdefault(e.step, {})[t] = e.value
      name = "%s__%s.csv" % (os.path.basename(os.path.normpath(exp)), os.path.basename(run))
      path = os.path.join(args.out, name)
      with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + have)
        for step in sorted(rows):
          w.writerow([step] + [rows[step].get(t, "") for t in have])
      print("%s: %d iterations (steps %d-%d)" % (path, len(rows), min(rows), max(rows)))


if __name__ == "__main__":
  main()
