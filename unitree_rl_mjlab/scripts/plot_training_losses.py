"""PPO training curves (losses, action std, reward) from RSL-RL TensorBoard logs.

Each --run is "label=source[,source...]", a source being a run directory with TensorBoard events or
a CSV from scripts/export_training_scalars.py; several sources are joined in step order (a run
resumed in a new directory). Plots, against training iteration, the value-function loss, the
surrogate (policy) loss, the entropy, the action standard deviation and the mean episode reward:
faint per-iteration values under a 50-iteration rolling mean.

Usage (from unitree_rl_mjlab/):
  python scripts/plot_training_losses.py \
      --run "Stairs specialist=logs/cluster_tb/go2_spec_stairs_v8b/<run>" \
      --run "Matched generalist=logs/cluster_tb/go2_generalist_v3/<run1>,logs/cluster_tb/go2_generalist_v3/<run2>" \
      --out ../report_content/figures/training_losses
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

PANELS = (  # tag, title, log scale
  ("Loss/value", "Value-function loss", True),
  ("Loss/surrogate", "Surrogate (policy) loss", False),
  ("Loss/entropy", "Entropy", False),
  ("Policy/mean_std", "Action standard deviation", False),
  ("Train/mean_reward", "Mean episode reward", False),
)
# Categorical slots of the validated reference palette, in fixed order.
COLORS = ("#e87ba4", "#eda100", "#2a78d6", "#1baf7a", "#eb6834", "#6250d6")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(dirs: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  """Each entry is a run directory with TensorBoard events, or a CSV from export_training_scalars.py."""
  series: dict[str, dict[int, float]] = {}
  for d in dirs:
    if d.endswith(".csv"):
      with open(d) as f:
        for row in csv.DictReader(f):
          for tag, _, _ in PANELS:
            if row.get(tag):
              series.setdefault(tag, {})[int(float(row["step"]))] = float(row[tag])
      continue
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    for tag, _, _ in PANELS:
      if tag in ea.Tags()["scalars"]:
        series.setdefault(tag, {}).update({e.step: e.value for e in ea.Scalars(tag)})
  return {t: (np.array(sorted(v)), np.array([v[k] for k in sorted(v)])) for t, v in series.items()}


def rolling(y: np.ndarray, n: int) -> np.ndarray:
  if len(y) < n:
    return y
  c = np.cumsum(np.insert(y, 0, 0.0))
  head = c[1:n] / np.arange(1, n)
  return np.concatenate([head, (c[n:] - c[:-n]) / n])


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--run", action="append", required=True, help='"label=dir[,dir...]"')
  ap.add_argument("--out", required=True, help="output path without extension (.png and .pdf are written)")
  ap.add_argument("--window", type=int, default=50)
  ap.add_argument("--width", type=float, default=7.1, help="figure width, inches")
  args = ap.parse_args()

  runs = []
  for spec in args.run:
    label, dirs = spec.split("=", 1)
    data = load(dirs.split(","))
    if not data:
      raise SystemExit("no scalars found for %s" % label)
    runs.append((label, data))
    print("%s: %s" % (label, {t: len(v[0]) for t, v in data.items()}))

  plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2,
                       "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                       "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
  fig, axes = plt.subplots(2, 3, figsize=(args.width, args.width * 0.52))
  axes = axes.ravel()
  for ax, (tag, title, log) in zip(axes, PANELS):
    for (label, data), color in zip(runs, COLORS):
      if tag not in data:
        continue
      x, y = data[tag]
      ax.plot(x, y, color=color, lw=0.5, alpha=0.25)
      ax.plot(x, rolling(y, args.window), color=color, lw=1.3, label=label)
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.set_xlabel("training iteration")
    if log:
      ax.set_yscale("log")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
  axes[-1].axis("off")
  handles, labels = axes[0].get_legend_handles_labels()
  axes[-1].legend(handles, labels, loc="center", frameon=False, fontsize=8,
                  title="faint: per iteration\nsolid: %d-iteration mean" % args.window, title_fontsize=7)
  fig.tight_layout(pad=0.4)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  for ext in ("png", "pdf"):
    fig.savefig("%s.%s" % (args.out, ext), dpi=300)
  print("wrote %s.png and .pdf" % args.out)


if __name__ == "__main__":
  main()
