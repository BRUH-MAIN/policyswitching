"""Course success by arm under two conditions, as grouped horizontal bars.

Each condition is a set of result files; an arm's trials are pooled over the files
of its condition.

Usage (from unitree_rl_mjlab/):
  python scripts/switch_follow_conditions.py \
      --cond "sensor noise off" eval_results/switch_follow/generalist_clean_s50*.json \
      --cond "sensor noise on" eval_results/switch_follow/generalist_noisy_s50*.json \
      --order fixed:generalist hard:0.3:label fixed:stairs \
      --out ../report_content/figures/switch_generalist.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from switch_follow_analyze import load, wilson  # noqa: E402
from switch_follow_bars import label  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2 (validated as a pair)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--cond", action="append", nargs="+", required=True, metavar=("NAME", "FILE"),
                  help="Condition name followed by its result files; give exactly two --cond.")
  ap.add_argument("--order", nargs="+", required=True, help="Arm specs, top to bottom.")
  ap.add_argument("--title", default="Course success by arm and condition")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  if len(args.cond) != 2:
    ap.error("exactly two --cond groups")
  conds = [(c[0], load(c[1:])) for c in args.cond]

  plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
                       "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2})
  fig, ax = plt.subplots(figsize=(8.2, 0.95 * len(args.order) + 1.9), facecolor=SURFACE)
  ax.set_facecolor(SURFACE)
  h = 0.34
  centres = list(range(len(args.order)))[::-1]
  for k, ((name, (arms, _)), color) in enumerate(zip(conds, COLORS)):
    for y0, spec in zip(centres, args.order):
      a = arms[spec]
      y = y0 + (0.5 - k) * (h + 0.04)
      p = a.pct("success")
      lo, hi = wilson(a.count("success"), a.n)
      ax.barh(y, p, height=h, color=color, zorder=2, label=name if spec == args.order[0] else None)
      ax.plot([100 * lo, 100 * hi], [y, y], color=INK, lw=1.1, zorder=3)
      ax.text(100 * hi + 1.5, y, f"{p:.1f}%", va="center", ha="left", fontsize=9, color=INK)
  ax.set_yticks(centres, [label(s) for s in args.order])
  ax.set_xlim(0, 108)
  ax.set_xlabel("trials that cross the whole course (%)")
  ax.grid(axis="x", color=GRID, lw=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right", "bottom"):
    ax.spines[side].set_visible(False)
  ax.spines["left"].set_color(AXIS)
  ax.tick_params(length=0)
  ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncols=2, frameon=False, fontsize=9, labelcolor=INK_2)
  n = [next(iter(arms.values())).n for _, (arms, _) in conds]
  seeds = sorted({c["seed"] for _, (_, cs) in conds for c in cs})
  fig.suptitle(args.title, x=0.012, ha="left", fontsize=11.5, color=INK, fontweight="bold")
  fig.text(0.012, 0.012, f"Course multi L1, follow task, seeds {seeds}, {n[0]} trials per bar; "
           "lines are 95% Wilson intervals.", color=MUTED, fontsize=8)
  fig.tight_layout(rect=(0, 0.05, 1, 0.93))
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=170, facecolor=SURFACE)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
