"""Course success by arm, as horizontal bars with 95% Wilson intervals.

Usage (from unitree_rl_mjlab/):
  python scripts/switch_follow_bars.py eval_results/switch_follow/confirm_s50*.json \
      --out ../report_content/figures/switch_confirm.png
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

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAR = "#2a78d6"  # one measure, one hue: the arms are named on the axis, not by colour

LABELS = {
  "fixed:flat": "Flat specialist only",
  "fixed:rough": "Rough specialist only",
  "fixed:stairs": "Stairs specialist only",
  "fixed:generalist": "Generalist only",
}


def label(spec: str) -> str:
  if spec in LABELS:
    return LABELS[spec]
  kind, *rest = spec.split(":")
  if kind == "hard":
    return f"Hard switch, {float(rest[0]):g} m ahead"
  return f"Cross-fade, {float(rest[0]):g} to {float(rest[1]):g} m ahead"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("files", nargs="+")
  ap.add_argument("--out", required=True)
  ap.add_argument("--title", default="Course success by arm")
  ap.add_argument("--order", nargs="*", default=None, help="Arm specs, top to bottom; default: file order.")
  args = ap.parse_args()
  arms, conds = load(args.files)
  order = [s for s in (args.order or list(arms)) if s in arms]

  plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
                       "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2})
  fig, ax = plt.subplots(figsize=(8.2, 0.52 * len(order) + 1.5), facecolor=SURFACE)
  ax.set_facecolor(SURFACE)
  ys = list(range(len(order)))[::-1]
  for y, spec in zip(ys, order):
    a = arms[spec]
    p = a.pct("success")
    lo, hi = wilson(a.count("success"), a.n)
    ax.barh(y, p, height=0.56, color=BAR, zorder=2)
    ax.plot([100 * lo, 100 * hi], [y, y], color=INK, lw=1.1, zorder=3)
    ax.text(100 * hi + 1.5, y, f"{p:.1f}%", va="center", ha="left", fontsize=9, color=INK)
  ax.set_yticks(ys, [label(s) for s in order])
  ax.set_xlim(0, 108)
  ax.set_xlabel("trials that cross the whole course (%)")
  ax.grid(axis="x", color=GRID, lw=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right", "bottom"):
    ax.spines[side].set_visible(False)
  ax.spines["left"].set_color(AXIS)
  ax.tick_params(length=0)
  n = arms[order[0]].n
  fig.suptitle(args.title, x=0.012, ha="left", fontsize=11.5, color=INK, fontweight="bold")
  fig.text(0.012, 0.012, f"Course multi L1, follow task, seeds {[c['seed'] for c in conds]}, {n} trials per arm, "
           f"{conds[0]['terminations']} terminations; lines are 95% Wilson intervals.", color=MUTED, fontsize=8)
  fig.tight_layout(rect=(0, 0.05, 1, 0.94))
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=170, facecolor=SURFACE)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
