"""Before / after for the onboard classifier (arm B): Addendum 8 (labels 0.3 m ahead, old bank)
against Addendum 9 (labels 0.8 m ahead, bank R). Different test layouts (1000-1023 vs 1100-1123),
same generator and riser rule, tipping-only falls (where few trials end before the stairs).

Usage (from unitree_rl_mjlab/):
  python scripts/presentation_round_compare.py --out ../report_content/figures/presentation_r2
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
R1, R2 = "#86b6ef", "#eb6834"  # light blue = before, orange (arm B's slot) = after
HEIGHTS = (0.09, 0.12, 0.15, 0.17)
ROUNDS = (("round 1: label 0.3 m ahead", "eval_results/switch_follow/present/saro/L*_main.json", R1),
          ("round 2: label 0.8 m ahead", "eval_results/switch_follow/present_r2/saro/L*_clf.json", R2))


def clf_runs(pattern: str):
  for f in sorted(glob.glob(pattern)):
    d = json.loads(Path(f).read_text())
    key = next(k for k in d["arms"] if k.startswith("clf:"))
    yield d, d["arms"][key]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="../report_content/figures/presentation_r2")
  args = ap.parse_args()
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=200)
  fig.patch.set_facecolor(SURFACE)
  md = ["## Arm B before / after (tipping-only falls)", "", "| round | " + " | ".join(f"{round(h * 100)} cm" for h in HEIGHTS)
        + " | never chose stairs before stopping, 17 cm |", "|---|" + "---|" * (len(HEIGHTS) + 1)]
  for name, pattern, c in ROUNDS:
    by_h = {h: [] for h in HEIGHTS}
    leads17, never17, n17 = [], 0, 0
    for d, arm in clf_runs(pattern):
      h = round(float(d["conditions"]["step_height"]), 2)
      by_h[h].append(arm["summary"]["success_pct"])
      if h == 0.17:
        kinds = [g["kind"] for g in d["course"]["segments"] if g["kind"] != "flat"]
        iu = kinds.index("stairs_up")
        for t in arm["trials"]:
          l = (t.get("clf_leads") or [None] * len(kinds))[iu]
          n17 += 1
          if l is None:
            never17 += 1
          elif l < 1.49:
            leads17.append(l)
    ys = [np.mean(by_h[h]) for h in HEIGHTS]
    axes[0].plot([h * 100 for h in HEIGHTS], ys, color=c, linewidth=2.4, marker="o", markersize=9,
                 markeredgecolor=SURFACE, markeredgewidth=2, label=name)
    for x, y in zip(HEIGHTS, ys):
      axes[0].text(x * 100, y + (3 if c == R2 else -7), f"{y:.0f}%", ha="center", color=INK, fontsize=10)
    nv = 100 * never17 / max(n17, 1)
    axes[1].hist(leads17, bins=np.arange(0.0, 1.0, 0.02), color=c, alpha=0.85, edgecolor=SURFACE,
                 label=f"{name}: never chose stairs {nv:.1f}%")
    md.append(f"| {name} | " + " | ".join(f"{y:.1f}" for y in ys) + f" | {nv:.1f}% |")
  axes[0].set_xticks([h * 100 for h in HEIGHTS], [f"{round(h * 100)} cm" for h in HEIGHTS])
  axes[0].set_ylim(0, 110)
  axes[0].set_ylabel("course success (%)", color=INK2)
  axes[0].set_title("Onboard switch success by riser height", loc="left", color=INK, fontsize=12)
  axes[0].legend(frameon=False, loc="lower left", labelcolor=INK)
  axes[1].set_xlabel("distance before the first 17 cm up-step when 'stairs' is first chosen (m)", color=INK2)
  axes[1].set_title("When it chooses stairs, at 17 cm", loc="left", color=INK, fontsize=12)
  axes[1].legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
  for ax in axes:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
      ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
      ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
  fig.suptitle("The classifier fix: train on labels that lead the edge by 0.8 m (tipping-only falls)",
               x=0.01, ha="left", color=INK, fontsize=13)
  fig.tight_layout()
  for ext in ("png", "svg"):
    fig.savefig(out / f"fig12_classifier_before_after.{ext}", facecolor=SURFACE)
  (out / "classifier_before_after.md").write_text("\n".join(md) + "\n")
  print("\n".join(md))


if __name__ == "__main__":
  main()
