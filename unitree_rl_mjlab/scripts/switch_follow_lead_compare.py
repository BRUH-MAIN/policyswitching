"""Success against hard-switch lead for two sets of runs on one axis.

Used to compare the same sweep with a different checkpoint in the stairs slot.
Each set also gets a dashed line at its own stairs-only success, so "switching
is no better than not switching" is visible as the line sitting on its dash.

Usage (from unitree_rl_mjlab/):
  python scripts/switch_follow_lead_compare.py \
      --set "stairs specialist as trained (iteration 9999)" eval_results/switch_follow/confirm_noise_s50*.json \
      --set "its checkpoint from before the collapse (iteration 4800)" eval_results/switch_follow/it4800_L1_noisy_s50*.json \
      --out ../report_content/figures/switch_lead_two_checkpoints.png
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
COLORS = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2 (validated as a pair)
SCAN_HORIZON = 0.8


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--set", action="append", nargs="+", required=True, metavar=("NAME", "FILE"),
                  help="Series name followed by its result files; give exactly two --set.")
  ap.add_argument("--mapping", default="label")
  ap.add_argument("--title", default="Success against switch lead")
  ap.add_argument("--note", default="")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  if len(args.set) != 2:
    ap.error("exactly two --set groups")
  sets = [(s[0], load(s[1:])) for s in args.set]

  plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS,
                       "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED})
  fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
  ax.set_facecolor(SURFACE)
  x_lo, x_hi = 1e9, -1e9
  series = []
  for (name, (arms, _)), color in zip(sets, COLORS):
    hard = sorted((float(s.split(":")[1]), a) for s, a in arms.items()
                  if s.startswith("hard:") and s.endswith(":" + args.mapping))
    series.append((name, color, hard, arms.get("fixed:stairs")))
    x_lo, x_hi = min(x_lo, hard[0][0]), max(x_hi, hard[-1][0])
  x_lo, x_hi = x_lo - 0.15, x_hi + 0.75

  ax.axvspan(SCAN_HORIZON, x_hi, color="#f1f0ea", zorder=0, lw=0)
  ax.axvline(0.0, color=AXIS, lw=1, zorder=1)
  ax.text(SCAN_HORIZON + 0.04, 3, "beyond the onboard scan:\nneeds the leader's preview", color=INK_2,
          fontsize=8.5, va="bottom", ha="left")
  top = max(series, key=lambda t: t[2][-1][1].pct("success"))
  for name, color, hard, fixed in series:
    xs = [d for d, _ in hard]
    ys = [a.pct("success") for _, a in hard]
    ci = [wilson(a.count("success"), a.n) for _, a in hard]
    ax.vlines(xs, [100 * lo for lo, _ in ci], [100 * hi for _, hi in ci], color=color, lw=1, alpha=0.55, zorder=2)
    ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6.5, mec=SURFACE, mew=1.5, zorder=3, label=name)
    above = (name, color, hard, fixed) == top  # upper series labelled above its last point, lower one below
    ax.text(xs[-1], ys[-1] + (3.0 if above else -3.2), f"{ys[-1]:.0f}%", color=INK, fontsize=9,
            va="bottom" if above else "top", ha="center")
    if fixed is not None:
      y = fixed.pct("success")
      ax.axhline(y, color=color, lw=1, ls=(0, (4, 3)), alpha=0.8, zorder=1)
      ax.text(x_hi - 0.03, y - 1.4, f"stairs policy alone, {y:.0f}%", color=INK_2, fontsize=8.5, va="top", ha="right")
  ax.set_xlim(x_lo, x_hi)
  ax.set_ylim(0, 104)
  ax.set_xlabel("switch lead: metres before the terrain boundary")
  ax.set_ylabel("trials that cross the whole course (%)")
  ax.grid(axis="y", color=GRID, lw=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
  ax.tick_params(length=0)
  ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.16), frameon=False, fontsize=8.5, labelcolor=INK_2,
            title="policy in the stairs slot", title_fontsize=8.5, alignment="left")
  n = next(iter(sets[0][1][0].values())).n
  seeds = sorted({c["seed"] for _, (_, cs) in sets for c in cs})
  fig.suptitle(args.title, x=0.012, ha="left", fontsize=11.5, color=INK, fontweight="bold")
  fig.text(0.012, 0.008, f"{args.note} Seeds {seeds}, {n} trials per point; bars are 95% Wilson intervals.",
           color=MUTED, fontsize=8)
  fig.tight_layout(rect=(0, 0.035, 1, 0.94))
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=170, facecolor=SURFACE)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
