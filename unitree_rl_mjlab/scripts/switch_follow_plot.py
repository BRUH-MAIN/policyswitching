"""Success vs. how far ahead of a terrain boundary the switch happens.

One panel per terrain-to-specialist mapping. Hard switches are a line over the
lead; each soft schedule is a horizontal segment spanning the stretch over which
it cross-fades, at the height of its success rate. The shaded region is the part
of the lead axis the onboard height scan cannot reach, i.e. where a schedule
would need the leader's preview.

Usage (from unitree_rl_mjlab/):
  python scripts/switch_follow_plot.py eval_results/switch_follow/calib_s40*.json \
      --out ../report_content/figures/switch_lead_sweep.png
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
HARD, SOFT = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated as a pair)
SCAN_HORIZON = 0.8
MAPPING_TITLE = {
  "label": "Specialist named after the terrain",
  "comp": "Up-stairs sent to the rough specialist",
}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("files", nargs="+")
  ap.add_argument("--out", required=True)
  ap.add_argument("--mappings", nargs="+", default=["label", "comp"])
  args = ap.parse_args()
  arms, conds = load(args.files)

  plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS,
                       "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED})
  fig, axes = plt.subplots(1, len(args.mappings), figsize=(5.6 * len(args.mappings), 4.4), sharey=True,
                           facecolor=SURFACE)
  axes = [axes] if len(args.mappings) == 1 else list(axes)
  fixed = {s.split(":")[1]: a for s, a in arms.items() if s.startswith("fixed:")}
  best_fixed = max(fixed.items(), key=lambda kv: kv[1].pct("success")) if fixed else None

  for ax, mapping in zip(axes, args.mappings):
    ax.set_facecolor(SURFACE)
    hard = sorted((float(s.split(":")[1]), a) for s, a in arms.items()
                  if s.startswith("hard:") and s.endswith(":" + mapping))
    soft = [(float(s.split(":")[1]), float(s.split(":")[2]), a) for s, a in arms.items()
            if s.startswith("soft:") and s.endswith(":" + mapping)]
    x_hi = max([d for d, _ in hard] + [a for a, _, _ in soft]) + 0.25
    x_lo = min(d for d, _ in hard) - 0.2

    ax.axvspan(SCAN_HORIZON, x_hi, color="#f1f0ea", zorder=0, lw=0)
    ax.axvline(0.0, color=AXIS, lw=1, zorder=1)
    ax.text(SCAN_HORIZON + 0.05, 4, "beyond the onboard scan:\nneeds the leader's preview", color=INK_2,
            fontsize=8.5, va="bottom", ha="left")
    ax.text(-0.04, 4, "after the\nboundary", color=MUTED, fontsize=8.5, va="bottom", ha="right")
    if best_fixed is not None:
      y = best_fixed[1].pct("success")
      ax.axhline(y, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
      ax.text(x_hi - 0.05, y - 4.5, f"best single specialist ({best_fixed[0]}), {y:.0f}%", color=INK_2,
              fontsize=8.5, va="top", ha="right")

    xs = [d for d, _ in hard]
    ys = [a.pct("success") for _, a in hard]
    ci = [wilson(a.count("success"), a.n) for _, a in hard]
    ax.vlines(xs, [100 * lo for lo, _ in ci], [100 * hi for _, hi in ci], color=HARD, lw=1, alpha=0.55, zorder=2)
    ax.plot(xs, ys, color=HARD, lw=2, marker="o", ms=6.5, mec=SURFACE, mew=1.5, zorder=3, label="hard switch")
    for i, (a, b, arm) in enumerate(sorted(soft, key=lambda t: (t[0], t[1]))):
      y = arm.pct("success")
      ax.plot([b, a], [y, y], color=SOFT, lw=1.6, marker="|", ms=6, mew=1.6, zorder=3,
              label="soft cross-fade (span = where it blends)" if i == 0 else None)

    peak = max(hard, key=lambda t: t[1].pct("success"))
    ax.annotate(f"{peak[1].pct('success'):.0f}% at {peak[0]:g} m", (peak[0], peak[1].pct("success")),
                xytext=(peak[0] + 0.28, peak[1].pct("success") + 3.5), color=INK, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 104)
    ax.set_xlabel("switch lead: metres before the terrain boundary")
    ax.set_title(MAPPING_TITLE.get(mapping, mapping), fontsize=10, color=INK, loc="left", pad=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
      ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
  axes[0].set_ylabel("trials that cross the whole course (%)")
  axes[0].legend(loc="lower center", frameon=False, fontsize=8.5, labelcolor=INK_2, ncols=1,
                 bbox_to_anchor=(0.62, 0.16))
  n = next(iter(arms.values())).n
  fig.suptitle("Switching at the boundary works best; switching earlier gives the gain back",
               x=0.012, ha="left", fontsize=11.5, color=INK, fontweight="bold")
  fig.text(0.012, 0.005, f"Course multi L1, follow task, seeds {[c['seed'] for c in conds]}, {n} trials per arm; "
           "bars are 95% Wilson intervals.", color=MUTED, fontsize=8)
  fig.tight_layout(rect=(0, 0.03, 1, 0.95))
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=170, facecolor=SURFACE)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
