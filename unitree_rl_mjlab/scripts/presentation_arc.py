"""Project arc for the presentation: stairs success across specialist versions.

Single straight flights with a leader (scripts/switch_follow_real_stairs.sh and
scripts/stairs_robustness.sh), tipping-only falls (`saro`), trials pooled over the seed files
present. Left: 5-step flights at 15 and 17 cm, stairs v2 -> v8a, with generalist v2 for
reference. Right: 10-step flights at 17 cm, v7a -> v8b, with and without a 20 ms actuator delay.

Usage (from unitree_rl_mjlab/):
  python scripts/presentation_arc.py --out ../report_content/figures/presentation
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

RS = "eval_results/switch_follow/real_stairs"
RB = "eval_results/switch_follow/robustness"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
UP, DOWN = "#2a78d6", "#eb6834"  # categorical slots 1-2

FIVE_STEP = (  # label, file prefix, arm
  ("generalist v2", f"{RS}/", "fixed:gen2"),
  ("stairs v2", f"{RS}/", "fixed:stairs"),
  ("v5a", f"{RS}/v5a_it9999_n5_", "fixed:stairs"),
  ("v6a", f"{RS}/v6alap_final_n5_", "fixed:stairs"),
  ("v7a", f"{RS}/v7acl_c1600_n5_", "fixed:stairs"),
  ("v8a", f"{RS}/v8acl_final_n5_", "fixed:stairs"),
)
TEN_STEP = (  # label, pattern for (direction, height), suffix-free
  ("v7a", lambda d, h: f"{RS}/v7acl_c1600_n10_stairs_{d}_h{h}_saro_s*.json"),
  ("v8a", lambda d, h: f"{RS}/v8acl_final_n10_stairs_{d}_h{h}_saro_s*.json"),
  ("v8b", lambda d, h: f"{RB}/v8b_cl_it1999_n10_stairs_{d}_h{h}_nominal.json"),
  ("v8a, 20 ms delay", lambda d, h: f"{RB}/v8a_final_n10_stairs_{d}_h{h}_delay1.json"),
  ("v8b, 20 ms delay", lambda d, h: f"{RB}/v8b_cl_it1999_n10_stairs_{d}_h{h}_delay1.json"),
)


def pooled(paths: list[str], arm: str = "fixed:stairs") -> tuple[float, int]:
  t = [x for p in paths for x in json.loads(Path(p).read_text())["arms"][arm]["trials"]]
  return (100 * sum(x["outcome"] == "success" for x in t) / len(t), len(t)) if t else (float("nan"), 0)


def five_step_files(prefix: str, d: str, h: str) -> list[str]:
  # The stairs-v2 run is untagged (prefix ends at the directory; 5-step by default).
  return sorted(glob.glob(f"{prefix}stairs_{d}_h{h}_saro_s*.json"))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="../report_content/figures/presentation")
  args = ap.parse_args()
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), dpi=200, gridspec_kw=dict(width_ratios=[1.35, 1]))
  fig.patch.set_facecolor(SURFACE)
  md = ["## Project arc: single flights, tipping-only success %", ""]

  # Left: 5-step flights, 15 and 17 cm.
  ax = axes[0]
  cols = [(d, h) for h in ("0.15", "0.17") for d in ("up", "down")]
  md += ["5-step flights | " + " | ".join(f"{d} {round(float(h) * 100)} cm" for d, h in cols), "|---|" + "---|" * len(cols)]
  w = 0.19
  for i, (label, prefix, arm) in enumerate(FIVE_STEP):
    vals = []
    for d, h in cols:
      v, n = pooled(five_step_files(prefix, d, h), arm)
      vals.append(v)
    md.append(f"| {label} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
    for j, ((d, h), v) in enumerate(zip(cols, vals)):
      x = i + (j - 1.5) * w
      ax.bar(x, v, width=w * 0.9, color=UP if d == "up" else DOWN, alpha=1.0 if h == "0.17" else 0.55,
             edgecolor=SURFACE, linewidth=1.2)
  ax.set_xticks(range(len(FIVE_STEP)), [l for l, _, _ in FIVE_STEP], color=INK)
  ax.set_ylim(0, 118)
  ax.set_yticks(range(0, 101, 20))
  ax.set_ylabel("flights crossed (%)", color=INK2)
  ax.set_title("5-step flights: each bar group = up 15, down 15, up 17, down 17 cm", loc="left", color=INK, fontsize=11)

  # Right: 10-step flights, 17 cm.
  ax = axes[1]
  md += ["", "10-step flights, 17 cm | up | down", "|---|---|---|"]
  for i, (label, pat) in enumerate(TEN_STEP):
    vals = [pooled(sorted(glob.glob(pat(d, "0.17"))))[0] for d in ("up", "down")]
    md.append(f"| {label} | {vals[0]:.1f} | {vals[1]:.1f} |")
    for j, (d, v) in enumerate(zip(("up", "down"), vals)):
      x = i + (j - 0.5) * 0.36
      ax.bar(x, v, width=0.33, color=UP if d == "up" else DOWN, edgecolor=SURFACE, linewidth=1.2)
      ax.text(x, (v if v == v else 0) + 1.5, f"{v:.0f}", ha="center", color=INK, fontsize=8)
  ax.set_xticks(range(len(TEN_STEP)), [l.replace(", ", "\n") for l, _ in TEN_STEP], color=INK, fontsize=9)
  ax.set_ylim(0, 108)
  ax.set_title("10-step flights at 17 cm (building-length stairs)", loc="left", color=INK, fontsize=11)

  handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=a) for c, a in ((UP, 0.55), (DOWN, 0.55), (UP, 1), (DOWN, 1))]
  axes[0].legend(handles, ["up 15 cm", "down 15 cm", "up 17 cm", "down 17 cm"], frameon=False, ncol=4,
                 loc="upper left", fontsize=9, labelcolor=INK)
  for ax in axes:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
      ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
      ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
  fig.suptitle("How the stairs specialist got there: success on real riser heights, tipping-only falls",
               x=0.01, ha="left", color=INK, fontsize=13)
  fig.tight_layout()
  for ext in ("png", "svg"):
    fig.savefig(out / f"fig10_project_arc.{ext}", facecolor=SURFACE)
  (out / "project_arc.md").write_text("\n".join(md) + "\n")
  print("\n".join(md))


if __name__ == "__main__":
  main()
