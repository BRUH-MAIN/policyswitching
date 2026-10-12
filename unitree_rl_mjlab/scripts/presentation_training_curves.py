"""Training curves from SLURM training logs (RSL-RL's per-iteration console block).

Fairness evidence for the presentation: GeneralistV3 trained to a plateau, like the stairs v8b
specialist it was warm-started from. Plots, against iteration (100-iteration rolling mean):
achieved / commanded speed, the fall rate, and the mean terrain row per population.

Usage (from unitree_rl_mjlab/):
  python scripts/presentation_training_curves.py \
      --run "stairs v8b=../go2-spec-12747.out" --run "generalist V3=../go2-spec-12816.out" \
      --out ../report_content/figures/presentation
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
RUN_COLORS = ("#e87ba4", "#eda100", "#2a78d6", "#1baf7a")  # stairs v8b alone, matched generalist: same slots as the main figures
ROW_TERMS = (  # log key suffix -> label
  ("terrain_row_mean_up", "up-flights"), ("terrain_row_mean_down", "down-flights"),
  ("terrain_row_mean_stairs_up", "up-flights"), ("terrain_row_mean_stairs_down", "down-flights"),
  ("terrain_row_mean_flat", "flat"), ("terrain_row_mean_rough", "rough"),
  ("terrain_row_mean", "all rows"),  # single-terrain runs log only this; used when nothing above is
)
LINESTYLES = {"up-flights": "-", "down-flights": "--", "flat": ":", "rough": "-.", "all rows": "-"}


def parse(path: Path) -> dict[str, np.ndarray]:
  text = path.read_text(errors="ignore")
  blocks = text.split("Learning iteration ")[1:]
  keys = ["actual_speed", "cmd_speed", "fell_over", "illegal_contact", "limb_contact"] + [k for k, _ in ROW_TERMS]
  out: dict[str, list] = {k: [] for k in keys}
  out["iteration"] = []
  for b in blocks:
    out["iteration"].append(int(b.split("/")[0]))
    for k in keys:
      m = re.search(r"/" + re.escape(k) + r":\s*(-?[\d.]+)", b)
      out[k].append(float(m.group(1)) if m else np.nan)
  return {k: np.array(v, dtype=float) for k, v in out.items()}


def roll(v: np.ndarray, n: int = 100) -> np.ndarray:
  out = np.full_like(v, np.nan)
  for i in range(len(v)):
    w = v[max(0, i - n + 1): i + 1]
    w = w[~np.isnan(w)]
    out[i] = w.mean() if len(w) else np.nan
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", action="append", required=True, metavar="LABEL=LOG")
  ap.add_argument("--out", default="../report_content/figures/presentation")
  ap.add_argument("--name", default="fig11_training_curves")
  ap.add_argument("--relative", action="store_true", help="x = iterations since each run's first logged one")
  ap.add_argument("--title", default="Both policies trained to a plateau (100-iteration rolling means)")
  args = ap.parse_args()
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  runs, colors = [], []
  for i, r in enumerate(args.run):  # LABEL=LOG or LABEL=LOG@#rrggbb
    lbl, _, rest = r.partition("=")
    path, _, color = rest.partition("@")
    d = parse(Path(path))
    if args.relative:
      d["iteration"] = d["iteration"] - d["iteration"][0]
    runs.append((lbl, d))
    colors.append(color or RUN_COLORS[i % len(RUN_COLORS)])

  fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=200)
  fig.patch.set_facecolor(SURFACE)
  md = ["## Training: last 200 iterations (means)", "", "| run | iterations | speed / commanded | fell_over | illegal_contact | limb_contact | row means |",
        "|---|---|---|---|---|---|---|"]
  for (label, d), c in zip(runs, colors):
    it = d["iteration"]
    speed = 100 * d["actual_speed"] / d["cmd_speed"]
    axes[0].plot(it, roll(speed), color=c, linewidth=2, label=label)
    axes[1].plot(it, roll(d["fell_over"]), color=c, linewidth=2, label=label)
    rows_txt = []
    seen = set()
    for key, rl in ROW_TERMS:
      if np.all(np.isnan(d[key])) or rl in seen or (rl == "all rows" and seen):
        continue
      seen.add(rl)
      axes[2].plot(it, roll(d[key]), color=c, linewidth=2, linestyle=LINESTYLES[rl], label=f"{label}: {rl}")
      rows_txt.append(f"{rl} {np.nanmean(d[key][-200:]):.2f}")
    last = slice(-200, None)
    md.append(f"| {label} | {int(it[-1]) + 1}{' (this run)' if args.relative else ''} | {np.nanmean(speed[last]):.0f}% | {np.nanmean(d['fell_over'][last]):.3f} | "
              f"{np.nanmean(d['illegal_contact'][last]):.3f} | {np.nanmean(d['limb_contact'][last]):.4f} | {', '.join(rows_txt)} |")
  titles = ("achieved / commanded speed (%)", "fell over per iteration (all robots; V3 includes\nits flat and rough robots, so not like-for-like)", "mean terrain row (0-9; higher = harder)")
  for ax, t in zip(axes, titles):
    ax.set_title(t, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("training iteration (from the warm start)", color=INK2)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
      ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
      ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2)
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
  axes[0].legend(frameon=False, labelcolor=INK)
  axes[2].legend(frameon=False, fontsize=8, labelcolor=INK)
  fig.suptitle(args.title, x=0.01, ha="left", color=INK, fontsize=13)
  fig.tight_layout()
  for ext in ("png", "svg"):
    fig.savefig(out / f"{args.name}.{ext}", facecolor=SURFACE)
  (out / f"{args.name}.md").write_text("\n".join(md) + "\n")
  print("\n".join(md))


if __name__ == "__main__":
  main()
