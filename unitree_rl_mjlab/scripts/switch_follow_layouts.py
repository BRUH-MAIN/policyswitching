"""Analyse the randomised-layout evaluation (Addendum 7): the layout is the unit.

For each arm: success per layout, then the mean over layouts with a bootstrap
interval over layouts. For each comparison: the per-layout paired difference,
its mean, a bootstrap interval, and in how many layouts the first arm is ahead.
Resampling trials instead of layouts would understate the uncertainty, because
trials on one layout share its geometry.

Usage (from unitree_rl_mjlab/):
  python scripts/switch_follow_layouts.py eval_results/switch_follow/random
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COMPARISONS = (
  ("A", "fixed:stairs", "A", "fixed:gen1", "stairs v2 alone - first generalist"),
  ("A", "fixed:stairs", "A", "fixed:gen2", "stairs v2 alone - generalist v2"),
  ("A", "fixed:gen2", "A", "fixed:gen1", "generalist v2 - first generalist"),
  ("A", "hard:0.3:label", "A", "fixed:stairs", "switch 0.3 (stairs v2 in bank) - stairs v2 alone"),
  ("A", "hard:1.5:label", "A", "hard:0.3:label", "early 1.5 - on-time 0.3, stairs v2"),
  ("A", "hard:-0.3:label", "A", "hard:0.3:label", "late -0.3 - on-time 0.3, stairs v2"),
  ("A", "blind:stairs", "A", "fixed:stairs", "stairs v2 without scan - with scan"),
  ("B", "hard:0.3:label", "B", "fixed:stairs", "switch 0.3 - as-trained stairs alone"),
  ("B", "hard:1.5:label", "B", "hard:0.3:label", "early 1.5 - on-time 0.3, as-trained stairs"),
  ("A", "fixed:stairs", "B", "fixed:stairs", "stairs v2 alone - as-trained stairs alone"),
)


def boot(values: np.ndarray, rng: np.random.Generator, n: int = 10000) -> tuple[float, float]:
  idx = rng.integers(0, len(values), size=(n, len(values)))
  means = values[idx].mean(axis=1)
  return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("directory")
  args = ap.parse_args()
  rng = np.random.default_rng(0)
  runs: dict[str, dict[int, dict]] = {"A": {}, "B": {}}
  for f in sorted(Path(args.directory).glob("[AB]_s*.json")):
    d = json.loads(f.read_text())
    runs[f.name[0]][int(f.stem.split("_s")[1])] = d
  seeds = sorted(set(runs["A"]) & set(runs["B"]))
  print(f"layouts with both runs complete: {len(seeds)} (seeds {seeds[0]}-{seeds[-1]})")
  levels = [runs["A"][s]["layout"]["level"] for s in seeds]
  print(f"levels: L1 x{levels.count('L1')}, L2 x{levels.count('L2')}; "
        f"leader speed {min(runs['A'][s]['layout']['leader_speed'] for s in seeds):.2f}-"
        f"{max(runs['A'][s]['layout']['leader_speed'] for s in seeds):.2f} m/s; "
        f"trials per layout {runs['A'][seeds[0]]['conditions']['num_envs']}\n")

  def success(run: str, arm: str) -> np.ndarray:
    return np.array([runs[run][s]["arms"][arm]["summary"]["success_pct"] for s in seeds])

  print("| run | arm | mean success over layouts (95% CI) | worst layout | best layout | L1 mean | L2 mean |")
  print("|---|---|---|---|---|---|---|")
  for run in ("A", "B"):
    for arm in runs[run][seeds[0]]["arms"]:
      v = success(run, arm)
      lo, hi = boot(v, rng)
      l1 = v[[lv == "L1" for lv in levels]]
      l2 = v[[lv == "L2" for lv in levels]]
      print(f"| {run} | `{arm}` | **{v.mean():.1f}** ({lo:.1f}-{hi:.1f}) | {v.min():.1f} | {v.max():.1f} | "
            f"{l1.mean():.1f} | {l2.mean():.1f} |")

  print("\n| comparison | mean per-layout difference (95% CI) | layouts ahead / behind / tied | range |")
  print("|---|---|---|---|")
  for ra, a, rb, b, name in COMPARISONS:
    d = success(ra, a) - success(rb, b)
    lo, hi = boot(d, rng)
    print(f"| {name} | **{d.mean():+.1f}** ({lo:+.1f} to {hi:+.1f}) | "
          f"{int((d > 0).sum())} / {int((d < 0).sum())} / {int((d == 0).sum())} | {d.min():+.1f} to {d.max():+.1f} |")


if __name__ == "__main__":
  main()
