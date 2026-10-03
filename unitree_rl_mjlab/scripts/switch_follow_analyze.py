"""Analyse switch_follow.py results under the pre-registered rules.

  calib    pool the calibration seeds, print every arm, pick the mapping and the
           best arm in each of the four cells (reactive/anticipatory x hard/soft)
  confirm  pool the confirmation seeds and print the pre-registered comparisons

Rules: `coordination/results/switch-follow-preregistration.md`. Trials are pooled
across the seed files given; per-seed success is printed next to every pooled
number so an effect carried by one seed is visible.

Usage (from unitree_rl_mjlab/):
  python scripts/switch_follow_analyze.py calib eval_results/switch_follow/calib_s40*.json
  python scripts/switch_follow_analyze.py confirm eval_results/switch_follow/confirm_s50*.json \
      --pairs "hard:0.3:label>fixed:stairs" ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

Z = 1.959964


def wilson(k: int, n: int) -> tuple[float, float]:
  if n == 0:
    return (float("nan"), float("nan"))
  p = k / n
  denom = 1 + Z * Z / n
  centre = (p + Z * Z / (2 * n)) / denom
  half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
  return centre - half, centre + half


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
  """Difference of proportions p1 - p2 with Newcombe's hybrid-score 95% interval."""
  p1, p2 = k1 / n1, k2 / n2
  l1, u1 = wilson(k1, n1)
  l2, u2 = wilson(k2, n2)
  d = p1 - p2
  return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
  """Difference of means a - b with a Welch 95% interval (normal quantile; n is in the hundreds)."""
  ma, mb = sum(a) / len(a), sum(b) / len(b)
  va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
  vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
  se = math.sqrt(va / len(a) + vb / len(b))
  return ma - mb, ma - mb - Z * se, ma - mb + Z * se


class Arm:
  def __init__(self, spec: str) -> None:
    self.spec = spec
    self.trials: list[dict] = []
    self.per_seed: list[float] = []
    self.smooth: list[dict] = []

  def add(self, block: dict) -> None:
    self.trials += block["trials"]
    self.per_seed.append(block["summary"]["success_pct"])
    self.smooth.append(block["smoothness"])

  @property
  def n(self) -> int:
    return len(self.trials)

  def count(self, outcome: str) -> int:
    return sum(t["outcome"] == outcome for t in self.trials)

  def pct(self, outcome: str) -> float:
    return 100 * self.count(outcome) / self.n

  def whole_mean(self, metric: str) -> float:
    num = sum(s[metric]["whole"]["mean"] * s[metric]["whole"]["n"] for s in self.smooth)
    return num / sum(s[metric]["whole"]["n"] for s in self.smooth)

  def window_rel(self, metric: str, kind: str, stat: str = "mean") -> float:
    """Boundary-window value over whole-rollout value. Means pool exactly (weighted
    by sample count); a p99 cannot be pooled from per-seed p99s, so it is their average."""
    if stat == "mean":
      num = sum(s[metric][kind]["mean"] * s[metric][kind]["n"] for s in self.smooth)
      return num / sum(s[metric][kind]["n"] for s in self.smooth) / self.whole_mean(metric)
    return sum(s[metric][kind]["p99_rel"] for s in self.smooth) / len(self.smooth)

  def per_trial_rel(self, key: str, metric: str) -> list[float]:
    """Per-trial boundary-window mean, normalised by this arm's whole-rollout mean."""
    whole = self.whole_mean(metric)
    return [t[key] / whole for t in self.trials if t.get(key) is not None]

  def mean(self, key: str) -> float:
    vals = [t[key] for t in self.trials if t.get(key) is not None]
    return sum(vals) / len(vals)


def load(paths: list[str]) -> tuple[dict[str, Arm], list[dict]]:
  arms: dict[str, Arm] = {}
  conds = []
  for p in paths:
    d = json.loads(Path(p).read_text())
    conds.append(d["conditions"])
    for spec, block in d["arms"].items():
      arms.setdefault(spec, Arm(spec)).add(block)
  return arms, conds


def cell(spec: str) -> str | None:
  kind, *rest = spec.split(":")
  if kind == "fixed":
    return None
  start = float(rest[0])
  return ("reactive" if start <= 0.8 else "anticipatory") + " " + kind


def row(a: Arm) -> str:
  lo, hi = wilson(a.count("success"), a.n)
  seeds = " / ".join(f"{s:.1f}" for s in a.per_seed)
  return (f"| `{a.spec}` | {a.n} | **{a.pct('success'):.1f}** ({100 * lo:.1f}-{100 * hi:.1f}) | {seeds} | "
          f"{a.pct('fall'):.1f} | {a.pct('lost'):.1f} | {a.pct('timeout'):.1f} | "
          f"{a.window_rel('action_rate', 'entry'):.3f} | {a.window_rel('action_rate', 'entry', 'p99'):.3f} | "
          f"{a.window_rel('force_rate', 'entry'):.3f} | {a.window_rel('force_rate', 'entry', 'p99'):.3f} | "
          f"{a.mean('track_err'):.3f} |")


HEADER = ("| arm | trials | success % (95% CI) | per seed | fall % | lost % | timeout % | "
          "action rate, entry (rel. mean) | (rel. p99) | force rate, entry (rel. mean) | (rel. p99) | "
          "track err m/s |\n|---|---|---|---|---|---|---|---|---|---|---|---|")


def best(arms: list[Arm]) -> Arm:
  """Highest success; among arms within 1 point of it, the lowest entry-boundary action rate."""
  top = max(a.pct("success") for a in arms)
  near = [a for a in arms if top - a.pct("success") <= 1.0]
  return min(near, key=lambda a: a.window_rel("action_rate", "entry"))


def calib(arms: dict[str, Arm]) -> None:
  print(HEADER)
  for a in arms.values():
    print(row(a))
  footprint = {m: arms[f"hard:0.3:{m}"] for m in ("label", "comp") if f"hard:0.3:{m}" in arms}
  mapping = max(footprint, key=lambda m: footprint[m].pct("success"))
  print(f"\nMapping (higher success at hard:0.3): **{mapping}** "
        + ", ".join(f"{m} {a.pct('success'):.1f}%" for m, a in footprint.items()))
  print("\n| cell | selected arm | success % |\n|---|---|---|")
  for c in ("reactive hard", "reactive soft", "anticipatory hard", "anticipatory soft"):
    pool = [a for a in arms.values() if cell(a.spec) == c and a.spec.endswith(":" + mapping)]
    if pool:
      b = best(pool)
      print(f"| {c} | `{b.spec}` | {b.pct('success'):.1f} |")


def compare(arms: dict[str, Arm], a_spec: str, b_spec: str) -> None:
  a, b = arms[a_spec], arms[b_spec]
  d, lo, hi = newcombe(a.count("success"), a.n, b.count("success"), b.n)
  print(f"\n**`{a_spec}` vs `{b_spec}`**")
  print(f"- success: {a.pct('success'):.1f}% vs {b.pct('success'):.1f}% -> difference "
        f"{100 * d:+.1f} points (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})")
  for label, key, metric in (("action rate", "ar_entry", "action_rate"), ("actuator-force rate", "fr_entry", "force_rate")):
    da, db = a.per_trial_rel(key, metric), b.per_trial_rel(key, metric)
    m, mlo, mhi = welch(da, db)
    print(f"- {label} in entry-boundary windows, relative to own whole-rollout mean: "
          f"{sum(da) / len(da):.3f} vs {sum(db) / len(db):.3f} -> difference {m:+.3f} (95% CI {mlo:+.3f} to {mhi:+.3f})")


def classifier_report(a: Arm) -> None:
  """Agreement with the footprint rule, and the lead at which the classifier first
  selected each non-flat segment's class (positive = before the boundary)."""
  agree = [t["clf_agree"] for t in a.trials]
  print(f"\n**`{a.spec}`**: agrees with the footprint rule on {100 * sum(agree) / len(agree):.1f}% of steps")
  n_seg = len(a.trials[0]["clf_leads"])
  for k in range(n_seg):
    # Only trials that got as far as the segment say anything about the switch into it.
    leads = [t["clf_leads"][k] for t in a.trials]
    got = sorted(v for v in leads if v is not None)
    if not got:
      print(f"- segment {k + 1}: never selected")
      continue
    q = lambda f: got[min(len(got) - 1, int(f * len(got)))]  # noqa: E731
    late = sum(v < 0 for v in got)
    print(f"- segment {k + 1}: selected in {len(got)}/{len(leads)} trials; lead median {q(0.5):+.2f} m, "
          f"10th-90th pct {q(0.1):+.2f} to {q(0.9):+.2f}; {100 * late / len(got):.0f}% after the boundary")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=("calib", "confirm"))
  ap.add_argument("files", nargs="+")
  ap.add_argument("--pairs", nargs="*", default=[], help='Comparisons "A>B" (A minus B).')
  args = ap.parse_args()
  arms, conds = load(args.files)
  print(f"files: {len(args.files)}, seeds: {[c['seed'] for c in conds]}, "
        f"trials/seed: {conds[0]['num_envs']}, terminations: {conds[0]['terminations']}\n")
  if args.mode == "calib":
    calib(arms)
  else:
    print(HEADER)
    for a in arms.values():
      print(row(a))
  for a in arms.values():
    if a.spec.startswith("clf:"):
      classifier_report(a)
  for pair in args.pairs:
    a_spec, b_spec = pair.split(">")
    compare(arms, a_spec, b_spec)


if __name__ == "__main__":
  main()
