"""Pool per-cell baseline/closed-loop JSONs into per-arm rates with Wilson intervals.

Trials are pooled across goal offsets and seeds; every cell is one env per trial, so
pooling is over independent trials (no episode pooling -- findings.md bug #15).
Also reports the pre-registered Gate-A/Gate-B checks when `--gate` is passed.

Usage (from unitree_rl_mjlab/):
  PYTHONPATH=$PWD python scripts/vlm_nav_summarize.py eval_results/vlm_nav/phase1_confirm/*.json --gate
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  if n == 0:
    return (0.0, 0.0)
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return (100 * (c - h), 100 * (c + h))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("files", nargs="+")
  ap.add_argument("--gate", action="store_true", help="Apply the pre-registered Gate A / Gate B checks.")
  ap.add_argument("--gate-margin", type=float, default=15.0, help="Gate B: percentage points oracle must exceed the best fixed arm.")
  args = ap.parse_args()

  by_course_arm: dict[tuple[str, str], list[dict]] = defaultdict(list)
  for f in args.files:
    d = json.loads(Path(f).read_text())
    course = d["course"]["name"] if "course" in d else d["conditions"]["course"]
    for arm, payload in d["arms"].items():
      by_course_arm[(course, arm)].extend(payload["trials"])

  rows = {}
  for (course, arm), trials in sorted(by_course_arm.items()):
    n = len(trials)
    k = sum(t["success"] for t in trials)
    lo, hi = wilson(k, n)
    falls = sum(t["fell"] for t in trials)
    timeouts = sum(t.get("timed_out", t.get("outcome") == "timeout") for t in trials)
    match = sum(t["policy_match_frac"] for t in trials) / max(1, n)
    rows[(course, arm)] = dict(n=n, success=100 * k / n, ci=(lo, hi), fall=100 * falls / n,
                               timeout=100 * timeouts / n, match=100 * match, k=k)
    print(f"{course:12s} {arm:16s} n={n:4d} success={100*k/n:5.1f}% (CI {lo:4.1f}-{hi:4.1f}) "
          f"fall={100*falls/n:5.1f}% timeout={100*timeouts/n:5.1f}% policy_match={100*match:5.1f}%")

  if args.gate:
    print()
    for course in sorted({c for c, _ in rows}):
      arms = {a: r for (c, a), r in rows.items() if c == course}
      if "oracle" not in arms:
        continue
      o = arms["oracle"]
      fixed = {a: r for a, r in arms.items() if a not in ("oracle",)}
      best = max(fixed.items(), key=lambda kv: kv[1]["success"], default=(None, None))
      gate_a = o["success"] >= 80
      print(f"[{course}] Gate A (oracle >= 80%): {'PASS' if gate_a else 'FAIL'} ({o['success']:.1f}%, CI {o['ci'][0]:.1f}-{o['ci'][1]:.1f})")
      if best[0] is not None:
        b = best[1]
        gap = o["success"] - b["success"]
        disjoint = o["ci"][0] > b["ci"][1]
        print(f"[{course}] Gate B (oracle beats best fixed '{best[0]}' by >= {args.gate_margin:g} pts, CIs disjoint): "
              f"{'PASS' if gap >= args.gate_margin and disjoint else 'FAIL'} "
              f"(gap {gap:+.1f} pts; oracle CI {o['ci'][0]:.1f}-{o['ci'][1]:.1f} vs {best[0]} CI {b['ci'][0]:.1f}-{b['ci'][1]:.1f})")


if __name__ == "__main__":
  main()
