"""Cross-terrain eval matrix: every trained policy x every terrain class x difficulty.

This is the experiment that tests the project's premise before the switching
module gets built. Switching between specialists only has something to gain if
specialists degrade off their own terrain -- if the diagonal of this matrix
doesn't dominate the off-diagonal, that is the most important result the
project has, and it arrives before Phases 4-5 rather than after.

Runs scripts/eval_checkpoint.py once per cell under pinned conditions (terrain
curriculum off, fixed difficulty, final-training command range), writing one
JSON per cell. Cells whose JSON already exists are skipped, so a job cut off by
walltime just gets resubmitted. Finally writes summary.md with:

  - fall% / per-step velocity error / stalled% for every policy x terrain,
    one table per difficulty
  - each specialist's home-terrain vs. off-terrain numbers
  - the height-scan ablation: how much each stock-PPO policy's numbers move when
    its exteroception is replaced by an uninformative constant

Policies are discovered, not hardcoded: each local experiment is included only
once its latest checkpoint reaches --min-iteration (so a half-trained run never
silently lands in the table), plus the final PAS checkpoint in both oracle and
estimator-only mode.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_ckpt_resume import latest_local_checkpoint  # noqa: E402

TERRAINS = ("flat", "rough", "stairs", "gaps", "mixed")

# label, task, experiment name under logs/rsl_rl, home terrain (None = generalist)
LOCAL_POLICIES = (
  ("flat", "Unitree-Go2-Spec-Flat", "go2_spec_flat", "flat"),
  ("rough", "Unitree-Go2-Spec-Rough", "go2_spec_rough", "rough"),
  ("stairs", "Unitree-Go2-Spec-Stairs", "go2_spec_stairs", "stairs"),
  ("gaps_blend", "Unitree-Go2-Spec-Gaps", "go2_spec_gaps", "gaps"),
  ("gaps_warm", "Unitree-Go2-Spec-GapsWarm", "go2_spec_gapswarm", "gaps"),
  ("generalist", "Unitree-Go2-Generalist", "go2_generalist", None),
)
PAS_TASK = "Unitree-Go2-PAS-Anneal"
# Cluster-local fast path -- if this exact file exists, use it directly and skip
# the HF round-trip. Not assumed to exist: the laptop side (docs/CLAUDE.laptop.md)
# pulls PAS straight from HF via --hf-repo into eval_ckpts/, never populating this
# path, so PAS_HF_REPO/PAS_HF_STAGE below is the path that actually runs there.
PAS_CHECKPOINT = "logs/rsl_rl/go2_pas/2026-09-05_17-59-43_stage2/model_79998.pt"
PAS_HF_REPO = "RohanRamesh/go2-pas-saro"
PAS_HF_STAGE = "stage2"


@dataclass
class Policy:
  label: str
  task: str
  home: str | None = None
  checkpoint: str | None = None  # local path; mutually exclusive with hf_repo
  hf_repo: str | None = None
  hf_stage: str | None = None
  extra_args: list[str] = field(default_factory=list)
  ablatable: bool = True  # reads raw height_scan through a stock MLP

  def checkpoint_args(self) -> list[str]:
    if self.checkpoint is not None:
      return ["--checkpoint", self.checkpoint]
    assert self.hf_repo is not None
    return ["--hf-repo", self.hf_repo, "--hf-stage", self.hf_stage]


def resolve_policies(mjlab_dir: Path, min_iteration: int, pas_checkpoint: str | None) -> list[Policy]:
  policies = []
  for label, task, experiment, home in LOCAL_POLICIES:
    found = latest_local_checkpoint(str(mjlab_dir), experiment)
    if found is None:
      print(f"[SKIP] {label}: no checkpoint under logs/rsl_rl/{experiment}/")
      continue
    iteration, run_dir = found
    if iteration < min_iteration:
      print(f"[SKIP] {label}: latest checkpoint is iteration {iteration} < --min-iteration {min_iteration}")
      continue
    ckpt = mjlab_dir / "logs" / "rsl_rl" / experiment / run_dir / f"model_{iteration}.pt"
    policies.append(Policy(label, task, home, checkpoint=str(ckpt)))
    print(f"[POLICY] {label}: {ckpt}")

  # PAS: prefer an explicit --pas-checkpoint override, then the cluster-local
  # fast path, then fall back to HF -- the only one that works unmodified on
  # the laptop, which never writes PAS_CHECKPOINT's literal path (see above).
  pas_local = Path(pas_checkpoint) if pas_checkpoint else mjlab_dir / PAS_CHECKPOINT
  if pas_local.is_file():
    for label, prob in (("pas_oracle", "1.0"), ("pas_estimator", "0.0")):
      policies.append(
        Policy(label, PAS_TASK, None, checkpoint=str(pas_local),
               extra_args=["--anneal-prob", prob], ablatable=False)
      )
      print(f"[POLICY] {label}: {pas_local} (--anneal-prob {prob})")
  else:
    print(f"[INFO] PAS: {pas_local} not found locally; falling back to "
          f"hf.co/{PAS_HF_REPO}/{PAS_HF_STAGE} (eval_checkpoint.py caches this once, "
          "under eval_ckpts/, then reuses it for every PAS cell)")
    for label, prob in (("pas_oracle", "1.0"), ("pas_estimator", "0.0")):
      policies.append(
        Policy(label, PAS_TASK, None, hf_repo=PAS_HF_REPO, hf_stage=PAS_HF_STAGE,
               extra_args=["--anneal-prob", prob], ablatable=False)
      )
      print(f"[POLICY] {label}: hf.co/{PAS_HF_REPO}/{PAS_HF_STAGE} (--anneal-prob {prob})")
  return policies


def cell_name(label: str, terrain: str, difficulty: float, seed: int, ablate: bool) -> str:
  return f"{label}__{terrain}__d{difficulty:g}__s{seed}{'__noscan' if ablate else ''}.json"


def run_matrix(args: argparse.Namespace, mjlab_dir: Path, out_dir: Path) -> int:
  policies = resolve_policies(mjlab_dir, args.min_iteration, args.pas_checkpoint)
  if args.only:
    policies = [p for p in policies if p.label in args.only]

  jobs = []
  for p in policies:
    for seed in args.seeds:
      for difficulty in args.difficulties:
        for terrain in TERRAINS:
          jobs.append((p, terrain, difficulty, seed, False))
      if p.ablatable and not args.no_ablation:
        for difficulty in args.ablation_difficulties:
          for terrain in TERRAINS[:-1]:
            jobs.append((p, terrain, difficulty, seed, True))

  print(f"[INFO] {len(jobs)} cells, {len(policies)} policies -> {out_dir}")
  failures = []
  for i, (p, terrain, difficulty, seed, ablate) in enumerate(jobs, 1):
    out = out_dir / cell_name(p.label, terrain, difficulty, seed, ablate)
    tag = f"[{i}/{len(jobs)}] {out.name}"
    if out.exists() and not args.force:
      print(f"{tag} exists, skipping")
      continue
    cmd = [
      sys.executable, "scripts/eval_checkpoint.py",
      "--task", p.task,
      *p.checkpoint_args(),
      "--terrain", terrain,
      "--difficulty", str(difficulty),
      "--seed", str(seed),
      "--num-envs", str(args.num_envs),
      "--steps", str(args.steps),
      "--label", p.label,
      "--json-out", str(out),
      *p.extra_args,
      *(["--ablate-height-scan"] if ablate else []),
    ]
    if args.dry_run:
      print(f"{tag} DRY RUN: {' '.join(cmd)}")
      continue
    print(f"{tag} running", flush=True)
    rc = subprocess.run(cmd, cwd=mjlab_dir).returncode
    if rc != 0 or not out.exists():
      print(f"{tag} FAILED (exit {rc})")
      failures.append(out.name)

  if failures:
    print(f"[ERROR] {len(failures)} cell(s) failed: {failures}")
  return 1 if failures else 0


def _fmt(value: float | None, spec: str) -> str:
  return "–" if value is None else format(value, spec)


def summarize(out_dir: Path) -> str:
  cells: dict[tuple, list[dict]] = {}
  order: list[str] = []
  for path in sorted(out_dir.glob("*.json")):
    r = json.loads(path.read_text())
    c = r["conditions"]
    key = (r["label"], c["terrain"], c["difficulty"], bool(c.get("ablate_height_scan")))
    cells.setdefault(key, []).append(r)
    if r["label"] not in order:
      order.append(r["label"])
  if not cells:
    return "No results yet.\n"

  wanted = [label for label, *_ in LOCAL_POLICIES] + ["pas_oracle", "pas_estimator"]
  order = sorted(order, key=lambda lab: wanted.index(lab) if lab in wanted else len(wanted))
  homes = {label: home for label, _, _, home in LOCAL_POLICIES}

  def metric(key: tuple, group: str, name: str) -> float | None:
    vals = [r[group].get(name) for r in cells.get(key, []) if r[group].get(name) is not None]
    return sum(vals) / len(vals) if vals else None

  lines = [
    "# Cross-terrain eval matrix",
    "",
    "Each cell: **fall %** · per-step linear velocity error (m/s) · stalled-while-commanded %.",
    "Terrain curriculum off, difficulty pinned, command range = final training stage;",
    "values averaged over seeds. Generated by `a100/eval_matrix.py`.",
    "",
  ]
  difficulties = sorted({k[2] for k in cells if not k[3]})
  for d in difficulties:
    lines += [f"## Difficulty {d:g}", "", "| policy | " + " | ".join(TERRAINS) + " |",
              "|---|" + "---|" * len(TERRAINS)]
    for label in order:
      row = []
      for t in TERRAINS:
        k = (label, t, d, False)
        if k not in cells:
          row.append("")
          continue
        cell = (f"{_fmt(metric(k, 'survival', 'fall_pct'), '.1f')} · "
                f"{_fmt(metric(k, 'locomotion', 'lin_vel_error_per_step'), '.2f')} · "
                f"{_fmt(metric(k, 'locomotion', 'stalled_pct'), '.0f')}")
        row.append(f"**{cell}**" if homes.get(label) == t else cell)
      lines.append(f"| {label} | " + " | ".join(row) + " |")
    lines.append("")

    lines += [f"### Home vs. off-terrain, difficulty {d:g}", "",
              "| specialist | home fall % | off-terrain fall % (mean) | home error | off-terrain error (mean) |",
              "|---|---|---|---|---|"]
    for label in order:
      home = homes.get(label)
      if home is None or (label, home, d, False) not in cells:
        continue
      off = [t for t in TERRAINS[:-1] if t != home and (label, t, d, False) in cells]

      def mean_over(ts: list[str], group: str, name: str) -> float | None:
        vals = [metric((label, t, d, False), group, name) for t in ts]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

      lines.append(
        f"| {label} | {_fmt(metric((label, home, d, False), 'survival', 'fall_pct'), '.1f')} "
        f"| {_fmt(mean_over(off, 'survival', 'fall_pct'), '.1f')} "
        f"| {_fmt(metric((label, home, d, False), 'locomotion', 'lin_vel_error_per_step'), '.2f')} "
        f"| {_fmt(mean_over(off, 'locomotion', 'lin_vel_error_per_step'), '.2f')} |"
      )
    lines.append("")

  ablation_keys = sorted({(k[0], k[1], k[2]) for k in cells if k[3]})
  if ablation_keys:
    lines += ["## Height-scan ablation", "",
              "Same cell with the actor's height_scan replaced by its normalizer mean. "
              "Near-zero deltas = the policy isn't using exteroception.", "",
              "| policy | terrain | difficulty | fall % (scan → none) | error (scan → none) |",
              "|---|---|---|---|---|"]
    for label, t, d in sorted(ablation_keys, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1], k[2])):
      with_scan, no_scan = (label, t, d, False), (label, t, d, True)
      lines.append(
        f"| {label} | {t} | {d:g} "
        f"| {_fmt(metric(with_scan, 'survival', 'fall_pct'), '.1f')} → {_fmt(metric(no_scan, 'survival', 'fall_pct'), '.1f')} "
        f"| {_fmt(metric(with_scan, 'locomotion', 'lin_vel_error_per_step'), '.2f')} → "
        f"{_fmt(metric(no_scan, 'locomotion', 'lin_vel_error_per_step'), '.2f')} |"
      )
    lines.append("")
  return "\n".join(lines)


def main() -> None:
  repo = Path(__file__).resolve().parent.parent
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--mjlab-dir", type=Path, default=repo / "unitree_rl_mjlab")
  parser.add_argument("--out-dir", type=Path, default=None, help="Default: <mjlab-dir>/eval_results/matrix")
  parser.add_argument("--difficulties", type=float, nargs="+", default=[0.25, 0.5, 0.75])
  parser.add_argument("--ablation-difficulties", type=float, nargs="+", default=[0.5])
  parser.add_argument("--no-ablation", action="store_true")
  parser.add_argument("--seeds", type=int, nargs="+", default=[0])
  parser.add_argument("--num-envs", type=int, default=1024,
                       help="Cluster A100 default; an 8GB laptop GPU should pass ~128-256.")
  parser.add_argument("--steps", type=int, default=1200)
  parser.add_argument("--min-iteration", type=int, default=9999)
  parser.add_argument("--pas-checkpoint", default=None,
                       help="Local PAS .pt path override. Default: PAS_CHECKPOINT if that "
                       "exact file exists (cluster), else fall back to HF -- see resolve_policies.")
  parser.add_argument("--only", nargs="+", help="Restrict to these policy labels.")
  parser.add_argument("--force", action="store_true", help="Re-run cells that already have JSON.")
  parser.add_argument("--dry-run", action="store_true", help="Print the cell commands without running.")
  parser.add_argument("--summarize-only", action="store_true")
  args = parser.parse_args()

  mjlab_dir = args.mjlab_dir.resolve()
  out_dir = (args.out_dir or mjlab_dir / "eval_results" / "matrix").resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  rc = 0 if args.summarize_only else run_matrix(args, mjlab_dir, out_dir)
  if not args.dry_run:
    summary = summarize(out_dir)
    (out_dir / "summary.md").write_text(summary)
    print(summary)
    print(f"[INFO] Wrote {out_dir / 'summary.md'}")
  sys.exit(rc)


if __name__ == "__main__":
  main()
