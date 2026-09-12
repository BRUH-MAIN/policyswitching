"""Premise gate 2a: is the height scan *terrain-discriminative*?

`objective.md`'s premise gate 2 asks whether the policies use `height_scan`, and
tests it by ablating the scan and watching fall rate / velocity error. That tests
whether the *locomotion policy* needs the scan, which is not the property Phase 4
depends on: the reactive terrain classifier and the gating network's current-terrain
input need the scan to be *terrain-discriminative*. Those dissociate in both
directions -- proprioception may be enough to walk on <=10 cm stairs while the scan
still separates the classes fine, and the scan may help foot placement without
separating them. See the 2026-09-12 (3) entry in coordination/inbox/to-cluster.md.

This measures the property directly and without RL: collect labelled `height_scan`
observations from each terrain class under pinned eval conditions, fit a multinomial
logistic regression, and report held-out accuracy and the confusion matrix. Chance
is 1/len(classes). Runs on the laptop in minutes and needs no checkpoint.

Two conditions are reported from a single rollout:

  clean  -- the scan as the simulator produces it (upper bound on discriminability)
  noisy  -- clean plus the observation noise the policy actually sees

The noisy condition is the one Phase 4 has to live with; clean-minus-noisy is how
much the injected noise costs, which is what tells you whether the noise level is
the thing to fix. Both come from one rollout because mjlab applies noise *before*
`scale` with no clip on this term (pipeline: compute -> noise -> clip -> scale), so
post-scale noise is exactly uniform over (n_min*scale, n_max*scale) and can be added
analytically. The magnitudes are read from the config rather than hardcoded.

Sampling notes that matter for reading the numbers:

  - Zero actions throughout, so the robot holds its default pose. That keeps the
    scan a measurement of terrain rather than of gait, and keeps the comparison
    across classes controlled.
  - Samples within one env are strongly correlated (a standing robot sees nearly
    the same patch every step), so the train/test split is **by env**, never by
    sample. Splitting randomly over samples would put the same terrain patch on
    both sides and report near-perfect accuracy that means nothing.
  - Effective sample size is therefore closer to the env count than the sample
    count; both are reported.

Usage (from the repo root, PYTHONPATH set to unitree_rl_mjlab/ as in a100/*.sh):

  python3 scripts/height_scan_classifier.py --num-envs 256 --json-out gate2a.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from src.tasks.velocity.config.go2.env_cfgs import TERRAIN_CLASSES  # noqa: E402


def find_height_scan_term(cfg):
  """Return (group_name, term_cfg) for the height_scan observation term."""
  for group_name, group in cfg.observations.items():
    terms = group if isinstance(group, dict) else vars(group)
    for term_name, term_cfg in terms.items():
      if term_name == "height_scan":
        return group_name, term_cfg
  raise SystemExit("[ERROR] no 'height_scan' observation term found in this task's config.")


def scan_slice_for_group(env, group: str) -> slice:
  """Locate height_scan's span within a flattened observation group."""
  om = env.observation_manager
  start = 0
  for name, shape in zip(om.active_terms[group], om.group_obs_term_dim[group]):
    size = math.prod(shape)
    if name == "height_scan":
      return slice(start, start + size)
    start += size
  raise SystemExit(f"[ERROR] 'height_scan' not active in observation group '{group}'.")


def collect(task: str, terrain: str, num_envs: int, steps: int, stride: int,
            warmup: int, seed: int, device: str):
  """Roll out zero actions on one terrain class; return (samples, env_ids, noise_halfwidth).

  Noise is disabled on the scan term so the collected values are clean; the
  post-scale noise half-width is returned so the caller can add it analytically.
  """
  # Imported lazily so --help works without a GPU/simulator present.
  from src.tasks.velocity.config.go2.env_cfgs import apply_eval_conditions

  cfg = load_env_cfg(task, play=False)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  # difficulty=None: spread envs uniformly over difficulty rows. Deliberately not
  # pinning difficulty -- the pinned path was findings.md bug #12, and its fix is
  # not necessarily in this checkout yet.
  apply_eval_conditions(cfg, terrain=terrain, difficulty=None, seed=seed)
  cfg.curriculum.pop("command_vel", None)

  group, term = find_height_scan_term(cfg)
  scale = term.scale if term.scale is not None else 1.0
  if not isinstance(scale, (int, float)):
    raise SystemExit(f"[ERROR] expected a scalar scale on height_scan, got {type(scale)}.")
  if getattr(term, "clip", None) is not None:
    raise SystemExit(
      "[ERROR] height_scan has a clip configured. Noise is applied before clip, so it "
      "can no longer be added analytically after the fact -- collect the noisy "
      "condition with a second rollout instead."
    )
  if term.noise is None:
    noise_half = 0.0
  else:
    n_min, n_max = float(term.noise.n_min), float(term.noise.n_max)
    if abs(n_min + n_max) > 1e-12:
      raise SystemExit(f"[ERROR] expected symmetric height_scan noise, got ({n_min}, {n_max}).")
    noise_half = n_max * float(scale)
  term.noise = None  # collect clean; noise is added analytically below

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    sl = scan_slice_for_group(env, group)
    obs, _ = env.reset()
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

    samples, env_ids = [], []
    for step in range(warmup + steps):
      obs, *_ = env.step(action)
      if step < warmup or (step - warmup) % stride:
        continue
      flat = obs[group] if isinstance(obs, dict) else obs
      samples.append(flat[:, sl].detach().clone().cpu())
      env_ids.append(torch.arange(env.num_envs))
  finally:
    env.close()

  return torch.cat(samples), torch.cat(env_ids), noise_half


def fit_logreg(x_tr, y_tr, x_te, y_te, num_classes: int, epochs: int, device: str):
  """Multinomial logistic regression. Standardised inputs, full-batch LBFGS."""
  mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-8)
  x_tr = ((x_tr - mean) / std).to(device)
  x_te = ((x_te - mean) / std).to(device)
  y_tr, y_te = y_tr.to(device), y_te.to(device)

  model = torch.nn.Linear(x_tr.shape[1], num_classes).to(device)
  opt = torch.optim.LBFGS(model.parameters(), max_iter=epochs, line_search_fn="strong_wolfe")
  loss_fn = torch.nn.CrossEntropyLoss()

  def closure():
    opt.zero_grad()
    loss = loss_fn(model(x_tr), y_tr)
    # Mild L2 so a 187-dim fit on correlated samples doesn't run away.
    loss = loss + 1e-4 * sum((p * p).sum() for p in model.parameters())
    loss.backward()
    return loss

  opt.step(closure)

  with torch.no_grad():
    pred = model(x_te).argmax(1)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_te.cpu(), pred.cpu()):
      confusion[t, p] += 1
  per_class = (confusion.diag().float() / confusion.sum(1).clamp_min(1).float()).tolist()
  return {
    "accuracy": (confusion.diag().sum().item() / max(confusion.sum().item(), 1)),
    "per_class_accuracy": per_class,
    "confusion": confusion.tolist(),
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", default="Unitree-Go2-Generalist",
                  help="Base task supplying the observation space (all specialists share it).")
  ap.add_argument("--classes", nargs="+", default=list(TERRAIN_CLASSES),
                  help=f"Terrain classes to discriminate (default: {list(TERRAIN_CLASSES)}).")
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=200, help="Sampled steps after warmup.")
  ap.add_argument("--stride", type=int, default=20, help="Keep every Nth step (samples are correlated).")
  ap.add_argument("--warmup", type=int, default=30, help="Discard steps while the robot settles.")
  ap.add_argument("--test-frac", type=float, default=0.25, help="Fraction of ENVS held out.")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--epochs", type=int, default=200, help="LBFGS iterations.")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--json-out", default=None)
  args = ap.parse_args()

  torch.manual_seed(args.seed)
  xs, ys, envs, noise_half = [], [], [], None
  for label, terrain in enumerate(args.classes):
    print(f"\n=== collecting '{terrain}' ({label + 1}/{len(args.classes)}) ===", flush=True)
    x, e, nh = collect(args.task, terrain, args.num_envs, args.steps,
                       args.stride, args.warmup, args.seed, args.device)
    if noise_half is None:
      noise_half = nh
    elif abs(noise_half - nh) > 1e-12:
      raise SystemExit("[ERROR] noise magnitude differs across classes; cannot share one noise model.")
    xs.append(x)
    ys.append(torch.full((x.shape[0],), label, dtype=torch.long))
    # Offset env ids per class so the split never pairs envs across classes.
    envs.append(e + label * args.num_envs)
    print(f"    {x.shape[0]} samples x {x.shape[1]} rays from {args.num_envs} envs")

  x, y, e = torch.cat(xs), torch.cat(ys), torch.cat(envs)

  # Split by env, not by sample: a standing robot's consecutive scans are near
  # duplicates, so a random split would leak the same terrain patch into both sides.
  uniq = torch.unique(e)
  perm = uniq[torch.randperm(len(uniq), generator=torch.Generator().manual_seed(args.seed))]
  n_test = max(1, int(round(args.test_frac * len(uniq))))
  test_envs = set(perm[:n_test].tolist())
  is_test = torch.tensor([int(v) in test_envs for v in e])

  noisy = x + (torch.rand_like(x) * 2 - 1) * noise_half

  results = {}
  for name, data in (("clean", x), ("noisy", noisy)):
    results[name] = fit_logreg(data[~is_test], y[~is_test], data[is_test], y[is_test],
                               len(args.classes), args.epochs, args.device)

  chance = 1.0 / len(args.classes)
  print("\n" + "=" * 62)
  print("PREMISE GATE 2a -- height-scan terrain discriminability")
  print("=" * 62)
  print(f"classes          : {args.classes}")
  print(f"chance accuracy  : {chance:.3f}")
  print(f"rays per sample  : {x.shape[1]}")
  print(f"samples          : {(~is_test).sum().item()} train / {is_test.sum().item()} test")
  print(f"envs (effective) : {len(uniq) - n_test} train / {n_test} test  <- split by env")
  print(f"noise half-width : +/-{noise_half:.5f} in scaled units")
  for name in ("clean", "noisy"):
    r = results[name]
    print(f"\n--- {name} ---")
    print(f"held-out accuracy: {r['accuracy']:.3f}  ({r['accuracy'] / chance:.2f}x chance)")
    for cls, acc in zip(args.classes, r["per_class_accuracy"]):
      print(f"    {cls:<8}: {acc:.3f}")
    print("  confusion (row = true, col = predicted):")
    print(f"    {'':<8}" + "".join(f"{c:>9}" for c in args.classes))
    for cls, row in zip(args.classes, r["confusion"]):
      print(f"    {cls:<8}" + "".join(f"{v:>9}" for v in row))
  drop = results["clean"]["accuracy"] - results["noisy"]["accuracy"]
  print(f"\ncost of injected noise: {drop:+.3f} accuracy")
  print("=" * 62)

  if args.json_out:
    payload = {
      "gate": "2a_height_scan_discriminability",
      "task": args.task,
      "classes": args.classes,
      "chance": chance,
      "num_envs_per_class": args.num_envs,
      "rays": x.shape[1],
      "noise_half_width_scaled": noise_half,
      "split": "by env",
      "train_envs": len(uniq) - n_test,
      "test_envs": n_test,
      "results": results,
    }
    pathlib.Path(args.json_out).write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
  main()
