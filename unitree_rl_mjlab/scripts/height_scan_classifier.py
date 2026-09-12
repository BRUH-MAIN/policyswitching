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
  """Return (group_name, term_cfg) for the height_scan observation term.

  Groups are `ObservationGroupCfg` dataclasses holding their terms in `.terms`,
  so this looks there rather than at the group's own fields. 'actor' is preferred
  over 'critic' when both carry a scan: the actor group is what a deployed
  classifier would read.
  """
  found = {}
  for group_name, group in cfg.observations.items():
    terms = group if isinstance(group, dict) else getattr(group, "terms", {})
    if "height_scan" in terms:
      found[group_name] = terms["height_scan"]
  if not found:
    raise SystemExit("[ERROR] no 'height_scan' observation term found in this task's config.")
  for preferred in ("actor", "policy"):
    if preferred in found:
      return preferred, found[preferred]
  return next(iter(found.items()))


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
            warmup: int, seed: int, device: str, spawn_spread: float = 3.0):
  """Roll out zero actions on one terrain class.

  Returns (samples, env_ids, noise_halfwidth, grid_shape).

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

  # Spread spawns across the terrain patch. The task's own reset_base randomizes
  # only +/-0.5 m around the patch origin, which is 1/64th of an 8x8 m patch and
  # sits dead centre -- and the centre of a `pyramid_stairs` patch is its flat top
  # platform, wider than the 1.6x1.0 m scan. Sampling there makes stairs
  # indistinguishable from flat by construction (measured: identical mean, ray-std
  # and across-sample std to 4 decimals), which is a property of where we stood,
  # not of the sensor. Keep a margin so the scan footprint stays on the patch.
  if spawn_spread > 0:
    reset_base = cfg.events.get("reset_base")
    if reset_base is None:
      raise SystemExit("[ERROR] no 'reset_base' event to widen; pass --spawn-spread 0 to skip.")
    half = min(spawn_spread, cfg.scene.terrain.terrain_generator.size[0] / 2 - 1.0)
    reset_base.params["pose_range"]["x"] = (-half, half)
    reset_base.params["pose_range"]["y"] = (-half, half)

  # Ray-grid shape, derived from the sensor pattern rather than hardcoded, so a
  # conv model reshapes correctly if the scan geometry ever changes.
  pattern = cfg.scene.sensors[0].pattern if hasattr(cfg.scene, "sensors") else None
  grid_shape = None
  for sensor in getattr(cfg.scene, "sensors", ()) or ():
    if getattr(sensor, "name", None) == "terrain_scan":
      pattern = sensor.pattern
      res = pattern.resolution
      grid_shape = (round(pattern.size[0] / res) + 1, round(pattern.size[1] / res) + 1)
      break

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

  x = torch.cat(samples)
  if grid_shape is not None and grid_shape[0] * grid_shape[1] != x.shape[1]:
    print(f"[WARN] ray-grid shape {grid_shape} doesn't match {x.shape[1]} rays; "
          "conv model will be skipped.")
    grid_shape = None
  return x, torch.cat(env_ids), noise_half, grid_shape


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


def fit_cnn(x_tr, y_tr, x_te, y_te, num_classes: int, grid_shape, epochs: int,
            device: str, seed: int = 0):
  """Small conv net over the ray grid.

  The linear probe can only key on per-ray mean differences, which is why it does
  badly on stairs: a staircase's signature is spatial structure (a repeating step
  edge), not a shift in average height. This tests whether that structure is
  recoverable at the same noise level. Train accuracy is reported alongside test
  so overfitting is visible rather than buried -- with ~7.7k samples and an
  env-wise split, a conv net has enough capacity to memorise.
  """
  torch.manual_seed(seed)
  rows, cols = grid_shape
  mean, std = x_tr.mean(), x_tr.std().clamp_min(1e-8)

  def prep(x):
    return ((x - mean) / std).reshape(-1, 1, rows, cols).to(device)

  x_tr_d, x_te_d = prep(x_tr), prep(x_te)
  y_tr_d, y_te_d = y_tr.to(device), y_te.to(device)

  model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(),
    torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(2), torch.nn.Flatten(),
    torch.nn.Linear(32 * 4, num_classes),
  ).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
  loss_fn = torch.nn.CrossEntropyLoss()

  for _ in range(epochs):
    model.train()
    perm = torch.randperm(x_tr_d.shape[0], device=device)
    for i in range(0, len(perm), 512):
      idx = perm[i:i + 512]
      opt.zero_grad()
      loss_fn(model(x_tr_d[idx]), y_tr_d[idx]).backward()
      opt.step()

  model.eval()
  with torch.no_grad():
    train_acc = (model(x_tr_d).argmax(1) == y_tr_d).float().mean().item()
    pred = model(x_te_d).argmax(1)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, q in zip(y_te_d.cpu(), pred.cpu()):
      confusion[t, q] += 1
  per_class = (confusion.diag().float() / confusion.sum(1).clamp_min(1).float()).tolist()
  return {
    "accuracy": (confusion.diag().sum().item() / max(confusion.sum().item(), 1)),
    "train_accuracy": train_acc,
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
  ap.add_argument("--spawn-spread", type=float, default=3.0,
                  help="Half-width (m) of the spawn box within each terrain patch. The task's "
                       "own reset_base uses 0.5, which keeps every robot on the flat top "
                       "platform of a pyramid_stairs patch. 0 leaves the task's value alone.")
  ap.add_argument("--test-frac", type=float, default=0.25, help="Fraction of ENVS held out.")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--epochs", type=int, default=200, help="LBFGS iterations (linear model).")
  ap.add_argument("--cnn-epochs", type=int, default=60, help="Epochs for the conv model.")
  ap.add_argument("--models", nargs="+", default=["linear", "cnn"], choices=["linear", "cnn"],
                  help="Run both by default, on the SAME collected data and split, so the "
                       "comparison isn't confounded by a different draw.")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--json-out", default=None)
  args = ap.parse_args()

  torch.manual_seed(args.seed)
  xs, ys, envs, noise_half, grid_shape = [], [], [], None, None
  for label, terrain in enumerate(args.classes):
    print(f"\n=== collecting '{terrain}' ({label + 1}/{len(args.classes)}) ===", flush=True)
    x, e, nh, gs = collect(args.task, terrain, args.num_envs, args.steps,
                           args.stride, args.warmup, args.seed, args.device, args.spawn_spread)
    grid_shape = gs if grid_shape is None else grid_shape
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
  for model_name in args.models:
    if model_name == "cnn" and grid_shape is None:
      print("[WARN] no ray-grid shape available; skipping the conv model.")
      continue
    for cond, data in (("clean", x), ("noisy", noisy)):
      key = cond if model_name == "linear" else f"{cond}_cnn"
      if model_name == "linear":
        results[key] = fit_logreg(data[~is_test], y[~is_test], data[is_test], y[is_test],
                                  len(args.classes), args.epochs, args.device)
      else:
        results[key] = fit_cnn(data[~is_test], y[~is_test], data[is_test], y[is_test],
                               len(args.classes), grid_shape, args.cnn_epochs,
                               args.device, args.seed)

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
  print(f"spawn spread     : +/-{args.spawn_spread:g} m within each 8x8 m patch")
  if grid_shape:
    print(f"ray grid         : {grid_shape[0]} x {grid_shape[1]}")
  for name in [k for k in ("clean", "noisy", "clean_cnn", "noisy_cnn") if k in results]:
    r = results[name]
    print(f"\n--- {name} ---")
    if "train_accuracy" in r:
      print(f"train accuracy   : {r['train_accuracy']:.3f}  (vs held-out below; a large "
            "gap means it memorised)")
    print(f"held-out accuracy: {r['accuracy']:.3f}  ({r['accuracy'] / chance:.2f}x chance)")
    for cls, acc in zip(args.classes, r["per_class_accuracy"]):
      print(f"    {cls:<8}: {acc:.3f}")
    print("  confusion (row = true, col = predicted):")
    print(f"    {'':<8}" + "".join(f"{c:>9}" for c in args.classes))
    for cls, row in zip(args.classes, r["confusion"]):
      print(f"    {cls:<8}" + "".join(f"{v:>9}" for v in row))
  for suffix, label in (("", "linear"), ("_cnn", "cnn")):
    if f"clean{suffix}" in results and f"noisy{suffix}" in results:
      drop = results[f"clean{suffix}"]["accuracy"] - results[f"noisy{suffix}"]["accuracy"]
      print(f"\ncost of injected noise ({label}): {drop:+.3f} accuracy")
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
      "grid_shape": grid_shape,
      "spawn_spread_m": args.spawn_spread,
      "train_envs": len(uniq) - n_test,
      "test_envs": n_test,
      "results": results,
    }
    pathlib.Path(args.json_out).write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
  main()
