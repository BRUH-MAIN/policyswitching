"""Switching while following: does switch timing, blending, or preview horizon matter?

The robot follows a kinematic leader down a mixed-terrain course. Every arm gets
the same leader, the same follow controller and the same ground-truth segment
layout; arms differ only in which specialist(s) produce the action at each point
of the course (`src/vlm_nav/schedule.py`):

  fixed:<policy>               one policy for the whole run
  hard:<lead>:<mapping>        hard switch <lead> m before each non-flat segment
  soft:<start>:<end>:<mapping> linear cross-fade from <start> to <end> m before it

A lead above the onboard scan horizon (0.8 m) needs terrain knowledge from beyond
it, i.e. the leader. Design and analysis rules:
`coordination/results/switch-follow-preregistration.md`.

One trial per env. A trial ends at the first of: success (base past the last
non-flat segment by --success-margin), fall (a termination), lost (range to the
leader above --lost-range), or the time limit. Smoothness is measured in a
geometric window around terrain-class boundaries, never keyed to the arm's own
switch (objective.md, "Metrics").

Results are written after every arm, so a reboot loses at most one arm; --resume
skips arms already in the file and refuses to continue under different conditions.

Usage (from unitree_rl_mjlab/):
  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/switch_follow.py --arm-set calib \
      --ckpt-root ../../../../unitree_rl_mjlab/eval_ckpts --seed 400 \
      --json-out eval_results/switch_follow/calib_s400.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

from src.vlm_nav.controllers import FollowGains, follow_command  # noqa: E402
from src.vlm_nav.course import saro_courses  # noqa: E402
from src.vlm_nav.policy_bank import POLICY_NAMES, PolicyBank, default_checkpoints  # noqa: E402
from src.vlm_nav.schedule import MAPPINGS, Schedule, class_boundaries, policy_weights, segment_spans  # noqa: E402
from src.vlm_nav.twin_env import BASE_TASK, make_twin_env_cfg, set_velocity_command  # noqa: E402

HARD_LEADS = (-0.6, -0.3, 0.0, 0.3, 0.8, 1.5, 2.5)
SOFT_LEADS = ((0.3, 0.0), (0.8, 0.0), (0.8, 0.3), (1.5, 0.3), (1.5, 0.8), (2.5, 0.3), (2.5, 0.8), (2.5, 1.5))
BOUNDARY_WINDOW = 0.5  # +- m of base travel around a terrain-class boundary (objective.md)


def arm_set(name: str) -> list[str]:
  fixed = [f"fixed:{p}" for p in POLICY_NAMES]
  if name == "fixed":
    return fixed
  if name == "calib":
    arms = list(fixed)
    for mapping in MAPPINGS:
      arms += [f"hard:{d}:{mapping}" for d in HARD_LEADS]
      arms += [f"soft:{a}:{b}:{mapping}" for a, b in SOFT_LEADS]
    return arms
  raise ValueError(f"unknown arm set {name!r}")


def parse_arm(spec: str) -> tuple[str | None, Schedule | None, str | None]:
  """-> (fixed policy, schedule, mapping name); exactly one of the first two is set."""
  kind, *rest = spec.split(":")
  if kind == "fixed" and len(rest) == 1:
    return rest[0], None, None
  if kind == "hard" and len(rest) == 2:
    return None, Schedule(float(rest[0]), float(rest[0])), rest[1]
  if kind == "soft" and len(rest) == 3:
    return None, Schedule(float(rest[0]), float(rest[1])), rest[2]
  raise ValueError(f"bad arm spec {spec!r}")


def _stats(values: torch.Tensor) -> dict:
  v = values[~torch.isnan(values)]
  if v.numel() == 0:
    return dict(n=0, mean=float("nan"), p99=float("nan"))
  # torch.quantile caps its input at 16M elements; a full rollout log stays well under.
  return dict(n=int(v.numel()), mean=float(v.mean()), p99=float(torch.quantile(v, 0.99)))


def run_arm(env, bank: PolicyBank, course, spec: str, args, device: str) -> dict:
  u = env.unwrapped
  robot = u.scene["robot"]
  n, dt = u.num_envs, u.step_dt
  names = bank.names
  fixed, schedule, mapping_name = parse_arm(spec)
  if fixed is not None and fixed not in names:
    raise KeyError(f"{spec}: no policy {fixed!r} in the bank ({names})")

  spans = segment_spans(course)
  success_x = max(x1 for _, x1, kind in spans if kind != "flat") + args.success_margin
  x_off = course.length / 2  # course x = world x + length / 2
  centre_y_w = float(course.course_to_world(np.array([0.0, course.width / 2, 0.0]))[1])
  gains = FollowGains()
  leader_vel = torch.tensor([args.leader_speed, 0.0], device=device).expand(n, 2)
  max_steps = int(args.time_limit / dt)

  torch.manual_seed(args.seed)
  obs, _ = env.reset()
  active = torch.ones(n, dtype=torch.bool, device=device)
  outcome = ["timeout"] * n
  t_end = torch.full((n,), float(args.time_limit), device=device)
  end_x = torch.full((n,), float("nan"), device=device)
  fall_term: list[list[str] | None] = [None] * n
  path = torch.zeros(n, device=device)
  prev_xy = robot.data.root_link_pos_w[:, :2].clone()
  prev_force = robot.data.actuator_force.clone()
  tm = u.termination_manager
  am = u.action_manager

  nan = float("nan")
  log_x = torch.full((max_steps, n), nan, device=device)
  log_action_rate = torch.full((max_steps, n), nan, device=device)
  log_force_rate = torch.full((max_steps, n), nan, device=device)
  log_track_err = torch.full((max_steps, n), nan, device=device)
  log_gap_err = torch.full((max_steps, n), nan, device=device)

  for step in range(max_steps):
    pos = robot.data.root_link_pos_w
    x_course = pos[:, 0] + x_off
    leader_x_w = course.start_x + args.gap + args.leader_speed * step * dt - x_off
    leader_xy = torch.tensor([leader_x_w, centre_y_w], device=device, dtype=pos.dtype).expand(n, 2)
    cmd, rng = follow_command(pos, robot.data.root_link_quat_w, leader_xy, args.gap, gains, leader_vel_w=leader_vel)

    reached = active & (x_course >= success_x)
    lost = active & ~reached & (rng > args.lost_range)
    for mask, label in ((reached, "success"), (lost, "lost")):
      for i in mask.nonzero().flatten().tolist():
        outcome[i] = label
      t_end[mask] = step * dt
      end_x[mask] = x_course[mask]
    active &= ~(reached | lost)
    if not active.any():
      break
    cmd[~active] = 0.0
    set_velocity_command(env, cmd)

    if fixed is not None:
      weights = torch.zeros(n, len(names), device=device)
      weights[:, names.index(fixed)] = 1.0
    else:
      weights = policy_weights(x_course, course, names, schedule, MAPPINGS[mapping_name])
    obs, _, dones, extras = env.step(bank.act_blend(obs, weights))

    done = dones.bool()
    timeouts = extras.get("time_outs", torch.zeros_like(done)).bool()
    new_fall = active & done & ~timeouts
    for i in new_fall.nonzero().flatten().tolist():
      outcome[i] = "fall"
      fall_term[i] = [name for name in tm.active_terms if bool(tm.get_term(name)[i])]
    t_end[new_fall] = step * dt
    end_x[new_fall] = x_course[new_fall]

    valid = active & ~done
    xy = robot.data.root_link_pos_w[:, :2]
    path += torch.where(valid, torch.linalg.norm(xy - prev_xy, dim=-1), torch.zeros_like(path))
    prev_xy = xy.clone()
    force = robot.data.actuator_force
    if step > 0:  # the first step's "previous action" is the reset's zero, not a policy output
      action_rate = torch.sum(torch.square(am.action - am.prev_action), dim=1)
      force_rate = torch.linalg.norm(force - prev_force, dim=1) / dt
      log_action_rate[step] = torch.where(valid, action_rate, torch.full_like(action_rate, nan))
      log_force_rate[step] = torch.where(valid, force_rate, torch.full_like(force_rate, nan))
    prev_force = force.clone()
    track_err = torch.linalg.norm(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=1)
    log_x[step] = torch.where(valid, x_course, torch.full_like(x_course, nan))
    log_track_err[step] = torch.where(valid, track_err, torch.full_like(track_err, nan))
    log_gap_err[step] = torch.where(valid, rng - args.gap, torch.full_like(rng, nan))
    active &= ~new_fall

  still = active.nonzero().flatten().tolist()
  if still:
    end_x[active] = (robot.data.root_link_pos_w[:, 0] + x_off)[active]

  # Geometric boundary windows, by boundary type.
  windows = {"entry": torch.zeros_like(log_x, dtype=torch.bool), "exit": torch.zeros_like(log_x, dtype=torch.bool)}
  for xb, kind in class_boundaries(course):
    windows[kind] |= (log_x - xb).abs() <= BOUNDARY_WINDOW  # NaN compares False

  def masked(log: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return log if mask is None else torch.where(mask, log, torch.full_like(log, nan))

  smooth = {}
  for metric, log in (("action_rate", log_action_rate), ("force_rate", log_force_rate)):
    whole = _stats(log)
    smooth[metric] = dict(whole=whole)
    for kind, mask in windows.items():
      s = _stats(masked(log, mask))
      s["mean_rel"] = s["mean"] / whole["mean"] if whole["n"] else nan
      s["p99_rel"] = s["p99"] / whole["p99"] if whole["n"] else nan
      smooth[metric][kind] = s

  def per_trial(log: torch.Tensor, mask: torch.Tensor | None = None) -> list[float]:
    return torch.nanmean(masked(log, mask), dim=0).tolist()

  cols = dict(
    track_err=per_trial(log_track_err),
    gap_rms=torch.sqrt(torch.nanmean(torch.square(log_gap_err), dim=0)).tolist(),
    ar_all=per_trial(log_action_rate), ar_entry=per_trial(log_action_rate, windows["entry"]),
    ar_exit=per_trial(log_action_rate, windows["exit"]),
    fr_all=per_trial(log_force_rate), fr_entry=per_trial(log_force_rate, windows["entry"]),
    fr_exit=per_trial(log_force_rate, windows["exit"]),
  )
  trials = []
  for i in range(n):
    row = dict(outcome=outcome[i], time_s=float(t_end[i]), end_x=float(end_x[i]), path_m=float(path[i]),
               fall_term=fall_term[i])
    row.update({k: (None if v[i] != v[i] else round(v[i], 5)) for k, v in cols.items()})
    trials.append(row)

  counts = {k: sum(t["outcome"] == k for t in trials) for k in ("success", "fall", "lost", "timeout")}
  total_path = float(path.sum())
  summary = dict(
    n=n, **{f"{k}_pct": 100 * v / n for k, v in counts.items()},
    falls_per_100m=100 * counts["fall"] / total_path if total_path > 0 else nan,
    track_err=float(torch.nanmean(log_track_err)),
    gap_rms=float(torch.sqrt(torch.nanmean(torch.square(log_gap_err)))),
    ar_entry_rel=smooth["action_rate"]["entry"]["mean_rel"],
  )
  return dict(summary=summary, smoothness=smooth, trials=trials)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--course", default="multi")
  ap.add_argument("--level", default="L1", help="Course difficulty level (course.DIFFICULTY_LEVELS).")
  ap.add_argument("--seed", type=int, required=True, help="Terrain + spawn seed.")
  ap.add_argument("--arms", nargs="+", default=None, help="Arm specs (see module docstring).")
  ap.add_argument("--arm-set", default=None, choices=("fixed", "calib"))
  ap.add_argument("--ckpt-root", required=True)
  ap.add_argument("--extra-policy", action="append", default=[], metavar="NAME=CKPT",
                  help="Add a policy to the bank (e.g. generalist=/path/model_9999.pt); use it as fixed:NAME.")
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--gap", type=float, default=3.0, help="Leader's lead over the robot (m): the preview horizon.")
  ap.add_argument("--leader-speed", type=float, default=0.5)
  ap.add_argument("--success-margin", type=float, default=1.5)
  ap.add_argument("--lost-range", type=float, default=6.0)
  ap.add_argument("--time-limit", type=float, default=None, help="Seconds; default 2 x nominal traverse + 10.")
  ap.add_argument("--spawn-xy-jitter", type=float, default=0.15)
  ap.add_argument("--spawn-yaw", type=float, default=0.3)
  ap.add_argument("--terminations", default="training", choices=("training", "saro"))
  ap.add_argument("--resume", action="store_true")
  ap.add_argument("--json-out", required=True)
  args = ap.parse_args()
  arms = (arm_set(args.arm_set) if args.arm_set else []) + (args.arms or [])
  if not arms:
    ap.error("give --arms and/or --arm-set")

  configure_torch_backends()
  device = "cuda:0"
  course = saro_courses(visual="plain", level=args.level)[args.course]
  success_x = max(x1 for _, x1, kind in segment_spans(course) if kind != "flat") + args.success_margin
  if args.time_limit is None:
    args.time_limit = 2.0 * (success_x - course.start_x) / args.leader_speed + 10.0

  conditions = {k: v for k, v in vars(args).items() if k not in ("arms", "arm_set", "resume", "json_out")}
  out_path = Path(args.json_out)
  result = dict(conditions=conditions, course=asdict(course), arms={})
  if out_path.exists():
    if not args.resume:
      raise SystemExit(f"{out_path} exists; pass --resume to add arms to it, or choose another file")
    result = json.loads(out_path.read_text())
    if result["conditions"] != json.loads(json.dumps(conditions)):
      raise SystemExit(f"{out_path} was produced under different conditions; refusing to mix:\n"
                       f"  file: {result['conditions']}\n  now:  {conditions}")

  cfg = make_twin_env_cfg(
    None, terrain_generator=course.generator_cfg(seed=args.seed), num_envs=args.num_envs, seed=args.seed,
    spawn_xy_jitter=args.spawn_xy_jitter, spawn_yaw_range=(-args.spawn_yaw, args.spawn_yaw),
    terminations=args.terminations,
  )
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=load_rl_cfg(BASE_TASK).clip_actions)
  checkpoints = dict(default_checkpoints(args.ckpt_root))
  for item in args.extra_policy:
    name, _, ckpt = item.partition("=")
    checkpoints[name] = Path(ckpt)
  bank = PolicyBank(env, checkpoints, device)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  for spec in arms:
    if spec in result["arms"]:
      print(f"[SKIP] {spec}: already in {out_path.name}")
      continue
    t0 = time.time()
    result["arms"][spec] = run_arm(env, bank, course, spec, args, device)
    s = result["arms"][spec]["summary"]
    print(f"[RESULT] {course.name} {args.level} s{args.seed} {spec:>20}: "
          + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items())
          + f"  ({time.time() - t0:.0f}s)", flush=True)
    out_path.write_text(json.dumps(result))


if __name__ == "__main__":
  main()
