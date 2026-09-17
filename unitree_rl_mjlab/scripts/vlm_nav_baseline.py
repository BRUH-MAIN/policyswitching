"""Phase-1 no-VLM baselines: does choosing the right specialist change course outcomes?

For one course build (course x goal offset x terrain seed) this runs every arm on
the same env instance, one trial per env:
  oracle   -- ground-truth terrain under the robot's footprint picks the specialist
              (CourseSpec.required_terrain)
  flat / rough / stairs -- that specialist for the whole run
All arms walk the same ground-truth goal-seeking controller, so the only thing
that differs between arms is which specialist is active.

A trial ends at the first of: base within --goal-radius of the goal (success), a
termination (fall: bad orientation or illegal non-foot contact, the specialists'
own training terminations), or the time limit. Metrics are per trial, so they
can't be skewed by episode pooling (findings.md bug #15): success, fall,
crossed-the-last-intermediation (SARO Table I's grey "across terrains" column),
path length, time, where on the course a fall happened, and the fraction of
steps the active specialist matched the oracle's.

Usage (from unitree_rl_mjlab/):
  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/vlm_nav_baseline.py --course multi \
      --ckpt-root ../../../../unitree_rl_mjlab/eval_ckpts --num-envs 32 \
      --json-out eval_results/vlm_nav/phase1/multi_g+0.0_s0.json
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

from src.vlm_nav.controllers import GotoGains, goto_command  # noqa: E402
from src.vlm_nav.course import FOOTPRINT_FRONT, FOOTPRINT_REAR, saro_courses  # noqa: E402
from src.vlm_nav.policy_bank import POLICY_NAMES, PolicyBank, default_checkpoints  # noqa: E402
from src.vlm_nav.twin_env import BASE_TASK, make_twin_env_cfg, set_velocity_command  # noqa: E402


def run_arm(env, bank: PolicyBank, course, arm: str, args, device: str) -> list[dict]:
  u = env.unwrapped
  robot = u.scene["robot"]
  n = u.num_envs
  dt = u.step_dt
  regions = course.regions()
  last_inter_x1 = max((r.x1 for r in regions if r.terrain != "flat"), default=0.0)
  goal_xy = torch.tensor(course.course_to_world(course.goal_course)[:2], device=device, dtype=torch.float32)
  goal_xy = goal_xy.expand(n, 2)
  gains = GotoGains(v_max=args.speed)

  def oracle_policy(x_course: torch.Tensor) -> torch.Tensor:
    # course.required_terrain, vectorized: footprint overlap with non-flat regions,
    # the region under the front point taking precedence.
    front, rear = args.footprint_front, args.footprint_rear
    out = torch.full(x_course.shape, bank.names.index("flat"), dtype=torch.long, device=device)
    for r in [r for r in regions if r.terrain != "flat"]:
      overlap = (x_course + front > r.x0) & (x_course - rear < r.x1)
      out[overlap] = bank.names.index(r.terrain)
    for r in [r for r in regions if r.terrain != "flat"]:
      under_front = (x_course + front >= r.x0) & (x_course + front < r.x1)
      out[under_front] = bank.names.index(r.terrain)
    return out

  torch.manual_seed(args.seed)
  obs, _ = env.reset()
  active = torch.ones(n, dtype=torch.bool, device=device)
  success = torch.zeros(n, dtype=torch.bool, device=device)
  fell = torch.zeros_like(success)
  crossed = torch.zeros_like(success)
  t_end = torch.full((n,), float("nan"), device=device)
  fall_x = torch.full((n,), float("nan"), device=device)
  path = torch.zeros(n, device=device)
  matched = torch.zeros(n, device=device)
  steps_active = torch.zeros(n, device=device)
  speed_sum = torch.zeros(n, device=device)
  prev_xy = robot.data.root_link_pos_w[:, :2].clone()
  max_steps = int(args.time_limit / dt)

  for step in range(max_steps):
    pos = robot.data.root_link_pos_w
    x_course = pos[:, 0] + course.length / 2
    oracle_idx = oracle_policy(x_course)
    pidx = oracle_idx if arm == "oracle" else torch.full_like(oracle_idx, bank.names.index(arm))
    cmd, dist = goto_command(pos, robot.data.root_link_quat_w, goal_xy, gains)
    crossed |= active & (x_course > last_inter_x1)
    reached = active & (dist < args.goal_radius)
    success |= reached
    t_end[reached] = step * dt
    active &= ~reached
    cmd[~active] = 0.0
    set_velocity_command(env, cmd)

    matched += (active & (pidx == oracle_idx)).float()
    steps_active += active.float()
    obs, _, dones, extras = env.step(bank.act_per_env(obs, pidx))

    done = dones.bool()
    timeouts = extras.get("time_outs", torch.zeros_like(done)).bool()
    new_fall = active & done & ~timeouts
    fell |= new_fall
    fall_x[new_fall] = x_course[new_fall]
    t_end[new_fall] = step * dt
    xy = robot.data.root_link_pos_w[:, :2]
    moved = active & ~done
    step_len = torch.linalg.norm(xy - prev_xy, dim=-1)
    path += torch.where(moved, step_len, torch.zeros_like(step_len))
    speed_sum += torch.where(moved, step_len / dt, torch.zeros_like(step_len))
    prev_xy = xy.clone()
    active &= ~new_fall
    if not active.any():
      break

  timed_out = active.clone()
  t_end[timed_out] = args.time_limit
  out = []
  for i in range(n):
    out.append(dict(
      arm=arm, env_id=i, success=bool(success[i]), fell=bool(fell[i]), timed_out=bool(timed_out[i]),
      crossed=bool(crossed[i]), time_s=float(t_end[i]), path_m=float(path[i]),
      mean_speed=float(speed_sum[i] / max(1.0, float(steps_active[i]))),
      fall_x_course=None if torch.isnan(fall_x[i]) else float(fall_x[i]),
      policy_match_frac=float(matched[i] / max(1.0, float(steps_active[i]))),
    ))
  return out


def summarize(trials: list[dict]) -> dict:
  n = len(trials)
  path = sum(t["path_m"] for t in trials)
  falls = sum(t["fell"] for t in trials)
  return dict(
    n=n,
    success_pct=100 * sum(t["success"] for t in trials) / n,
    fall_pct=100 * falls / n,
    timeout_pct=100 * sum(t["timed_out"] for t in trials) / n,
    crossed_pct=100 * sum(t["crossed"] for t in trials) / n,
    falls_per_100m=100 * falls / path if path > 0 else float("nan"),
    mean_speed=float(np.mean([t["mean_speed"] for t in trials])),
    policy_match_pct=100 * float(np.mean([t["policy_match_frac"] for t in trials])),
  )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--course", required=True)
  ap.add_argument("--goal-y-offset", type=float, default=0.0)
  ap.add_argument("--level", default="L2", help="Course difficulty level (course.DIFFICULTY_LEVELS).")
  ap.add_argument("--seed", type=int, default=0, help="Terrain + spawn seed.")
  ap.add_argument("--arms", nargs="+", default=["oracle", *POLICY_NAMES])
  ap.add_argument("--ckpt-root", required=True)
  ap.add_argument("--num-envs", type=int, default=32)
  ap.add_argument("--speed", type=float, default=0.5)
  ap.add_argument("--footprint-front", type=float, default=FOOTPRINT_FRONT)
  ap.add_argument("--footprint-rear", type=float, default=FOOTPRINT_REAR)
  ap.add_argument("--goal-radius", type=float, default=0.3)
  ap.add_argument("--time-limit", type=float, default=None, help="Seconds; default 2.5 x length / speed + 10.")
  ap.add_argument("--spawn-xy-jitter", type=float, default=0.15)
  ap.add_argument("--spawn-yaw", type=float, default=0.3)
  ap.add_argument("--json-out", required=True)
  args = ap.parse_args()

  configure_torch_backends()
  device = "cuda:0"
  course = saro_courses(goal_y_offset=args.goal_y_offset, level=args.level)[args.course]
  if args.time_limit is None:
    args.time_limit = 2.5 * course.length / args.speed + 10.0
  cfg = make_twin_env_cfg(
    None, terrain_generator=course.generator_cfg(seed=args.seed), num_envs=args.num_envs, seed=args.seed,
    spawn_xy_jitter=args.spawn_xy_jitter, spawn_yaw_range=(-args.spawn_yaw, args.spawn_yaw),
  )
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=load_rl_cfg(BASE_TASK).clip_actions)
  bank = PolicyBank(env, default_checkpoints(args.ckpt_root), device)

  result = dict(conditions=vars(args), course=asdict(course), arms={})
  for arm in args.arms:
    t0 = time.time()
    trials = run_arm(env, bank, course, arm, args, device)
    summary = summarize(trials)
    result["arms"][arm] = dict(summary=summary, trials=trials)
    print(f"[RESULT] {course.name} {args.level} g{args.goal_y_offset:+.1f} s{args.seed} {arm:>6}: "
          + " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items())
          + f"  ({time.time() - t0:.0f}s)")
  Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.json_out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
  main()
