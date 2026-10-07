"""Which reward terms pay for crossing stairs, and which pay for standing still?

Runs a checkpoint in its own TRAINING env (rewards, noise, pushes, command range as trained)
with the riser height pinned and the terrain curriculum off, and reports the mean of every
reward term per step, split by

  * direction: the env's patch is a pyramid (spawn on top, every way out is DOWN) or an
    inverted pyramid (spawn in the pit, every way out is UP), told apart by spawn height;
  * what the robot was doing on that step, among steps commanded to move (> 0.3 m/s):
    "moving" (planar speed >= half the command), "stalled" (planar speed < 0.1 m/s), or
    "on_stairs" (1.6-3.0 m from the spawn point, i.e. on the flight rather than the platform).

Values are weight * raw term, i.e. reward per second; multiply by dt = 0.02 for per step.
The question it answers: on a riser the policy refuses, does the reward function itself
prefer the stalled steps to the moving ones, and through which term? Compare a checkpoint
that refuses a riser with one that crosses it (e.g. v5a model_9999 vs model_400 going down
15 cm). Observational: a term that differs is a candidate, a training run with it changed
is the test.

  python scripts/diag_reward_terms.py --task Unitree-Go2-Spec-StairsV5a \
      --checkpoint eval_ckpts/go2_spec_stairs_v5a/model_9999.pt --step-height 0.15
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

import src.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from src.tasks.velocity.config.go2.env_cfgs import apply_eval_conditions


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Unitree-Go2-Spec-StairsV5a")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--step-height", type=float, required=True)
  ap.add_argument("--step-width", type=float, default=None)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=1000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--json-out", default=None)
  args = ap.parse_args()

  configure_torch_backends()
  torch.manual_seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  apply_eval_conditions(
    env_cfg, terrain="native", seed=args.seed, step_height=args.step_height, step_width=args.step_width
  )
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task)(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  u = env.unwrapped
  robot = u.scene["robot"]
  rm = u.reward_manager
  names = list(rm.active_terms)
  origin_z = u.scene.terrain.env_origins[:, 2]
  # Pyramid: spawn platform is the top of the flight. Inverted: it is the bottom of the pit.
  split = 0.5 * (origin_z.max() + origin_z.min())
  groups = {"down": origin_z > split, "up": origin_z <= split}
  print(f"[INFO] spawn heights: min {origin_z.min():.2f} max {origin_z.max():.2f} m; "
        f"down envs {int(groups['down'].sum())}, up envs {int(groups['up'].sum())}")

  kinds = ("moving", "stalled", "on_stairs", "all_cmd")
  origins_xy = u.scene.terrain.env_origins[:, :2]
  acc = {g: {k: torch.zeros(len(names) + 1, device=device) for k in kinds} for g in groups}
  cnt = {g: {k: 0.0 for k in kinds} for g in groups}
  speed_sum = {g: 0.0 for g in groups}
  cmd_sum = {g: 0.0 for g in groups}

  obs, _ = env.reset()
  for _ in range(args.steps):
    with torch.no_grad():
      obs, _, _, _ = env.step(policy(obs))
    cmd = u.command_manager.get_command("twist")
    cmd_xy = torch.norm(cmd[:, :2], dim=1)
    speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
    step_r = rm._step_reward  # [B, T], weight * raw (reward per second)
    row = torch.cat([step_r, step_r.sum(dim=1, keepdim=True)], dim=1)
    commanded = cmd_xy > 0.3
    cheb = (robot.data.root_link_pos_w[:, :2] - origins_xy).abs().max(dim=1).values
    masks = {
      "moving": commanded & (speed >= 0.5 * cmd_xy),
      "stalled": commanded & (speed < 0.1),
      # The flight spans 1.5-3.0 m (Chebyshev) from the spawn point; inside it is the platform.
      "on_stairs": commanded & (cheb > 1.6) & (cheb < 3.0),
      "all_cmd": commanded,
    }
    for g, gm in groups.items():
      for k, m in masks.items():
        mm = gm & m
        acc[g][k] += row[mm].sum(dim=0)
        cnt[g][k] += float(mm.sum())
      gc = gm & commanded
      speed_sum[g] += float(speed[gc].sum())
      cmd_sum[g] += float(cmd_xy[gc].sum())

  out = {"checkpoint": args.checkpoint, "task": args.task, "step_height": args.step_height,
         "step_width": args.step_width, "num_envs": args.num_envs, "steps": args.steps, "groups": {}}
  cols = names + ["TOTAL"]
  for g in groups:
    n_all = max(cnt[g]["all_cmd"], 1.0)
    res = {
      "speed_over_cmd": speed_sum[g] / max(cmd_sum[g], 1e-9),
      "frac_moving": cnt[g]["moving"] / n_all,
      "frac_stalled": cnt[g]["stalled"] / n_all,
      "frac_on_stairs": cnt[g]["on_stairs"] / n_all,
      "terms": {},
    }
    print(f"\n== {g.upper()} at {args.step_height * 100:.0f} cm: speed/cmd {res['speed_over_cmd']:.2f}, "
          f"moving {100 * res['frac_moving']:.0f}% of commanded steps, stalled {100 * res['frac_stalled']:.0f}%, on the flight {100 * res['frac_on_stairs']:.0f}%")
    print(f"{'term':26s} {'moving':>9s} {'stalled':>9s} {'on_stairs':>9s} {'all':>9s}")
    for i, name in enumerate(cols):
      vals = {k: float(acc[g][k][i]) / max(cnt[g][k], 1.0) for k in kinds}
      res["terms"][name] = vals
      print(f"{name:26s} {vals['moving']:9.3f} {vals['stalled']:9.3f} {vals['on_stairs']:9.3f} {vals['all_cmd']:9.3f}")
    out["groups"][g] = res
  if args.json_out:
    with open(args.json_out, "w") as f:
      json.dump(out, f, indent=1)
  env.close()


if __name__ == "__main__":
  main()
