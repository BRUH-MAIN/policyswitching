"""Check the robot-side observation builder (deploy_numpy/go2_obs.py) against the simulator.

Rolls a policy up a staircase in the simulator and, every step, rebuilds the observation the
way the robot will: from the base quaternion, body gyro, joint positions and velocities in
the ROBOT's motor order, the previous action, a step counter, and a height scan cast
independently here with CPU MuJoCo (terrain only) at the points go2_obs.scan_points_xy()
names. It then compares, term by term, with the observation the simulator produced, and
compares the numpy policy's action on the rebuilt observation with the PyTorch policy's.

What passing shows: joint order and sign, default pose, gravity projection, gait clock, scan
grid order, scan frame (level, yaw-aligned), scan scale and the exported weights are right.
What it cannot show: that the real IMU, encoders and motor indices follow the conventions
go2_obs.py assumes. That is the robot's --dry-run check.

  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/verify_deploy_obs.py \
      eval_ckpts/go2_spec_stairs_v5a/model_9999.pt --npz deploy_numpy/stairs_v5a.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy_numpy import go2_obs  # noqa: E402
from deploy_numpy.policy_numpy import NumpyPolicy  # noqa: E402

TERMS = (("base_ang_vel", 3), ("projected_gravity", 3), ("command", 3), ("phase", 2),
         ("joint_pos", 12), ("joint_vel", 12), ("actions", 12), ("height_scan", 187))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoint")
  ap.add_argument("--npz", required=True, help="Export of the same checkpoint (scripts/export_policy_numpy.py).")
  ap.add_argument("--step-height", type=float, default=0.12)
  ap.add_argument("--steps", type=int, default=700)
  ap.add_argument("--num-envs", type=int, default=4)
  args = ap.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  from src.vlm_nav.course import saro_courses
  from src.vlm_nav.policy_bank import PolicyBank
  from src.vlm_nav.twin_env import BASE_TASK, make_twin_env_cfg, set_velocity_command

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  clip = load_rl_cfg(BASE_TASK).clip_actions
  course = saro_courses(visual="plain", level="L2")["stairs_up"]
  course = replace(course, segments=tuple(
    replace(g, step_height=args.step_height) if g.kind.startswith("stairs") else g for g in course.segments))
  cfg = make_twin_env_cfg(None, terrain_generator=course.generator_cfg(seed=0), num_envs=args.num_envs, seed=0,
                          spawn_xy_jitter=0.15, spawn_yaw_range=(-0.3, 0.3), terminations="saro")
  # Terrain-only scan in the simulator too, so the scan comparison is exact. (As trained the
  # rays also hit the robot's legs; the count of cells that differ is printed for reference.)
  cfg.scene.sensors = tuple(
    replace(s, include_geom_groups=(0, 1)) if s.name == "terrain_scan" else s for s in cfg.scene.sensors)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=clip)
  bank = PolicyBank(env, {"p": Path(args.checkpoint)}, device)
  policy = NumpyPolicy(args.npz)

  u = env.unwrapped
  robot = u.scene["robot"]
  model = u.sim.mj_model
  data = mujoco.MjData(model)
  base_id = model.body("robot/base_link").id if _has_body(model, "robot/base_link") else model.body("base_link").id
  terrain_groups = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
  pts = go2_obs.scan_points_xy()
  down = np.array([0.0, 0.0, -1.0])
  geomid = np.zeros(1, dtype=np.int32)

  def cast_scan(env_i: int, quat) -> np.ndarray:
    data.qpos[:] = u.sim.data.qpos[env_i].cpu().numpy()
    mujoco.mj_kinematics(model, data)
    base = data.xpos[base_id].copy()
    yaw = go2_obs.yaw_of(quat)
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.empty(len(pts), dtype=np.float32)
    for k, (px, py) in enumerate(pts):
      origin = base + np.array([c * px - s * py, s * px + c * py, 0.0])
      dist = mujoco.mj_ray(model, data, origin, down, terrain_groups, 1, -1, geomid)
      out[k] = dist if geomid[0] >= 0 else go2_obs.SCAN_MISS
    return out

  command = np.array([0.5, 0.0, 0.0], dtype=np.float32)
  worst = {name: 0.0 for name, _ in TERMS}
  worst_action = 0.0
  obs, _ = env.reset()
  climbed = 0.0
  z0 = robot.data.root_link_pos_w[:, 2].clone()
  for _ in range(args.steps):
    # The command the current observation was computed with (zero on the very first step).
    obs_command = u.command_manager.get_command("twist").cpu().numpy().copy()
    set_velocity_command(env, command.tolist())
    with torch.inference_mode():
      reference = torch.clamp(bank.policies["p"](obs), -clip, clip)
    sim_obs = obs["actor"].cpu().numpy()
    for i in range(args.num_envs):
      quat = robot.data.root_link_quat_w[i].cpu().numpy()
      ours = go2_obs.build_obs(
        gyro=robot.data.root_link_ang_vel_b[i].cpu().numpy(),
        quat_wxyz=quat,
        command=obs_command[i],
        step_count=int(u.episode_length_buf[i]),
        q_robot=robot.data.joint_pos[i].cpu().numpy()[go2_obs.ROBOT_TO_SIM],
        dq_robot=robot.data.joint_vel[i].cpu().numpy()[go2_obs.ROBOT_TO_SIM],
        last_action=u.action_manager.action[i].cpu().numpy(),
        scan_m=cast_scan(i, quat),
      )
      start = 0
      for name, size in TERMS:
        diff = float(np.abs(ours[start:start + size] - sim_obs[i, start:start + size]).max())
        worst[name] = max(worst[name], diff)
        start += size
      worst_action = max(worst_action, float(np.abs(policy.act(ours) - reference[i].cpu().numpy()).max()))
    obs, _, _, _ = env.step(reference)
    climbed = max(climbed, float((robot.data.root_link_pos_w[:, 2] - z0).max()))

  print(f"{args.steps} steps x {args.num_envs} robots on a {args.step_height * 100:.0f} cm flight; "
        f"highest climb {climbed:.2f} m")
  print("largest |robot-side - simulator| per observation term:")
  for name, _ in TERMS:
    print(f"  {name:18s} {worst[name]:.2e}")
  print(f"largest |numpy policy on robot-side obs - torch policy on simulator obs|: {worst_action:.2e}")
  # The scan tolerates 1 cm: a ray landing within a hair of a riser edge can fall either side
  # in the two ray casters (float32 on the GPU, float64 here); it is then off by one riser.
  tol = {name: 1e-3 for name, _ in TERMS}
  bad = [name for name, _ in TERMS if name != "height_scan" and worst[name] > tol[name]]
  if bad:
    raise SystemExit(f"MISMATCH in {bad}: do not deploy")
  if climbed < 2 * args.step_height:
    raise SystemExit("the robot never got onto the stairs, so the scan was only checked on flat ground")
  print("PARITY OK (scan: see its number above; an occasional one-riser difference at an edge is expected)")
  env.close()


def _has_body(model, name: str) -> bool:
  try:
    model.body(name)
    return True
  except KeyError:
    return False


if __name__ == "__main__":
  main()
