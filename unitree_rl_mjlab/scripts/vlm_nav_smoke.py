"""Phase-0 smoke test for the VLM navigation twin (not an eval).

Checks, in one process, everything the pipeline assumes before any VLM is in
the loop:
  1. the ego RGB-D camera renders on this GPU alongside the specialists;
  2. the depth convention: deprojecting flat ground must give a constant world z.
     Tested under both conventions (unit-ray distance vs. optical-axis depth) so
     the check can't pass by construction;
  3. the externally written velocity command actually moves the robot;
  4. hot-swapping specialists mid-episode runs and the robot keeps walking;
  5. per-step wall time and GPU memory with the VLM server also loaded.

Usage (from unitree_rl_mjlab/):
  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/vlm_nav_smoke.py \
      --ckpt-root /path/to/eval_ckpts --terrain flat --out-dir /tmp/smoke
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
from PIL import Image  # noqa: E402

import src  # noqa: E402
from src.tasks.velocity.config.go2.env_cfgs import apply_eval_conditions  # noqa: E402
from src.vlm_nav.camera import CameraSpec, body_to_world  # noqa: E402
from src.vlm_nav.policy_bank import PolicyBank, default_checkpoints  # noqa: E402
from src.vlm_nav.twin_env import (  # noqa: E402
  BASE_TASK,
  base_pose,
  get_rgbd,
  make_twin_env_cfg,
  set_velocity_command,
)


def depth_convention_check(camera: CameraSpec, depth: np.ndarray, pos, quat) -> dict:
  """World z of every ground pixel's deprojection, under both depth conventions."""
  h, w = depth.shape
  vv, uu = np.mgrid[0:h, 0:w]
  valid = np.isfinite(depth) & (depth > 0.05) & (depth < 20.0)
  u, v, d = uu[valid], vv[valid], depth[valid]
  out = {"valid_px": int(valid.sum())}
  rays = camera.unit_rays_cam(u, v)
  as_ray = rays * d[:, None]
  as_axis = rays / np.abs(rays[:, 2:3]) * d[:, None]  # scale so |z_cam| == d
  for name, p_cam in (("ray_distance", as_ray), ("optical_axis", as_axis)):
    p_body = p_cam @ camera.rotation_body_from_cam.T + np.asarray(camera.pos)
    z = body_to_world(p_body, pos, quat)[:, 2]
    out[name] = {"z_p5": float(np.percentile(z, 5)), "z_median": float(np.median(z)),
                 "z_p95": float(np.percentile(z, 95)), "z_spread_p95_p5": float(np.ptp(np.percentile(z, [5, 95])))}
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt-root", required=True)
  ap.add_argument("--terrain", default="flat", help="apply_eval_conditions terrain class")
  ap.add_argument("--difficulty", type=float, default=0.5)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--switch-step", type=int, default=150)
  ap.add_argument("--first", default="flat")
  ap.add_argument("--second", default="stairs")
  ap.add_argument("--speed", type=float, default=0.6)
  ap.add_argument("--no-shadows", action="store_true")
  ap.add_argument("--out-dir", required=True)
  args = ap.parse_args()

  print(f"[INFO] src package: {src.__file__}")
  configure_torch_backends()
  device = "cuda:0"
  out = Path(args.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  free0, total = torch.cuda.mem_get_info()

  camera = CameraSpec()
  cfg = make_twin_env_cfg(camera, use_shadows=not args.no_shadows)
  apply_eval_conditions(cfg, terrain=args.terrain, difficulty=args.difficulty, seed=0)
  agent_cfg = load_rl_cfg(BASE_TASK)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=agent_cfg.clip_actions)
  bank = PolicyBank(env, default_checkpoints(args.ckpt_root), device)
  bank.select(args.first, step=0)

  obs, _ = env.reset()
  set_velocity_command(env, [args.speed, 0.0, 0.0])
  report: dict = {"args": vars(args), "camera": asdict(camera), "camera_quat": camera.quat}

  rgb, depth = get_rgbd(env)
  pos, quat = base_pose(env)
  report["depth_check_step0"] = depth_convention_check(camera, depth, pos, quat)
  start_xy = pos[:2].copy()

  step_times, falls = [], 0
  for step in range(args.steps):
    if step == args.switch_step:
      bank.select(args.second, step=step)
    t = time.perf_counter()
    actions = bank.act(obs)
    obs, _, dones, extras = env.step(actions)
    torch.cuda.synchronize()
    step_times.append(time.perf_counter() - t)
    if bool(dones[0]):
      timeout = bool(extras.get("time_outs", torch.zeros(1))[0])
      falls += 0 if timeout else 1
      print(f"[WARN] step {step}: episode ended ({'timeout' if timeout else 'termination'})")
    if step in (0, args.switch_step, args.steps - 1):
      rgb, depth = get_rgbd(env)
      Image.fromarray(rgb).save(out / f"{args.terrain}_step{step:04d}_{bank.active}.png")
      d = np.clip(np.nan_to_num(depth, nan=0.0, posinf=0.0), 0, 6) / 6
      Image.fromarray((d * 255).astype(np.uint8)).save(out / f"{args.terrain}_step{step:04d}_depth.png")

  pos, _ = base_pose(env)
  free1, _ = torch.cuda.mem_get_info()
  report.update(
    net_displacement_xy=float(np.linalg.norm(pos[:2] - start_xy)),
    expected_displacement_xy=args.speed * args.steps * env.unwrapped.step_dt,
    falls=falls,
    switch_log=bank.switch_log,
    step_ms_median=1000 * float(np.median(step_times[5:])),
    step_ms_p95=1000 * float(np.percentile(step_times[5:], 95)),
    gpu_free_before_mb=free0 / 2**20,
    gpu_free_after_mb=free1 / 2**20,
    gpu_total_mb=total / 2**20,
  )
  print(json.dumps(report, indent=2, default=str))
  (out / f"smoke_{args.terrain}.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
  main()
