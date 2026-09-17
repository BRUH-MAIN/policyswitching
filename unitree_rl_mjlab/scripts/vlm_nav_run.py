"""Closed-loop SARO-style navigation with VLM specialist selection, on one course build.

Arms (--arms, any subset):
  vlm            VLM navigation + VLM policy selection  (the modification under test)
  vlm+oracle     VLM navigation + ground-truth policy   (isolates navigation errors)
  vlm+fixed:X    VLM navigation + specialist X always   (SARO's single low-level policy)
  gt+oracle      ground-truth navigation + policy       (upper bound; = Phase-1 oracle)
  ovlm[+...]     the SARO executor fed by a ground-truth "VLM" (src/vlm_nav/oracle_vlm.py):
                 pipeline logic with perfect perception, to separate executor faults from model errors

Timing is lockstep: while the agents query the VLM, simulation time is frozen,
so results measure decision quality independently of this laptop's inference
speed. VLM latency is recorded per trial so a real-time budget can be read off.

Each trial ends at success (base within the goal radius, ground truth), a fall
(training termination), or the time limit. Outputs: per-trial metrics JSON,
per-agent event logs, the VLM transcript with every queried image, and an
optional ego-camera video of env 0.

Usage (from unitree_rl_mjlab/, VLM server running):
  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/vlm_nav_run.py --course stairs_up --level L1 \
      --arms vlm gt+oracle --num-envs 4 --ckpt-root <eval_ckpts> --out logs/vlm_nav/run1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
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

from src.vlm_nav.camera import CameraSpec  # noqa: E402
from src.vlm_nav.course import saro_courses  # noqa: E402
from src.vlm_nav.executor import ExecutorConfig, SaroAgent  # noqa: E402
from src.vlm_nav.oracle_vlm import OracleVLM  # noqa: E402
from src.vlm_nav.policy_bank import PolicyBank, default_checkpoints  # noqa: E402
from src.vlm_nav.twin_env import BASE_TASK, CAMERA_SENSOR_NAME, make_twin_env_cfg, set_velocity_command  # noqa: E402
from src.vlm_nav.vlm_backend import OpenAICompatVLM, health  # noqa: E402


def parse_arm(arm: str) -> tuple[str, str]:
  if arm in ("vlm", "ovlm"):
    return arm, "vlm"
  nav, _, pol = arm.partition("+")
  if nav not in ("vlm", "gt", "ovlm") or not pol:
    raise ValueError(f"bad arm {arm!r}")
  return nav, pol


def run_arm(env, bank, course, camera, arm: str, args, out: Path, device: str) -> dict:
  nav, policy_source = parse_arm(arm)
  u = env.unwrapped
  robot = u.scene["robot"]
  n, dt = u.num_envs, u.step_dt
  arm_dir = out / arm.replace(":", "_").replace("+", "_")
  arm_dir.mkdir(parents=True, exist_ok=True)
  vlm = None
  state_cache: dict = {}
  box_convention = args.box_convention
  if nav == "ovlm":
    vlm = OracleVLM(course, camera, lambda i: state_cache[i], goal_radius=args.goal_radius)
    box_convention = "xyxy_px"  # the oracle answers in pixels
  elif nav == "vlm" or policy_source == "vlm":
    vlm = OpenAICompatVLM(base_url=f"{args.base_url}/v1", transcript=arm_dir / "vlm_transcript.jsonl",
                          image_dir=arm_dir / "vlm_images" if args.save_vlm_images else None)
  cfg = ExecutorConfig(policy_source=policy_source, nav_source="gt" if nav == "gt" else "vlm",
                       box_convention=box_convention, speed=args.speed)
  agents = [SaroAgent(i, course, camera, vlm, cfg, dt) for i in range(n)]

  torch.manual_seed(args.seed)
  obs, _ = env.reset()
  active = np.ones(n, dtype=bool)
  outcome = ["timeout"] * n
  t_end = np.full(n, args.time_limit)
  fall_term: list = [None] * n
  regions = course.regions()
  last_x1 = max((r.x1 for r in regions if r.terrain != "flat"), default=0.0)
  crossed = np.zeros(n, dtype=bool)
  match = np.zeros(n)
  steps_active = np.zeros(n)
  goal_xy = course.course_to_world(course.goal_course)[:2]
  cam = u.scene[CAMERA_SENSOR_NAME]
  tm = u.termination_manager
  frames = []
  pool = ThreadPoolExecutor(max(1, args.vlm_workers))
  wall0 = time.time()
  max_steps = int(args.time_limit / dt)

  for step in range(max_steps):
    pos = robot.data.root_link_pos_w
    quat = robot.data.root_link_quat_w
    if step % args.vlm_period == 0 and vlm is not None:
      rgb_all = cam.data.rgb.cpu().numpy()
      depth_all = cam.data.depth[..., 0].cpu().numpy()
      pos_np, quat_np = pos.cpu().numpy(), quat.cpu().numpy()
      ids = [i for i in range(n) if active[i]]
      for i in ids:
        state_cache[i] = (depth_all[i], pos_np[i], quat_np[i])
      list(pool.map(lambda i: agents[i].think(step, rgb_all[i], depth_all[i], pos_np[i], quat_np[i]), ids))
    cmds = torch.zeros(n, 3, device=device)
    pidx = torch.zeros(n, dtype=torch.long, device=device)
    for i in range(n):
      if not active[i]:
        continue
      cmd, pol = agents[i].control(step, pos[i], quat[i])
      cmds[i] = cmd
      pidx[i] = bank.names.index(pol)
      x_course = course.world_to_course_x(float(pos[i, 0]))
      match[i] += pol == course.required_terrain(x_course)
      steps_active[i] += 1
      crossed[i] |= x_course > last_x1
      if np.linalg.norm(pos[i, :2].cpu().numpy() - goal_xy) < args.goal_radius:
        outcome[i], t_end[i], active[i] = "success", step * dt, False
    set_velocity_command(env, cmds)
    if args.video and active[0] and step % 2 == 0:
      frames.append(cam.data.rgb[0].cpu().numpy())
    obs, _, dones, extras = env.step(bank.act_per_env(obs, pidx))
    done = dones.bool().cpu().numpy()
    timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool().cpu().numpy()
    for i in np.nonzero(active & done & ~timeouts)[0]:
      outcome[i], t_end[i], active[i] = "fall", step * dt, False
      fall_term[i] = [name for name in tm.active_terms if bool(tm.get_term(name)[i])]
    if not active.any():
      break

  pool.shutdown()
  trials = []
  for i, a in enumerate(agents):
    s = a.summary()
    trials.append(dict(
      arm=arm, env_id=i, outcome=outcome[i], success=outcome[i] == "success", fell=outcome[i] == "fall",
      crossed=bool(crossed[i]), time_s=float(t_end[i]), fall_term=fall_term[i],
      policy_match_frac=float(match[i] / max(1.0, steps_active[i])),
      vlm_calls=s["vlm_calls"], vlm_time_s=s["vlm_time_s"], plan=s["plan"], replans=s["replans"],
      switches=[e["detail"] for e in s["events"] if e["kind"] == "switch"],
    ))
    (arm_dir / f"agent{i}_events.json").write_text(json.dumps(s, indent=1, default=str))
  if frames:
    import imageio.v2 as imageio  # noqa: PLC0415
    imageio.mimsave(arm_dir / "env0_ego.mp4", frames, fps=int(round(1 / (2 * dt))))
  summary = dict(
    n=n, success_pct=100 * np.mean([t["success"] for t in trials]), fall_pct=100 * np.mean([t["fell"] for t in trials]),
    crossed_pct=100 * np.mean([t["crossed"] for t in trials]),
    policy_match_pct=100 * np.mean([t["policy_match_frac"] for t in trials]),
    vlm_calls_per_trial=float(np.mean([t["vlm_calls"] for t in trials])),
    vlm_s_per_call=float(sum(t["vlm_time_s"] for t in trials) / max(1, sum(t["vlm_calls"] for t in trials))),
    wall_s=time.time() - wall0,
  )
  return dict(summary=summary, trials=trials)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--course", required=True)
  ap.add_argument("--level", default="L1")
  ap.add_argument("--visual", default="tiled")
  ap.add_argument("--goal-y-offset", type=float, default=0.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--arms", nargs="+", default=["vlm", "vlm+oracle", "gt+oracle"])
  ap.add_argument("--ckpt-root", required=True)
  ap.add_argument("--num-envs", type=int, default=4)
  ap.add_argument("--speed", type=float, default=0.5)
  ap.add_argument("--goal-radius", type=float, default=0.3)
  ap.add_argument("--time-limit", type=float, default=None)
  ap.add_argument("--vlm-period", type=int, default=25, help="Control steps between VLM ticks (25 = 0.5 s).")
  ap.add_argument("--vlm-workers", type=int, default=2)
  ap.add_argument("--base-url", default="http://127.0.0.1:8091")
  ap.add_argument("--box-convention", default="xyxy_px")
  ap.add_argument("--spawn-xy-jitter", type=float, default=0.15)
  ap.add_argument("--spawn-yaw", type=float, default=0.3)
  ap.add_argument("--terminations", default="training", choices=("training", "saro"),
                  help="Fall definition: specialists' training terminations, or SARO's orientation-only.")
  ap.add_argument("--save-vlm-images", action="store_true")
  ap.add_argument("--video", action="store_true")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  if any(parse_arm(a)[0] == "vlm" or (parse_arm(a) == ("vlm", "vlm")) for a in args.arms) and not health(args.base_url):
    raise SystemExit(f"[ERROR] VLM server not healthy at {args.base_url} (scripts/vlm_server.sh)")
  configure_torch_backends()
  device = "cuda:0"
  course = saro_courses(visual=args.visual, goal_y_offset=args.goal_y_offset, level=args.level)[args.course]
  if args.time_limit is None:
    args.time_limit = 2.5 * course.length / args.speed + 10.0
  camera = CameraSpec()
  cfg = make_twin_env_cfg(
    camera, terrain_generator=course.generator_cfg(seed=args.seed), num_envs=args.num_envs, seed=args.seed,
    spawn_xy_jitter=args.spawn_xy_jitter, spawn_yaw_range=(-args.spawn_yaw, args.spawn_yaw),
    terminations=args.terminations,
  )
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=load_rl_cfg(BASE_TASK).clip_actions)
  bank = PolicyBank(env, default_checkpoints(args.ckpt_root), device)
  out = Path(args.out) / f"{course.name}_{args.level}_g{args.goal_y_offset:+.1f}_s{args.seed}"
  out.mkdir(parents=True, exist_ok=True)
  result = dict(conditions=vars(args), course=asdict(course), camera=asdict(camera), arms={})
  for arm in args.arms:
    r = run_arm(env, bank, course, camera, arm, args, out, device)
    result["arms"][arm] = r
    print(f"[RESULT] {course.name} {args.level} g{args.goal_y_offset:+.1f} s{args.seed} {arm}: "
          + " ".join(f"{k}={v:.2f}" for k, v in r["summary"].items()), flush=True)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
  main()
