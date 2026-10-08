"""Record a third-person video of a stairs policy following a walking leader up and down a flight.

One robot, one trial: flat approach, a flight up, a landing, a flight down, flat run-out, at a
chosen riser height and step count, with the scripted person of `src.vlm_nav.leader` walking
ahead and the same follow controller the evaluation uses. Training sensor noise is on.

A video is one sample. It shows what a crossing looks like; how often the policy crosses is
what `scripts/switch_follow_real_stairs.sh` measures (256 trials per cell).

  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/record_stairs_demo.py \
      --checkpoint eval_ckpts/go2_spec_stairs_v8a/model_3999.pt --step-height 0.17 --steps 10 \
      --out ../report_content/videos/v8a_17cm_10steps.mp4 --caption "stairs v8a, 17 cm x 10 steps"
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.viewer import ViewerConfig  # noqa: E402

from src.vlm_nav.controllers import FollowGains, follow_command  # noqa: E402
from src.vlm_nav.course import STEP_WIDTH, Segment, saro_courses  # noqa: E402
from src.vlm_nav.leader import write_leader_pose  # noqa: E402
from src.vlm_nav.policy_bank import PolicyBank  # noqa: E402
from src.vlm_nav.twin_env import BASE_TASK, make_twin_env_cfg, set_velocity_command  # noqa: E402

LEADER_GROUP = 4  # the leader's geoms are in the camera-only group


def ground_height(course, x: float) -> float:
  """Height of the walking surface at course coordinate x (the top of the step under x)."""
  for reg, seg in zip(course.regions(), course.segments):
    if x < reg.x1 or reg is course.regions()[-1]:
      if not seg.kind.startswith("stairs"):
        return reg.z_start
      k = int(np.clip(np.floor((x - reg.x0) / STEP_WIDTH), 0, round(seg.length / STEP_WIDTH) - 1))
      # Going up, step k is k+1 risers above the start; going down, the first tread is one riser below.
      return reg.z_start + (k + 1) * seg.step_height * (1 if seg.kind == "stairs_up" else -1)
  return course.regions()[-1].z_end


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--step-height", type=float, default=0.17)
  ap.add_argument("--steps", type=int, default=10, help="Steps per flight.")
  ap.add_argument("--direction", choices=("up_down", "up", "down"), default="up_down")
  ap.add_argument("--leader-speed", type=float, default=0.5)
  ap.add_argument("--gap", type=float, default=2.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--no-noise", action="store_true")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--azimuth-up", type=float, default=50.0, help="Camera azimuth while climbing (risers face it).")
  ap.add_argument("--azimuth-down", type=float, default=130.0, help="Camera azimuth while descending.")
  ap.add_argument("--elevation", type=float, default=-12.0)
  ap.add_argument("--distance", type=float, default=5.5)
  ap.add_argument("--max-seconds", type=float, default=60.0)
  ap.add_argument("--caption", default=None)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  flight = STEP_WIDTH * args.steps
  h = args.step_height
  segs = [Segment("flat", 3.0, h)]
  if args.direction in ("up_down", "up"):
    segs += [Segment("stairs_up", flight, h), Segment("flat", 2.5, h)]
  if args.direction in ("up_down", "down"):
    segs += [Segment("stairs_down", flight, h), Segment("flat", 3.5, h)]
  base = saro_courses(visual="plain", level="L2")["stairs_up"]
  rise = args.steps * h
  course = replace(base, name="demo", segments=tuple(segs), goal_marker=False,
                   base_height=rise if args.direction == "down" else 0.0)

  cfg = make_twin_env_cfg(None, terrain_generator=course.generator_cfg(seed=args.seed), num_envs=1, seed=args.seed,
                          terminations="saro", leader=True)
  cfg.observations["actor"].enable_corruption = not args.no_noise
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base_link",
    distance=args.distance, elevation=args.elevation, azimuth=args.azimuth_up, width=args.width, height=args.height,
  )
  raw = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(raw, clip_actions=load_rl_cfg(BASE_TASK).clip_actions)
  bank = PolicyBank(env, {"p": Path(args.checkpoint)}, device)
  u = env.unwrapped
  robot = u.scene["robot"]
  dt = u.step_dt

  x_off = course.length / 2
  centre_y = float(course.course_to_world(np.array([0.0, course.width / 2, 0.0]))[1])
  stop_x = course.length - 1.0  # where the leader stops, in course x
  gains = FollowGains()

  def leader_at(t: float):
    x = min(course.start_x + args.gap + args.leader_speed * t, stop_x)
    moving = x < stop_x
    return x, moving

  obs, _ = env.reset()
  x0, _ = leader_at(0.0)
  write_leader_pose(env, np.array([x0 - x_off, centre_y, ground_height(course, x0)]), 0.0)
  u.render()  # creates the renderer
  rend = u._offline_renderer
  import mujoco
  scene_opt = mujoco.MjvOption()
  scene_opt.geomgroup[LEADER_GROUP] = 1  # the leader's geoms are in a group the default view hides
  # Light the scene from ahead and to the side, so risers are darker than treads in both
  # directions; with the stock light the steps of an up-flight read as a ramp.
  m = rend._model
  print(f"[INFO] lights in the model: {m.nlight}; headlight diffuse {m.vis.headlight.diffuse}")
  for i in range(m.nlight):
    d = np.array([-0.3, 0.5, -0.8])
    m.light_dir[i] = d / np.linalg.norm(d)
    m.light_type[i] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    m.light_diffuse[i] = (0.9, 0.9, 0.9)
    m.light_ambient[i] = (0.15, 0.15, 0.15)
    m.light_castshadow[i] = 1
  m.vis.headlight.diffuse[:] = 0.15
  m.vis.headlight.ambient[:] = 0.25

  # The camera swings from behind-left while climbing to ahead-left while descending, so the
  # risers face it both ways; it turns over the landing between the flights.
  regions = course.regions()
  landing = next((r for r, g in zip(regions, course.segments) if g.kind == "flat" and r.x0 > 0.1
                  and args.direction == "up_down" and r is not regions[-1]), None)

  def azimuth_at(x_course: float) -> float:
    if args.direction == "down":
      return args.azimuth_down
    if landing is None:
      return args.azimuth_up
    w = float(np.clip((x_course - landing.x0) / (landing.x1 - landing.x0), 0.0, 1.0))
    w = w * w * (3 - 2 * w)
    return args.azimuth_up + (args.azimuth_down - args.azimuth_up) * w

  def render_frame(x_course: float) -> np.ndarray:
    rend._cam.azimuth = azimuth_at(x_course)
    rend.update(u.sim.data)
    rend._renderer.update_scene(rend._data, camera=rend._cam, scene_option=scene_opt)
    return rend._renderer.render()

  font = None
  if args.caption:
    from PIL import ImageFont
    try:
      font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
      font = ImageFont.load_default()

  frames, outcome = [], "time limit"
  top = float(robot.data.root_link_pos_w[0, 2])
  z_start = top
  for step in range(int(args.max_seconds / dt)):
    t = step * dt
    lx, moving = leader_at(t)
    write_leader_pose(env, np.array([lx - x_off, centre_y, ground_height(course, lx)]), 0.0)
    pos = robot.data.root_link_pos_w
    leader_xy = torch.tensor([[lx - x_off, centre_y]], device=device, dtype=pos.dtype)
    leader_vel = torch.tensor([[args.leader_speed if moving else 0.0, 0.0]], device=device, dtype=pos.dtype)
    cmd, _ = follow_command(pos, robot.data.root_link_quat_w, leader_xy, args.gap, gains, leader_vel_w=leader_vel)
    set_velocity_command(env, cmd)
    with torch.inference_mode():
      obs, _, dones, _ = env.step(bank.policies["p"](obs))
    if step % 2 == 0:
      frame = render_frame(float(pos[0, 0]) + x_off)
      if font is not None:
        from PIL import Image, ImageDraw
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, args.width, 40], fill=(0, 0, 0))
        draw.text((14, 8), args.caption, fill=(255, 255, 255), font=font)
        draw.text((args.width - 250, 8), f"simulation  t = {t:4.1f} s", fill=(200, 200, 200), font=font)
        frame = np.asarray(img)
      frames.append(frame)
    top = max(top, float(robot.data.root_link_pos_w[0, 2]))
    if bool(dones[0]):
      outcome = "FELL (episode ended)"
      break
    if float(pos[0, 0]) + x_off > stop_x - args.gap - 0.3 and not moving:
      outcome = "reached the leader at the end of the course"
      break

  import imageio.v2 as imageio

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(out, frames, fps=int(round(1 / (2 * dt))), macro_block_size=None)
  print(f"{out}: {len(frames) / (1 / (2 * dt)):.1f} s, outcome: {outcome}; climbed {top - z_start:+.2f} m "
        f"(one flight is {rise:.2f} m)")
  env.close()


if __name__ == "__main__":
  main()
