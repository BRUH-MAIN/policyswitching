"""Record a demo of policy switching: one robot follows a walking leader over a course of flat
ground, rough ground and staircases while the active locomotion policy is switched between a
flat, a rough and a stairs specialist by the terrain under its footprint.

The switch is the evaluation's "footprint rule" (`CourseSpec.required_terrain`): the class of
the terrain under the robot from 0.30 m ahead of the base to 0.35 m behind it, non-flat
winning, so the stairs policy stays on until the hind feet are off the last step. It reads
the course layout (ground truth), as the study's hard-switch arm did.

The frame shows which policy is active, and a strip along the bottom shows the course
(coloured by terrain class), the robot's position, and which policy was active where.

What this video is and is not. It shows the switching mechanism working. It does not show
that switching is needed: over randomised layouts the study found that switching between
specialists adds nothing over the best single policy (`objective.md`, "What the result
was"), and the stairs policy used here crosses the whole course alone (`--single stairs`
records that for comparison).

  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/record_switching_demo.py \
      --out ../report_content/videos/switching_demo/switching_demo.mp4
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.viewer import ViewerConfig  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from record_stairs_demo import LEADER_GROUP, ground_height  # noqa: E402
from src.vlm_nav.controllers import FollowGains, follow_command  # noqa: E402
from src.vlm_nav.course import DIFFICULTY_LEVELS, STEP_WIDTH, Segment, saro_courses  # noqa: E402
from src.vlm_nav.leader import write_leader_pose  # noqa: E402
from src.vlm_nav.policy_bank import PolicyBank  # noqa: E402
from src.vlm_nav.twin_env import BASE_TASK, make_twin_env_cfg, set_velocity_command  # noqa: E402

CLASS_RGB = {"flat": (150, 150, 150), "rough": (96, 150, 60), "stairs": (70, 110, 200)}
LABEL = {"flat": "FLAT specialist", "rough": "ROUGH specialist", "stairs": "STAIRS specialist"}


def build_course(riser: float, long_steps: int, short_steps: int, level: str):
  noise = DIFFICULTY_LEVELS[level]["noise_range"]
  f = lambda n: Segment("flat", n, riser)  # noqa: E731
  rough = lambda n: Segment("rough", n, riser, noise_range=noise)  # noqa: E731
  segs = (
    f(4.0), rough(5.0), f(3.0),
    Segment("stairs_up", STEP_WIDTH * long_steps, riser), f(3.0),
    Segment("stairs_down", STEP_WIDTH * long_steps, riser), f(3.0),
    rough(5.0), f(3.0),
    Segment("stairs_up", STEP_WIDTH * short_steps, riser), f(3.0),
    Segment("stairs_down", STEP_WIDTH * short_steps, riser), f(4.5),
  )
  base = saro_courses(visual="class_colors", level=level)["multi"]
  return replace(base, name="switching_demo", segments=segs, goal_marker=False, base_height=0.0)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--ckpt-root", default="eval_ckpts")
  ap.add_argument("--flat", default=None, help="Default: <ckpt-root>/go2_spec_flat/model_9999.pt")
  ap.add_argument("--rough", default=None, help="Default: <ckpt-root>/go2_spec_rough/model_9999.pt")
  ap.add_argument("--stairs", default=None, help="Default: <ckpt-root>/go2_spec_stairs_v8a/model_3999.pt")
  ap.add_argument("--single", default=None, choices=("flat", "rough", "stairs"),
                  help="No switching: run this one policy over the whole course (for comparison).")
  ap.add_argument("--step-height", type=float, default=0.15)
  ap.add_argument("--long-steps", type=int, default=10)
  ap.add_argument("--short-steps", type=int, default=5)
  ap.add_argument("--level", default="L2", help="Rough-ground roughness level (course.DIFFICULTY_LEVELS).")
  ap.add_argument("--leader-speed", type=float, default=0.55)
  ap.add_argument("--gap", type=float, default=2.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--no-noise", action="store_true")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--distance", type=float, default=4.6)
  ap.add_argument("--elevation", type=float, default=-14.0)
  ap.add_argument("--max-seconds", type=float, default=150.0)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  root = Path(args.ckpt_root)
  ckpts = {
    "flat": Path(args.flat or root / "go2_spec_flat/model_9999.pt"),
    "rough": Path(args.rough or root / "go2_spec_rough/model_9999.pt"),
    "stairs": Path(args.stairs or root / "go2_spec_stairs_v8a/model_3999.pt"),
  }
  course = build_course(args.step_height, args.long_steps, args.short_steps, args.level)

  cfg = make_twin_env_cfg(None, terrain_generator=course.generator_cfg(seed=args.seed), num_envs=1, seed=args.seed,
                          terminations="saro", leader=True)
  cfg.observations["actor"].enable_corruption = not args.no_noise
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base_link",
    distance=args.distance, elevation=args.elevation, azimuth=90.0, width=args.width, height=args.height,
  )
  raw = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(raw, clip_actions=load_rl_cfg(BASE_TASK).clip_actions)
  bank = PolicyBank(env, ckpts, device)
  u = env.unwrapped
  robot = u.scene["robot"]
  dt = u.step_dt

  x_off = course.length / 2
  centre_y = float(course.course_to_world(np.array([0.0, course.width / 2, 0.0]))[1])
  stop_x = course.length - 1.0
  gains = FollowGains()
  regions = course.regions()

  obs, _ = env.reset()
  lx0 = course.start_x + args.gap
  write_leader_pose(env, np.array([lx0 - x_off, centre_y, ground_height(course, lx0)]), 0.0)
  u.render()
  rend = u._offline_renderer
  scene_opt = mujoco.MjvOption()
  scene_opt.geomgroup[LEADER_GROUP] = 1
  m = rend._model
  for i in range(m.nlight):  # one raking light: risers darker than treads both ways, relief on rough ground
    d = np.array([-0.3, 0.5, -0.8])
    m.light_dir[i] = d / np.linalg.norm(d)
    m.light_type[i] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    m.light_diffuse[i] = (0.7, 0.7, 0.7)
    m.light_ambient[i] = (0.15, 0.15, 0.15)
    m.light_castshadow[i] = 1
  m.vis.headlight.diffuse[:] = 0.15
  m.vis.headlight.ambient[:] = 0.3

  def target_azimuth(x: float) -> float:
    """Behind-left of the robot while it climbs, ahead-left while it descends, side-on otherwise."""
    for reg, seg in zip(regions, course.segments):
      if seg.kind == "stairs_up" and reg.x0 - 2.0 < x < reg.x1 + 0.6:
        return 52.0
      if seg.kind == "stairs_down" and reg.x0 - 1.2 < x < reg.x1 + 1.0:
        return 128.0
    return 90.0

  font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
  font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
  font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
  W, H = args.width, args.height
  strip_x0, strip_x1 = 40, W - 40
  px = lambda x: strip_x0 + (strip_x1 - strip_x0) * float(np.clip(x / course.length, 0.0, 1.0))  # noqa: E731
  history: list[tuple[float, str]] = []  # (course x, active policy) per recorded frame

  def annotate(frame: np.ndarray, t: float, x: float, active: str, n_switches: int) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 40], fill=(0, 0, 0, 255))
    title = (f"Go2 follows a leader over flat, rough and {args.step_height * 100:.0f} cm stairs  |  "
             + ("one policy, no switching" if args.single else "policy switching"))
    d.text((14, 8), title, fill=(255, 255, 255), font=font)
    d.text((W - 262, 8), f"simulation  t = {t:5.1f} s", fill=(200, 200, 200), font=font)
    # Active-policy badge.
    d.rectangle([24, 58, 470, 122], fill=CLASS_RGB[active] + (235,), outline=(255, 255, 255, 255), width=2)
    d.text((40, 62), "ACTIVE POLICY", fill=(255, 255, 255), font=font_small)
    d.text((40, 80), LABEL[active], fill=(255, 255, 255), font=font_big)
    if not args.single:
      d.text((486, 78), f"switches so far: {n_switches}", fill=(230, 230, 230), font=font_small)
      d.text((486, 100), "rule: terrain class under the robot's footprint", fill=(190, 190, 190), font=font_small)
    # Bottom strip: terrain along the course, then which policy was active where.
    y0 = H - 92
    d.rectangle([0, y0 - 26, W, H], fill=(0, 0, 0, 200))
    d.text((strip_x0, y0 - 22), "terrain along the course", fill=(220, 220, 220), font=font_small)
    for reg in regions:
      d.rectangle([px(reg.x0), y0, px(reg.x1), y0 + 18], fill=CLASS_RGB[reg.terrain] + (255,))
    d.text((strip_x0, y0 + 24), "policy that was active there", fill=(220, 220, 220), font=font_small)
    for (xa, pol), (xb, _) in zip(history, history[1:] + [(x, active)]):
      d.rectangle([px(xa), y0 + 46, max(px(xb), px(xa) + 1), y0 + 64], fill=CLASS_RGB[pol] + (255,))
    d.rectangle([strip_x0, y0 + 46, strip_x1, y0 + 64], outline=(120, 120, 120, 255))
    xr = px(x)
    d.polygon([(xr, y0 - 2), (xr - 7, y0 - 14), (xr + 7, y0 - 14)], fill=(255, 255, 255, 255))
    d.line([(xr, y0 - 2), (xr, y0 + 66)], fill=(255, 255, 255, 255), width=2)
    lx = strip_x1 - 330
    for i, c in enumerate(("flat", "rough", "stairs")):
      d.rectangle([lx + 112 * i, y0 - 22, lx + 112 * i + 14, y0 - 8], fill=CLASS_RGB[c] + (255,))
      d.text((lx + 112 * i + 20, y0 - 24), c, fill=(230, 230, 230), font=font_small)
    return np.asarray(img.convert("RGB"))

  frames, outcome = [], "time limit"
  az = 90.0
  for step in range(int(args.max_seconds / dt)):
    t = step * dt
    lx = min(course.start_x + args.gap + args.leader_speed * t, stop_x)
    moving = lx < stop_x
    write_leader_pose(env, np.array([lx - x_off, centre_y, ground_height(course, lx)]), 0.0)
    pos = robot.data.root_link_pos_w
    x = float(pos[0, 0]) + x_off
    bank.select(args.single or course.required_terrain(x), step)
    leader_xy = torch.tensor([[lx - x_off, centre_y]], device=device, dtype=pos.dtype)
    leader_vel = torch.tensor([[args.leader_speed if moving else 0.0, 0.0]], device=device, dtype=pos.dtype)
    cmd, _ = follow_command(pos, robot.data.root_link_quat_w, leader_xy, args.gap, gains, leader_vel_w=leader_vel)
    set_velocity_command(env, cmd)
    obs, _, dones, _ = env.step(bank.act(obs))
    if step % 2 == 0:
      az += 0.06 * (target_azimuth(x) - az)
      rend._cam.azimuth = az
      rend.update(u.sim.data)
      rend._renderer.update_scene(rend._data, camera=rend._cam, scene_option=scene_opt)
      history.append((x, bank.active))
      frames.append(annotate(rend._renderer.render(), t, x, bank.active, len(bank.switch_log)))
    if bool(dones[0]):
      outcome = "FELL (episode ended)"
      break
    if not moving and x > stop_x - args.gap - 0.3:
      outcome = "reached the leader at the end of the course"
      break

  import imageio.v2 as imageio

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(out, frames, fps=int(round(1 / (2 * dt))), macro_block_size=None)
  print(f"{out}: {len(frames) * 2 * dt:.1f} s, course {course.length:.1f} m, outcome: {outcome}; "
        f"{len(bank.switch_log)} switches; policies {dict((k, str(v)) for k, v in ckpts.items())}")
  for s, a, b in bank.switch_log:
    print(f"   t = {s * dt:5.1f} s: {a} -> {b}")
  env.close()


if __name__ == "__main__":
  main()
