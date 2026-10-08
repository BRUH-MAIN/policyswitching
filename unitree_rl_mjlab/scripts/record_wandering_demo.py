"""Record a policy-switching demo in which the leader wanders: circles, weaves, and stairs.

Like `record_switching_demo.py`, but the scripted person does not walk a straight line. On a
wider course (flat plaza, rough ground, a 10-step flight up to a raised landing, the flight
down, another plaza, more rough ground) the leader:

  * loops one and a quarter times round the first plaza (radius 2.5 m),
  * crosses the first rough patch on a diagonal,
  * walks straight up the stairs, circles on the landing (radius 2 m), and walks straight down,
  * weaves through the second rough patch.

The robot follows with the evaluation's follow controller (turn to face the leader, hold the
gap), so it cuts inside the leader's curves. Terrain class depends only on distance along
the course, so the leader's sideways motion changes the robot's heading and path, not which
terrain it is on.

Switching is by `--classifier` (the robot's own height scan; see `record_switching_demo.py`)
or, without it, by the ground-truth footprint rule. The overlay is a top-down map: the course
coloured by terrain, the leader's path, and the robot's trail coloured by the policy that was
active. One trial, in simulation.

  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/record_wandering_demo.py \
      --classifier logs/switch_follow/clf_demo15/clf_noisy_15cm.pt --alpha 0.3 --hold 3 \
      --out ../report_content/videos/switching_demo/wandering_leader_own_scan.mp4
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
from record_switching_demo import CLASS_RGB, LABEL  # noqa: E402
from src.vlm_nav.controllers import FollowGains, follow_command  # noqa: E402
from src.vlm_nav.course import DIFFICULTY_LEVELS, STEP_WIDTH, Segment, saro_courses  # noqa: E402
from src.vlm_nav.leader import write_leader_pose  # noqa: E402
from src.vlm_nav.policy_bank import PolicyBank  # noqa: E402
from src.vlm_nav.scan_classifier import CLASSES, SwitchFilter, load_classifier, scan_slice  # noqa: E402
from src.vlm_nav.twin_env import BASE_TASK, make_twin_env_cfg, set_velocity_command  # noqa: E402

WIDTH = 10.0  # course width (m)
PLAZA_A, ROUGH_A, PRE, LANDING, PLAZA_B, ROUGH_B, END = 9.0, 6.0, 3.0, 8.0, 4.0, 6.0, 3.5
R_PLAZA, R_LANDING = 2.5, 2.0  # leader's circle radii (m); both larger than the follow gap


def build_course(riser: float, steps: int, level: str):
  noise = DIFFICULTY_LEVELS[level]["noise_range"]
  flight = STEP_WIDTH * steps
  segs = (
    Segment("flat", PLAZA_A, riser), Segment("rough", ROUGH_A, riser, noise_range=noise), Segment("flat", PRE, riser),
    Segment("stairs_up", flight, riser), Segment("flat", LANDING, riser),
    Segment("stairs_down", flight, riser), Segment("flat", PLAZA_B, riser),
    Segment("rough", ROUGH_B, riser, noise_range=noise), Segment("flat", END, riser),
  )
  base = saro_courses(visual="class_colors", level=level)["multi"]
  return replace(base, name="wandering_demo", segments=segs, width=WIDTH, goal_marker=False, base_height=0.0)


def arc(cx: float, cy: float, r: float, a0: float, a1: float, n_per_turn: int = 240) -> np.ndarray:
  """Points on a circle from angle a0 to a1 (degrees; a1 > a0 is counter-clockwise)."""
  n = max(2, int(abs(a1 - a0) / 360.0 * n_per_turn))
  a = np.radians(np.linspace(a0, a1, n))
  return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def leader_path(course, start_x: float, mirror: bool = False) -> np.ndarray:
  """(N, 2) points of the leader's path in course coordinates (x along the course, y across).

  `mirror` flips it across the centre line (used to collect classifier training data on a
  path that is not the one filmed)."""
  mid = WIDTH / 2
  x = {}
  edge = 0.0
  for name, seg in zip(("plaza_a", "rough_a", "pre", "up", "landing", "down", "plaza_b", "rough_b", "end"), course.segments):
    x[name] = (edge, edge + seg.length)
    edge += seg.length
  pts = []
  # Plaza A: a quarter turn and then a full counter-clockwise loop round the middle of the
  # plaza, leaving along +x from the bottom of the circle.
  r = R_PLAZA
  pts.append(arc(start_x + r, mid, r, 180.0, 180.0 + 90.0 + 360.0))
  pts.append(np.array([[start_x + r, mid - r], [x["rough_a"][0], mid - r]]))
  # First rough patch: one smooth diagonal sweep from that side back to the centre line.
  xs = np.linspace(x["rough_a"][0], x["rough_a"][1], 120)
  pts.append(np.stack([xs, mid - r + r * 0.5 * (1 - np.cos(np.pi * (xs - xs[0]) / (xs[-1] - xs[0])))], axis=1))
  # Straight up the stairs, on to the landing.
  lx0, lx1 = x["landing"]
  lc = (lx0 + lx1) / 2
  pts.append(np.array([[x["rough_a"][1], mid], [lc, mid]]))
  # Landing: a full counter-clockwise circle, tangent to the centre line at its lowest point.
  pts.append(arc(lc, mid + R_LANDING, R_LANDING, 270.0, 270.0 + 360.0))
  # Straight down the stairs and across the second plaza.
  pts.append(np.array([[lc, mid], [x["rough_b"][0], mid]]))
  # Second rough patch: an S across both sides.
  xs = np.linspace(x["rough_b"][0], x["rough_b"][1], 160)
  pts.append(np.stack([xs, mid + 2.0 * np.sin(2 * np.pi * (xs - xs[0]) / (xs[-1] - xs[0]))], axis=1))
  pts.append(np.array([[x["rough_b"][1], mid], [x["end"][1] - 1.0, mid]]))
  path = np.concatenate(pts)
  if mirror:
    path[:, 1] = WIDTH - path[:, 1]
  keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-6])
  return path[keep]


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--ckpt-root", default="eval_ckpts")
  ap.add_argument("--flat", default=None)
  ap.add_argument("--rough", default=None)
  ap.add_argument("--stairs", default=None)
  ap.add_argument("--classifier", default=None, help="Scan-classifier weights: the robot picks the policy itself.")
  ap.add_argument("--alpha", type=float, default=0.3)
  ap.add_argument("--hold", type=int, default=3)
  ap.add_argument("--single", default=None, choices=("flat", "rough", "stairs"))
  ap.add_argument("--step-height", type=float, default=0.15)
  ap.add_argument("--steps", type=int, default=10)
  ap.add_argument("--level", default="L2")
  ap.add_argument("--leader-speed", type=float, default=0.7)
  ap.add_argument("--gap", type=float, default=1.5)
  ap.add_argument("--mirror", action="store_true", help="Mirror the leader's path across the centre line.")
  ap.add_argument("--collect", default=None, metavar="OUT.pt",
                  help="No video: run --num-envs robots with ground-truth switching and save their height "
                       "scans with the footprint rule's class, for scan_classifier_train.py.")
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--no-noise", action="store_true")
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--distance", type=float, default=6.4)
  ap.add_argument("--elevation", type=float, default=-20.0)
  ap.add_argument("--max-seconds", type=float, default=240.0)
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
  course = build_course(args.step_height, args.steps, args.level)
  path = leader_path(course, course.start_x + args.gap, args.mirror)
  seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
  cum = np.concatenate([[0.0], np.cumsum(seg_len)])
  total = float(cum[-1])

  def leader_at(t: float):
    """Course-frame position, heading and velocity of the leader at time t."""
    s = min(args.leader_speed * t, total)
    i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg_len) - 1))
    w = (s - cum[i]) / seg_len[i]
    p = path[i] + w * (path[i + 1] - path[i])
    tangent = (path[i + 1] - path[i]) / seg_len[i]
    moving = s < total
    return p, float(np.arctan2(tangent[1], tangent[0])), tangent * (args.leader_speed if moving else 0.0), moving

  n_envs = args.num_envs if args.collect else 1
  cfg = make_twin_env_cfg(None, terrain_generator=course.generator_cfg(seed=args.seed), num_envs=n_envs, seed=args.seed,
                          terminations="saro", leader=not args.collect,  # the leader body is one-env only
                          spawn_xy_jitter=0.25 if args.collect else 0.0,
                          spawn_yaw_range=(-0.4, 0.4) if args.collect else (0.0, 0.0))
  cfg.observations["actor"].enable_corruption = not args.no_noise
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base_link",
    distance=args.distance, elevation=args.elevation, azimuth=90.0, width=args.width, height=args.height,
  )
  raw = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None if args.collect else "rgb_array")
  env = RslRlVecEnvWrapper(raw, clip_actions=load_rl_cfg(BASE_TASK).clip_actions)
  bank = PolicyBank(env, ckpts, device)
  u = env.unwrapped
  robot = u.scene["robot"]
  dt = u.step_dt

  clf = filt = scan_sl = None
  probs_now = np.array([1.0, 0.0, 0.0])
  if args.classifier:
    clf = load_classifier(args.classifier, device)
    filt = SwitchFilter(args.num_envs if args.collect else 1, args.alpha, args.hold, device,
                        initial=CLASSES.index("flat"))
    scan_sl = scan_slice(env)
  agree_steps = total_steps = 0

  off = np.array([course.length / 2, course.width / 2])  # course xy = world xy + off
  gains = FollowGains()
  regions = course.regions()

  obs, _ = env.reset()
  p0, yaw0, _, _ = leader_at(0.0)
  if not args.collect:
    write_leader_pose(env, np.array([p0[0] - off[0], p0[1] - off[1], ground_height(course, p0[0])]), yaw0)

  if args.collect:
    names = bank.names
    scan_sl = scan_slice(env)
    off_t = torch.tensor(off, device=device, dtype=torch.float32)
    alive = torch.ones(n_envs, dtype=torch.bool, device=device)
    rec_scan, rec_label, rec_trial = [], [], []
    steps = int(min(args.max_seconds, total / args.leader_speed + 4.0) / dt)
    for step in range(steps):
      lp, lyaw, lvel, _ = leader_at(step * dt)  # followed as a point; the scan never sees the leader
      pos = robot.data.root_link_pos_w
      xs = (pos[:, 0] + off_t[0]).cpu().numpy()
      labels = [course.required_terrain(float(v)) for v in xs]
      label_idx = torch.tensor([CLASSES.index(c) for c in labels], device=device)
      policy_idx = torch.tensor([names.index(c) for c in labels], device=device)
      if clf is not None:  # let the classifier drive, to count how many robots finish that way
        with torch.inference_mode():
          sel = filt.step(torch.softmax(clf(obs["actor"][:, scan_sl]), dim=1))
        policy_idx = torch.tensor([names.index(c) for c in CLASSES], device=device)[sel]
        agree_steps += int(((sel == label_idx) & alive).sum())
        total_steps += int(alive.sum())
      if step % 3 == 0:
        keep = alive.nonzero().flatten()
        rec_scan.append(obs["actor"][keep][:, scan_sl].detach().cpu())
        rec_label.append(label_idx[keep].cpu())
        rec_trial.append(keep.cpu())
      leader_xy = torch.tensor(lp - off, device=device, dtype=pos.dtype).expand(n_envs, 2)
      leader_vel = torch.tensor(lvel, device=device, dtype=pos.dtype).expand(n_envs, 2)
      cmd, _ = follow_command(pos, robot.data.root_link_quat_w, leader_xy, args.gap, gains, leader_vel_w=leader_vel)
      set_velocity_command(env, cmd)
      obs, _, dones, _ = env.step(bank.act_per_env(obs, policy_idx))
      alive &= ~dones.bool()
    Path(args.collect).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(scan=torch.cat(rec_scan), label=torch.cat(rec_label), trial=torch.cat(rec_trial),
                    classes=list(CLASSES), conditions=vars(args)), args.collect)
    counts = torch.bincount(torch.cat(rec_label), minlength=len(CLASSES)).tolist()
    print(f"{args.collect}: {sum(counts)} scans {dict(zip(CLASSES, counts))}; "
          f"{int(alive.sum())} of {n_envs} robots finished without falling"
          + (f"; classifier-driven, agreeing with the footprint rule on {100 * agree_steps / max(total_steps, 1):.1f}% of steps"
             if clf is not None else "; ground-truth switching"))
    env.close()
    return

  u.render()
  rend = u._offline_renderer
  scene_opt = mujoco.MjvOption()
  scene_opt.geomgroup[LEADER_GROUP] = 1
  m = rend._model
  for i in range(m.nlight):
    d = np.array([-0.3, 0.5, -0.8])
    m.light_dir[i] = d / np.linalg.norm(d)
    m.light_type[i] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    m.light_diffuse[i] = (0.7, 0.7, 0.7)
    m.light_ambient[i] = (0.15, 0.15, 0.15)
    m.light_castshadow[i] = 1
  m.vis.headlight.diffuse[:] = 0.15
  m.vis.headlight.ambient[:] = 0.3

  def target_azimuth(x: float) -> float:
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
  # Top-down map along the bottom: 1 m = `ppm` pixels, course y up.
  ppm = 18.0
  map_h = int(course.width * ppm)
  map_w = int(course.length * ppm)
  mx0, my1 = 30, H - 14
  to_px = lambda p: (mx0 + p[0] * ppm, my1 - p[1] * ppm)  # noqa: E731
  path_px = [to_px(p) for p in path[::4]]
  trail: list[tuple[float, float, str]] = []

  def annotate(frame: np.ndarray, t: float, rxy: np.ndarray, lxy: np.ndarray, active: str, n_switches: int):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 40], fill=(0, 0, 0, 255))
    title = ("Go2 following a wandering leader  |  "
             + ("one policy, no switching" if args.single else "policy chosen from its own height scan"
                if clf is not None else "policy switched by ground truth"))
    d.text((14, 8), title, fill=(255, 255, 255), font=font)
    d.text((W - 262, 8), f"simulation  t = {t:5.1f} s", fill=(200, 200, 200), font=font)
    d.rectangle([24, 58, 470, 122], fill=CLASS_RGB[active] + (235,), outline=(255, 255, 255, 255), width=2)
    d.text((40, 62), "ACTIVE POLICY", fill=(255, 255, 255), font=font_small)
    d.text((40, 80), LABEL[active], fill=(255, 255, 255), font=font_big)
    if clf is not None:
      d.text((486, 60), f"switches so far: {n_switches}", fill=(230, 230, 230), font=font_small)
      d.text((486, 80), "terrain classifier on the height scan:", fill=(190, 190, 190), font=font_small)
      for i, c in enumerate(CLASSES):
        bx = 486 + 150 * i
        d.text((bx, 102), c, fill=(230, 230, 230), font=font_small)
        d.rectangle([bx + 50, 104, bx + 130, 118], outline=(150, 150, 150, 255))
        d.rectangle([bx + 50, 104, bx + 50 + 80 * float(probs_now[i]), 118], fill=CLASS_RGB[c] + (255,))
    elif not args.single:
      d.text((486, 78), f"switches so far: {n_switches}", fill=(230, 230, 230), font=font_small)
      d.text((486, 100), "rule: terrain class under the robot's footprint", fill=(190, 190, 190), font=font_small)
    # Map.
    top = my1 - map_h
    d.rectangle([0, top - 30, W, H], fill=(0, 0, 0, 190))
    d.text((mx0, top - 26), "course from above" + (" (not given to the robot)" if clf is not None else ""),
           fill=(220, 220, 220), font=font_small)
    for reg in regions:
      c = CLASS_RGB[reg.terrain]
      d.rectangle([mx0 + reg.x0 * ppm, top, mx0 + reg.x1 * ppm, my1], fill=(c[0] // 3, c[1] // 3, c[2] // 3, 255))
    d.line(path_px, fill=(255, 255, 255, 110), width=1)
    for tx, ty, pol in trail:
      d.ellipse([tx - 2, ty - 2, tx + 2, ty + 2], fill=CLASS_RGB[pol] + (255,))
    lx, ly = to_px(lxy)
    d.ellipse([lx - 6, ly - 6, lx + 6, ly + 6], fill=(255, 150, 30, 255), outline=(0, 0, 0, 255))
    rx, ry = to_px(rxy)
    d.ellipse([rx - 6, ry - 6, rx + 6, ry + 6], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
    lx0 = mx0 + map_w + 28
    d.text((lx0, top - 2), "terrain / active policy", fill=(200, 200, 200), font=font_small)
    for i, name in enumerate(("flat", "rough", "stairs")):
      d.rectangle([lx0, top + 24 + 24 * i, lx0 + 14, top + 38 + 24 * i], fill=CLASS_RGB[name] + (255,))
      d.text((lx0 + 22, top + 21 + 24 * i), name, fill=(230, 230, 230), font=font_small)
    for i, (c, name) in enumerate((((255, 150, 30), "leader"), ((255, 255, 255), "robot"))):
      d.ellipse([lx0 + 110, top + 24 + 24 * i, lx0 + 124, top + 38 + 24 * i], fill=c + (255,), outline=(0, 0, 0, 255))
      d.text((lx0 + 132, top + 21 + 24 * i), name, fill=(230, 230, 230), font=font_small)
    d.text((lx0, top + 104), "thin line: the leader's path", fill=(200, 200, 200), font=font_small)
    d.text((lx0, top + 126), "thick trail: the robot, coloured", fill=(200, 200, 200), font=font_small)
    d.text((lx0, top + 146), "by the policy it was running", fill=(200, 200, 200), font=font_small)
    return np.asarray(img.convert("RGB"))

  frames, outcome = [], "time limit"
  az = 90.0
  for step in range(int(args.max_seconds / dt)):
    t = step * dt
    lp, lyaw, lvel, moving = leader_at(t)
    write_leader_pose(env, np.array([lp[0] - off[0], lp[1] - off[1], ground_height(course, lp[0])]), lyaw)
    pos = robot.data.root_link_pos_w
    rxy = pos[0, :2].cpu().numpy() + off
    truth = course.required_terrain(float(rxy[0]))
    if clf is not None:
      with torch.inference_mode():
        sel = int(filt.step(torch.softmax(clf(obs["actor"][:, scan_sl]), dim=1))[0])
      probs_now = filt.probs[0].cpu().numpy()
      choice = CLASSES[sel]
      agree_steps += int(choice == truth)
      total_steps += 1
    else:
      choice = args.single or truth
    bank.select(choice, step)
    leader_xy = torch.tensor([lp - off], device=device, dtype=pos.dtype)
    leader_vel = torch.tensor([lvel], device=device, dtype=pos.dtype)
    cmd, rng = follow_command(pos, robot.data.root_link_quat_w, leader_xy, args.gap, gains, leader_vel_w=leader_vel)
    set_velocity_command(env, cmd)
    obs, _, dones, _ = env.step(bank.act(obs))
    if step % 2 == 0:
      az += 0.06 * (target_azimuth(float(rxy[0])) - az)
      rend._cam.azimuth = az
      rend.update(u.sim.data)
      rend._renderer.update_scene(rend._data, camera=rend._cam, scene_option=scene_opt)
      if step % 6 == 0:
        trail.append((*to_px(rxy), bank.active))
      frames.append(annotate(rend._renderer.render(), t, rxy, lp, bank.active, len(bank.switch_log)))
    if bool(dones[0]):
      outcome = "FELL (episode ended)"
      break
    if not moving and float(rng[0]) < args.gap + 0.3:
      outcome = "reached the leader at the end of its path"
      break

  import imageio.v2 as imageio

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(out, frames, fps=int(round(1 / (2 * dt))), macro_block_size=None)
  print(f"{out}: {len(frames) * 2 * dt:.1f} s, leader path {total:.1f} m on a {course.length:.1f} x {course.width:.0f} m course, "
        f"outcome: {outcome}; {len(bank.switch_log)} switches")
  if clf is not None:
    print(f"   classifier {args.classifier} (alpha {args.alpha}, hold {args.hold}): its choice matched the "
          f"footprint rule on {100 * agree_steps / max(total_steps, 1):.1f}% of steps")
  stairs_sw = [(s * dt, a, b) for s, a, b in bank.switch_log if "stairs" in (a, b)]
  print("   switches to or from the stairs policy:", ", ".join(f"{t:.1f} s {a}->{b}" for t, a, b in stairs_sw))
  env.close()


if __name__ == "__main__":
  main()
