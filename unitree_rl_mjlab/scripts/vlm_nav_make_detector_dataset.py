"""Auto-labelled YOLO dataset of the scripted leader, from simulator ground truth.

Stock COCO YOLO does not recognise this project's leader: it reads the capsule legs
as "baseball bat" (0.79 conf) and the head sphere as "sports ball", at every camera
pitch and range tested, and finds nothing at all at the 6 m follow distance. That is
a simulation appearance gap, not an architecture problem -- on real hardware a real
person is exactly what COCO YOLO is good at.

Labels need no annotation: the leader's pose is set by us and its geom extents are
fixed, so the exact 2D box is the projection of its 3D box through the camera
(`CameraSpec.project_body`, pinned by a round-trip test).

Usage (from unitree_rl_mjlab/):
  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/vlm_nav_make_detector_dataset.py \
      --out /tmp/leader_ds --train 1400 --val 300
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

import src.tasks  # noqa: F401,E402
from src.vlm_nav.camera import CameraSpec, world_to_body  # noqa: E402
from src.vlm_nav.course import CourseSpec, Segment  # noqa: E402
from src.vlm_nav.leader import write_leader_pose  # noqa: E402
from src.vlm_nav.twin_env import BASE_TASK, CAMERA_SENSOR_NAME, make_twin_env_cfg, set_velocity_command  # noqa: E402

# Leader geom envelope from src/vlm_nav/leader.py (legs + torso + head).
HALF_X, HALF_Y, Z_LO, Z_HI = 0.135, 0.145, 0.011, 1.625
MIN_BOX_PX = 8.0


def leader_box_corners(pos_w: np.ndarray) -> np.ndarray:
  x, y, _ = pos_w
  return np.array([[x + sx * HALF_X, y + sy * HALF_Y, z]
                   for sx in (-1, 1) for sy in (-1, 1) for z in (Z_LO, Z_HI)])


def box_in_image(pos_w, base_pos_w, base_quat_w, camera: CameraSpec):
  """Tight pixel box of the leader, or None if not usefully visible."""
  corners_b = world_to_body(leader_box_corners(pos_w), base_pos_w, base_quat_w)
  uv = camera.project_body(corners_b)
  if np.isnan(uv).any():  # any corner behind the image plane -> skip, don't guess
    return None
  u0, v0 = uv[:, 0].min(), uv[:, 1].min()
  u1, v1 = uv[:, 0].max(), uv[:, 1].max()
  cu0, cv0 = max(0.0, u0), max(0.0, v0)
  cu1, cv1 = min(camera.width - 1.0, u1), min(camera.height - 1.0, v1)
  if cu1 - cu0 < MIN_BOX_PX or cv1 - cv0 < MIN_BOX_PX:
    return None
  # Mostly out of frame -> the visible sliver is a bad label.
  if (cu1 - cu0) * (cv1 - cv0) < 0.35 * max(1.0, (u1 - u0) * (v1 - v0)):
    return None
  return cu0, cv0, cu1, cv1


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--train", type=int, default=1400)
  ap.add_argument("--val", type=int, default=300)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--neg-frac", type=float, default=0.10, help="Fraction of frames with no leader in view.")
  args = ap.parse_args()

  out = Path(args.out)
  for split in ("train", "val"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

  rng = np.random.default_rng(args.seed)
  course = CourseSpec(name="ds", segments=(Segment("flat", 40.0),), intermediation="none",
                      instruction="detector dataset", width=8.0, visual="tiled", goal_marker=False)
  camera = CameraSpec()
  cfg = make_twin_env_cfg(camera, terrain_generator=course.generator_cfg(seed=args.seed),
                          num_envs=1, seed=args.seed, leader=True, spawn_yaw_range=(-0.4, 0.4))
  configure_torch_backends()
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda"), clip_actions=load_rl_cfg(BASE_TASK).clip_actions)
  u = env.unwrapped
  robot, cam = u.scene["robot"], u.scene[CAMERA_SENSOR_NAME]
  obs, _ = env.reset()
  zero = torch.zeros(1, 3, device="cuda")
  zero_act = torch.zeros(1, 12, device="cuda")
  for _ in range(20):  # let it settle into a stand
    set_velocity_command(env, zero)
    obs, *_ = env.step(zero_act)

  total = args.train + args.val
  kept = neg = 0
  i = 0
  while kept < total and i < total * 6:
    i += 1
    base_pos = robot.data.root_link_pos_w[0].cpu().numpy()
    base_quat = robot.data.root_link_quat_w[0].cpu().numpy()
    negative = rng.random() < args.neg_frac
    if negative:
      # Behind the robot: renders an empty scene, teaching "no leader here".
      rel = np.array([-rng.uniform(2.0, 8.0), rng.uniform(-3.0, 3.0)])
    else:
      rng_m = rng.uniform(1.2, 16.0)
      bearing = np.radians(rng.uniform(-32.0, 32.0))
      rel = np.array([rng_m * np.cos(bearing), rng_m * np.sin(bearing)])
    yaw = float(np.arctan2(*base_quat[[3, 0]][::-1])) * 0  # world-frame placement below
    from src.vlm_nav.camera import body_to_world  # noqa: PLC0415
    lead_w = body_to_world(np.array([[rel[0], rel[1], 0.0]]), base_pos, base_quat)[0]
    lead_w[2] = base_pos[2] - 0.33  # stand on the floor
    write_leader_pose(env, lead_w, float(rng.uniform(-np.pi, np.pi)))
    for _ in range(2):
      set_velocity_command(env, zero)
      obs, *_ = env.step(zero_act)

    base_pos = robot.data.root_link_pos_w[0].cpu().numpy()
    base_quat = robot.data.root_link_quat_w[0].cpu().numpy()
    box = None if negative else box_in_image(lead_w, base_pos, base_quat, camera)
    if not negative and box is None:
      continue
    split = "train" if kept < args.train else "val"
    name = f"{kept:05d}"
    rgb = cam.data.rgb[0].cpu().numpy().astype(np.uint8)
    Image.fromarray(rgb).save(out / "images" / split / f"{name}.png")
    lbl = out / "labels" / split / f"{name}.txt"
    if box is None:
      lbl.write_text("")  # YOLO background image
      neg += 1
    else:
      cu0, cv0, cu1, cv1 = box
      cx, cy = (cu0 + cu1) / 2 / camera.width, (cv0 + cv1) / 2 / camera.height
      bw, bh = (cu1 - cu0) / camera.width, (cv1 - cv0) / camera.height
      lbl.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    kept += 1
    if kept % 200 == 0:
      print(f"  {kept}/{total} ({neg} negatives)", flush=True)

  (out / "data.yaml").write_text(
    f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n"
  )
  print(f"DATASET_DONE kept={kept} negatives={neg} attempts={i} -> {out}")
  env.close()


if __name__ == "__main__":
  main()
