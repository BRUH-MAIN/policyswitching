"""Render labelled ego-camera frames at controlled poses on one navigation course.

Two uses: a quick visual check of a course (`--xs 1.0 2.5 4.0`), and building the
offline VLM perception dataset (plan Phase 2) by sweeping x / lateral offset /
yaw. Output: <out>/<course>_<visual>/frame_XXXX.png (+ _bbox.png overlays with
--overlay) and labels.jsonl, one FrameLabel per frame.

Usage (from unitree_rl_mjlab/):
  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/vlm_nav_render_frames.py \
      --course stairs_up --visual plain --out logs/vlm_nav/frames --overlay
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.vlm_nav.camera import CameraSpec  # noqa: E402
from src.vlm_nav.course import saro_courses  # noqa: E402
from src.vlm_nav.frames import label_dict, label_frame, place_robot  # noqa: E402
from src.vlm_nav.twin_env import make_twin_env_cfg  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--course", required=True)
  ap.add_argument("--visual", default="tiled", choices=("plain", "tiled", "class_colors"))
  ap.add_argument("--goal-y-offset", type=float, default=0.0)
  ap.add_argument("--level", default="L2", help="Course difficulty level (course.DIFFICULTY_LEVELS).")
  ap.add_argument("--xs", type=float, nargs="*", default=None, help="Course x positions; default: sweep.")
  ap.add_argument("--x-step", type=float, default=0.5)
  ap.add_argument("--ys", type=float, nargs="*", default=[0.0], help="Lateral offsets from centreline.")
  ap.add_argument("--yaws", type=float, nargs="*", default=[0.0], help="Yaw offsets (rad).")
  ap.add_argument("--seed", type=int, default=0, help="Terrain seed (rough heightfield).")
  ap.add_argument("--overlay", action="store_true", help="Also save frames with the GT bbox drawn.")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  configure_torch_backends()
  course = saro_courses(visual=args.visual, goal_y_offset=args.goal_y_offset, level=args.level)[args.course]
  camera = CameraSpec()
  cfg = make_twin_env_cfg(camera, terrain_generator=course.generator_cfg(seed=args.seed))
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.reset()

  out = Path(args.out) / f"{course.name}_{args.visual}_g{args.goal_y_offset:+.1f}_s{args.seed}"
  out.mkdir(parents=True, exist_ok=True)
  xs = args.xs if args.xs is not None else list(np.arange(course.start_x, course.length - 1.0, args.x_step))
  n = 0
  with open(out / "labels.jsonl", "w") as f:
    for x in xs:
      for dy in args.ys:
        for yaw in args.yaws:
          y = course.width / 2 + dy
          place_robot(env, course, float(x), y, float(yaw))
          rgb, _, label = label_frame(env, course, camera, float(x), y, float(yaw))
          name = f"frame_{n:04d}"
          Image.fromarray(rgb).save(out / f"{name}.png")
          if args.overlay:
            im = Image.fromarray(rgb)
            if label.intermediation_bbox:
              ImageDraw.Draw(im).rectangle(label.intermediation_bbox, outline=(255, 0, 255), width=3)
            im.save(out / f"{name}_bbox.png")
          f.write(json.dumps({"image": f"{name}.png", **label_dict(label)}) + "\n")
          n += 1
  print(f"[INFO] wrote {n} frames to {out}")


if __name__ == "__main__":
  main()
