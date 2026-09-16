"""Place the robot at a chosen pose on a course, render the ego camera, and label the frame.

Used for the offline VLM perception gate (plan Phase 2) and for course sanity
checks. Labels come from the course definition, not the renderer, except the
intermediation bounding box, which is the pixel extent of the intermediation's
surface points that are actually *visible* (projected range agrees with the
rendered depth) -- so a staircase hidden behind a platform edge, or outside the
field of view, has no box rather than a box drawn through the occluder.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch

from src.vlm_nav.camera import CameraSpec
from src.vlm_nav.course import STEP_WIDTH, CourseSpec, Region
from src.vlm_nav.twin_env import get_rgbd

STANDING_BASE_HEIGHT = 0.33


def surface_z(course: CourseSpec, x_course: float) -> float:
  """Nominal walking-surface height at course x (rough: the segment's base plane)."""
  for r in course.regions():
    if r.x0 <= x_course < r.x1:
      if r.terrain != "stairs":
        return r.z_start
      n = max(1, int(round((r.x1 - r.x0) / STEP_WIDTH)))
      k = min(n - 1, int((x_course - r.x0) / ((r.x1 - r.x0) / n)))
      step = (r.z_end - r.z_start) / n
      return r.z_start + (k + 1) * step
  return 0.0


def yaw_quat(yaw: float) -> np.ndarray:
  return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def place_robot(env, course: CourseSpec, x: float, y: float, yaw: float, env_id: int = 0) -> None:
  """Teleport the robot (standing default joint pose) and refresh sensors, no physics step."""
  u = env.unwrapped
  robot = u.scene["robot"]
  device = robot.data.root_link_pos_w.device
  z = surface_z(course, x) + STANDING_BASE_HEIGHT
  pos_w = course.course_to_world(np.array([x, y, z]))
  state = torch.zeros(1, 13, device=device)
  state[0, :3] = torch.as_tensor(pos_w, device=device)
  state[0, 3:7] = torch.as_tensor(yaw_quat(yaw), device=device)
  ids = torch.tensor([env_id], device=device)
  robot.write_root_state_to_sim(state, ids)
  robot.write_joint_state_to_sim(
    robot.data.default_joint_pos[ids].clone(), torch.zeros_like(robot.data.default_joint_vel[ids]), env_ids=ids
  )
  u.sim.forward()
  u.sim.sense()


@dataclass
class FrameLabel:
  course: str
  visual: str
  x: float
  y: float
  yaw: float
  terrain_under: str
  terrain_ahead_1m: str
  next_intermediation: str | None
  dist_to_intermediation: float | None
  """Base x to the start of the next non-flat region; 0 while on it; None if none remain."""
  intermediation_bbox: list[int] | None
  """[x0, y0, x1, y1] pixels of the visible surface of the next/current intermediation."""
  intermediation_visible_frac: float
  goal_dist: float
  goal_in_view: bool


def _next_intermediation(course: CourseSpec, x: float) -> Region | None:
  for r in course.regions():
    if r.terrain != "flat" and r.x1 > x:
      return r
  return None


def project_world(camera: CameraSpec, p_world: np.ndarray, base_pos: np.ndarray, base_quat: np.ndarray):
  """World points -> (u, v, ray_distance, in_front) for the ego camera."""
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, np.asarray(base_quat, dtype=np.float64))
  p_body = (np.asarray(p_world) - base_pos) @ rot.reshape(3, 3)
  p_cam = (p_body - np.asarray(camera.pos)) @ camera.rotation_body_from_cam
  in_front = p_cam[:, 2] < -1e-3
  zc = np.where(in_front, -p_cam[:, 2], 1.0)
  f = camera.focal_px
  u = f * p_cam[:, 0] / zc + camera.width / 2.0 - 0.5
  v = -f * p_cam[:, 1] / zc + camera.height / 2.0 - 0.5
  return u, v, np.linalg.norm(p_cam, axis=1), in_front


def visible_region_bbox(
  course: CourseSpec, region: Region, camera: CameraSpec, depth: np.ndarray, base_pos, base_quat, tol: float = 0.12
) -> tuple[list[int] | None, float]:
  xs = np.linspace(region.x0 + 0.02, region.x1 - 0.02, 40)
  ys = np.linspace(0.05, course.width - 0.05, 30)
  gx, gy = np.meshgrid(xs, ys)
  gz = np.vectorize(lambda x: surface_z(course, x))(gx)
  if region.terrain == "rough":
    gz = gz + 0.04  # mean of the default 0.02-0.08 noise; within tol either way
  pts = course.course_to_world(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1))
  u, v, dist, front = project_world(camera, pts, base_pos, base_quat)
  ui, vi = np.round(u).astype(int), np.round(v).astype(int)
  inside = front & (ui >= 0) & (ui < camera.width) & (vi >= 0) & (vi < camera.height)
  vis = np.zeros_like(inside)
  idx = np.nonzero(inside)[0]
  vis[idx] = np.abs(depth[vi[idx], ui[idx]] - dist[idx]) < tol
  if vis.sum() < 5:
    return None, float(vis.mean())
  return [int(ui[vis].min()), int(vi[vis].min()), int(ui[vis].max()), int(vi[vis].max())], float(vis.mean())


def label_frame(env, course: CourseSpec, camera: CameraSpec, x: float, y: float, yaw: float) -> tuple[np.ndarray, np.ndarray, FrameLabel]:
  rgb, depth = get_rgbd(env)
  robot = env.unwrapped.scene["robot"]
  base_pos = robot.data.root_link_pos_w[0].cpu().numpy()
  base_quat = robot.data.root_link_quat_w[0].cpu().numpy()
  region = _next_intermediation(course, x)
  bbox, frac = (None, 0.0)
  if region is not None:
    bbox, frac = visible_region_bbox(course, region, camera, depth, base_pos, base_quat)
  goal_w = course.course_to_world(course.goal_course + np.array([0.0, 0.0, 0.3]))
  gu, gv, _, gfront = project_world(camera, goal_w[None], base_pos, base_quat)
  label = FrameLabel(
    course=course.name,
    visual=course.visual,
    x=x, y=y, yaw=yaw,
    terrain_under=course.terrain_at(x + 0.25),
    terrain_ahead_1m=course.terrain_at(x + 1.0),
    next_intermediation=None if region is None else region.terrain,
    dist_to_intermediation=None if region is None else max(0.0, region.x0 - x),
    intermediation_bbox=bbox,
    intermediation_visible_frac=frac,
    goal_dist=float(np.linalg.norm(course.goal_course[:2] - np.array([x, y]))),
    goal_in_view=bool(gfront[0] and 0 <= gu[0] < camera.width and 0 <= gv[0] < camera.height),
  )
  return rgb, depth, label


def label_dict(label: FrameLabel) -> dict:
  return asdict(label)
