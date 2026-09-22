"""Ego camera geometry: mount pose, pinhole intrinsics, pixel+depth -> 3D.

SARO (arXiv:2407.16412, Appendix A, Fig. 8) turns the VLM's bounding box into a
3D sub-goal by deprojecting the box centre through an aligned depth image:
  {X, Y, Z} = {(i - x0) * d / fx, (j - y0) * d / fy, d}
where d is *optical-axis* depth (RealSense convention). mujoco_warp's depth
buffer is different: it stores the hit distance along each pixel's unit ray
(`render.py`: `depth_out[...] = dist` from `cast_ray` with a normalized
direction), so the deprojection here scales the unit ray, not the pixel's
z-extent. Mixing the two conventions puts off-centre points too close by a
factor of cos(angle off axis) -- ~10% at the image corner for a 42 deg FOV,
enough to stop a sub-goal short of a staircase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraSpec:
  """Forward-facing RGB-D camera rigidly mounted on the Go2 base.

  Defaults follow SARO's lower RealSense D435i (848x480 aligned RGB-D, colour
  vertical FOV ~42.5 deg), mounted at the Go2 head and pitched down so the
  image spans the ground from ~0.5 m out to the horizon.
  """

  parent_body: str = "robot/base_link"
  pos: tuple[float, float, float] = (0.37, 0.0, 0.03)
  """Mount position in the base_link frame (x forward, y left, z up), metres.

  Just ahead of the head: the head collision cylinder (`base2_collision`, centre
  x=0.285, radius 0.05) ends at x=0.335, and a camera behind that renders the
  inside of the head's visual mesh instead of the scene.
  """
  pitch_down_deg: float = 15.0
  fovy_deg: float = 42.5
  width: int = 848
  height: int = 480

  @property
  def quat(self) -> tuple[float, float, float, float]:
    """(w, x, y, z) of the camera frame in the parent body frame.

    A MuJoCo camera looks along its local -z with +y as image-up. Level mount:
    cam x = body -y (image right), cam y = body +z, cam z = body -x. Pitching
    down is a positive rotation about body +y (tips the view direction +x
    toward -z).
    """
    return tuple(float(q) for q in _mat_to_quat(self.rotation_body_from_cam))

  @property
  def rotation_body_from_cam(self) -> np.ndarray:
    level = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    t = math.radians(self.pitch_down_deg)
    pitch = np.array(
      [[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]]
    )
    return pitch @ level

  @property
  def focal_px(self) -> float:
    """Square-pixel focal length from the vertical FOV (MuJoCo's convention)."""
    return 0.5 * self.height / math.tan(math.radians(self.fovy_deg) / 2.0)

  def unit_rays_cam(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unit ray directions in the MuJoCo camera frame for pixel centres (u, v).

    u runs left->right, v top->bottom (image convention). Camera frame: +x
    image right, +y image up, looking along -z.
    """
    f = self.focal_px
    x = (np.asarray(u, dtype=np.float64) + 0.5 - self.width / 2.0) / f
    y = -(np.asarray(v, dtype=np.float64) + 0.5 - self.height / 2.0) / f
    d = np.stack([x, y, -np.ones_like(x)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)

  def deproject_body(self, u, v, ray_depth) -> np.ndarray:
    """Pixel(s) + mujoco_warp ray-distance depth -> points in the base_link frame."""
    p_cam = self.unit_rays_cam(u, v) * np.asarray(ray_depth, dtype=np.float64)[..., None]
    return p_cam @ self.rotation_body_from_cam.T + np.asarray(self.pos)


  def project_body(self, points_body: np.ndarray) -> np.ndarray:
    """Base_link-frame points -> pixel (u, v). Exact inverse of `deproject_body`'s
    ray model. Points at or behind the image plane come back as NaN."""
    p = np.atleast_2d(np.asarray(points_body, dtype=np.float64)) - np.asarray(self.pos)
    p_cam = p @ self.rotation_body_from_cam
    z = p_cam[:, 2]
    out = np.full((p_cam.shape[0], 2), np.nan)
    front = z < -1e-6  # camera looks along -z
    f = self.focal_px
    out[front, 0] = f * p_cam[front, 0] / (-z[front]) + self.width / 2.0 - 0.5
    out[front, 1] = -f * p_cam[front, 1] / (-z[front]) + self.height / 2.0 - 0.5
    return out


def world_to_body(points_w: np.ndarray, base_pos_w: np.ndarray, base_quat_w: np.ndarray) -> np.ndarray:
  """Inverse of `body_to_world`."""
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, np.asarray(base_quat_w, dtype=np.float64))
  return (np.atleast_2d(np.asarray(points_w)) - np.asarray(base_pos_w)) @ rot.reshape(3, 3)


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mat, dtype=np.float64).reshape(9))
  return quat


def body_to_world(points_body: np.ndarray, base_pos_w: np.ndarray, base_quat_w: np.ndarray) -> np.ndarray:
  """Transform base_link-frame points to world using the base pose (quat w,x,y,z)."""
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, np.asarray(base_quat_w, dtype=np.float64))
  return np.asarray(points_body) @ rot.reshape(3, 3).T + np.asarray(base_pos_w)
