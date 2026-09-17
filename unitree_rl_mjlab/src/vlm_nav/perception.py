"""Bounding box + depth -> where the intermediation is, in the world frame.

SARO (Appendix A, Fig. 8, Algorithm 1) deprojects the box centre through the
aligned depth image to get the intermediation's 3D position. Switching
specialists needs more than a centre: *when* to switch depends on where the
intermediation starts and ends along the robot's path. So every valid-depth
pixel inside the box is deprojected, and the near / far edge are taken as low /
high percentiles of distance along the robot's heading.

The percentiles are chosen so that a loose box errs in the safe direction: a
box that spills onto the floor in front makes the near edge too close (the
specialist is engaged early -- every specialist walks flat ground without
falling, gate 1), and a box that spills onto the platform beyond makes the far
edge too far (the flat policy is re-engaged late). The unsafe errors -- engaging
late, or dropping back to the flat policy with feet still on the steps -- need
the box to miss the intermediation, which `valid` flags when too few box pixels
have usable depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from src.vlm_nav.camera import CameraSpec


@dataclass
class IntermediationEstimate:
  valid: bool
  center_w: np.ndarray | None = None
  """Median 3D point of the box's visible surface (SARO's box-centre target)."""
  near_along: float | None = None
  far_along: float | None = None
  """Distances along the heading at observation time, from the base."""
  lateral: float | None = None
  """Median lateral offset (base frame, +left) of the box's surface points."""
  origin_w: np.ndarray | None = None
  heading_w: np.ndarray | None = None
  """Base xy and unit heading when observed, so the edges can be re-expressed as the robot moves."""
  center_u: float | None = None
  n_points: int = 0

  def along_from(self, base_xy_w: np.ndarray, edge: float) -> float:
    """Remaining distance from a (moved) base to an edge, measured along the observation heading."""
    return float(edge - np.dot(np.asarray(base_xy_w) - self.origin_w, self.heading_w))


def estimate_from_box(
  box_px: list[float],
  depth: np.ndarray,
  camera: CameraSpec,
  base_pos_w: np.ndarray,
  base_quat_w: np.ndarray,
  grid: int = 24,
  near_pct: float = 10.0,
  far_pct: float = 90.0,
  max_range: float = 8.0,
  min_points: int = 12,
) -> IntermediationEstimate:
  h, w = depth.shape
  x0, y0, x1, y1 = (int(round(v)) for v in box_px)
  x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w - 1))
  y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h - 1))
  if x1 - x0 < 2 or y1 - y0 < 2:
    return IntermediationEstimate(valid=False)
  us, vs = np.meshgrid(np.linspace(x0, x1, grid), np.linspace(y0, y1, grid))
  ui, vi = us.round().astype(int).ravel(), vs.round().astype(int).ravel()
  d = depth[vi, ui]
  ok = np.isfinite(d) & (d > 0.05) & (d < max_range)
  if ok.sum() < min_points:
    return IntermediationEstimate(valid=False, n_points=int(ok.sum()))
  p_body = camera.deproject_body(ui[ok], vi[ok], d[ok])
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, np.asarray(base_quat_w, dtype=np.float64))
  rot = rot.reshape(3, 3)
  p_world = p_body @ rot.T + np.asarray(base_pos_w)
  heading = rot[:2, 0] / max(1e-9, np.linalg.norm(rot[:2, 0]))
  origin = np.asarray(base_pos_w)[:2]
  along = (p_world[:, :2] - origin) @ heading
  lateral = (p_world[:, :2] - origin) @ np.array([-heading[1], heading[0]])
  return IntermediationEstimate(
    valid=True,
    center_w=np.median(p_world, axis=0),
    near_along=float(np.percentile(along, near_pct)),
    far_along=float(np.percentile(along, far_pct)),
    lateral=float(np.median(lateral)),
    origin_w=origin.copy(),
    heading_w=heading,
    center_u=(x0 + x1) / 2.0,
    n_points=int(ok.sum()),
  )


def ground_blind_distance(camera: CameraSpec, base_height: float = 0.33) -> float:
  """Distance ahead of the base where the camera's lowest image row meets level ground.

  Nothing nearer is visible, so a near-edge estimate cannot come closer than this:
  approached from 2 m, a staircase's near edge reads ~0.8 m and then stops moving.
  """
  import math

  h = base_height + camera.pos[2]
  lowest = math.radians(camera.pitch_down_deg + camera.fovy_deg / 2)
  return float(camera.pos[0] + h / math.tan(lowest))


class EdgeTracker:
  """World-frame near/far edge of one intermediation, fused over observations.

  The intermediation doesn't move, so every box observation is converted to world
  points along a reference heading (the first observation's) and fused:
  - near edge: median of observations taken while the edge was beyond the camera's
    blind distance (+ margin); once the robot is closer, the stored value is used
    rather than the saturated reading;
  - far edge: median of the most recent `far_window` observations (far pixels are
    at grazing angles and jitter by ~1 m tick to tick).
  """

  def __init__(self, blind_distance: float, margin: float = 0.15, far_window: int = 5):
    self.blind = blind_distance
    self.margin = margin
    self.far_window = far_window
    self.heading: np.ndarray | None = None
    self.near_w: list[np.ndarray] = []
    self.far_w: list[np.ndarray] = []
    self.lateral_w: list[np.ndarray] = []
    self.center_u: float | None = None

  @property
  def seen(self) -> bool:
    return bool(self.far_w)

  def update(self, est: IntermediationEstimate) -> None:
    if not est.valid:
      return
    if self.heading is None:
      self.heading = est.heading_w
    self.center_u = est.center_u
    if est.near_along > self.blind + self.margin:
      self.near_w.append(est.origin_w + est.heading_w * est.near_along)
    self.far_w.append(est.origin_w + est.heading_w * est.far_along)
    left = np.array([-est.heading_w[1], est.heading_w[0]])
    self.lateral_w.append(est.origin_w + left * est.lateral)

  def _along(self, points: list[np.ndarray], base_xy: np.ndarray) -> float | None:
    if not points or self.heading is None:
      return None
    return float(np.median([np.dot(p - base_xy, self.heading) for p in points]))

  def near_remaining(self, base_xy: np.ndarray) -> float | None:
    """Distance to the near edge; None if never seen beyond the blind zone (it is closer than that)."""
    return self._along(self.near_w, base_xy)

  def far_remaining(self, base_xy: np.ndarray) -> float | None:
    return self._along(self.far_w[-self.far_window :], base_xy)

  def lateral_offset(self, base_xy: np.ndarray) -> float:
    if not self.lateral_w or self.heading is None:
      return 0.0
    left = np.array([-self.heading[1], self.heading[0]])
    return float(np.median([np.dot(p - base_xy, left) for p in self.lateral_w[-self.far_window :]]))
