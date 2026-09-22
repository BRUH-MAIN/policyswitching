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

from src.vlm_nav.camera import CameraSpec, body_to_world


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


def depth_edge_estimate(
  depth: np.ndarray,
  camera: CameraSpec,
  base_pos_w: np.ndarray,
  base_quat_w: np.ndarray,
  band: tuple[float, float] = (0.3, 0.7),
  col_stride: int = 4,
  max_along: float = 5.0,
  step_thresh: float = 0.025,
  spread_thresh: float = 0.025,
  window_m: float = 0.4,
  flat_run_m: float = 0.5,
  cam_height: float = 0.36,
) -> IntermediationEstimate:
  """Where the ground ahead stops being level floor, from depth alone (class-agnostic).

  Fallback for when the VLM's box is unusable (Gemma-4-E4B mostly answers
  [0,0,0,0] or the whole frame; Phase-2 probe). The VLM still decides *what* the
  terrain is and which specialist to run; this only decides *where*.

  Works per image row of a central column band -- rows are already ordered by
  distance, and each row spans the band laterally. A row is "structured" if
  - its lateral height spread is large (rough ground, or a step edge crossing it),
  - its median height departs from the trailing window of level rows (a riser,
    or the drop past a platform edge), or
  - its distance jumps by more than the expected row spacing at that range (an
    occlusion: the hidden strip beyond a down-step's edge).
  Near edge = the first structured row; far edge = the last structured row before
  `flat_run_m` of level rows.

  (A first version binned points by distance in 5 cm bins. Beyond ~2 m adjacent
  pixel rows are further apart than a bin, empty bins read as occlusion gaps, and
  every flat frame "detected" an intermediation.)
  """
  h, w = depth.shape
  cols = np.arange(int(band[0] * w), int(band[1] * w), col_stride)
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, np.asarray(base_quat_w, dtype=np.float64))
  rot = rot.reshape(3, 3)
  heading = rot[:2, 0] / max(1e-9, np.linalg.norm(rot[:2, 0]))
  origin = np.asarray(base_pos_w)[:2]
  lateral_axis = np.array([-heading[1], heading[0]])

  rows = []  # (along, z_median, z_spread, lateral_median), nearest first
  for v in range(h - 1, -1, -1):
    d = depth[v, cols]
    ok = np.isfinite(d) & (d > 0.05) & (d < max_along + 3.0)
    if ok.sum() < max(5, 0.5 * len(cols)):
      continue
    p = camera.deproject_body(cols[ok], np.full(ok.sum(), v), d[ok]) @ rot.T + np.asarray(base_pos_w)
    along = (p[:, :2] - origin) @ heading
    a = float(np.median(along))
    if a > max_along:
      break
    rows.append((a, float(np.median(p[:, 2])), float(np.percentile(p[:, 2], 90) - np.percentile(p[:, 2], 10)),
                 float(np.median((p[:, :2] - origin) @ lateral_axis))))
  if len(rows) < 10:
    return IntermediationEstimate(valid=False, n_points=len(rows))

  f = camera.focal_px
  structured = []
  level_window: list[tuple[float, float]] = []
  prev_a = None
  for a, z, spread, _ in rows:
    expected_gap = a * a / (f * cam_height)
    occluded = prev_a is not None and a - prev_a > max(0.15, 4.0 * expected_gap)
    ref = [zz for aa, zz in level_window if a - aa <= window_m]
    stepped = bool(ref) and abs(z - float(np.median(ref))) > step_thresh
    s_flag = spread > spread_thresh or stepped or occluded
    structured.append(s_flag)
    if not s_flag:
      level_window.append((a, z))
    else:
      level_window = [(a, z)]  # a new level starts after structure (e.g. the next tread)
    prev_a = a

  hits = [k for k, fl in enumerate(structured) if fl]
  if not hits:
    return IntermediationEstimate(valid=False, n_points=len(rows))
  k0 = hits[0]
  near = rows[k0 - 1][0] if k0 > 0 and rows[k0][0] - rows[k0 - 1][0] > 0.15 else rows[k0][0]
  far = rows[k0][0]
  run_start = None
  for k in range(k0, len(rows)):
    if structured[k]:
      far, run_start = rows[k][0], None
    else:
      run_start = rows[k][0] if run_start is None else run_start
      if rows[k][0] - run_start >= flat_run_m:
        break
  return IntermediationEstimate(
    valid=True,
    near_along=float(near),
    far_along=float(far),
    lateral=float(np.median([r[3] for r in rows])),
    origin_w=origin.copy(),
    heading_w=heading,
    center_u=w / 2.0,
    n_points=len(rows),
  )


def target_from_bearing_column(
  depth: np.ndarray,
  camera: CameraSpec,
  col_px: float,
  base_pos_w: np.ndarray,
  base_quat_w: np.ndarray,
  min_height_m: float = 0.35,
  half_width_px: int = 6,
) -> np.ndarray | None:
  """World-frame (x, y) of the target at image column `col_px`, from depth alone.

  Works for whatever pointed at that column -- the VLM's box centre, or a
  detector's. The column is the only pixel information used.

  The VLM supplies only the *column* (`col_px`): Phase-2 measurement on the
  follow course found its box centre horizontally exact (0.08 deg bearing error
  at 6 m) while its vertical extent was wrong -- it boxed ground below the
  figure. So the box's y range is discarded and the range is recovered from
  geometry: deproject a narrow strip of that column and keep returns standing
  more than `min_height_m` above the local ground, which on flat terrain is the
  person and nothing else. Their median distance is the range.

  Returns None when the strip holds no such returns (person out of view, or the
  VLM pointed at empty floor).
  """
  h, w = depth.shape[:2]
  c = int(round(col_px))
  lo, hi = max(0, c - half_width_px), min(w, c + half_width_px + 1)
  us, vs = np.meshgrid(np.arange(lo, hi), np.arange(h), indexing="xy")
  us = us.reshape(-1).astype(np.float64)
  vs = vs.reshape(-1).astype(np.float64)
  d = depth[vs.astype(int), us.astype(int)].reshape(-1).astype(np.float64)
  ok = np.isfinite(d) & (d > 1e-3)
  if not ok.any():
    return None
  pts_body = camera.deproject_body(us[ok], vs[ok], d[ok])
  pts_w = body_to_world(pts_body, base_pos_w, base_quat_w)
  # Ground is flat here, so "standing above the floor" is height over the base's
  # own ground plane (base_pos_w[2] is ~0.33 m above it).
  ground_z = base_pos_w[2] - 0.33
  tall = pts_w[:, 2] > ground_z + min_height_m
  if not tall.any():
    return None
  return np.median(pts_w[tall][:, :2], axis=0)


# Back-compatible alias: the follow runner's VLM-only path predates the detector seam.
person_from_bearing_column = target_from_bearing_column
