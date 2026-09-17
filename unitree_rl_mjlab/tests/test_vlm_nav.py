"""Unit tests for the VLM navigation pipeline's pure logic (no simulator, no VLM server).

Run from unitree_rl_mjlab/:  PYTHONPATH=$PWD python -m pytest tests/test_vlm_nav.py -q
"""

import math

import numpy as np
import pytest

from src.vlm_nav import prompts as P
from src.vlm_nav.camera import CameraSpec, body_to_world
from src.vlm_nav.course import FOOTPRINT_FRONT, FOOTPRINT_REAR, saro_courses
from src.vlm_nav.perception import estimate_from_box
from src.vlm_nav.vlm_backend import parse_json


def _project_body(cam: CameraSpec, p_body: np.ndarray):
  p_cam = (p_body - np.asarray(cam.pos)) @ cam.rotation_body_from_cam
  zc = -p_cam[:, 2]
  u = cam.focal_px * p_cam[:, 0] / zc + cam.width / 2 - 0.5
  v = -cam.focal_px * p_cam[:, 1] / zc + cam.height / 2 - 0.5
  return u, v, np.linalg.norm(p_cam, axis=1)


def test_camera_quat_looks_forward_and_down():
  cam = CameraSpec(pitch_down_deg=15.0)
  view_dir_body = cam.rotation_body_from_cam @ np.array([0.0, 0.0, -1.0])
  assert view_dir_body[0] > 0.9 and view_dir_body[2] == pytest.approx(-math.sin(math.radians(15)), abs=1e-9)
  image_right_body = cam.rotation_body_from_cam @ np.array([1.0, 0.0, 0.0])
  assert image_right_body[1] == pytest.approx(-1.0)  # image right = body -y


def test_deprojection_round_trip_uses_ray_distance():
  cam = CameraSpec()
  rng = np.random.default_rng(0)
  pts = np.stack([rng.uniform(1, 5, 50), rng.uniform(-1, 1, 50), rng.uniform(-0.4, 0.2, 50)], axis=1)
  u, v, ray = _project_body(cam, pts)
  back = cam.deproject_body(u, v, ray)
  np.testing.assert_allclose(back, pts, atol=1e-6)


def test_body_to_world_yaw():
  q = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])  # +90 deg yaw
  out = body_to_world(np.array([[1.0, 0.0, 0.0]]), np.array([2.0, 3.0, 0.5]), q)
  np.testing.assert_allclose(out[0], [2.0, 4.0, 0.5], atol=1e-9)


@pytest.mark.parametrize("conv,raw,expect", [
  ("xyxy_px", [10, 20, 110, 220], [10, 20, 110, 220]),
  ("xyxy_1000", [100, 250, 200, 500], [84.8, 120, 169.6, 240]),
  ("yxyx_1000", [250, 100, 500, 200], [84.8, 120, 169.6, 240]),
  ("xyxy_unit", [0.1, 0.25, 0.2, 0.5], [84.8, 120, 169.6, 240]),
])
def test_box_conventions(conv, raw, expect):
  np.testing.assert_allclose(P.box_to_pixels(raw, conv, 848, 480), expect, atol=1e-6)


def test_iou():
  assert P.iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
  assert P.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
  assert P.iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_reply_parsers():
  assert parse_json('```json\n{"answer": "yes"}\n```') == {"answer": "yes"}
  assert parse_json('Sure: {"policy": "stairs"}') == {"policy": "stairs"}
  assert parse_json("no json here") is None
  assert P.parse_yes_no("Yes.") is True and P.parse_yes_no("no") is False and P.parse_yes_no("maybe") is None
  assert P.parse_policy("stairs") == "stairs"
  assert P.parse_policy("flat or rough") is None  # ambiguous -> unparsed, never a guess
  assert P.parse_box("[12, 40.5, 300, 470]") == [12.0, 40.5, 300.0, 470.0]
  assert P.parse_box("[1, 2]") is None


def test_required_terrain_holds_specialist_until_hind_feet_clear():
  c = saro_courses(level="L1")["stairs_up"]  # stairs span x in [3.0, 4.5)
  assert c.required_terrain(3.0 - FOOTPRINT_FRONT - 0.01) == "flat"
  assert c.required_terrain(3.0 - FOOTPRINT_FRONT + 0.01) == "stairs"
  assert c.required_terrain(4.5 + FOOTPRINT_REAR - 0.01) == "stairs"
  assert c.required_terrain(4.5 + FOOTPRINT_REAR + 0.01) == "flat"
  m = saro_courses(level="L1")["multi"]
  assert m.required_terrain(2.9) == "rough"


def test_course_regions_and_goal_consistent():
  for c in saro_courses(level="L2").values():
    regions = c.regions()
    assert regions[0].x0 == 0 and regions[-1].x1 == pytest.approx(c.length)
    assert all(a.x1 == pytest.approx(b.x0) and a.z_end == pytest.approx(b.z_start) for a, b in zip(regions, regions[1:]))
    assert c.goal_course[2] == pytest.approx(regions[-1].z_end)


def test_estimate_from_box_near_far_edges_on_synthetic_ground_plane():
  """Render a flat floor analytically; a box around a floor stripe 2-3 m ahead must
  give near ~2 m and far ~3 m along the heading from the base."""
  cam = CameraSpec()
  base_pos = np.array([0.0, 0.0, 0.33])
  quat = np.array([1.0, 0, 0, 0])
  h, w = cam.height, cam.width
  vv, uu = np.mgrid[0:h, 0:w]
  rays_body = cam.unit_rays_cam(uu.ravel(), vv.ravel()) @ cam.rotation_body_from_cam.T
  origin_world_z = base_pos[2] + cam.pos[2]
  t = np.where(rays_body[:, 2] < -1e-6, -origin_world_z / np.minimum(rays_body[:, 2], -1e-6), np.inf)
  depth = t.reshape(h, w).astype(np.float32)
  hit_x = (cam.pos[0] + t * rays_body[:, 0]).reshape(h, w)
  rows = np.nonzero(((hit_x > 2.0) & (hit_x < 3.0)).any(axis=1))[0]
  box = [0, rows.min(), w - 1, rows.max()]
  est = estimate_from_box(box, depth, cam, base_pos, quat, near_pct=0, far_pct=100)
  assert est.valid
  assert est.near_along == pytest.approx(2.0, abs=0.08)
  assert est.far_along == pytest.approx(3.0, abs=0.08)
  assert est.along_from(np.array([1.0, 0.0]), est.near_along) == pytest.approx(est.near_along - 1.0)
