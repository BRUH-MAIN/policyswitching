"""Blend-weight schedules (src/vlm_nav/schedule.py). No simulator needed."""

import pytest
import torch

from src.vlm_nav.course import CourseSpec, Segment, saro_courses
from src.vlm_nav.schedule import MAPPINGS, SCAN_HORIZON, Schedule, class_boundaries, policy_weights

NAMES = ["flat", "rough", "stairs"]


def _weights(course, xs, schedule, mapping="label"):
  return policy_weights(torch.tensor(xs, dtype=torch.float32), course, NAMES, schedule, MAPPINGS[mapping])


def test_hard_footprint_lead_reproduces_the_existing_oracle():
  # hard:0.3 with the label mapping must be the rule every earlier oracle arm used,
  # or the new runs are not comparable with the Phase-1 numbers.
  course = saro_courses(level="L1")["multi"]
  xs = [0.005 + 0.01 * k for k in range(int(course.length * 100))]
  w = _weights(course, xs, Schedule(0.3, 0.3))
  assert torch.all((w == 0) | (w == 1))
  got = [NAMES[i] for i in w.argmax(dim=1).tolist()]
  assert got == [course.required_terrain(x) for x in xs]


def test_rows_sum_to_one_for_every_schedule():
  course = saro_courses(level="L1")["multi"]
  xs = [0.01 * k for k in range(int(course.length * 100))]
  for sched in (Schedule(-0.6, -0.6), Schedule(0.8, 0.0), Schedule(2.5, 0.3), Schedule(2.5, 2.5)):
    for mapping in MAPPINGS:
      w = _weights(course, xs, sched, mapping)
      assert torch.allclose(w.sum(dim=1), torch.ones(len(xs)), atol=1e-6)
      assert (w >= 0).all()


def test_soft_ramp_is_linear_between_the_two_leads():
  course = saro_courses(level="L1")["rough"]  # rough segment starts at x = 3.0
  w = _weights(course, [3.0 - 0.8, 3.0 - 0.55, 3.0 - 0.3, 3.5], Schedule(0.8, 0.3))
  rough = w[:, NAMES.index("rough")]
  assert rough.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.0], abs=1e-6)
  assert w[:, NAMES.index("stairs")].abs().max() == 0


def test_negative_lead_switches_late_and_releases_late():
  course = saro_courses(level="L1")["rough"]  # rough on [3.0, 6.0)
  w = _weights(course, [3.5, 3.7, 6.0 + 0.35 + 0.5, 6.0 + 0.35 + 0.7], Schedule(-0.6, -0.6))
  assert w[:, NAMES.index("rough")].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_adjacent_non_flat_segments_blend_between_their_specialists():
  course = CourseSpec(
    name="adjacent", intermediation="x", instruction="x", base_height=0.25,
    segments=(Segment("flat", 3.0), Segment("rough", 3.0), Segment("stairs_down", 1.5, step_height=0.05),
              Segment("flat", 3.0)),
  )
  w = _weights(course, [6.0 - 0.4], Schedule(0.8, 0.0))[0]
  assert w[NAMES.index("flat")] == 0
  assert w[NAMES.index("rough")] == pytest.approx(0.5)
  assert w[NAMES.index("stairs")] == pytest.approx(0.5)


def test_comp_mapping_sends_up_stairs_to_the_rough_specialist():
  course = saro_courses(level="L1")["multi"]  # stairs_up on [8.0, 9.5), stairs_down on [11.5, 13.0)
  w = _weights(course, [8.5, 12.0], Schedule(0.3, 0.3), mapping="comp")
  assert [NAMES[i] for i in w.argmax(dim=1).tolist()] == ["rough", "stairs"]


def test_reactive_feasibility_is_decided_by_where_the_switch_starts():
  assert Schedule(SCAN_HORIZON, 0.0).reactive_feasible
  assert not Schedule(1.5, 0.3).reactive_feasible
  with pytest.raises(ValueError):
    Schedule(0.3, 0.8)


def test_class_boundaries_label_entries_and_exits():
  course = saro_courses(level="L1")["multi"]
  assert class_boundaries(course) == [
    (3.0, "entry"), (6.0, "exit"), (8.0, "entry"), (9.5, "exit"), (11.5, "entry"), (13.0, "exit"),
  ]
