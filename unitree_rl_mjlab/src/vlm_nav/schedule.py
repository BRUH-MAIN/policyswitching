"""When, along a course, each specialist is in control -- as blend weights.

A switch into a non-flat segment is described by two leads, in metres of base
travel before the segment's first edge:

  lead_start  where the incoming specialist starts to take over
  lead_end    where it has fully taken over

`lead_start == lead_end` is a hard switch; otherwise the weight ramps linearly
in between and the action is the weighted sum of the specialists' actions. A
negative lead means the switch happens only after the base is already that far
into the segment (a detector that needs to be on the terrain to recognise it).

The onboard height scan reaches 0.8 m ahead of the base, so a schedule with
`lead_start <= SCAN_HORIZON` could be driven by the robot's own sensing, and one
with a larger lead needs terrain knowledge from beyond it -- the followed person.
That is the only thing the leads encode; every schedule reads the same
ground-truth segment layout.

Exits are not part of the comparison: a specialist releases when the rear of the
footprint clears its segment (`course.FOOTPRINT_REAR`), later by the same lag for
a negative lead, since a detector that is late noticing a terrain is late noticing
it has ended.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.vlm_nav.course import FOOTPRINT_REAR, CourseSpec

SCAN_HORIZON = 0.8
"""Metres ahead of the base covered by `terrain_scan` (a 1.6 m x 1.0 m grid centred on it)."""

MAPPINGS: dict[str, dict[str, str]] = {
  # Segment kind -> specialist. `label` is the specialist named after the terrain.
  "label": {"flat": "flat", "rough": "rough", "stairs_up": "stairs", "stairs_down": "stairs"},
  # `comp` sends up-stairs to the rough specialist, which crossed 0.05 m up-stairs
  # more reliably than the stairs specialist (97% vs 81%, exploratory, 2026-09-17).
  "comp": {"flat": "flat", "rough": "rough", "stairs_up": "rough", "stairs_down": "stairs"},
}


@dataclass(frozen=True)
class Schedule:
  lead_start: float
  lead_end: float

  def __post_init__(self) -> None:
    if self.lead_start < self.lead_end:
      raise ValueError(f"lead_start ({self.lead_start}) must be >= lead_end ({self.lead_end})")

  @property
  def hard(self) -> bool:
    return self.lead_start == self.lead_end

  @property
  def reactive_feasible(self) -> bool:
    return self.lead_start <= SCAN_HORIZON


def segment_spans(course: CourseSpec) -> list[tuple[float, float, str]]:
  """(x0, x1, segment kind) in course x, in course order."""
  out, x = [], 0.0
  for seg in course.segments:
    out.append((x, x + seg.length, seg.kind))
    x += seg.length
  return out


def class_boundaries(course: CourseSpec) -> list[tuple[float, str]]:
  """(x, "entry" | "exit") for every crossing between terrain classes.

  "entry" is a crossing onto non-flat ground, "exit" a crossing onto flat. A
  crossing between two different non-flat classes counts as an entry.
  """
  regions = course.regions()
  out = []
  for a, b in zip(regions, regions[1:]):
    if a.terrain != b.terrain:
      out.append((b.x0, "exit" if b.terrain == "flat" else "entry"))
  return out


def policy_weights(
  x_course: torch.Tensor,
  course: CourseSpec,
  names: list[str],
  schedule: Schedule,
  mapping: dict[str, str],
  rear: float = FOOTPRINT_REAR,
) -> torch.Tensor:
  """(N, K) blend weights over `names` for bases at course x (N,). Rows sum to 1.

  Segments are applied in course order and each takes its share from whatever
  was in control before it, so where two non-flat segments meet the blend runs
  between their two specialists, not through the flat one.
  """
  w = torch.zeros(x_course.shape[0], len(names), dtype=torch.float32, device=x_course.device)
  w[:, names.index(mapping["flat"])] = 1.0
  lag = max(0.0, -schedule.lead_end)
  for x0, x1, kind in segment_spans(course):
    if kind == "flat":
      continue
    if schedule.hard:
      share = (x_course >= x0 - schedule.lead_start).float()
    else:
      share = ((x_course - (x0 - schedule.lead_start)) / (schedule.lead_start - schedule.lead_end)).clamp(0.0, 1.0)
    share = torch.where(x_course < x1 + rear + lag, share, torch.zeros_like(share))
    w = w * (1.0 - share).unsqueeze(-1)
    w[:, names.index(mapping[kind])] += share
  return w
