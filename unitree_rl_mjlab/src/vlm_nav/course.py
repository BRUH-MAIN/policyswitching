"""Navigation courses for SARO-style goal tracking across a terrain intermediation.

SARO's task (arXiv:2407.16412, Sec. III-A): start on platform P1, cross one
intermediation I, reach goal G on platform P2, given G's coordinates and a
language description L. Here the intermediations are the ones a trained
specialist exists for -- stairs (up or down) and rough ground -- and a course is
a straight strip of segments along +x, so every metre of it has a known ground
truth terrain class for scoring the policy choice.

Geometry is kept inside the specialists' training distribution on purpose
(`ROUGH_TERRAINS_CFG`): stairs use the 0.3 m tread and a riser within the
0-0.10 m training range, and rough segments call mjlab's own
`HfRandomUniformTerrainCfg` (noise 0.02-0.10 m, 0.02 m steps, 0.25 m flat
border). A course the specialists can't physically cross would make every
policy choice look equally bad and the VLM result uninformative.

Visual schemes exist because terrain colour is a shortcut: mjlab's default
height colouring paints stairs and heightfields differently, so a VLM could
read the class off the colour instead of the geometry. "plain" and "tiled"
give every segment identical materials; "class_colors" deliberately leaks the
class and is only for measuring how much that shortcut would inflate accuracy.

"tiled" exists because untextured geometry hides one intermediation almost
completely: a down-staircase seen from its top is uniform tread colour, since
the strip below each step edge that a rear light would shadow is exactly the
strip the step itself hides from the camera. mujoco_warp only texture-maps
planes and meshes (boxes and heightfields sample a single texel), so tiling is
built from thin grout-line geoms on a fixed 0.5 m world grid over every flat
surface and stair tread. Lines are class-agnostic (floors and treads alike;
natural rough ground gets none), non-colliding, and in the camera-only geom
group, so the specialists' height scan never sees them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from mjlab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

SegmentKind = Literal["flat", "rough", "stairs_up", "stairs_down"]
TerrainClass = Literal["flat", "rough", "stairs"]
VisualScheme = Literal["plain", "tiled", "class_colors"]

SEGMENT_CLASS: dict[str, TerrainClass] = {
  "flat": "flat",
  "rough": "rough",
  "stairs_up": "stairs",
  "stairs_down": "stairs",
}
STEP_WIDTH = 0.3  # ROUGH_TERRAINS_CFG pyramid stairs tread
# Go2 footprint along the body axis relative to the base origin: front feet
# ~0.19 m ahead of the hips at x=+0.19, hind feet ~0.19 behind x=-0.19, plus stride.
FOOTPRINT_FRONT = 0.30
FOOTPRINT_REAR = 0.35
GOAL_GEOM_GROUP = 4  # camera-only visuals (goal flag, grout lines): invisible to the height-scan rays
_SUPPORT_BOTTOM = -1.0

_PLAIN_RGBA = (0.42, 0.40, 0.37, 1.0)
_CLASS_RGBA = {"flat": (0.40, 0.40, 0.40, 1.0), "rough": (0.32, 0.40, 0.20, 1.0), "stairs": (0.20, 0.30, 0.52, 1.0)}


@dataclass(frozen=True)
class Segment:
  kind: SegmentKind
  length: float
  step_height: float = 0.07
  """Stair riser (m). Training range 0-0.10."""
  noise_range: tuple[float, float] = (0.02, 0.08)
  """Rough-ground height noise (m). Training range 0.02-0.10."""


@dataclass(frozen=True)
class Region:
  """A stretch of course with one ground-truth terrain class, in course x (m)."""

  x0: float
  x1: float
  terrain: TerrainClass
  z_start: float
  z_end: float


@dataclass(frozen=True)
class CourseSpec:
  name: str
  segments: tuple[Segment, ...]
  intermediation: str
  """What SARO's planning prompt calls the intermediation, e.g. "stairs"."""
  instruction: str
  """SARO's language description L of the goal."""
  width: float = 6.0
  start_x: float = 1.0
  goal_y_offset: float = 0.0
  """Lateral goal offset from the course centreline (SARO varies goal direction)."""
  visual: VisualScheme = "tiled"
  border_width: float = 12.0
  base_height: float = 0.0
  """Height of the first segment above the surrounding floor (a stairs-down
  course starts on a raised platform)."""

  @property
  def length(self) -> float:
    return float(sum(s.length for s in self.segments))

  def regions(self) -> list[Region]:
    out, x, z = [], 0.0, self.base_height
    for seg in self.segments:
      dz = _segment_rise(seg)
      out.append(Region(x, x + seg.length, SEGMENT_CLASS[seg.kind], z, z + dz))
      x += seg.length
      z += dz
    return out

  def required_terrain(self, x_base: float, front: float = FOOTPRINT_FRONT, rear: float = FOOTPRINT_REAR) -> TerrainClass:
    """Ground-truth policy choice for a base at course x: the terrain under the
    robot's footprint, not under one point. Non-flat terrain anywhere in
    [x - rear, x + front] wins (the front point's class first), so the stairs
    policy stays active until the rear feet are off the last step. A point rule
    switched back to the flat specialist with the hind legs still on the stairs,
    and it stalled there."""
    ahead = self.terrain_at(x_base + front)
    if ahead != "flat":
      return ahead
    for r in self.regions():
      if r.terrain != "flat" and r.x0 < x_base + front and r.x1 > x_base - rear:
        return r.terrain
    return "flat"

  def terrain_at(self, x_course: float) -> TerrainClass:
    for r in self.regions():
      if r.x0 <= x_course < r.x1:
        return r.terrain
    return "flat"  # the surrounding border floor

  @property
  def goal_course(self) -> np.ndarray:
    """Goal (x, y, z) in course coordinates: middle of the last segment."""
    last = self.regions()[-1]
    return np.array([(last.x0 + last.x1) / 2, self.width / 2 + self.goal_y_offset, last.z_end])

  @property
  def start_course(self) -> np.ndarray:
    return np.array([self.start_x, self.width / 2, self.base_height])

  def course_to_world(self, p_course: np.ndarray) -> np.ndarray:
    """The generator centres its single-patch grid on the world origin."""
    return np.asarray(p_course, dtype=np.float64) - np.array([self.length / 2, self.width / 2, 0.0])

  def world_to_course_x(self, x_world: float) -> float:
    return float(x_world + self.length / 2)

  def generator_cfg(self, seed: int = 0) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
      seed=seed,
      size=(self.length, self.width),
      border_width=self.border_width,
      num_rows=1,
      num_cols=1,
      curriculum=True,
      color_scheme="height",  # applies the per-geom colours set below
      sub_terrains={"course": CourseTerrainCfg(course=self)},
      difficulty_range=(1.0, 1.0),
      add_lights=False,
    )


def _segment_rise(seg: Segment) -> float:
  n = _num_steps(seg)
  if seg.kind == "stairs_up":
    return n * seg.step_height
  if seg.kind == "stairs_down":
    return -n * seg.step_height
  return 0.0


def _num_steps(seg: Segment) -> int:
  return int(round(seg.length / STEP_WIDTH)) if seg.kind.startswith("stairs") else 0


@dataclass(kw_only=True)
class CourseTerrainCfg(SubTerrainCfg):
  course: CourseSpec
  goal_marker: bool = True

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty
    c = self.course
    body = spec.body("terrain")
    geoms: list[TerrainGeometry] = []
    w = c.width

    def rgba_for(kind: str) -> tuple[float, float, float, float]:
      return _CLASS_RGBA[SEGMENT_CLASS[kind]] if c.visual == "class_colors" else _PLAIN_RGBA

    def box(x0: float, x1: float, top: float, kind: str) -> None:
      g = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=((x1 - x0) / 2, w / 2, (top - _SUPPORT_BOTTOM) / 2),
        pos=((x0 + x1) / 2, w / 2, (top + _SUPPORT_BOTTOM) / 2),
      )
      geoms.append(TerrainGeometry(geom=g, color=rgba_for(kind)))
      if c.visual == "tiled":
        _add_grout_lines(body, geoms, x0, x1, top, w)

    x, z = 0.0, c.base_height
    for seg in c.segments:
      if seg.kind == "flat":
        box(x, x + seg.length, z, seg.kind)
      elif seg.kind in ("stairs_up", "stairs_down"):
        n = _num_steps(seg)
        sign = 1.0 if seg.kind == "stairs_up" else -1.0
        if z + sign * n * seg.step_height < -1e-9:
          raise ValueError(f"{c.name}: stairs_down below floor level at x={x:.2f}")
        tread = seg.length / n
        for k in range(n):
          box(x + k * tread, x + (k + 1) * tread, z + sign * (k + 1) * seg.step_height, seg.kind)
      elif seg.kind == "rough":
        hf_cfg = HfRandomUniformTerrainCfg(
          size=(seg.length, w), noise_range=seg.noise_range, noise_step=0.02, border_width=0.25
        )
        out = hf_cfg.function(1.0, spec, rng)
        for tg in out.geometries:
          tg.geom.pos = np.array(tg.geom.pos) + np.array([x, 0.0, z])
          tg.geom.material = ""
          geoms.append(TerrainGeometry(geom=tg.geom, hfield=tg.hfield, color=rgba_for(seg.kind)))
        # Solid support under the heightfield slab so nothing shows through below it.
        box(x, x + seg.length, z - 1e-3, seg.kind)
      else:
        raise ValueError(f"unknown segment kind {seg.kind!r}")
      x += seg.length
      z += _segment_rise(seg)

    # The course's own lights (twin_env drops the terrain entity's default
    # overhead sun). mujoco_warp shading ignores light intensity -- each light
    # adds base_colour * cos(incidence) -- which is why the base colour is dark
    # (_PLAIN_RGBA): brightness sums over lights and saturates to white
    # otherwise, erasing rough-ground relief. Two oblique suns from opposite
    # sides: the one travelling toward the robot (-x) leaves the risers of an
    # up-staircase in shade; the one from behind (+x) casts each step's shadow
    # onto the tread below, without which a down-staircase seen from its top
    # renders as uniform floor. The rear sun is low (~25 deg elevation) so that
    # shadow is ~0.15 m on a 0.07 m riser rather than a 4 cm line invisible at
    # range; it also rakes across rough ground.
    for direction in ((-0.3, 0.5, -0.8), (0.88, -0.2, -0.42)):
      body.add_light(
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=(c.length / 2, w / 2, 10.0),
        dir=direction,
        castshadow=True,
      )

    if self.goal_marker:
      # A flag like SARO Fig. 1's. Non-colliding, and in a geom group the
      # height-scan rays skip, so it can't alter the specialists' observations.
      # Returned in `geometries` so the generator offsets it into world frame.
      gx, gy, gz = c.goal_course
      for rgba, kw in (
        ((0.85, 0.85, 0.85, 1.0), dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.015, 0.35, 0.0), pos=(gx, gy, gz + 0.35))),
        ((0.9, 0.1, 0.1, 1.0), dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.005, 0.12, 0.08), pos=(gx, gy + 0.12, gz + 0.62))),
      ):
        g = body.add_geom(contype=0, conaffinity=0, group=GOAL_GEOM_GROUP, **kw)
        geoms.append(TerrainGeometry(geom=g, color=rgba))

    return TerrainOutput(origin=c.start_course.copy(), geometries=geoms)


_GROUT_SPACING = 0.5
_GROUT_HALF_WIDTH = 0.008
_GROUT_RGBA = (0.16, 0.15, 0.14, 1.0)


def _add_grout_lines(body, geoms: list, x0: float, x1: float, top: float, w: float) -> None:
  """Tile joints on one horizontal surface [x0, x1] x [0, w] at height `top`."""

  def line(cx: float, cy: float, hx: float, hy: float) -> None:
    g = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX, size=(hx, hy, 0.001), pos=(cx, cy, top + 0.001),
      contype=0, conaffinity=0, group=GOAL_GEOM_GROUP,
    )
    geoms.append(TerrainGeometry(geom=g, color=_GROUT_RGBA))

  for k in range(1, int(round(w / _GROUT_SPACING))):  # joints running along x
    line((x0 + x1) / 2, k * _GROUT_SPACING, (x1 - x0) / 2, _GROUT_HALF_WIDTH)
  gx = np.ceil(x0 / _GROUT_SPACING + 1e-6) * _GROUT_SPACING
  while gx < x1 - 1e-6:  # joints running along y, on the global grid
    line(gx, w / 2, _GROUT_HALF_WIDTH, w / 2)
    gx += _GROUT_SPACING


# SARO's language description L says where the goal is ("the goal on the wooden box
# in front of the wall"), not what lies in between. An instruction that names the
# intermediation ("...at the top of the stairs") lets the VLM plan from the text
# without looking -- the first perception pilot scored 8/8 planning that way -- so
# every course uses the same neutral goal description.
NEUTRAL_INSTRUCTION = "reach the red goal flag ahead of you"

# Course difficulty levels for Phase-1 calibration, all inside the specialists'
# training ranges (riser 0-0.10 m, rough noise 0.02-0.10 m).
DIFFICULTY_LEVELS: dict[str, dict] = {
  "L1": dict(step_height=0.05, noise_range=(0.02, 0.06)),
  "L2": dict(step_height=0.07, noise_range=(0.02, 0.08)),
  "L3": dict(step_height=0.09, noise_range=(0.02, 0.10)),
}


def saro_courses(
  visual: VisualScheme = "tiled", goal_y_offset: float = 0.0, level: str = "L2"
) -> dict[str, CourseSpec]:
  """The course set. Single-intermediation courses follow SARO's P1 -> I -> P2
  task definition; `multi` chains several so the right policy changes more than
  once per run (this project's extension -- SARO uses one intermediation).

  The intermediation starts 2 m ahead of the spawn point, inside the camera's
  view (ground visible from ~0.5 m to the horizon) from the first frame.
  """
  lv = DIFFICULTY_LEVELS[level]
  common = dict(visual=visual, goal_y_offset=goal_y_offset)
  stairs = dict(step_height=lv["step_height"])
  rough = dict(noise_range=lv["noise_range"])
  rise = 5 * lv["step_height"]  # 1.5 m of 0.3 m treads
  return {
    "flat": CourseSpec(
      name="flat", intermediation="none", segments=(Segment("flat", 9.0),),
      instruction=NEUTRAL_INSTRUCTION, **common),
    "stairs_up": CourseSpec(
      name="stairs_up", intermediation="stairs",
      segments=(Segment("flat", 3.0), Segment("stairs_up", 1.5, **stairs), Segment("flat", 3.5)),
      instruction=NEUTRAL_INSTRUCTION, **common),
    "stairs_down": CourseSpec(
      name="stairs_down", intermediation="stairs", base_height=rise,
      segments=(Segment("flat", 3.0), Segment("stairs_down", 1.5, **stairs), Segment("flat", 3.5)),
      instruction=NEUTRAL_INSTRUCTION, **common),
    "rough": CourseSpec(
      name="rough", intermediation="rough ground",
      segments=(Segment("flat", 3.0), Segment("rough", 3.0, **rough), Segment("flat", 3.0)),
      instruction=NEUTRAL_INSTRUCTION, **common),
    "multi": CourseSpec(
      name="multi", intermediation="rough ground and stairs",
      segments=(
        Segment("flat", 3.0), Segment("rough", 3.0, **rough), Segment("flat", 2.0),
        Segment("stairs_up", 1.5, **stairs), Segment("flat", 2.0),
        Segment("stairs_down", 1.5, **stairs), Segment("flat", 3.0),
      ),
      instruction=NEUTRAL_INSTRUCTION,
      **common),
  }
