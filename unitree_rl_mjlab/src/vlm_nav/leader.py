"""Scripted kinematic leader (the followed person) for person-following runs.

`objective.md`'s scope decision: "a scripted kinematic leader body with a known
trajectory, not a simulated perception pipeline", so the leader's pose is ground
truth during sim ablations and real person-tracking is deferred to hardware.

The leader is a fixed-base mjlab `Entity`, which mjlab auto-wraps into a mocap
body (`mjlab.utils.spec.auto_wrap_fixed_base_mocap`), so its pose is written
each control step with `EntityData.write_mocap_pose` -- batched and on-device,
no per-step model surgery.

Two properties of its geoms matter and are not stylistic:

- `contype=0, conaffinity=0`: the leader is kinematic scenery. A colliding body
  dragged through the world at a scripted pose would inject contact forces the
  specialists never saw in training, and `illegal_contact` would fire on a
  robot that merely caught up.
- `group=GOAL_GEOM_GROUP`: the height-scan rays read groups 0-2 only (see
  `twin_env`'s `enabled_geom_groups`), so a person walking a few metres ahead
  cannot perturb the 187-dim `height_scan` the specialists consume. The camera
  *does* render group 4, so the leader is visible to the VLM but invisible to
  the policy -- the same trick the goal flag already uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg

from src.vlm_nav.course import GOAL_GEOM_GROUP

LEADER_ENTITY = "leader"

_TORSO_RGBA = (0.90, 0.45, 0.10, 1.0)
_HEAD_RGBA = (0.95, 0.80, 0.65, 1.0)
_LEG_RGBA = (0.20, 0.25, 0.45, 1.0)


def leader_spec() -> mujoco.MjSpec:
  """A person-sized figure: legs, torso, head. No joints -> fixed base -> mocap."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="leader")
  common = dict(contype=0, conaffinity=0, group=GOAL_GEOM_GROUP)
  for dx in (-0.09, 0.09):
    g = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=(0.055, 0.34, 0.0),
      pos=(0.0, dx, 0.40), quat=(1.0, 0.0, 0.0, 0.0), **common,
    )
    g.rgba = _LEG_RGBA
  torso = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=(0.135, 0.26, 0.0), pos=(0.0, 0.0, 1.06), **common
  )
  torso.rgba = _TORSO_RGBA
  head = body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.105, 0.0, 0.0), pos=(0.0, 0.0, 1.52), **common)
  head.rgba = _HEAD_RGBA
  return spec


def leader_entity_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=leader_spec)


@dataclass(frozen=True)
class SpeedChange:
  """Leader speed (m/s) taking effect at `t` seconds into the run."""

  t: float
  speed: float


# Occasional speed changes, including a full stop: the interesting cases for a
# standoff controller are the robot having to close a gap that opened (leader
# accelerates), and having to stop or back off without overshooting (leader
# halts). A monotone walk exercises neither.
# Top speed stays under FollowGains.v_max (1.0, the specialists' command limit) so
# the follower retains authority to close a gap rather than merely not lose more.
DEFAULT_SCHEDULE: tuple[SpeedChange, ...] = (
  SpeedChange(0.0, 0.70),
  SpeedChange(5.0, 0.30),
  SpeedChange(10.0, 0.00),
  SpeedChange(14.0, 0.85),
  SpeedChange(20.0, 0.45),
  SpeedChange(26.0, 0.80),
  SpeedChange(33.0, 0.25),
)


@dataclass
class LeaderPath:
  """Piecewise-constant-speed walk down the course centreline, with a gentle weave.

  Positions are in *course* frame (x along the strip, y across it); the caller
  converts to world with `CourseSpec.course_to_world`.
  """

  start_x: float
  centre_y: float
  schedule: tuple[SpeedChange, ...] = field(default_factory=lambda: DEFAULT_SCHEDULE)
  weave_amplitude: float = 0.55
  weave_period_s: float = 13.0
  z: float = 0.0

  def speed_at(self, t: float) -> float:
    s = 0.0
    for ch in self.schedule:
      if t >= ch.t:
        s = ch.speed
      else:
        break
    return s

  def distance_at(self, t: float) -> float:
    """Integral of the piecewise-constant speed schedule up to `t`."""
    d, prev_t, prev_v = 0.0, 0.0, 0.0
    for ch in self.schedule:
      if ch.t >= t:
        break
      d += prev_v * (ch.t - prev_t)
      prev_t, prev_v = ch.t, ch.speed
    return d + prev_v * max(0.0, t - prev_t)

  def pos_at(self, t: float) -> np.ndarray:
    """Course-frame (x, y, z) of the leader at time `t`."""
    y = self.centre_y + self.weave_amplitude * np.sin(2.0 * np.pi * t / self.weave_period_s)
    return np.array([self.start_x + self.distance_at(t), y, self.z], dtype=np.float64)

  def yaw_at(self, t: float, eps: float = 0.05) -> float:
    """Heading, from a finite difference of the path (so it faces where it walks)."""
    a, b = self.pos_at(max(0.0, t - eps)), self.pos_at(t + eps)
    d = b - a
    return float(np.arctan2(d[1], d[0])) if np.hypot(d[0], d[1]) > 1e-9 else 0.0


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
  return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def write_leader_pose(env, pos_w: np.ndarray, yaw: float) -> None:
  """Place the leader at a world-frame position, facing `yaw`, in every env."""
  leader = env.unwrapped.scene[LEADER_ENTITY]
  n = env.unwrapped.num_envs
  qw, qx, qy, qz = _yaw_quat(yaw)
  pose = torch.tensor(
    [[pos_w[0], pos_w[1], pos_w[2], qw, qx, qy, qz]], dtype=torch.float32, device=env.unwrapped.device
  ).repeat(n, 1)
  leader.data.write_mocap_pose(pose)
