"""Velocity-command controllers for the sub-task executor.

SARO (Appendix A, Algorithms 1-3) drives the low-level policy toward a sub-goal
"by PD control based on the sub-goal" until within 0.1 m. The specialists only
ever saw commands inside their training range, so outputs are clipped well
inside it (|vx| <= 1, |vy| <= 1, |wz| <= 1 at the final curriculum stage).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
  return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
  """Yaw of (w, x, y, z) quaternions, shape (..., 4)."""
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass(frozen=True)
class GotoGains:
  v_max: float = 0.5
  vy_max: float = 0.3
  wz_max: float = 0.8
  k_dist: float = 1.0
  k_yaw: float = 1.2
  turn_in_place_rad: float = 1.0
  """Heading error beyond which forward motion stops (turn first)."""


def goto_command(
  base_pos_w: torch.Tensor, base_quat_w: torch.Tensor, target_xy_w: torch.Tensor, gains: GotoGains = GotoGains()
) -> tuple[torch.Tensor, torch.Tensor]:
  """(N, 3) base-frame (vx, vy, wz) toward target_xy_w (N, 2), and distance (N,).

  P control: the commanded planar velocity points at the target in the base
  frame (vx = v cos e, vy = v sin e, clipped) while wz turns the base to face
  it, so the path is a straight line even before the heading has converged.
  (Scaling vy by distance instead saturates it far from the target and walks
  the robot off the goal line -- observed on stairs, where the base yawed and
  drifted sideways across the steps.)
  """
  delta = target_xy_w - base_pos_w[:, :2]
  dist = torch.linalg.norm(delta, dim=-1)
  yaw = yaw_from_quat(base_quat_w)
  err = wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - yaw)
  speed = torch.clamp(gains.k_dist * dist, max=gains.v_max)
  gate = (err.abs() < gains.turn_in_place_rad).float()
  vx = speed * torch.clamp(torch.cos(err), min=0.0) * gate
  vy = torch.clamp(speed * torch.sin(err), -gains.vy_max, gains.vy_max) * gate
  wz = torch.clamp(gains.k_yaw * err, -gains.wz_max, gains.wz_max)
  return torch.stack([vx, vy, wz], dim=-1), dist


@dataclass(frozen=True)
class FollowGains:
  v_max: float = 1.0
  """Must exceed the leader's top speed or the robot cannot close a gap that opens:
  at v_max=0.9 against a 0.95 m/s leader the range error grew monotonically for the
  whole burst (+1.19 m peak, measured) and only recovered when the leader slowed.
  1.0 is the specialists' final-curriculum command limit, so this is the ceiling."""
  v_back_max: float = 0.35
  """Backing up is slower than closing: the specialists saw |vx| <= 1 in training,
  but reverse gaits are far less practised than forward ones."""
  vy_max: float = 0.3
  wz_max: float = 0.8
  k_range: float = 0.9
  k_yaw: float = 1.2
  deadband: float = 0.15
  """Range error (m) inside which the robot holds station instead of shuffling."""
  turn_in_place_rad: float = 1.0


def follow_command(
  base_pos_w: torch.Tensor,
  base_quat_w: torch.Tensor,
  leader_xy_w: torch.Tensor,
  gap: float,
  gains: FollowGains = FollowGains(),
  leader_vel_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """(N, 3) base-frame (vx, vy, wz) to follow `leader_xy_w` at `gap` metres, and range (N,).

  Unlike `goto_command`, the robot always turns to *face* the leader and regulates
  range with signed forward speed. Driving to a point `gap` short of the leader
  would work while closing, but once inside the gap that point sits behind the
  robot and the yaw controller spins it around to walk away -- losing sight of the
  leader, which is the one thing a follower must not do.

  A deadband around the set-point keeps the robot from shuffling when the leader
  stops: without it the range error dithers about zero and the command sign flips
  every few steps.

  `leader_vel_w` (N, 2) is the leader's world velocity, projected onto the line of
  sight and fed forward. Without it this is pure proportional control against a
  moving set-point, which droops: the loop settles where `k_range * err` equals the
  leader's speed, leaving a standing error of `v_leader / k_range` (~0.8 m at
  0.7 m/s with the default gain, measured). Every speed change then parks the robot
  at a different wrong distance, which is exactly what a follow task must not do.
  The caller estimates it by differencing observed leader positions, as a real
  tracker would, rather than reading the script.
  """
  delta = leader_xy_w - base_pos_w[:, :2]
  rng = torch.linalg.norm(delta, dim=-1)
  yaw = yaw_from_quat(base_quat_w)
  err = wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - yaw)

  range_err = rng - gap
  range_err = torch.where(range_err.abs() < gains.deadband, torch.zeros_like(range_err), range_err)
  v_ff = torch.zeros_like(rng)
  if leader_vel_w is not None:
    los = delta / rng.clamp_min(1e-6).unsqueeze(-1)
    v_ff = (leader_vel_w * los).sum(dim=-1)
  v = torch.clamp(v_ff + gains.k_range * range_err, -gains.v_back_max, gains.v_max)

  # Only translate once roughly facing the leader; always keep turning toward it.
  gate = (err.abs() < gains.turn_in_place_rad).float()
  vx = v * torch.cos(err) * gate
  vy = torch.clamp(v * torch.sin(err), -gains.vy_max, gains.vy_max) * gate
  wz = torch.clamp(gains.k_yaw * err, -gains.wz_max, gains.wz_max)
  return torch.stack([vx, vy, wz], dim=-1), rng
