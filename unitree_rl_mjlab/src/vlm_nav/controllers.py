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
