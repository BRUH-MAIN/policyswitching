"""Camera-equipped Go2 simulation env for the SARO-style VLM navigation pipeline.

Built on a specialist's *play* config so the three frozen specialists see
exactly the observation space they were trained on (234-dim actor obs,
height_scan included) -- this module only adds things around the policy:

- an ego RGB-D camera (not an actor observation; read straight off the sensor),
- an externally driven velocity command (SARO's sub-task executor sends
  (vx, vy, wz); the task's random resampling, heading control and "stand still"
  branch are all disabled so nothing overwrites it),
- a caller-supplied terrain (a navigation course, or a pinned eval terrain),
- a deterministic spawn pose, since a course has a defined start and heading.

Nothing here touches a registered task config: `load_env_cfg` returns a deep copy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

import src.tasks  # noqa: F401  (registers Unitree-Go2-* tasks)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains import TerrainGeneratorCfg

from src.vlm_nav.camera import CameraSpec

CAMERA_SENSOR_NAME = "ego_cam"
COMMAND_NAME = "twist"
# Any Spec-* task works as the base: they differ only in terrain, which is replaced.
BASE_TASK = "Unitree-Go2-Spec-Flat"


def make_twin_env_cfg(
  camera: CameraSpec,
  terrain_generator: TerrainGeneratorCfg | None = None,
  num_envs: int = 1,
  seed: int = 0,
  spawn_xy_jitter: float = 0.0,
  spawn_yaw_range: tuple[float, float] = (0.0, 0.0),
  use_shadows: bool = True,
) -> ManagerBasedRlEnvCfg:
  cfg = load_env_cfg(BASE_TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  cfg.curriculum = {}

  # Externally driven command. heading_command=False also stops _update_command
  # from rewriting wz every step; resampling pushed past any episode length.
  cmd = cfg.commands[COMMAND_NAME]
  cmd.heading_command = False
  cmd.ranges.heading = None
  cmd.rel_standing_envs = 0.0
  cmd.init_velocity_prob = 0.0
  cmd.resampling_time_range = (1e9, 1e9)
  cmd.ranges.lin_vel_x = (0.0, 0.0)
  cmd.ranges.lin_vel_y = (0.0, 0.0)
  cmd.ranges.ang_vel_z = (0.0, 0.0)
  cmd.debug_vis = False

  if terrain_generator is not None:
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = terrain_generator
    cfg.scene.terrain.max_init_terrain_level = None
    # play mode re-randomizes the env's terrain patch on reset; a course is fixed.
    cfg.events.pop("randomize_terrain", None)

  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (-spawn_xy_jitter, spawn_xy_jitter),
    "y": (-spawn_xy_jitter, spawn_xy_jitter),
    "z": (0.0, 0.0),
    "yaw": spawn_yaw_range,
  }

  sensors = []
  for s in cfg.scene.sensors or ():
    # findings.md bug #17: the ray overlay renders as a broken robot pose far
    # from spawn. It is a viewer overlay only; the camera sensor never sees it.
    sensors.append(replace(s, debug_vis=False) if s.name == "terrain_scan" else s)
  sensors.append(
    CameraSensorCfg(
      name=CAMERA_SENSOR_NAME,
      parent_body=camera.parent_body,
      pos=camera.pos,
      quat=camera.quat,
      fovy=camera.fovy_deg,
      width=camera.width,
      height=camera.height,
      data_types=("rgb", "depth"),
      use_textures=True,
      use_shadows=use_shadows,
    )
  )
  cfg.scene.sensors = tuple(sensors)
  return cfg


def set_velocity_command(env, cmd_vel: torch.Tensor | np.ndarray | list) -> None:
  """Write (vx, vy, wz) in the base frame for every env; shape (3,) or (N, 3).

  Takes effect in the observation produced by the next env.step() (one 20 ms
  control-step lag), which is how a real command topic behaves too.
  """
  term = env.unwrapped.command_manager.get_term(COMMAND_NAME)
  value = torch.as_tensor(cmd_vel, dtype=term.vel_command_b.dtype, device=term.vel_command_b.device)
  term.vel_command_b[:] = value


def get_rgbd(env, env_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
  """(rgb uint8 HxWx3, ray-distance depth float32 HxW) for one env, as numpy."""
  data = env.unwrapped.scene[CAMERA_SENSOR_NAME].data
  rgb = data.rgb[env_id].detach().cpu().numpy()
  depth = data.depth[env_id, ..., 0].detach().cpu().numpy()
  return rgb, depth


def base_pose(env, env_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
  """(position xyz, quaternion wxyz) of the robot base in the world frame."""
  robot = env.unwrapped.scene["robot"]
  pos = robot.data.root_link_pos_w[env_id].detach().cpu().numpy()
  quat = robot.data.root_link_quat_w[env_id].detach().cpu().numpy()
  return pos, quat
