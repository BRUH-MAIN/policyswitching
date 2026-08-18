"""Unitree Go2 velocity environment configurations."""

from dataclasses import replace
from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import ALL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.velocity.mdp.pas import foot_friction
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


def unitree_go2_pas_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go2 config for replicating SARO's PAS low-level policy (arXiv:2407.16412).

  Same task as Rough, plus: a "privileged_state" obs group (base linear
  velocity + foot friction) feeding the PAS actor's oracle latent, a
  gap-crossing terrain (stepping stones, absent from ROUGH_TERRAINS_CFG), and
  the paper's energy / joint-velocity reward terms. Pair with
  `unitree_go2_pas_ppo_runner_cfg`: Stage 1 with `enable_annealing=False`,
  Stage 2 with `enable_annealing=True` resumed from the Stage-1 checkpoint.
  """
  cfg = unitree_go2_rough_env_cfg(play=play)

  # Gap terrain: ROUGH_TERRAINS_CFG has stairs and ramps but no gap-crossing
  # terrain. Make room for stepping-stones (from ALL_TERRAINS_CFG) by scaling
  # the existing sub-terrain proportions down so they still sum to 1.0.
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  gap_proportion = 0.15
  scaled_sub_terrains = {
    name: replace(sub_cfg, proportion=sub_cfg.proportion * (1.0 - gap_proportion))
    for name, sub_cfg in ROUGH_TERRAINS_CFG.sub_terrains.items()
  }
  scaled_sub_terrains["stepping_stones"] = replace(
    ALL_TERRAINS_CFG.sub_terrains["stepping_stones"], proportion=gap_proportion
  )
  cfg.scene.terrain.terrain_generator = replace(
    cfg.scene.terrain.terrain_generator, sub_terrains=scaled_sub_terrains
  )

  # Privileged state: base linear velocity + foot friction (paper's s_t in R^4).
  # foot_friction reads back the same geoms randomized by the "foot_friction"
  # domain-randomization event term.
  foot_geom_names = cfg.events["foot_friction"].params["asset_cfg"].geom_names
  cfg.observations["privileged_state"] = ObservationGroupCfg(
    terms={
      "base_lin_vel": ObservationTermCfg(
        func=envs_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
        noise=Unoise(n_min=-0.05, n_max=0.05),
      ),
      "foot_friction": ObservationTermCfg(
        func=foot_friction,
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names)},
      ),
    },
    concatenate_terms=True,
    enable_corruption=True,
    history_length=1,
  )

  # Paper's Energy and Joint-velocity reward terms (Table V) -- present in
  # mjlab's reward library but not wired into the stock Go2 tasks.
  cfg.rewards["energy"] = RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1.0e-6)
  cfg.rewards["joint_vel_l2"] = RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-0.002)

  return cfg
