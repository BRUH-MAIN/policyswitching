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

# The project's four geometry-based terrain classes (see objective.md), as mjlab
# sub-terrain names. Single source of truth for the specialists, the matched
# generalist and the pinned eval terrains, so "stairs" is the same geometry in
# training and in every cell of the cross-terrain eval matrix.
TERRAIN_CLASSES: dict[str, tuple[str, ...]] = {
  "flat": ("flat",),
  "rough": ("random_rough", "wave_terrain"),
  "stairs": ("pyramid_stairs", "pyramid_stairs_inv"),
  "gaps": ("stepping_stones",),
}

# --terrain choices for scripts/eval_checkpoint.py: the task's own mix, one
# class, or all four classes at equal weight.
EVAL_TERRAINS: tuple[str, ...] = ("native", *TERRAIN_CLASSES, "mixed")


def _available_sub_terrains() -> dict:
  # ROUGH_TERRAINS_CFG is the default set, with ALL_TERRAINS_CFG covering the
  # ones it omits (gaps). Where both define a sub-terrain, ROUGH's wins.
  available = dict(ALL_TERRAINS_CFG.sub_terrains)
  available.update(ROUGH_TERRAINS_CFG.sub_terrains)
  return available


def _class_weighted_proportions(class_weights: dict[str, float]) -> dict[str, float]:
  """Sub-terrain proportions giving each terrain class its weight, split evenly
  across that class's sub-terrains (so rough's two sub-terrains don't count
  double against flat's one)."""
  return {
    name: weight / len(TERRAIN_CLASSES[cls])
    for cls, weight in class_weights.items()
    for name in TERRAIN_CLASSES[cls]
  }


def apply_eval_conditions(
  cfg: ManagerBasedRlEnvCfg,
  terrain: str = "native",
  difficulty: float | None = None,
  seed: int | None = None,
) -> None:
  """Pin the terrain for cross-policy numeric eval (scripts/eval_checkpoint.py).

  Always removes the `terrain_levels` curriculum. Left in, it promotes an env
  to harder terrain whenever it walks far enough and demotes it otherwise,
  during the eval rollout itself -- so a better policy is pushed onto harder
  ground until it too starts failing, and survival ends up measuring distance
  to the curriculum's equilibrium instead of capability.

  - `terrain`: "native" keeps the task's own sub-terrain mix; a TERRAIN_CLASSES
    key swaps in exactly that class; "mixed" uses all four at equal weight.
  - `difficulty`: None spreads envs uniformly over every difficulty row; a
    float in [0, 1] collapses the grid to a single row generated at exactly
    that difficulty.
  """
  if terrain not in EVAL_TERRAINS:
    raise ValueError(f"terrain must be one of {EVAL_TERRAINS}, got {terrain!r}")
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  gen = cfg.scene.terrain.terrain_generator

  if terrain != "native":
    classes = TERRAIN_CLASSES if terrain == "mixed" else (terrain,)
    proportions = _class_weighted_proportions({c: 1.0 for c in classes})
    available = _available_sub_terrains()
    gen = replace(
      gen,
      sub_terrains={
        name: replace(available[name], proportion=p) for name, p in proportions.items()
      },
    )

  if difficulty is None:
    cfg.scene.terrain.max_init_terrain_level = None  # uniform over all rows
  else:
    if not 0.0 <= difficulty <= 1.0:
      raise ValueError(f"difficulty must be in [0, 1], got {difficulty}")
    gen = replace(gen, num_rows=1, difficulty_range=(difficulty, difficulty))

  # curriculum=True here is the generator's column-per-type layout (proportions
  # become exact column counts rather than per-patch samples), not the
  # terrain_levels curriculum removed below.
  gen = replace(gen, curriculum=True, seed=seed if seed is not None else gen.seed)
  cfg.scene.terrain.terrain_generator = gen
  cfg.curriculum.pop("terrain_levels", None)


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


def _unitree_go2_specialist_env_cfg(
  sub_terrain_names: tuple[str, ...],
  play: bool = False,
  proportions: dict[str, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Base config for a single-terrain specialist policy.

  Every specialist derives from the *rough* config and differs ONLY in which
  sub-terrains are active, which matters for a specific downstream reason: the
  switching module blends the frozen specialists, so they must all share one
  observation space and one network shape.

  That rules out building the flat specialist on `unitree_go2_flat_env_cfg()`,
  which switches to a `"plane"` terrain and then deletes the `height_scan`
  terms from both the actor and critic observation groups (there is no terrain
  to scan). A specialist trained that way would have a different input width
  than the other three and could not be blended with them. Using a
  flat-only *generator* instead keeps the terrain scan present and the
  observation space identical across all four.

  Rewards are deliberately left identical across specialists too -- per-terrain
  reward shaping (e.g. extra foot clearance on gaps, stricter orientation
  penalty on slopes) is a later tuning step, and keeping them uniform for now
  keeps the specialist-vs-generalist comparison fair.

  `proportions` overrides the default equal-weight split across
  `sub_terrain_names` -- e.g. `{"stepping_stones": 0.2, "flat": 0.4,
  "random_rough": 0.4}` for a specialist that needs an easy on-ramp during
  training but is still evaluated/deployed as "the gaps policy" (see the
  Gaps specialist below: 100% stepping_stones plateaued in two independent
  training runs -- reward flat, ~10-step episodes, no improvement over
  thousands of iterations each time -- almost certainly because a cold-start
  policy has no easy terrain to learn basic locomotion on before it also has
  to solve gap-crossing. This mirrors PAS's own gap terrain, which mixes
  stepping_stones at only 15% into 85% easier ground for the same reason.)
  """
  cfg = unitree_go2_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None

  available = _available_sub_terrains()

  missing = [n for n in sub_terrain_names if n not in available]
  if missing:
    raise KeyError(
      f"Unknown sub-terrain(s) {missing}. Available: {sorted(available)}"
    )
  if proportions is not None and set(proportions) != set(sub_terrain_names):
    raise ValueError(
      f"proportions keys {sorted(proportions)} must exactly match "
      f"sub_terrain_names {sorted(sub_terrain_names)}"
    )

  # Equal weight by default; proportions are relative weights, so they need
  # not sum to 1.0 either way.
  sub_terrains = {
    name: replace(available[name], proportion=(proportions or {}).get(name, 1.0))
    for name in sub_terrain_names
  }
  cfg.scene.terrain.terrain_generator = replace(
    cfg.scene.terrain.terrain_generator, sub_terrains=sub_terrains
  )
  return cfg


def unitree_go2_spec_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat specialist. Flat-only generator (NOT a plane) so height_scan survives."""
  return _unitree_go2_specialist_env_cfg(TERRAIN_CLASSES["flat"], play=play)


def unitree_go2_spec_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough specialist: continuous uneven ground (noise + waves), no discrete steps."""
  return _unitree_go2_specialist_env_cfg(TERRAIN_CLASSES["rough"], play=play)


def unitree_go2_spec_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs specialist: ascending and descending pyramid stairs."""
  return _unitree_go2_specialist_env_cfg(TERRAIN_CLASSES["stairs"], play=play)


def unitree_go2_spec_gaps_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Gap specialist: stepping stones, blended with easier terrain during training.

  100% stepping_stones plateaued in two independent training runs (see
  `_unitree_go2_specialist_env_cfg`'s docstring) -- blending in flat/rough
  gives the policy somewhere to learn basic locomotion before it also has to
  solve gap-crossing, matching PAS's own ~15% gap-terrain mix. The terrain
  curriculum's difficulty-by-row scaling still applies within each
  sub-terrain, so this isn't purely an easier task, just a less narrow one.
  """
  return _unitree_go2_specialist_env_cfg(
    ("stepping_stones", "flat", "random_rough"),
    play=play,
    proportions={"stepping_stones": 0.2, "flat": 0.4, "random_rough": 0.4},
  )


def unitree_go2_spec_gaps_warm_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Gap specialist, second design: 100% stepping_stones, warm-started.

  The blended Gaps task above trains 80% on flat/rough -- nearly the union of
  the Flat and Rough specialists' terrain -- so the result is barely a gap
  specialist and hands the gating network a near-redundant expert. The plateau
  it was fixing was a cold-start problem, which is what initialization is for:
  this task keeps the terrain 100% gaps and is meant to start from an existing
  specialist's weights (a100/warm_start_ckpt.py, via INIT_FROM in
  a100/train_specialist_slurm.sh) rather than from random init.
  """
  return _unitree_go2_specialist_env_cfg(TERRAIN_CLASSES["gaps"], play=play)


def unitree_go2_generalist_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Sensing-matched generalist baseline (arm 1 of the switching comparison).

  One stock-PPO policy over the union of the four specialist terrain classes,
  each class at equal weight. Built on the same base as the specialists, so it
  has the identical observation space (raw height_scan included), rewards and
  runner config -- the only thing that differs from a specialist is terrain
  breadth. PAS is not a clean arm 1: its deployable estimator-only mode has no
  height_scan at all, and it adds reward terms (energy, joint_vel_l2) and ~8x
  the training compute that no specialist got.
  """
  proportions = _class_weighted_proportions({c: 1.0 for c in TERRAIN_CLASSES})
  return _unitree_go2_specialist_env_cfg(
    tuple(proportions), play=play, proportions=proportions
  )


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
