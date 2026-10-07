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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
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
from src.tasks.velocity.mdp.rewards import (
  climb_progress,
  feet_clearance_relative,
  feet_gait_gated,
  variable_posture_gated,
)
from src.tasks.velocity.mdp.terrain_curriculum import (
  PROGRESS_ACROSS_M,
  PROGRESS_STALL_CMD,
  PROGRESS_STALL_M,
  actual_speed,
  cmd_moving,
  cmd_speed,
  limb_contact,
  stalled,
  terrain_levels_progress,
  terrain_levels_progress_mixed,
  terrain_row_fraction,
  terrain_row_mean,
  terrain_row_mean_by_direction,
)
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
  step_height: float | None = None,
  step_width: float | None = None,
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
    float in [0, 1] pins every row to exactly that difficulty (rows still
    exist -- see the bug note below for why collapsing them was wrong).
  - `step_height` / `step_width` (metres): pin every stairs sub-terrain's riser
    height and/or tread, overriding the class defaults (risers 0-10 cm, 0.3 m
    tread) so a policy can be evaluated at real stair dimensions. Riser height
    then no longer depends on `difficulty` or the row. Raises if the selected
    terrain has no stairs.
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

  if step_height is not None or step_width is not None:
    stairs = {n: s for n, s in gen.sub_terrains.items() if hasattr(s, "step_height_range")}
    if not stairs:
      raise ValueError(
        f"step_height/step_width given but terrain {terrain!r} has no stairs sub-terrain "
        f"(sub-terrains: {sorted(gen.sub_terrains)})"
      )
    overrides: dict = {}
    if step_height is not None:
      overrides["step_height_range"] = (step_height, step_height)
    if step_width is not None:
      overrides["step_width"] = step_width
    gen = replace(
      gen,
      sub_terrains={
        n: (replace(s, **overrides) if n in stairs else s) for n, s in gen.sub_terrains.items()
      },
    )

  if difficulty is None:
    cfg.scene.terrain.max_init_terrain_level = None  # uniform over all rows
  else:
    if not 0.0 <= difficulty <= 1.0:
      raise ValueError(f"difficulty must be in [0, 1], got {difficulty}")
    # NOT num_rows=1. mjlab's curriculum generator computes each row's
    # difficulty as `lower + (upper - lower) * frac` -- with
    # difficulty_range=(d, d), (upper - lower) is 0, so every row already
    # gets exactly d regardless of num_rows; collapsing the grid was
    # unnecessary for pinning. It was also actively wrong: forcing every env
    # onto one 8m-wide row, combined with max_init_terrain_level (still 5
    # from the base config -- this branch never touched it) clamping to
    # min(5, num_rows-1)=0, produced near-total immediate failure independent
    # of policy or terrain class (a 0.8m-tall, 1-row grid, not a real
    # difficulty effect). Confirmed via direct rollout on PAS-oracle,
    # --terrain flat: 98.9% fall / ~30-step episodes with num_rows=1 vs. 0%
    # fall / full-length episodes with num_rows left alone -- see findings.md
    # bug #12. Only reset max_init_terrain_level, so envs spread across every
    # (now same-difficulty) row instead of piling into row 0 of an untouched
    # multi-row grid.
    gen = replace(gen, difficulty_range=(difficulty, difficulty))
    assert gen.difficulty_range == (difficulty, difficulty), (
      "difficulty pin didn't take -- would silently produce plausible-looking "
      "wrong numbers rather than a loud failure (this is exactly how bug #12 "
      "went undetected); check TerrainGeneratorCfg field name/type."
    )
    cfg.scene.terrain.max_init_terrain_level = None

  # curriculum=True here is the generator's column-per-type layout (proportions
  # become exact column counts rather than per-patch samples), not the
  # terrain_levels curriculum removed below.
  gen = replace(gen, curriculum=True, seed=seed if seed is not None else gen.seed)
  cfg.scene.terrain.terrain_generator = gen
  cfg.curriculum.pop("terrain_levels", None)
  # Tasks that re-draw each env's row/column at reset (StairsV3) must not do so under a
  # pinned eval: it would undo the pinned layout and the per-sub-terrain breakdown.
  cfg.events.pop("randomize_terrain", None)


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


def _hold_command_range_at_stage0(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Cut the `command_vel` curriculum to its first stage (the "v2" change)."""
  # Play mode has no curriculum at all.
  if "command_vel" in cfg.curriculum:
    params = cfg.curriculum["command_vel"].params
    params["velocity_stages"] = params["velocity_stages"][:1]
  return cfg


def unitree_go2_spec_stairs_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs specialist v2: Stairs with the command range held at its first stage.

  Differs from `unitree_go2_spec_stairs_env_cfg` ONLY in the `command_vel`
  curriculum, which keeps stage 0 (lin_vel_x up to 1 m/s) for the whole run
  instead of widening to 2 m/s at iteration 5000. In the v1 run that widening
  dropped the terrain curriculum from level ~1.9 to ~0.9 for the remaining
  half of training (a robot that covers less than half its commanded distance
  is demoted), and the policy lost stairs ability it already had: on pinned
  pyramid stairs model_9999 falls 2.6-3.6x as often per 100 m as model_4800,
  the last checkpoint before the widening
  (coordination/results/2026-10-03-stairs-precollapse-checkpoint-eval.md).

  The terrain mix is deliberately unchanged, so a v1-vs-v2 difference has one
  cause. Evaluate at the range it trains on: eval_checkpoint.py's default
  command range is the widened one, so pass
  `--lin-vel-x -0.5 1.0 --lin-vel-y -0.5 0.5`.
  """
  return _hold_command_range_at_stage0(
    _unitree_go2_specialist_env_cfg(TERRAIN_CLASSES["stairs"], play=play)
  )


# Stairs V3 geometry: risers across the range of real stairs (5 cm up to 20 cm; building
# stairs are 15-18 cm) at two treads (real treads are 25-30 cm). Row r of the 10-row grid
# is generated at difficulty ~(r + U[0,1)) / 10, so riser = 5 + 15 * d cm:
#   row 0: 5-6.5   1: 6.5-8   2: 8-9.5   3: 9.5-11   4: 11-12.5
#   row 5: 12.5-14 6: 14-15.5 7: 15.5-17 8: 17-18.5  9: 18.5-20 cm
STAIRS_V3_STEP_HEIGHT_RANGE = (0.05, 0.20)
STAIRS_V3_TREADS = (0.30, 0.26)


def unitree_go2_spec_stairs_v3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs specialist v3: real stair heights, rows spread uniformly all run.

  StairsV2 (stage-0 command range, same observations, rewards, terminations, runner)
  with three changes:

  1. Risers 5-20 cm (v2 was 0-10 cm) on pyramid_stairs and pyramid_stairs_inv, each at
     two treads (0.30 m and 0.26 m), the four sub-terrains at equal weight.
  2. No promote/demote terrain curriculum. Every env is re-drawn uniformly over all 10
     rows (and all 20 columns, i.e. the four sub-terrains) at every reset, so 60% of
     training is on rows 4-9 (11-20 cm) from the first iteration and the share never
     moves. Why: terrain_levels_vel needs 4 m of net displacement (the patch edge) to
     promote and demotes anything under |command| * 10 m, so for commands above 0.4 m/s
     every episode that does not leave the patch is demoted. Replayed on the stairs-v2
     policy it is demoted ~48% and promoted ~42% of the time on rows 1-3, where it almost
     never falls, and the rule's stationary mean row is ~2.4 (logged: 1.9-2.1)
     (coordination/results/2026-10-05-terrain-curriculum-diagnosis.md).
  3. Monitor terms `terrain_rows_*` log what fraction of envs sit on rows 0-1 ... 8-9, and
     `terrain_row_mean` the mean row (uniform draws: ~4.5), because a mean level that
     looks fine can hide a run that never visits the tall rows.

  Meant to be warm-started from v2's final checkpoint (a100/train_specialist_slurm.sh does
  this by default for SPEC=StairsV3). Evaluate at `--lin-vel-x -0.5 1.0 --lin-vel-y -0.5
  0.5` with `--step-height` (and `--step-width`) for pinned riser heights.
  """
  cfg = _hold_command_range_at_stage0(
    _unitree_go2_specialist_env_cfg(TERRAIN_CLASSES["stairs"], play=play)
  )
  assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
  gen = cfg.scene.terrain.terrain_generator
  base = gen.sub_terrains
  sub_terrains = {}
  for tread in STAIRS_V3_TREADS:
    suffix = "" if tread == 0.30 else f"_w{round(tread * 100)}"
    for name in TERRAIN_CLASSES["stairs"]:
      sub_terrains[name + suffix] = replace(
        base[name],
        proportion=1.0,
        step_height_range=STAIRS_V3_STEP_HEIGHT_RANGE,
        step_width=tread,
      )
  cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=sub_terrains)

  if not play:  # play mode already re-draws terrain at reset and has no curriculum
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.pop("terrain_levels", None)
    # Must run BEFORE the root-state reset, which places the robot at env_origins.
    cfg.events = {
      "randomize_terrain": EventTermCfg(func=envs_mdp.randomize_terrain, mode="reset"),
      **cfg.events,
    }
    cfg.curriculum["terrain_row_mean"] = CurriculumTermCfg(func=terrain_row_mean)
    for lo in range(0, 10, 2):
      cfg.curriculum[f"terrain_rows_{lo}_{lo + 1}"] = CurriculumTermCfg(
        func=terrain_row_fraction, params={"lo": lo, "hi": lo + 1}
      )
  return cfg


# StairsV4b: a thigh/calf touching a step costs this much per step in contact (rewards are
# scaled by dt = 0.02, so -2.0 is -0.04 per step, about one step's whole positive reward).
# A termination costs -200 * dt = -4 on the step AND forfeits the rest of the episode
# (~0.04 per remaining step), so a 10-step knee brush (-0.4) is a tenth of the penalty
# alone and far under the full price, while a scrape held for a whole climb (100 steps, -4)
# is still discouraged.
STAIRS_V4B_LIMB_CONTACT_WEIGHT = -2.0


def _stairs_v4_env_cfg(play: bool, limb_contact_penalised: bool) -> ManagerBasedRlEnvCfg:
  """Shared body of StairsV4a/V4b: StairsV3's terrain and command range, adaptive rows.

  Identical to `unitree_go2_spec_stairs_v3_env_cfg` (risers 5-20 cm at treads 0.30/0.26 m,
  four sub-terrains, stage-0 commands, observations, rewards, runner) except:

  * Rows are NOT re-drawn uniformly. Robots start on rows 0-3 (5-11 cm) and move by
    `terrain_levels_progress`, evaluated when an episode ends. With cheb = max(|dx|, |dy|)
    from the spawn (the stairs span 1.5-3.0 m, so cheb >= 3.2 m means all 5 steps are
    behind it) and cmd = |commanded xy velocity| at the last step:

      promote : ran to the time limit (not terminated) AND cheb >= 3.2 m
      demote  : terminated, OR (ran to the time limit AND cmd > 0.3 m/s AND cheb < 1.9 m)
      stay    : anything else (part-way up the flight; told to stand)

    A robot standing on the platform all episode is demoted, never promoted and never
    left in place. StairsV3 did exactly that on rows it could not climb, from uniform rows.
    (A promotion past row 9 re-draws the row uniformly, mjlab's own behaviour.)
  * Per-iteration `Episode_Metrics/{cmd_speed, actual_speed, cmd_moving, stalled}` are
    logged: achieved/commanded speed ~ actual_speed / cmd_speed and the stalled fraction
    among commanded-to-move steps ~ stalled / cmd_moving. Judge the run on these, the row
    histogram (`Curriculum/terrain_rows_*`, `terrain_row_mean`) and crossing, not on
    reward, episode length or falls: a robot that stands still scores well on all three.
  * `limb_contact_penalised` (V4b): see `unitree_go2_spec_stairs_v4b_env_cfg`.
  """
  cfg = unitree_go2_spec_stairs_v3_env_cfg(play=play)
  cfg.metrics = {
    **(cfg.metrics or {}),
    "cmd_speed": MetricsTermCfg(func=cmd_speed),
    "actual_speed": MetricsTermCfg(func=actual_speed),
    "cmd_moving": MetricsTermCfg(func=cmd_moving),
    "stalled": MetricsTermCfg(func=stalled),
  }
  if not play:
    cfg.scene.terrain.max_init_terrain_level = 3
    cfg.events.pop("randomize_terrain")  # rows move by the curriculum instead
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=terrain_levels_progress,
      params={
        "across_m": PROGRESS_ACROSS_M,
        "stall_m": PROGRESS_STALL_M,
        "stall_cmd": PROGRESS_STALL_CMD,
      },
    )

  if limb_contact_penalised:
    # The base sensor keeps terminating, but only on trunk geoms (base1-3) and hips;
    # thigh and calf geoms move to their own sensor that feeds a penalty instead.
    sensors = list(cfg.scene.sensors or ())
    idx = next(i for i, s in enumerate(sensors) if s.name == "nonfoot_ground_touch")
    trunk = sensors[idx]
    sensors[idx] = replace(
      trunk, primary=replace(trunk.primary, pattern=r"^(base[123]|[FR][LR]_hip)_collision$", exclude=())
    )
    sensors.append(
      replace(
        trunk,
        name="limb_ground_touch",
        primary=replace(trunk.primary, pattern=r"^[FR][LR]_(thigh|calf[12])_collision$", exclude=()),
      )
    )
    cfg.scene.sensors = tuple(sensors)
    cfg.rewards["limb_contact"] = RewardTermCfg(
      func=limb_contact,
      weight=STAIRS_V4B_LIMB_CONTACT_WEIGHT,
      params={"sensor_name": "limb_ground_touch", "force_threshold": 10.0},
    )
  return cfg


def unitree_go2_spec_stairs_v4a_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs v4a: StairsV3's terrain with progress-gated adaptive rows (see `_stairs_v4_env_cfg`).

  Rewards and terminations are exactly StairsV2/V3's: any non-foot contact above 10 N
  (knee, calf, thigh, base, hip) ends the episode. Warm-start from stairs v2.
  """
  return _stairs_v4_env_cfg(play, limb_contact_penalised=False)


def unitree_go2_spec_stairs_v4b_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs v4b: v4a, and a thigh or calf touching a step is penalised, not terminal.

  `nonfoot_ground_touch` (the `illegal_contact` termination, >10 N) now matches only the
  trunk (base1-3) and hip geoms; a new `limb_ground_touch` sensor covers thigh and calf
  geoms and feeds a `limb_contact` reward at STAIRS_V4B_LIMB_CONTACT_WEIGHT (-2.0 per step
  in contact, > 10 N). `fell_over` (70 deg) is unchanged. This changes the task definition
  relative to every earlier specialist: its `illegal_contact` and falls/100 m are not
  comparable with theirs, and a shin brushing a nosing no longer ends an episode.
  """
  return _stairs_v4_env_cfg(play, limb_contact_penalised=True)


def unitree_go2_spec_stairs_v5a_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs v5a: v4b with the `foot_clearance` reward REMOVED.

  `mdp.feet_clearance` charges |foot_z_world - 0.10| * foot speed. On a staircase the robot is
  up to five risers away from z = 0, so every moving foot is charged for the staircase's height
  and the cheapest response is to move less. In the v4/v3 logs that term is 1.5-2x v2's and
  about half the tracking reward (coordination/results/2026-10-06-stairs-v4-stop-test.md).
  Everything else is v4b (progress-gated rows from 0-3, limb contact penalised, speed/stall
  metrics, warm start from v2, normalizer kept). One change, to test one hypothesis.
  """
  cfg = _stairs_v4_env_cfg(play, limb_contact_penalised=True)
  cfg.rewards.pop("foot_clearance")
  return cfg


def unitree_go2_spec_stairs_v5b_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs v5b: v4b with `foot_clearance` measured relative to the lowest foot.

  Same term, weight (-1.0), target (0.10 m), speed weighting and command gate as the stock
  reward, but a foot's height is taken above the robot's lowest foot instead of world z
  (`mdp.feet_clearance_relative`), so a staircase's height is not charged and a swing still is
  asked to clear. Otherwise identical to v5a/v4b.
  """
  cfg = _stairs_v4_env_cfg(play, limb_contact_penalised=True)
  cfg.rewards["foot_clearance"] = replace(cfg.rewards["foot_clearance"], func=feet_clearance_relative)
  return cfg


def unitree_go2_spec_stairs_v5c_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs v5c: v5a with the progress rule mixed 50/50 with uniformly drawn rows.

  v5a (clearance removed) walks at ~68% of commanded speed but its rows settled at a mean of
  ~3.7 and drifted down, with ~6% of robots on rows 6-7 and none on 8-9 by iteration 4,400, so
  the 15-20 cm risers get almost no data. Here `terrain_levels_progress_mixed` applies the same
  progress rule and then re-draws the row of half of the resetting robots uniformly over all ten
  rows: 5% on each row at all times, the other half at the frontier. Everything else is v5a.
  Meant to be warm-started from a v5a checkpoint (the train script defaults to the latest).
  """
  cfg = unitree_go2_spec_stairs_v5a_env_cfg(play=play)
  if not play:
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=terrain_levels_progress_mixed,
      params={"explore_frac": 0.5, "across_m": PROGRESS_ACROSS_M, "stall_m": PROGRESS_STALL_M, "stall_cmd": PROGRESS_STALL_CMD},
    )
    # Mean row of the descending and ascending populations separately (they keep their column).
    for direction in ("down", "up"):
      cfg.curriculum[f"terrain_row_mean_{direction}"] = CurriculumTermCfg(
        func=terrain_row_mean_by_direction, params={"direction": direction}
      )
  return cfg


def unitree_go2_spec_stairs_v6a_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs v6a: v5a with the posture and gait rewards paid only in proportion to progress.

  Final v5a (model_9999) climbs 12 cm and refuses 15 cm and up, in both directions, by
  stalling. Measured per reward term in its own training env at a pinned 15 cm riser
  (scripts/diag_reward_terms.py): a robot commanded to move that stands on the platform
  earns ~2.6 reward/s (pose 0.96 + angular tracking 0.97 + gait 0.45 + linear tracking 0.30),
  a robot walking on flat ground ~3.0, and a robot on the flight ~1.8 (model_400, which still
  crossed 15 cm going down). So the reward pays more for refusing a tall flight than for
  crossing it, and neither the row rule nor uniform rows can change that.

  Here `pose` and `foot_gait` are multiplied by `progress_gate` (achieved / commanded planar
  velocity along the command, clamped to [0, 1]; 1 when told to stand). A stalled robot then
  keeps ~1.3/s and a walking one loses nothing. Everything else is v5a. Meant to be
  warm-started from v5a's final checkpoint with the normaliser kept.
  """
  cfg = unitree_go2_spec_stairs_v5a_env_cfg(play=play)
  cfg.rewards["pose"] = replace(cfg.rewards["pose"], func=variable_posture_gated)
  cfg.rewards["foot_gait"] = replace(cfg.rewards["foot_gait"], func=feet_gait_gated)
  return cfg


# StairsV7a: reward per m/s of height gained on an up-flight. A five-step 17 cm flight is
# 0.85 m, so the whole climb is worth 4.25 reward-seconds, about one termination (-4), and a
# robot climbing at 0.2 m/s along the ground earns ~0.57/s for it against a flight-versus-stall
# difference of 0.09/s without it.
STAIRS_V7_CLIMB_WEIGHT = 5.0


def unitree_go2_spec_stairs_v7a_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stairs v7a: v6a plus a reward for height gained on up-flights (`climb_progress`).

  Final v6a (6,000 laptop iterations) goes down 9-17 cm flights 99-100% of the time and up
  12 cm 99.6%, 15 cm 85%, 17 cm 3% (256 trials per cell), and its 15 cm ascent swings between
  0% and 98% from checkpoint to checkpoint. Per reward term at a pinned 17 cm riser, a robot
  on the up-flight earns 0.86/s and a stalled one 0.77/s: the climb itself is not paid. This
  adds STAIRS_V7_CLIMB_WEIGHT x vertical base velocity on inverted-pyramid columns only
  (signed, so only net height gained counts). Also logs the up and down row means, since the
  overall mean row hid everything that happened to ascent in v6a. Warm-start from v6a.
  """
  cfg = unitree_go2_spec_stairs_v6a_env_cfg(play=play)
  if not play:  # the play terrain has no fixed up/down columns to key the term on
    cfg.rewards["climb_progress"] = RewardTermCfg(
      func=climb_progress, weight=STAIRS_V7_CLIMB_WEIGHT, params={"command_name": "twist"}
    )
    for direction in ("down", "up"):
      cfg.curriculum[f"terrain_row_mean_{direction}"] = CurriculumTermCfg(
        func=terrain_row_mean_by_direction, params={"direction": direction}
      )
  return cfg


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


def unitree_go2_generalist_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Generalist v2: the generalist with the command range held at its first stage.

  Differs from `unitree_go2_generalist_env_cfg` ONLY in the `command_vel`
  curriculum -- the same change `unitree_go2_spec_stairs_v2_env_cfg` makes to
  Stairs. The v1 generalist's terrain curriculum collapsed at iteration 5000
  like the specialists' (terrain_levels 1.52 -> 0.4-0.5, job 12479), so against
  Stairs v2 it is the handicapped arm; this is the matched one. As with Stairs
  v2, evaluate at `--lin-vel-x -0.5 1.0 --lin-vel-y -0.5 0.5`.
  """
  return _hold_command_range_at_stage0(unitree_go2_generalist_env_cfg(play=play))


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
