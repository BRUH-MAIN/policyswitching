# Manager configuration deep dive

This is the part you'll touch most often: tuning or adding terms in each of
the 8 managers. Every example below is either taken directly from, or a
minor simplification of, real code in `unitree_rl_mjlab/src/tasks/`. Doc
sources: mjlab's
[Observations](https://mujocolab.github.io/mjlab/main/source/observations.html),
[Actions](https://mujocolab.github.io/mjlab/main/source/actions.html),
[Rewards](https://mujocolab.github.io/mjlab/main/source/rewards.html),
[Terminations](https://mujocolab.github.io/mjlab/main/source/terminations.html),
[Commands](https://mujocolab.github.io/mjlab/main/source/commands.html),
[Events](https://mujocolab.github.io/mjlab/main/source/events.html),
[Domain Randomization](https://mujocolab.github.io/mjlab/main/source/randomization.html),
[Curriculum](https://mujocolab.github.io/mjlab/main/source/curriculum.html).

Every manager term follows the same shape: a `<X>TermCfg(func=..., params={...}, ...)`
where `func(env, **params)` gets called by the manager. `env` is always the
first positional argument to any term function.

## Observations

`ObservationTermCfg` fields: `func`, `params`, `noise` (a `NoiseCfg`, applied
only if the group's `enable_corruption=True`), `scale`, `clip: (lo, hi)`,
`history_length` (overrides group setting), `delay_max_lag`.

`ObservationGroupCfg` fields: `terms: dict[str, ObservationTermCfg]`,
`concatenate_terms` (default `True` — set `False` to get a dict of tensors
instead of one concatenated tensor), `enable_corruption`, `history_length`.

Per-term processing pipeline, in order: **compute → noise → clip → scale →
delay → history**.

**Actor/critic asymmetry** is the standard pattern (privileged critic obs):
```python
observations = {
  "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True,
                                enable_corruption=True, history_length=1),
  "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True,
                                 enable_corruption=False, history_length=1),
}
```
`actor_terms` gets noise; `critic_terms` (a superset, in the velocity task —
adds `base_lin_vel`, `foot_height`, `foot_air_time`, `foot_contact`,
`foot_contact_forces`) does not.

Real term from `src/tasks/velocity/velocity_env_cfg.py`:
```python
"base_ang_vel": ObservationTermCfg(
  func=mdp.builtin_sensor,
  params={"sensor_name": "robot/imu_ang_vel"},
  noise=Unoise(n_min=-0.2, n_max=0.2),
),
"height_scan": ObservationTermCfg(
  func=envs_mdp.height_scan,
  params={"sensor_name": "terrain_scan"},
  noise=Unoise(n_min=-0.1, n_max=0.1),
  scale=1 / terrain_scan.max_distance,
),
```

**Custom observation function:**
```python
def my_observation(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_b   # shape [num_envs, D]
```
For stateful observations, write a class instead:
`__init__(self, cfg, env)`, `__call__(self, env, ...)`, optional
`reset(env_ids)`.

## Actions

The action manager slices the policy's output tensor across registered
terms (in registration order) and routes each slice to the matching
entity's actuators.

Built-in action types (all inherit `entity_name`, `actuator_names` (regex
tuple), `scale`, `offset`, `clip`; joint actions also get
`use_default_offset`):

| Type | Behavior |
|---|---|
| `JointPositionActionCfg` | Commands joint position targets. |
| `RelativeJointPositionActionCfg` | Position commands relative to current joint state. |
| `JointVelocityActionCfg` | Joint velocity targets. |
| `JointEffortActionCfg` | Direct torque/effort. |
| `TendonLengthActionCfg` / `TendonVelocityActionCfg` / `TendonEffortActionCfg` | Tendon-space equivalents. |
| `SiteEffortActionCfg` | Forces/torques at a named site (e.g. quadrotor thrust). |
| `DifferentialIKActionCfg` | Cartesian commands → joint positions via damped-least-squares IK. |

This repo's standard action term (used by both velocity and tracking
tasks):
```python
actions = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=0.25,             # overridden per-robot
    use_default_offset=True,  # zero policy output == default pose
  )
}
```
Actuator targets update every physics substep during decimation, not just
once per policy step.

## Rewards

`RewardTermCfg(func, weight, params)`. Each term returns `[num_envs]`;
manager computes `sum(weight_i * term_i)`. If
`scale_rewards_by_dt=True` (default), the whole sum is scaled by `step_dt`
so returns stay comparable across different physics rates.

Built-ins in `mjlab.envs.mdp.rewards`: `is_alive`, `is_terminated`,
`joint_torques_l2`, `joint_vel_l2`, `joint_acc_l2`, `action_rate_l2`,
`action_acc_l2`, `posture`, `electrical_power_cost`, `flat_orientation_l2`.

This repo's velocity task adds many task-specific terms in
`src/tasks/velocity/mdp/rewards.py` (`track_linear_velocity`,
`track_angular_velocity`, `body_orientation_l2`, `variable_posture`,
`feet_gait`, `feet_clearance`, `feet_slip`, `soft_landing`, `stand_still`,
...). Representative snippet:
```python
rewards = {
  "track_linear_velocity": RewardTermCfg(
    func=mdp.track_linear_velocity, weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  ),
  "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
  "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
  "foot_gait": RewardTermCfg(
    func=mdp.feet_gait, weight=0.5,
    params={"period": 0.6, "offset": [0.0, 0.5], "threshold": 0.56,
            "command_threshold": 0.1, "command_name": "twist",
            "sensor_name": "feet_ground_contact"},
  ),
}
```
Negative weight = penalty (most terms here besides tracking/gait are
penalties). **Tuning rewards is almost always the fastest lever for changing
learned behavior** — start by nudging existing weights before adding new
terms.

Custom reward function: same signature as observations, returns
`[num_envs]`; use a class + `ManagerTermBase` for anything needing
per-episode state.

## Terminations

`TerminationTermCfg(func, time_out: bool, params)`. `time_out=True` = a
**truncation** (hit the time limit — the agent should bootstrap value past
it, maps to Gym's `truncated`). `time_out=False` (default) = a genuine
**failure** (maps to `terminated`) — no value bootstrapping.

Built-ins in `mjlab.envs.mdp.terminations`: `time_out`, `bad_orientation`
(`limit_angle` param), `root_height_below_minimum`, `nan_detection`.

Repo example (`velocity_env_cfg.py`):
```python
terminations = {
  "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  "fell_over": TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
}
```
Go2's config adds a robot-specific one for illegal body-ground contact:
```python
cfg.terminations["illegal_contact"] = TerminationTermCfg(
  func=mdp.illegal_contact,
  params={"sensor_name": "nonfoot_ground_touch", "force_threshold": 10.0},
)
```

## Commands

Every command is a **class** inheriting `CommandTerm`, paired with a
`CommandTermCfg`. `resampling_time_range: (min, max)` seconds controls how
often a fresh goal is drawn; commands also resample unconditionally on
episode reset.

| Command | Use case |
|---|---|
| `UniformVelocityCommandCfg` | Planar `[v_x, v_y, ω_z]` commands. Supports standing envs (`rel_standing_envs`) and heading-mode control (`heading_command=True`, replaces raw yaw-rate with proportional heading control). Used by the velocity task. |
| `LiftingCommandCfg` | 3D target position for a manipulated object (manipulation tasks, not used by this repo). |
| `MotionCommandCfg` | Streams reference joint/body targets from a `.npz` mocap clip. Used by the tracking task — see [05-training-and-evaluation.md](05-training-and-evaluation.md#motion-imitation). |

Repo example, velocity task:
```python
commands = {
  "twist": UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.05,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi),
    ),
  )
}
```
`debug_vis=True` draws ghost/arrow visualization in the viewer.

Custom commands need `_resample_command(env_ids)`, `_update_command(env_ids)`
(called each step; `env_ids=None` means all envs, or specific IDs right
after their reset), `_update_metrics()`, and a `command` property. The
paired `CommandTermCfg` implements `build(env)` to construct the term.

## Events (and where domain randomization hooks in)

`EventTermCfg(func, mode, params, interval_range_s, is_global_time,
min_step_count_between_reset)`.

| Mode | Fires | Typical use |
|---|---|---|
| `"startup"` | once, at init | fixed per-env parameters (link mass, armature, friction) |
| `"reset"` | every episode reset | state init, per-episode DR |
| `"interval"` | every `interval_range_s` seconds | mid-episode perturbations (pushes), drifting params |
| `"step"` | every env step | continuous per-step effects |

Built-ins in `mjlab.envs.mdp.events`: `reset_scene_to_default`,
`reset_root_state_uniform`, `reset_joints_by_offset`,
`push_by_setting_velocity`, `apply_body_impulse`,
`reset_root_state_from_flat_patches`.

**Domain randomization** is just events registered with functions from
`mjlab.envs.mdp.dr`. Repo example (`velocity_env_cfg.py`):
```python
events = {
  "reset_base": EventTermCfg(
    func=mdp.reset_root_state_uniform, mode="reset",
    params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5),
                            "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {}},
  ),
  "push_robot": EventTermCfg(
    func=mdp.push_by_setting_velocity, mode="interval",
    interval_range_s=(5.0, 6.0),
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5),
                                "z": (-0.4, 0.4), "roll": (-0.52, 0.52),
                                "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}},
  ),
  "foot_friction": EventTermCfg(
    mode="startup", func=dr.geom_friction,
    params={"asset_cfg": SceneEntityCfg("robot", geom_names=()),
            "operation": "abs", "ranges": (0.3, 1.6),
            "shared_random": True},  # all foot geoms share one sampled value
  ),
  "encoder_bias": EventTermCfg(
    mode="startup", func=dr.encoder_bias,
    params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
  ),
}
```

### `dr` module reference

| Category | Functions |
|---|---|
| Geometry | `geom_friction`, `geom_size`, `geom_pos`, `geom_quat`, `geom_rgba`, `geom_matid` |
| Body | `body_mass`, `body_com_offset` (alias `body_ipos`), `body_pos`, `body_quat`, `pseudo_inertia` |
| Joint/DOF | `joint_damping`, `joint_armature`, `joint_friction`, `joint_stiffness`, `joint_limits`, `joint_default_pos` |
| Entity-level | `pd_gains`, `effort_limits`, `encoder_bias` |
| Contact pair | `pair_friction` (needs explicit `<contact><pair>` in XML; `isotropic=True` forces symmetric friction) |

`operation`: `"abs"` (default, set directly), `"scale"` (multiply
*original compile-time default*), `"add"` (offset from *original default*).
`"scale"`/`"add"` always reference the original default, so repeated calls
don't accumulate. `distribution`: `"uniform"` (default), `"log_uniform"`,
`"gaussian"` (range = mean, std), or a custom `dr.Distribution`.
`shared_random=True` makes every geom/body matched by one `asset_cfg` in one
env draw the *same* sampled value (independent per env, shared within env) —
used above so all 4 Go2 feet get identical friction.

**Gotchas** (from the FAQ, worth remembering before you randomize something
and get confused by the result):
- Randomizing `body_mass` alone breaks physical consistency (inertia doesn't
  scale with it) — use `pseudo_inertia` with `alpha_range` instead.
- All `*_quat` DR functions expect **radians**; passing degrees silently
  gives ~57× too much rotation.
- `dr.geom_friction` defaults to axis 0 (tangential) only — pass
  `axes=[0,1,2]` explicitly if you want torsional/rolling friction
  randomized too.
- `dr.geom_size` errors on non-primitive geoms (mesh/plane/heightfield).

## Curriculum

`CurriculumTermCfg(func, params)`. Called at every reset, receives `env` +
the resetting env IDs, inspects a performance signal, mutates env config
in-place. Return value is logged under `Curriculum/<term_name>`.

Built-ins: `terrain_levels_vel` (promotes/demotes an env's terrain row based
on distance traveled), `commands_vel` (widens command ranges as training
progresses), `reward_curriculum` (stage reward weights/params over time),
`termination_curriculum` (tighten termination thresholds over time).

Repo example:
```python
curriculum = {
  "terrain_levels": CurriculumTermCfg(func=mdp.terrain_levels_vel,
                                       params={"command_name": "twist"}),
  "command_vel": CurriculumTermCfg(
    func=mdp.commands_vel,
    params={"command_name": "twist",
            "velocity_stages": [
              {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5),
               "ang_vel_z": (-1.0, 1.0)},
              {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
            ]},
  ),
}
```
Note `step: 5000 * 24` — curriculum "steps" here are counted in
environment-steps, and `24` is `num_steps_per_env` from the PPO runner config
(rollout length), so this stage kicks in after 5000 PPO iterations' worth of
env steps. Play-mode configs typically clear `curriculum = {}` entirely (see
Go2's `play=True` branch) so evaluation always uses full difficulty/range.

## Metrics

`MetricsTermCfg(func)` — logs an arbitrary per-step scalar as an
episode-averaged training metric, without affecting the reward the policy
optimizes. Repo example: `"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)`.
Useful for tracking diagnostics (e.g. smoothness) you want in TensorBoard/
wandb but don't want to directly shape behavior.
