# Architecture and core concepts

Source: mjlab docs — [Architecture Overview](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html),
[Entity](https://mujocolab.github.io/mjlab/main/source/entity/index.html),
[Actuators](https://mujocolab.github.io/mjlab/main/source/actuators.html),
[Sensors](https://mujocolab.github.io/mjlab/main/source/sensors/index.html),
[Scene](https://mujocolab.github.io/mjlab/main/source/scene.html),
[Terrain](https://mujocolab.github.io/mjlab/main/source/terrain.html).

## Two-layer design

mjlab splits cleanly into:

1. **Simulation layer** — physics and robots. Independent of any RL
   concepts. Built from `Entity`, `Actuators`, `Sensors`, `Scene`, on top of
   MuJoCo Warp.
2. **Manager layer** — the RL problem definition, built *on top of* the
   simulation layer via `ManagerBasedRlEnv` / `ManagerBasedRlEnvCfg`. This is
   where observations/actions/rewards/etc. live (see
   [03-manager-configuration.md](03-manager-configuration.md)).

This separation is why you can freely restructure a task's reward/observation
terms without touching how the robot's physics model is built, and vice
versa.

## Simulation layer

### Scene pipeline

1. Each `Entity` (robot, object, terrain) loads from an **MJCF** file via
   `mujoco.MjSpec.from_file(...)`.
2. Python dataclasses (`EntityCfg` and friends) programmatically extend that
   spec — actuators, collision rules, sensors, lights, materials — *without
   editing the original XML*.
3. `SceneCfg` composes all entities' specs into one root `MjSpec`. Each
   entity's internal elements get **prefixed with the entity's name** to
   avoid collisions: a robot named `"robot"` turns `base_link` into
   `robot/base_link`, `joint0` into `robot/joint0`. Terrain is the exception —
   it attaches with **no prefix**, living in the global namespace.
4. The composed spec compiles once to an `MjModel` (CPU), then uploads to GPU
   via **MuJoCo Warp**.

### MuJoCo Warp backend

- Preserves the familiar `MjModel`/`MjData` paradigm but adds a **world**
  dimension: one `MjData` holds `N` parallel simulation instances
  (`num_envs`).
- **CUDA graphs** capture the simulation step sequence once, eliminating
  per-frame CPU dispatch overhead — this is most of why mjlab is fast at
  scale.
- Currently **all environments share the same compiled `MjModel`** (same
  meshes/geometry/kinematic tree). Environments are independent worlds with
  separate state but cannot physically interact with each other.
  `VariantEntityCfg` is the escape hatch for heterogeneous meshes across
  worlds (see below).

### Entity

An `Entity` is "a physical object in the simulation: a robot, a manipulated
object, or a fixed fixture like a table," classified along two axes:
fixed vs. floating base, articulated vs. non-articulated.

`EntityCfg` fields:

| Field | Purpose |
|---|---|
| `spec_fn` | Callable returning an `mujoco.MjSpec` — usually `MjSpec.from_file(...)` plus optional programmatic edits (add bodies, change joint limits, swap materials, or build a model entirely in code). |
| `init_state` | `EntityCfg.InitialStateCfg` — root pose/velocity + joint positions/velocities as regex-keyed dicts. Patterns match in order; later entries override earlier ones. |
| `articulation` | `EntityArticulationInfoCfg` — actuator configs + `soft_joint_pos_limit_factor` (default 1.0), which shrinks joint ranges *for soft-limit penalty computation only*, without touching the real MuJoCo limits. |
| `collisions`, `lights`, `cameras`, `textures`, `materials`, `geoms` | Optional "spec editor" tuples applied to the `MjSpec` before compilation. |

Spec editor distinction worth remembering: `GeomCfg` is a **sparse patch**
(only touches attributes you set), while `CollisionCfg` is a **policy**
(required fields always written, and any geom *not* matched by the collision
config's regex is disabled by default). This is why the Go2 config below
needs an explicit `FULL_COLLISION` / `FEET_ONLY_COLLISION` choice — leaving
it out silently disables all collisions.

**Runtime access:**
- `entity.data` (`EntityData`) — kinematic state, actuator forces, derived
  quantities as PyTorch tensors shaped `(num_envs, ...)`.
- `env.scene[entity_name]` — look up an entity from within an `mdp` function.
- `env.sim.data` / `env.sim.model` — raw MuJoCo arrays, indexed by *global*
  MuJoCo IDs (not per-entity).
- `find_bodies()`, `find_joints()`, `find_geoms()`, `find_sites()`,
  `find_tendons()` — regex → indices/names lookups on an entity.

**`SceneEntityCfg`** is how manager terms *reference* part of an entity —
e.g. `SceneEntityCfg("robot", joint_names=(".*hip.*",))`. Regex patterns
resolve to integer indices once at env initialization, so there's no
per-step regex cost.

**`VariantEntityCfg`** enables heterogeneous worlds — different mesh assets
per parallel environment, each world getting a variant proportional to
configured weights; mesh-dependent compiled constants are stored per-world.

### Actuators

Configured via `EntityArticulationInfoCfg.actuators`, a tuple of actuator
configs matched to joints by `target_names_expr` regex.

**Built-in** (wrap native MuJoCo elements, implicit integration → best
stability):
- `BuiltinPositionActuatorCfg` — PD via `<position>`
- `BuiltinVelocityActuatorCfg` — via `<velocity>`
- `BuiltinMotorActuatorCfg` — direct torque via `<motor>`
- `BuiltinPdActuatorCfg` — closes on both position and velocity target
- `BuiltinDcMotorActuatorCfg`, `BuiltinMuscleActuatorCfg`

**Explicit** (torque computed in Python, forwarded through a motor
passthrough — more flexible, less numerically robust):
- `IdealPdActuatorCfg` — `tau = Kp * pos_error + Kd * vel_error`; the
  recommended base class if you're writing a custom explicit actuator
- `DcMotorActuatorCfg` — adds velocity-dependent torque saturation
- `LearnedMlpActuatorCfg` — NN-predicted torque for capturing real motor
  dynamics

**`XmlActuatorCfg`** — wraps an actuator already defined in the robot's XML
(matched by joint name regex) instead of creating one from Python. Used in
the cartpole tutorial.

Common fields across actuator configs: `stiffness` (Kp), `damping` (Kd),
`effort_limit`, `armature` (reflected rotor inertia), `frictionloss`,
`target_names_expr`. mjlab's convention is **one config per actuator group**
(e.g. hip/thigh/calf), not per individual joint, mirroring how real hardware
is specified. See Go2's actuator config in
[04-repo-structure-and-tasks.md](04-repo-structure-and-tasks.md) for a
concrete example.

**Delay modeling** — any actuator config supports simulated command latency:
```python
IdealPdActuatorCfg(
    delay_min_lag=2,      # min physics steps of lag
    delay_max_lag=5,      # max physics steps of lag
    delay_hold_prob=0.3,  # probability of holding the current lag value
)
```
Delays are quantized to physics timesteps.

**Hardware parameter helpers**: `reflected_inertia_from_two_stage_planetary()`
computes `armature` from gearbox specs; PD gains derive from natural
frequency/damping ratio via `Kp = J·ωn²`, `Kd = 2·ζ·J·ωn`.

### Sensors

Sensors live on **`SceneCfg`**, not on `EntityCfg` — even though they can
reference specific entity elements (a site/body/joint on `"robot"`).

| Type | What it does |
|---|---|
| `BuiltinSensorCfg` | Wraps native MuJoCo sensors (site: accelerometer/velocimeter/gyro/force/torque; joint: jointpos/jointvel/jointlimit*/jointactuatorfrc; frame: framepos/framequat/...). Sensors already defined in an entity's XML are auto-discovered and prefixed. |
| `ContactSensorCfg` | Filters MuJoCo's flat contact list into structured batched tensors via `primary`/`secondary` regex `ContactMatch`es (mode `"geom"`/`"body"`/`"subtree"`). `reduce`: `"none"` (fastest, non-deterministic), `"mindist"`, `"maxforce"`, `"netforce"`. Supports `track_air_time=True` for gait rewards (`compute_first_contact()`, `compute_first_air()`) and a rolling `history_length` buffer. |
| `RayCastSensorCfg` | GPU raycasting — terrain height scans (`GridPatternCfg`) or pinhole-camera-style depth patterns. |
| `CameraSensorCfg` | RGB/depth rendering from a MuJoCo camera. |

Runtime access: `env.scene["sensor_name"].data`. Sensors cache per-step —
multiple reads in the same step cost one computation.

Custom sensors subclass `Sensor[T]` and override `edit_spec`, `initialize`,
`update`, `reset`, `_compute_data`.

### Scene

`SceneCfg` fields: `num_envs`, `env_spacing` (or `extent`, seen in this
repo's `velocity_env_cfg.py` — spacing depends on terrain type), `terrain`
(`TerrainEntityCfg`), `entities` (`dict[str, EntityCfg]`), `sensors` (tuple
of sensor configs), `spec_fn` (optional custom MJCF callback for global
scene edits).

Environments are placed via `scene.env_origins` — a regular grid for flat
terrain, or terrain-patch centers for procedural terrain.

### Terrain

`TerrainEntityCfg.terrain_type`:
- `"plane"` — a single flat MuJoCo plane geom, no procedural geometry. Grid
  spacing is `ceil(sqrt(num_envs))` square, spaced by `env_spacing`.
- `"generator"` — procedural, driven by `TerrainGeneratorCfg`
  (`size`, `num_rows`, `num_cols`, `border_width`, `curriculum: bool`,
  `sub_terrains: dict[str, SubTerrainCfg]`, each with a `proportion` weight).

**Curriculum mode**: difficulty is a function of row —
`difficulty = lower + (upper - lower) * row / max(num_rows - 1, 1)`.
Row 0 = easiest, last row = hardest. Environments are promoted/demoted
between rows automatically based on a performance metric (see
`terrain_levels_vel` curriculum term in
[03-manager-configuration.md](03-manager-configuration.md)).

**Presets** in `mjlab.terrains.config`, all customizable via
`dataclasses.replace(...)`:
- `ROUGH_TERRAINS_CFG` — 10×20 random-mode grid, 7 terrain types, general
  locomotion training (this is what `unitree_rl_mjlab`'s velocity task uses
  by default — see `velocity_env_cfg.py`).
- `STAIRS_TERRAINS_CFG` — 10-row curriculum focused on stairs.
- `ALL_TERRAINS_CFG` — 10-row random-mode grid, every terrain type.

Sub-terrains are either **primitive** (box-geom based: pyramid stairs,
stepping stones, random/tilted grids) or **heightfield-based** (Perlin
noise, waves — smooth terrain box geoms can't represent).

`scripts/visualize_terrain.py` in this repo is a Viser-based interactive
tool for browsing/tuning these terrain generators live — run it and drag
sliders rather than guessing at parameters blind.

## Manager layer: `ManagerBasedRlEnv`

Built directly on Isaac Lab's manager-based design. You compose small,
reusable **terms** — plain functions, or `ManagerTermBase` subclasses when
you need per-episode state via a `reset()` hook — and register them by name
in a dict on `ManagerBasedRlEnvCfg`.

### The 8 managers

| Manager | Role |
|---|---|
| `ObservationManager` | Assembles named observation groups (e.g. `"actor"`, `"critic"`) from terms, with noise/scale/clip/history/delay processing. |
| `ActionManager` | Splits the policy's flat action tensor across registered action terms, routes each slice to an entity's actuators. |
| `RewardManager` | Weighted sum of reward terms → one scalar per env per step. |
| `TerminationManager` | Per-env boolean "should this episode end", distinguishing true failures from timeouts/truncations. |
| `EventManager` | Fires lifecycle terms (`startup`/`reset`/`interval`/`step`) — this is where domain randomization hooks in. |
| `CommandManager` | Generates goal signals (velocity targets, motion-tracking targets, ...), handles resampling. |
| `CurriculumManager` | Adjusts task difficulty/parameters based on policy performance, evaluated at each reset. |
| `MetricsManager` | Logs arbitrary per-step scalars as episode-averaged training metrics (distinct from reward — doesn't affect the policy). |

### Configuration: `ManagerBasedRlEnvCfg`

Top-level dataclass fields:

| Field | Meaning |
|---|---|
| `scene: SceneCfg` | Required — terrain, entities, sensors, `num_envs`. |
| `sim: SimulationCfg` | Physics params, wraps `MujocoCfg` (timestep, `iterations`, `ls_iterations`, `ccd_iterations`, `nconmax`, `njmax`, `contact_sensor_maxmatch`, `nan_guard`, ...). |
| `decimation: int` | Physics steps per policy step. |
| `episode_length_s: float` | Episode duration in seconds. |
| `is_finite_horizon: bool` | Whether the time limit is a real task boundary (default `False`). |
| `scale_rewards_by_dt: bool` | Multiply each reward term by step duration before summing (default `True`) — keeps returns comparable across different physics rates. |
| `observations`, `actions`, `rewards`, `terminations`, `events`, `commands`, `curriculum`, `metrics` | `dict[str, <TermCfg>]` — one dict per manager. |
| `seed`, `viewer` | RNG seed, camera config (`ViewerConfig`). |

**Timing relationship**: `step_dt = sim.mujoco.timestep * decimation`. Max
policy steps per episode = `ceil(episode_length_s / step_dt)`. This repo's
velocity task uses `timestep=0.005` (200 Hz physics) with `decimation=4` →
50 Hz policy, `episode_length_s=20.0` → 1000 policy steps/episode.

**Registration**: tasks register with `register_mjlab_task(task_id=..., env_cfg=..., play_env_cfg=..., rl_cfg=..., runner_cls=...)`, importable at module import time so `scripts/list_envs.py` / `train.py` / `play.py` can discover them (`mjlab.tasks.registry.list_tasks/load_env_cfg/load_rl_cfg/load_runner_cls`). See
[04-repo-structure-and-tasks.md](04-repo-structure-and-tasks.md) for exactly
how this repo wires that up per-robot.

### Environment lifecycle

1. **Build** — scene composition → `MjModel` compile → GPU upload → CUDA
   graph capture.
2. **Initialize** — managers construct from configs; regex patterns resolve
   to indices; domain-randomized fields expand from shared to per-world GPU
   storage.
3. **Reset** — `EventManager` fires `mode="reset"` terms (incl. per-episode
   DR); commands resample; observation buffers clear.
4. **Step** — `ActionManager` processes policy output → physics advances
   `decimation` substeps → termination/reward computed → interval events
   fire → terminated envs reset → sensors update → `ObservationManager`
   assembles the next observation.

Design takeaway: simulation stays fully decoupled from RL logic; everything
RL-specific is a **term** you can add/remove/swap independently. This is the
main lever you'll pull when modifying a task — see the next two docs.
