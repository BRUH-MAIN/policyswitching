# Repo structure, task registration, and how to modify/add things

## Layout of `unitree_rl_mjlab/`

```
unitree_rl_mjlab/
├── setup.py                  # pip package "unitree_rl_mjlab"; deps: mjlab==1.2.0, mujoco-warp==3.5.0
├── scripts/
│   ├── train.py               # RSL-RL training entry point
│   ├── play.py                # checkpoint playback / viewer
│   ├── list_envs.py           # print all registered task IDs
│   ├── csv_to_npz.py          # mocap CSV -> mjlab motion .npz converter
│   └── visualize_terrain.py   # interactive Viser terrain-generator browser
├── src/
│   ├── assets/
│   │   ├── robots/            # one subpackage per robot: MJCF + actuator/collision cfg
│   │   │   ├── unitree_go2/  unitree_a2/  unitree_as2/
│   │   │   ├── unitree_g1/   unitree_h1_2/  unitree_h2/  unitree_r1/
│   │   └── motions/            # reference mocap clips (csv) for motion imitation, per robot
│   └── tasks/
│       ├── velocity/           # velocity-tracking locomotion task
│       │   ├── velocity_env_cfg.py   # make_velocity_env_cfg(): robot-agnostic base config
│       │   ├── mdp/                   # task-specific observation/reward/termination/curriculum fns
│       │   ├── rl/runner.py           # VelocityOnPolicyRunner (adds ONNX export on save)
│       │   └── config/<robot>/        # per-robot: env_cfgs.py, rl_cfg.py, __init__.py (registration)
│       └── tracking/           # motion-imitation ("mimic") task
│           ├── tracking_env_cfg.py   # make_tracking_env_cfg(): robot-agnostic base config
│           ├── mdp/                   # commands.py (MotionCommandCfg), rewards, observations, ...
│           ├── rl/runner.py
│           └── config/<robot>/        # currently: g1, g1_23dof
├── deploy/                    # C++ sim2real deployment stack, per-robot build dirs
├── simulate/                  # vendored unitree_mujoco simulator (for deploy-side testing)
└── logs/rsl_rl/<experiment_name>/<timestamp>/   # training output: checkpoints, params, onnx export
```

`src/tasks/__init__.py` auto-imports every task subpackage
(`mjlab.utils.lab_api.tasks.importer.import_packages`), which is what
actually triggers each `config/<robot>/__init__.py`'s
`register_mjlab_task(...)` calls. **This means: to add a new robot/task, you
just need a new subpackage under the right `config/` folder — nothing else
needs to import it manually.**

## The config-composition pattern

Every task follows the same three-level structure, which is the main thing
to internalize before modifying anything:

1. **Task-level base config** (`velocity_env_cfg.py` / `tracking_env_cfg.py`)
   — a `make_*_env_cfg()` factory returning a robot-agnostic
   `ManagerBasedRlEnvCfg` with all managers populated, but robot-specific
   fields left as **empty placeholders** (`site_names=()`, `body_names=()`,
   `std_standing={}`, etc.) with a `# Set per-robot.` comment marking them.
2. **Per-robot env config** (`config/<robot>/env_cfgs.py`) — calls the base
   factory, then mutates the returned dataclass in place to fill in
   robot-specific names/regexes/weights and swap in the robot's `EntityCfg`.
   Usually exposes two functions: `unitree_<robot>_rough_env_cfg(play=False)`
   and `unitree_<robot>_flat_env_cfg(play=False)` (flat calls rough and then
   strips terrain-related bits).
3. **Registration** (`config/<robot>/__init__.py`) — calls
   `register_mjlab_task(task_id=..., env_cfg=..., play_env_cfg=..., rl_cfg=..., runner_cls=...)`
   once per task variant.

Because mutation happens on **already-constructed dataclass instances**
(not subclassing), every robot config is very literally "take the shared
recipe, patch in these specific values" — reading `config/go2/env_cfgs.py`
top-to-bottom pretty much tells you every axis that varies by robot: sensor
frame names, foot geom/site names, body names for orientation/COM
randomization, per-robot posture std-devs, gait phase offsets, terrain
tuning, sim tolerances (`ccd_iterations`, `contact_sensor_maxmatch`,
`njmax`, `nconmax`).

### Worked example: Go2 velocity task

`src/tasks/velocity/config/go2/env_cfgs.py` (see full listing already read
in this session) does, in order:

```python
cfg = make_velocity_env_cfg()                      # 1. shared base

cfg.sim.mujoco.ccd_iterations = 500                 # 2. sim tolerance tuning
cfg.sim.contact_sensor_maxmatch = 500
cfg.scene.entities = {"robot": get_go2_robot_cfg()} # 3. plug in the robot asset

for sensor in cfg.scene.sensors or ():               # 4. point the raycast sensor at Go2's base body
    if sensor.name == "terrain_scan":
        sensor.frame.name = "base_link"

feet_ground_cfg = ContactSensorCfg(...)              # 5. add robot-specific contact sensors
cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, nonfoot_ground_cfg)

cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names
cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
cfg.rewards["pose"].params["std_standing"] = {...}   # 6. fill in the "Set per-robot" placeholders
...
cfg.terminations["illegal_contact"] = TerminationTermCfg(...)  # 7. add a robot-specific term

if play:                                              # 8. play-mode overrides
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    ...
```

`unitree_go2_flat_env_cfg()` then calls `unitree_go2_rough_env_cfg()` and
strips terrain-generator-specific bits (raycast sensor, `height_scan`
observation, `terrain_levels` curriculum term) to get the flat-ground
variant. This "flat = rough minus terrain stuff" pattern is used for every
robot's velocity task.

`config/go2/rl_cfg.py` defines the PPO/network hyperparameters
independently (see [05-training-and-evaluation.md](05-training-and-evaluation.md)).

`config/go2/__init__.py` registers two task IDs, `Unitree-Go2-Rough` and
`Unitree-Go2-Flat`, both using `VelocityOnPolicyRunner` (adds ONNX export on
checkpoint save — needed for the sim2real deploy step).

## How to modify an existing task

- **Change reward shaping / observation noise / termination thresholds for
  one robot only** → edit that robot's `config/<robot>/env_cfgs.py`.
- **Change something shared across all robots of a task type** (e.g. add a
  new observation term to every velocity task) → edit
  `velocity_env_cfg.py`'s `make_velocity_env_cfg()`; every robot config
  picks it up automatically next time it calls the factory. If the new term
  needs a robot-specific parameter, initialize it as an empty placeholder
  (`()`/`{}`) with a `# Set per-robot.` comment and fill it in in each
  `config/<robot>/env_cfgs.py` (or accept the same default everywhere).
- **Change PPO hyperparameters / network size** → edit that robot's
  `config/<robot>/rl_cfg.py`, or override from the CLI (see
  [05-training-and-evaluation.md](05-training-and-evaluation.md)) for a
  one-off experiment without touching the file.
- **Add a brand-new `mdp` function** (custom reward/observation/termination/
  curriculum) → add it to `src/tasks/<task>/mdp/*.py`, export it from
  `mdp/__init__.py`, then reference it via `func=mdp.my_new_term` in the
  relevant `TermCfg`.

## How to add a new robot

Using `src/assets/robots/unitree_go2/` as the template:

1. **Add the MJCF.** Create `src/assets/robots/unitree_<name>/xmls/<name>.xml`
   plus a `xmls/assets/` folder for mesh files (STL/OBJ). MuJoCo Menagerie is
   a good source of ready-made MJCF for common robots if you're not
   converting from URDF/CAD yourself. Note from mjlab's FAQ: **`<option>`
   elements in entity XML are silently dropped during scene composition** —
   set physics options (timestep, solver, etc.) through `MujocoCfg` in
   Python instead, not in the robot's own XML.
2. **Write `<name>_constants.py`**, following `go2_constants.py`'s shape:
   - `get_assets(meshdir)` / `get_spec()` — load the MJCF + mesh assets via
     `mujoco.MjSpec.from_file` and `update_assets`.
   - One `Builtin*ActuatorCfg` per actuator group (hip/thigh/calf, or
     whatever your robot's joint groups are), with `stiffness`, `damping`,
     `effort_limit`, `armature` sourced from the datasheet (use
     `reflected_inertia_from_two_stage_planetary()` / `ElectricActuator` if
     you only have gearbox/motor specs, not a directly-measured armature).
   - `INIT_STATE = EntityCfg.InitialStateCfg(pos=..., joint_pos={...regex: value...}, joint_vel={".*": 0.0})`
     for a sane default standing pose.
   - A `CollisionCfg` — decide feet-only vs. full-body collision, condim/
     friction/solimp per geom group. **Remember: `CollisionCfg` is a
     policy** — any geom whose name doesn't match one of its
     `geom_names_expr` patterns gets collisions disabled by default, so a
     too-narrow regex here silently makes parts of the robot ghost through
     the floor.
   - `get_<name>_robot_cfg() -> EntityCfg` — assembles `init_state`,
     `collisions`, `spec_fn`, `articulation` into one `EntityCfg`. Return a
     **fresh instance every call** (as Go2 does) to avoid config-sharing
     mutation bugs across multiple task registrations.
   - Optionally a `if __name__ == "__main__":` block that opens the robot
     alone in `mujoco.viewer` — useful for a first visual sanity check
     before wiring it into a full task (`python -m src.assets.robots.unitree_<name>.<name>_constants`).
3. **Register it** in `src/assets/robots/__init__.py`
   (`from .unitree_<name>.<name>_constants import get_<name>_robot_cfg as get_<name>_robot_cfg`).
4. **Create a task config** under `src/tasks/velocity/config/<name>/`
   (copy `config/go2/` as a starting point): `env_cfgs.py` calling
   `make_velocity_env_cfg()` and filling in every `# Set per-robot.`
   placeholder for your robot's joint/body/site/geom names, `__init__.py`
   registering `Unitree-<Name>-Rough` / `Unitree-<Name>-Flat`, `rl_cfg.py`
   with a starting PPO config (copying an existing robot's is a reasonable
   default; humanoids vs. quadrupeds in this repo don't differ much here).
5. **Sanity-check before training**: `python scripts/list_envs.py` should
   show your new task IDs; `python scripts/play.py Unitree-<Name>-Flat --agent zero`
   opens the viewer with a zero-action policy so you can check the robot
   spawns correctly, doesn't clip through the floor, and joints move in
   sane directions before spending GPU time training it.

## How to add a brand-new task type (not velocity/tracking)

Follow the mjlab cartpole tutorial's structure
(https://mujocolab.github.io/mjlab/main/source/tutorials/cartpole.html) as
the minimal reference — an XML + a `*_env_cfg.py` defining
`observations`/`actions`/`rewards`/`terminations`/`events`/`commands`
dicts, assembled into a `ManagerBasedRlEnvCfg`, registered via
`register_mjlab_task`. In this repo, mirror the `velocity/`/`tracking/`
folder shape (`<task>_env_cfg.py` + `mdp/` + `rl/` + `config/<robot>/`) so
it's discovered the same way via `src/tasks/__init__.py`'s auto-import.

## Full list of registered task IDs (as of this checkout)

**Velocity tracking** (`src/tasks/velocity/config/`) — each robot has
`-Rough` (procedural terrain + terrain curriculum) and `-Flat` variants:

| Robot | Rough | Flat |
|---|---|---|
| Go2 (quadruped) | `Unitree-Go2-Rough` | `Unitree-Go2-Flat` |
| A2 (quadruped) | `Unitree-A2-Rough` | `Unitree-A2-Flat` |
| AS2 (quadruped) | `Unitree-As2-Rough` | `Unitree-As2-Flat` |
| G1 (humanoid) | `Unitree-G1-Rough` | `Unitree-G1-Flat` |
| G1 23-DoF (humanoid) | `Unitree-G1-23Dof-Rough` | `Unitree-G1-23Dof-Flat` |
| H1_2 (humanoid) | `Unitree-H1_2-Rough` | `Unitree-H1_2-Flat` |
| H2 (humanoid) | `Unitree-H2-Rough` | `Unitree-H2-Flat` |
| R1 (humanoid) | `Unitree-R1-Rough` | `Unitree-R1-Flat` |

**Motion imitation / tracking** (`src/tasks/tracking/config/`) — currently
G1 only:

| Robot | With state estimation | No state estimation |
|---|---|---|
| G1 | `Unitree-G1-Tracking` | `Unitree-G1-Tracking-No-State-Estimation` |
| G1 23-DoF | `Unitree-G1-23Dof-Tracking` | `Unitree-G1-23Dof-Tracking-No-State-Estimation` |

("No-State-Estimation" variants presumably drop privileged/estimated state
from the observation set to better match what's available on real hardware
at deploy time — check `config/g1/env_cfgs.py` for the exact observation
diff between the two if you need to know precisely.)

Run `python scripts/list_envs.py [keyword]` at any time to get this list
live from the registry rather than trusting this table, since it will drift
as robots/tasks are added.
