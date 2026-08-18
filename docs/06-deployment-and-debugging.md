# Viewers, debugging, sim2real deployment, and troubleshooting

Doc sources: mjlab's
[Viewers](https://mujocolab.github.io/mjlab/main/source/viewers.html),
[NaN Guard](https://mujocolab.github.io/mjlab/main/source/debugging/nan_guard.html),
[FAQ](https://mujocolab.github.io/mjlab/main/source/faq.html),
plus this repo's top-level `README.md` (§4, real deployment) and
`deploy/`/`simulate/` folders.

## Viewers

Two interactive viewers, sharing one `ViewerConfig` (set on
`ManagerBasedRlEnvCfg.viewer`) and the same simulation loop:

```bash
python scripts/play.py <task_id> --viewer native   # desktop window
python scripts/play.py <task_id> --viewer viser    # browser at localhost:8080
python scripts/play.py <task_id> --viewer auto      # default: native if a display is attached, else viser
python scripts/play.py <task_id> --agent zero        # no checkpoint needed, quick sanity check
```

**`ViewerConfig` fields**: `lookat` (default world origin), `distance`,
`elevation`, `azimuth`, `fovy`; `origin_type` — `WORLD` (free camera),
`ASSET_ROOT` (tracks entity root body, good for locomotion), `ASSET_BODY`
(tracks a specific body, good for close-ups — this repo uses it, targeting
`"base_link"`/pelvis-equivalent per robot); `enable_shadows`,
`enable_reflections`; `height`/`width` (offscreen render resolution);
`env_idx`; `entity_name`, `body_name`.

**Native MuJoCo viewer** — fastest, full visual fidelity, supports
click-drag body perturbation (test balance recovery without writing code).
Keys: `Space` pause/resume, `Enter` reset, `+`/`-` playback speed, `<`/`>`
cycle environments, `A` toggle all-envs view, `P` toggle live reward plots,
`R` toggle debug visualization (contact points, sensor rays, command
arrows, motion-tracking ghosts).

**Viser (browser) viewer** — works over SSH (no X forwarding needed),
auto-discovers `CameraSensor`s and shows their RGB/depth feeds in tabbed
panels (Rewards / Metrics / Camera Feeds / Groups). No interactive
perturbation support; domain-randomization visualization is body-pose-only
(doesn't reflect geom property changes like friction/size).

**Offscreen renderer** (`--video`) is hard-capped at 32 environments.

**Terrain tuning tool**: `python scripts/visualize_terrain.py` — a
standalone Viser app for live-tuning `TerrainGeneratorCfg` parameters (see
its slider hints in the script, e.g. `octaves`, `persistence`,
`step_height_range`) against `mjlab.terrains.config.ALL_TERRAINS_CFG`, with
a robot dropdown for scale reference. Much faster than editing a terrain cfg
blind, retraining, and eyeballing the result.

## NaN guard

Physics or policy divergence in a 4096-env parallel rollout is very hard to
debug from a crash message alone — the NaN guard exists to make it
reproducible.

```bash
python scripts/train.py <task_id> --enable-nan-guard True
```
or programmatically via `SimulationCfg.nan_guard: NanGuardCfg` —
`enabled` (default `False`), `buffer_size` (rolling state history kept,
default 100 steps), `output_dir` (default `/tmp/mjlab/nan_dumps`),
`max_envs_to_dump` (default 5, keeps dump size manageable).

**Mechanism**: captures `qpos`/`qvel`/actuator activations/mocap state
before each step; after each step, checks positions/velocities/
accelerations/sensor outputs for NaN/Inf; on first detection, dumps the
rolling buffer + a compiled `.mjb` model snapshot (timestamped, with a
"latest" symlink for convenience).

**Inspect a dump**: `viz-nan` (interactive viewer with step/env sliders and
an affected-environment info panel).

Typical symptom this is diagnosing: training crashing with
`RuntimeError: normal expects all elements of std >= 0.0` — that's usually
NaN/Inf propagating from physics into the policy's action distribution, not
an RL-side bug. Fixes: enable the guard to find *where* it started, and
consider adding an `nan_detection` termination term so affected
environments reset instead of poisoning the batch.

## FAQ / troubleshooting gotchas worth knowing up front

- **`device="cpu"` still touches the GPU.** To fully avoid claiming GPU
  memory (e.g. running on a shared machine), use
  `CUDA_VISIBLE_DEVICES="" python scripts/train.py ...` rather than a
  device flag.
- **Reproducibility is not exact.** MuJoCo Warp doesn't yet guarantee
  determinism across runs even with a fixed seed (tracked upstream as
  issue #562).
- **`<option>` in an entity's XML is silently ignored** once composed into
  a scene. Set physics options through `MujocoCfg` in Python.
- **Stale derived quantities**: `sim.forward()` syncs `mjData` from
  `qpos`/`qvel` once per step, before observations, after events — so event
  functions see state from the *previous* substep by design. Only call
  `forward()` manually if you write state and need to read a derived
  quantity back in the same function.
- **Contact sensors can miss brief contacts** when `decimation > 1`, since a
  contact can appear and vanish within a substep that never gets sampled.
  Set `ContactSensorCfg.history_length` equal to `decimation` and check the
  force history in your reward/termination logic rather than a single-step
  read.
- **Fixed-base robots stack at the origin** if you forget a reset event —
  add `reset_root_state_uniform` with an empty `pose_range` (still applies
  `env_origins`).
- **Native viewer geom limit**: MuJoCo's viewer buffer caps out around
  10,000 geoms — very large `num_envs` in native-viewer play mode may hit
  this.
- **This repo adds far more robots than upstream mjlab ships.** Upstream
  mjlab's own asset zoo (`mjlab.asset_zoo.robots`, referenced by
  `scripts/visualize_terrain.py`) only includes Go1 and G1 out of the box —
  everything else (Go2, A2, AS2, G1-23dof, H1_2, H2, R1) is this repo's
  addition under `src/assets/robots/`.
- **Only MJCF is natively supported.** URDF/USD assets need conversion
  first; MuJoCo Menagerie is a good source of pre-converted MJCF.

## Sim2real deployment

This is specific to `unitree_rl_mjlab` (not part of mjlab itself) — a C++
stack under `deploy/` that loads the ONNX policy exported during training
and runs it on (or against a simulated) real robot via Unitree's SDK.

### Prerequisites
```bash
# Unitree's DDS-based comms stack
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git
git clone https://github.com/unitreerobotics/unitree_sdk2.git
```

### Pipeline
1. **Power on** the robot in suspended state; wait for `zero-torque` mode.
2. **Enter debug mode**: hold `L2 + R2` on the controller (enables joint
   damping).
3. **Network**: connect PC↔robot via Ethernet, set your PC to
   `192.168.123.222` / netmask `255.255.255.0`. Find the interface name with
   `ifconfig`.
4. **Place the exported policy**: copy `policy.onnx` + `policy.onnx.data`
   (written automatically by this repo's RSL-RL runners during training —
   see [05-training-and-evaluation.md](05-training-and-evaluation.md)) into
   `deploy/robots/<robot>/config/policy/<task_type>/v0/exported/`.
5. **Build**:
   ```bash
   cd deploy/robots/<robot>
   mkdir build && cd build
   cmake .. && make
   ```
6. **Simulate before touching hardware** (strongly recommended — this repo
   vendors [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
   under `simulate/` for exactly this):
   ```bash
   cd simulate && mkdir build && cd build && cmake .. && make -j8
   ./simulate/build/unitree_mujoco     # needs a gamepad connected
   # in another terminal:
   cd deploy/robots/<robot>/build && ./<robot>_ctrl --network=lo
   ```
   Pick the robot model to simulate via `simulate/config`.
7. **Real robot**:
   ```bash
   cd deploy/robots/<robot>/build
   ./<robot>_ctrl --network=enp5s0    # your actual ethernet interface, from ifconfig
   ```

`deploy/` has a subfolder per supported robot (`a2`, `g1`, `g1_23dof`,
`go2`, `h1_2`, `r1`), each with its own `config/`, plus shared FSM/IsaacLab
compatibility headers under `deploy/include/`. If you need to trace exactly
how an ONNX policy's inputs map to real sensor readings (state estimation,
IMU frame conventions, joint ordering), that mapping lives in this C++ code
and in the ONNX metadata attached by `attach_metadata_to_onnx`/
`get_base_metadata` at export time (`mjlab.rl.exporter_utils`) — worth
diffing against the observation term list in the task's `env_cfgs.py` if
sim and real behavior disagree.

## Where to go next

- Modifying reward/observation/termination behavior →
  [03-manager-configuration.md](03-manager-configuration.md)
- Adding a robot or new task variant →
  [04-repo-structure-and-tasks.md](04-repo-structure-and-tasks.md)
- CLI flags for training/eval →
  [05-training-and-evaluation.md](05-training-and-evaluation.md)
