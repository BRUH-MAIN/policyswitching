# Training and evaluation

Doc sources: mjlab's
[Training with RSL-RL](https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html),
[Motion Imitation](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html),
[Distributed Training](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html),
plus this repo's `scripts/train.py`, `scripts/play.py`, `scripts/csv_to_npz.py`.

## `train.py` walkthrough

```bash
python scripts/train.py <task_id> [--env.<path> ...] [--agent.<path> ...] [other TrainConfig flags]
```

`TrainConfig` (a frozen dataclass parsed by [tyro](https://brentyi.github.io/tyro/))
wraps:
- `env: ManagerBasedRlEnvCfg` — defaulted from `load_env_cfg(task_id)`
- `agent: RslRlBaseRunnerCfg` — defaulted from `load_rl_cfg(task_id)`
- `motion_file: str | None` — required for tracking tasks
- `video: bool`, `video_length`, `video_interval` — periodic training video capture
- `enable_nan_guard: bool`
- `gpu_ids: list[int] | "all" | None` — defaults to `[0]`

Because tyro exposes **every nested dataclass field** as a flag, you can
override *anything* in the environment or agent config from the command
line without editing Python — this is the main way you'll iterate on
hyperparameters. Field names convert `snake_case` → `--kebab-case`, nested
fields use dots:

```bash
python scripts/train.py Unitree-G1-Flat \
  --env.scene.num-envs 4096 \
  --env.sim.mujoco.timestep 0.004 \
  --agent.algorithm.learning-rate 3e-4 \
  --agent.max-iterations 20000
```

Booleans need explicit values (`--agent.resume True`, not just
`--agent.resume`).

**What happens on a run**: picks `device` from `CUDA_VISIBLE_DEVICES`
(falls back to `"cpu"` if unset/empty), constructs `ManagerBasedRlEnv`,
wraps it in `RslRlVecEnvWrapper`, builds the runner (this repo's
`VelocityOnPolicyRunner` or `MjlabOnPolicyRunner` for tracking — chosen via
each task's `runner_cls` at registration), optionally resumes from a
checkpoint, dumps `env.yaml`/`agent.yaml` under
`logs/rsl_rl/<experiment_name>/<timestamp>/params/`, then calls
`runner.learn(num_learning_iterations=cfg.agent.max_iterations, ...)`.

### Multi-GPU

```bash
python scripts/train.py Unitree-G1-Flat --gpu-ids 0 1 --env.scene.num-envs=4096
```
More than one GPU ID switches from a direct in-process run to
[`torchrunx`](https://github.com/apple/torchrunx)-launched workers (one
process per GPU, `backend=None` lets rsl_rl own process-group init). Seeds
are diversified per rank (`agent.seed + local_rank`). `--gpu-ids all` uses
every visible GPU. `torchrunx` logs land under
`logs/rsl_rl/<experiment_name>/<timestamp>/torchrunx/` by default
(override with `--torchrunx-log-dir`).

### Checkpoints and resuming

Runs write to `logs/rsl_rl/<agent.experiment_name>/<YYYY-MM-DD_HH-MM-SS>[_<run_name>]/`.
`experiment_name` comes from the robot's `rl_cfg.py` (e.g. `"go2_velocity"`
for Go2's velocity task — this is also the folder `play.py` looks under).

```bash
# resume the most recent checkpoint for this task's experiment_name
python scripts/train.py Unitree-G1-Flat --agent.resume True

# narrow the search
python scripts/train.py Unitree-G1-Flat --agent.resume True \
  --agent.load-run "2026-08-.*" --agent.load-checkpoint "model_5000.*"

# W&B-based resume
python scripts/train.py Unitree-G1-Flat --agent.resume True --wandb-run-path <entity>/<project>/<run-id>
```
`--agent.max-iterations` sets the number of *additional* iterations to run
past the checkpoint.

### NaN guard

```bash
python scripts/train.py Unitree-G1-Flat --enable-nan-guard True
```
See [06-deployment-and-debugging.md](06-deployment-and-debugging.md) for
what this captures and how to inspect a dump.

## RSL-RL configuration (`RslRlOnPolicyRunnerCfg`)

Set per-robot in `config/<robot>/rl_cfg.py`. Go2's, as a concrete reference:

```python
RslRlOnPolicyRunnerCfg(
  actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu",
                       obs_normalization=True,
                       distribution_cfg={"class_name": "GaussianDistribution",
                                         "init_std": 1.0, "std_type": "scalar"}),
  critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu",
                        obs_normalization=True),
  algorithm=RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
    entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
    learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
    desired_kl=0.01, max_grad_norm=1.0,
  ),
  experiment_name="go2_velocity",
  save_interval=100,
  num_steps_per_env=24,     # PPO rollout length (per env) before an update
  max_iterations=10001,
)
```
Actor and critic have **independent** architectures (asymmetric actor-critic
is common here since the critic observation group carries privileged info —
see [03-manager-configuration.md](03-manager-configuration.md#observations)).
`schedule="adaptive"` auto-adjusts LR to hit `desired_kl` each update.
`save_interval` is in PPO iterations, not env steps.

To try a different network size or PPO hyperparameter for a one-off
experiment, prefer CLI overrides over editing the file — e.g.
`--agent.actor.hidden-dims "(256, 128, 64)"`,
`--agent.algorithm.entropy-coef 0.02`.

## `play.py` walkthrough

```bash
python scripts/play.py <task_id> --checkpoint_file=<path/to/model_N.pt> [--viewer native|viser|auto] [--video]
```

Loads the task's `play_env_cfg` (not the training config — typically
disables push events, sets `episode_length_s` to effectively-infinite,
disables observation corruption, clears curriculum, widens/re-randomizes
terrain — see the Go2 `play=True` branch in
[04-repo-structure-and-tasks.md](04-repo-structure-and-tasks.md)). Options:

- `--agent trained|zero|random` — `trained` needs `--checkpoint-file` (or
  `--wandb-run-path`); `zero`/`random` need neither and are the fastest way
  to sanity-check a new robot/task config visually before training.
- `--no-terminations` — disables all termination conditions; handy for
  scrubbing through a motion-imitation clip without the episode ending
  early on a tracking-error termination.
- `--num-envs N` — override env count for playback (default from config).
- `--viewer auto|native|viser` — `auto` picks `native` if a display is
  attached, else `viser` (browser at `localhost:8080` — useful over SSH).
- `--video --video-length N --video-height H --video-width W` — offscreen
  render to `logs/rsl_rl/<experiment>/<run>/videos/play/`.

See [06-deployment-and-debugging.md](06-deployment-and-debugging.md) for
viewer keybindings.

**During training**, checkpoints saved by this repo's runners
(`VelocityOnPolicyRunner` / tracking's runner) also export
`policy.onnx` + `policy.onnx.data` alongside each `model_<iter>.pt` — that's
the artifact the C++ deploy stack actually loads (see
[06-deployment-and-debugging.md](06-deployment-and-debugging.md#sim2real-deployment)).

## Motion imitation

The tracking task (`src/tasks/tracking/`) trains a policy to reproduce a
reference mocap clip rather than track a velocity command. This repo's
implementation is a re-port of
[BeyondMimic](https://beyondmimic.github.io/)/`whole_body_tracking`
(see the header comment in `tracking_env_cfg.py` for the exact commit
pinned).

### 1. Convert a CSV mocap clip to mjlab's `.npz` format

```bash
python scripts/csv_to_npz.py \
  --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz \
  --input-fps 30 --output-fps 50 \
  --robot g1     # or g1_23dof
```
`csv_to_npz.py` (`scripts/csv_to_npz.py:312`, function `main`) accepts
`robot`, `input_file`, `output_name`, `input_fps` (default 30), `output_fps`
(default 50), `device` (default `cuda:0`), `render` (bool, save a debug
video), `line_range` (optional `(start, end)` line subset). Internally it
replays the CSV through MuJoCo Warp physics (using each robot's own
`joint_names` ordering — see the `robot == "g1"` branch for the 29-DoF list)
to compute forward kinematics and produces an `.npz` with
`joint_pos`/`joint_vel`/`body_pos_w`/`body_quat_w`/`body_lin_vel_w`/
`body_ang_vel_w`. Input CSVs are expected in **Unitree's generalized
coordinate convention**: base position, base quaternion (xyzw), then joint
angles.

> **Important (from mjlab docs):** you must use *mjlab's own* converter.
> npz files produced by other frameworks' converters have incompatible body
> orderings.

Output lands in `src/motions/<robot>/...` per the README (double-check the
exact path your `--output-name` resolves to — `csv_to_npz.py` may also
accept a full output path).

### 2. Train

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz \
  --env.scene.num-envs=4096
```
`train.py` detects a tracking task by checking whether
`cfg.env.commands["motion"]` is a `MotionCommandCfg`; if so, `--motion-file`
(or an already-set `motion_cmd.motion_file`) is required and resolved to an
absolute path before training starts.

`MotionCommandCfg` key fields (from `tracking_env_cfg.py`):
`entity_name`, `resampling_time_range` (set to `(1e9, 1e9)` — effectively
never resample mid-episode, since the whole point is to track one
continuous clip), `debug_vis` (ghost overlay of the reference pose),
`pose_range`/`velocity_range`/`joint_position_range` (randomize the
*starting* offset from the reference trajectory), `motion_file`,
`anchor_body_name` (the body whose pose anchors the tracking error — set
per-robot), `body_names` (which bodies' positions/orientations are tracked
and penalized).

`play.py`'s `MotionCommandCfg.sampling_mode` controls where in the clip
playback starts: `"start"` (frame 0), `"uniform"` (random frame — used in
demo mode for diversity across parallel envs), `"adaptive"` (biases toward
historically-difficult regions of the clip during training).

### 3. Play back / verify

```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz \
  --checkpoint_file=logs/rsl_rl/g1_tracking/<run>/model_<iter>.pt
```
Add `--no-terminations` if you just want to eyeball a clip/checkpoint
without episodes cutting out early on tracking-error terminations.

### Motion registry (WandB)

`play.py` also supports `--registry-name your-org/motions/motion-name` to
pull a motion clip from a WandB artifact registry instead of a local file
(needed in `"zero"`/`"random"` dummy-agent mode with no local `--motion-file`).
Setting this up follows BeyondMimic's own instructions — see
[whole_body_tracking's README](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup),
linked from this repo's top-level README.

## `list_envs.py`

```bash
python scripts/list_envs.py            # all registered task IDs
python scripts/list_envs.py tracking   # filter by substring
```
Trivial wrapper around `mjlab.tasks.registry.list_tasks()` — the
authoritative, always-current source of truth for what task IDs exist,
better than the table in
[04-repo-structure-and-tasks.md](04-repo-structure-and-tasks.md) if that
doc has drifted from the code.
