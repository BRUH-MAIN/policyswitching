# mjlab / unitree_rl_mjlab documentation

This folder documents [`unitree_rl_mjlab`](../unitree_rl_mjlab), the RL
training/deployment project vendored into this repo, and
[mjlab](https://mujocolab.github.io/mjlab/main/index.html), the underlying
simulation + RL framework it's built on (`mjlab==1.2.0`,
`mujoco-warp==3.5.0`). It's written from a from-scratch read of the mjlab
docs site plus the actual source in `unitree_rl_mjlab/` as of 2026-08-18, so
that you can freely set up, modify, and extend environments/training without
re-deriving all of this yourself.

mjlab itself is **not vendored** in this repo — it's a pip dependency
(`pip show mjlab` finds nothing locally). To read its actual source
(managers, mdp functions, actuator/sensor implementations), install it first
(see [01-installation.md](01-installation.md)) and then look in your
environment's `site-packages/mjlab`.

## How to read this

| File | Covers |
|---|---|
| [01-installation.md](01-installation.md) | System requirements, install steps (this repo's conda+pip flow, and mjlab's own uv/pip/docker flows), platform gotchas |
| [02-architecture-and-concepts.md](02-architecture-and-concepts.md) | mjlab's two-layer design: simulation layer (Entity, Actuators, Sensors, Scene, Terrain, MuJoCo Warp) and the manager-based RL env |
| [03-manager-configuration.md](03-manager-configuration.md) | Deep dive on the 8 managers (observations, actions, rewards, terminations, commands, events, curriculum, domain randomization) with real config snippets from this repo |
| [04-repo-structure-and-tasks.md](04-repo-structure-and-tasks.md) | How `unitree_rl_mjlab/src` is organized, the task registration pattern, and a worked walkthrough of adding/modifying a robot or task |
| [05-training-and-evaluation.md](05-training-and-evaluation.md) | `train.py` / `play.py`, RSL-RL PPO config, CLI overrides via tyro, multi-GPU, checkpoints/resuming, motion imitation workflow |
| [06-deployment-and-debugging.md](06-deployment-and-debugging.md) | Sim2real deployment pipeline, viewers, NaN guard, FAQ/troubleshooting gotchas |
| [07-pas-implementation.md](07-pas-implementation.md) | SARO's PAS (Probability Annealing Selection) low-level policy, replicated for Go2: architecture diagrams, class map, two-stage training pipeline, known gaps |

## 30-second orientation

- **mjlab** = MuJoCo Warp (GPU-parallel MuJoCo) physics + an Isaac-Lab-style
  "manager-based" config API for defining RL environments as plain
  dataclasses/dictionaries of composable terms (no USD, no Omniverse).
- **unitree_rl_mjlab** = this repo. It uses mjlab as a library and adds:
  - Unitree robot assets (Go2, A2, AS2, G1, G1-23dof, H1_2, H2, R1) under
    `src/assets/robots/`.
  - Task configs (velocity tracking + motion imitation/tracking) under
    `src/tasks/`.
  - Training/play/conversion scripts under `scripts/`.
  - A full C++ sim-to-real deployment stack under `deploy/` and `simulate/`.
- **Workflow**: `Train` (`scripts/train.py`) → `Play` (`scripts/play.py`,
  visualize in MuJoCo) → `Sim2Real` (export `policy.onnx`, deploy via
  `deploy/robots/<robot>`).

## Quick command cheat sheet

```bash
# List all registered tasks
python scripts/list_envs.py

# Train (velocity tracking)
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096

# Train on multiple GPUs
python scripts/train.py Unitree-G1-Flat --gpu-ids 0 1 --env.scene.num-envs=4096

# Resume training
python scripts/train.py Unitree-G1-Flat --agent.resume True

# Play back a checkpoint
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/<run>/model_<iter>.pt

# Motion imitation: convert a CSV mocap clip to mjlab's npz format
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/dance1_subject2.csv \
  --output-name dance1_subject2.npz --input-fps 30 --output-fps 50 --robot g1

# Train motion imitation
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096
```

See [05-training-and-evaluation.md](05-training-and-evaluation.md) for the
full flag reference and [04-repo-structure-and-tasks.md](04-repo-structure-and-tasks.md)
for the complete list of registered task IDs.
