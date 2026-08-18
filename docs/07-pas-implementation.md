# PAS implementation (SARO, arXiv:2407.16412)

This documents the code added to replicate the low-level locomotion training
technique from **SARO** ("Space-Aware Robot System for Terrain Crossing via
Vision-Language Model", arXiv:2407.16412) — the paper's "PAS" (Probability
Annealing Selection) method — for the Unitree Go2, inside
`unitree_rl_mjlab/`. It does **not** implement SARO's VLM-based high-level
task planner (LLaVA prompting, real-camera sub-task execution) — only the
low-level policy, which is the part that is actually "a policy" trainable in
sim. See [../.claude/plans/2407-16412v3-pdf-contains-the-policy-hidden-hejlsberg.md](../.claude/plans/2407-16412v3-pdf-contains-the-policy-hidden-hejlsberg.md)
for the original feasibility assessment this was built from.

## 1. What PAS actually is

PAS is a **two-stage training-time distillation**, not a runtime switch
between frozen policies:

- **Stage 1 ("oracle")**: train a privileged actor that sees proprioception
  plus a "full latent state" `l_t` — a 32-dim encoding of a 187-point terrain
  height-scan concatenated with a 4-dim privileged state (base linear
  velocity + foot friction). A separate estimator (LSTM + MLP) is trained
  concurrently to *predict* `l_t` from proprioception history alone, via an
  auxiliary MSE loss — it has no effect on the actor yet.
- **Stage 2 ("anneal")**: resume from the Stage-1 checkpoint. Each step, each
  env independently uses either the real `l_t` or the estimator's prediction,
  with the probability of using the real one decaying as `P_t =
  0.9998^iteration`. Both branches stay in the autograd graph, so PPO's own
  backward pass fine-tunes the estimator jointly with the rest of the actor
  whenever its branch was sampled. Training ends with a policy that only
  needs proprioception history to run.

## 2. Stage 1 (oracle) data flow

```mermaid
flowchart TB
    subgraph obs["Per-step observations"]
        HS["height_scan (raw, noisy)\n187-dim"]
        PS["privileged_state\nbase_lin_vel(3) + foot_friction(1)"]
        PR["proprioception\n47-dim"]
    end

    HS --> TE["terrain_encoder\nMLP 187 to 128 to 64 to 32"]
    TE --> CAT1["real l_t (36-dim)\nconcat(terrain_latent, privileged_state)"]
    PS --> CAT1

    PR --> EST["estimator\nLSTM(2x256) + MLP head 256 to 256 to 128 to 36"]
    EST --> PRED["predicted l_t (36-dim)"]

    CAT1 -->|"anneal_prob = 1.0\nalways selected"| MIX["mixed l_t"]
    PRED -.->|"masked out→zero gradient"| MIX

    PR --> CAT2["concat(proprio, mixed l_t)\n83-dim"]
    MIX --> CAT2
    CAT2 --> MLP["actor MLP\n83 to 512 to 256 to 128 to 12"]
    MLP --> ACTION["action: 12 joint position targets"]

    PRED -.->|"MSE vs real l_t.detach()"| AUX["PasPPO estimator\nauxiliary optimizer\n(separate from PPO's optimizer)"]
    CAT1 -.->|"detached target"| AUX
```

In Stage 1 the Bernoulli selector is degenerate (`p=1.0`), so `real l_t`
deterministically wins every step; `terrain_encoder` gets ordinary
policy-gradient updates through the actor MLP, exactly like the paper's Fig.
4. The estimator's *only* training signal here is the auxiliary MSE loss —
it contributes nothing to the action, hence gets no RL gradient, matching
"Oracle Policy Training" in the paper.

## 3. Stage 2 (anneal) data flow

```mermaid
flowchart TB
    subgraph obs["Per-step observations"]
        HS["height_scan (raw, noisy)\n187-dim"]
        PS["privileged_state\nbase_lin_vel(3) + foot_friction(1)"]
        PR["proprioception\n47-dim"]
    end

    HS --> TE["terrain_encoder"]
    TE --> CAT1["real l_t (36-dim)"]
    PS --> CAT1

    PR --> EST["estimator (LSTM + MLP head)"]
    EST --> PRED["predicted l_t (36-dim)"]

    CAT1 --> BERN{{"per-env Bernoulli(P_t)\nP_t = 0.9998^iteration"}}
    PRED --> BERN
    BERN -->|"real (prob P_t)"| MIX["mixed l_t"]
    BERN -->|"predicted (prob 1-P_t)"| MIX

    PR --> CAT2["concat(proprio, mixed l_t)"]
    MIX --> CAT2
    CAT2 --> MLP["actor MLP"]
    MLP --> ACTION["action"]

    MLP -.->|"PPO gradient flows back\nthrough whichever branch\nwas selected, per env"| TE
    MLP -.-> EST
```

As `P_t → 0` over training, the estimator increasingly supplies the actor's
input and receives gradient directly from the RL loss — the paper's "the
loss of RL is utilized to simultaneously optimize both the estimator and the
low-level MLP." The critic is unaffected by any of this: it always sees the
full, unencoded privileged observation (see `"critic"` + `"privileged_state"`
obs groups) and is a stock `MLPModel`.

## 4. Class map

```mermaid
classDiagram
    class MLPModel {
        <<rsl_rl>>
        +obs_normalizer
        +mlp
        +get_latent(obs, masks, hidden_state)
        +is_recurrent = False
    }
    class PPO {
        <<rsl_rl>>
        +actor
        +critic
        +storage
        +update()
    }
    class VelocityOnPolicyRunner {
        <<this repo, pre-existing>>
        +save(): ONNX export
    }

    class PasActorModel {
        +terrain_encoder: MLP
        +estimator_rnn: RNN(LSTM, 2x256)
        +estimator_head: MLP
        +anneal_prob: float
        +is_recurrent = True
        +real_latent(obs, masks)
        +predicted_latent(proprio, masks, hidden_state)
        +get_latent(obs, masks, hidden_state)
    }
    class PasPPO {
        +estimator_optimizer: Adam
        +enable_annealing: bool
        +anneal_base: float
        -_estimator_aux_step()
        +update()
    }
    class PasOnPolicyRunner {
        +save(): try/except around ONNX export
    }

    MLPModel <|-- PasActorModel
    PPO <|-- PasPPO
    VelocityOnPolicyRunner <|-- PasOnPolicyRunner
    PasPPO --> PasActorModel : actor
    PasOnPolicyRunner --> PasPPO : self.alg
```

All five new classes live in one file, `src/tasks/velocity/mdp/pas.py`. They
plug into `mjlab`/`rsl_rl` entirely through existing, documented extension
points (`class_name` string resolution for actor/critic/algorithm classes,
and the same runner-subclassing pattern `VelocityOnPolicyRunner` already
uses) — no fork of either library.

## 5. Two-stage training pipeline

```mermaid
sequenceDiagram
    participant U as You
    participant S1 as Stage 1: Unitree-Go2-PAS-Oracle
    participant CKPT as logs/rsl_rl/go2_pas/<run>/model_N.pt
    participant S2 as Stage 2: Unitree-Go2-PAS-Anneal

    U->>S1: scripts/train.py Unitree-Go2-PAS-Oracle
    Note over S1: anneal_prob pinned at 1.0.<br/>terrain_encoder + actor MLP trained by PPO.<br/>estimator trained by auxiliary MSE only.
    S1->>CKPT: save every save_interval iterations
    U->>S2: scripts/train.py Unitree-Go2-PAS-Anneal --agent.resume --agent.load-run <S1 run>
    CKPT->>S2: load_state_dict (actor, critic, optimizer)
    Note over S2: P_t = 0.9998^iteration, decaying from 1.0.<br/>iteration counter restarts at 0 for Stage 2<br/>(tracked in PasPPO, not in the checkpoint).
    S2->>CKPT: save every save_interval iterations
    Note over CKPT: Final checkpoint's actor ≈ proprioception-only<br/>policy (anneal_prob ≈ 0).
```

## 6. Files touched

| File | Change |
|---|---|
| `src/tasks/velocity/mdp/pas.py` | **New.** `foot_friction` obs term, `PasModelCfg`, `PasPpoAlgorithmCfg`, `PasActorModel`, `PasPPO`, `PasOnPolicyRunner`. |
| `src/tasks/velocity/mdp/__init__.py` | Re-exports `pas.py`. |
| `src/tasks/velocity/config/go2/env_cfgs.py` | New `unitree_go2_pas_env_cfg()`: adds the `"privileged_state"` obs group, a gap-crossing terrain (stepping-stones, rebalanced into `ROUGH_TERRAINS_CFG`'s proportions), and the paper's `energy`/`joint_vel_l2` reward terms. |
| `src/tasks/velocity/config/go2/rl_cfg.py` | New `unitree_go2_pas_ppo_runner_cfg(enable_annealing)`: wires `PasModelCfg`/`PasPpoAlgorithmCfg` via `class_name` string resolution, and `obs_groups = {"actor": ("actor","privileged_state"), "critic": ("critic","privileged_state")}`. |
| `src/tasks/velocity/config/go2/__init__.py` | Registers `Unitree-Go2-PAS-Oracle` (`enable_annealing=False`) and `Unitree-Go2-PAS-Anneal` (`enable_annealing=True`), both on `PasOnPolicyRunner`. |

## 7. Known gaps

- **ONNX export isn't PAS-aware yet.** `MLPModel.as_onnx()`'s generic wrapper
  assumes `get_latent()` is a plain concat-and-normalize, which is false for
  `PasActorModel`. `PasOnPolicyRunner.save()` catches this so checkpointing
  never crashes a training run, but a real deployable `policy.onnx` needs a
  custom `PasActorModel.as_onnx()`/`as_jit()` (mirroring how `RNNModel`
  provides its own) that bakes in `proprioception → estimator → action`. Not
  needed for training or for the Stage 1 → Stage 2 handoff (`.pt` checkpoints
  are unaffected) — only for eventual real-robot deployment.
- **`P_t` is keyed to environment steps via `PasPPO`'s own iteration
  counter**, not literal PPO iterations restored from a checkpoint — it
  always restarts at 0 when a new `PasPPO` is constructed (i.e., once per
  `train.py` invocation), which is the correct behavior for Stage 2 starting
  its own fresh anneal, but means interrupting and resuming *mid-Stage-2*
  would also restart the anneal schedule. Not an issue for the intended
  Stage 1 → Stage 2 handoff, but worth knowing if a Stage 2 run itself needs
  to be resumed after a crash.
- **Gap terrain uses `stepping_stones`**, which approximates but isn't
  geometrically identical to the paper's "gap" intermediation.
- **Environment quirk (not code):** this conda env's editable
  `unitree_rl_mjlab` install resolves `import src` to a different, unrelated
  project (`/home/rohan/unitree_rl_mjlab`). Run everything below with
  `PYTHONPATH` pointed at *this* repo's `unitree_rl_mjlab/`, as shown.

## 8. Running it

```bash
cd /home/rohan/rl/policyswitching/unitree_rl_mjlab
export MUJOCO_GL=egl
export PYTHONPATH=/home/rohan/rl/policyswitching/unitree_rl_mjlab:$PYTHONPATH

# Stage 1 (oracle)
python scripts/train.py Unitree-Go2-PAS-Oracle \
  --env.scene.num-envs <N> \
  --agent.logger tensorboard

# Stage 2 (anneal), resuming from Stage 1
python scripts/train.py Unitree-Go2-PAS-Anneal \
  --env.scene.num-envs <N> \
  --agent.logger tensorboard \
  --agent.experiment-name go2_pas \
  --agent.resume True \
  --agent.run-name stage2

# Watch training
tensorboard --logdir logs/rsl_rl/go2_pas

# Visually evaluate a checkpoint
python scripts/play.py Unitree-Go2-PAS-Oracle \
  --checkpoint_file logs/rsl_rl/go2_pas/<run>/model_<iter>.pt
```

Both tasks default to the paper's `max_iterations=40000` and
`num_steps_per_env=24`; override `--agent.max-iterations` for a shorter
first run. `--env.scene.num-envs` has no default baked into the PAS
configs beyond whatever the base Go2-Rough config uses — pick a value your
GPU's VRAM supports (see the next section).
