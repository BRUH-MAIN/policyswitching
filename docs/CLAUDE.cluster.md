# Role: cluster session (training)

Host: the SLURM login node, which is `asaicomputemaster` — the same box is
simultaneously the RTX 6000 Ada compute node (2×), so "the login node" and
"a compute node" are not distinct machines here. The A100 nodes are
`asaicomputenode02`/`03` (2× A100 each), per findings.md. Run `hostname` if
unsure which machine a given session is on.

**As of 2026-09-12 all three nodes are `drained`** ("Kill task failed", since
2026-09-10), so nothing runs until an admin resumes them; jobs 11914/11918/11919
are PENDING behind that. Login-node work (git, HF uploads, analysis) is
unaffected — see the HF backfill note below, which is deliberately GPU-free for
this reason.

**Pushing to GitHub from this checkout does not work yet.** There is no
credential helper and no `gh` here, so `git push` fails with "could not read
Username". The user has chosen a repo-scoped SSH **deploy key** as the fix
(`ssh-keygen -t ed25519 -f ~/.ssh/policyswitching_deploy -N ""`, public half
added at BRUH-MAIN/policyswitching → Settings → Deploy keys with **write access
ticked**, then `git remote set-url origin git@github.com:BRUH-MAIN/policyswitching.git`).
Until that is in place, a commit here reaches nobody — see
`coordination/log/2026-09-12-laptop.md`.

**Canonical repo**: `/dist_home/d_palmani/c-08/policyswitching` (see
findings.md "Infra facts" — a second, divergent clone was merged in and
deleted on 2026-09-05; don't recreate one). venv:
`/dist_home/d_palmani/.venvs/policyswitching-pas`.

**Model pin**: copy `coordination/settings-examples/settings.local.cluster.json`
to `.claude/settings.local.json` in this checkout (gitignored, machine-local)
if it isn't there yet — pins this role to Sonnet, matching the laptop
session's Opus pin for the analysis/review role split.

Read the root [`CLAUDE.md`](../CLAUDE.md), [`objective.md`](../objective.md)
and [`findings.md`](../findings.md) first if you haven't this session —
this file is only the cross-machine coordination layer on top of that, not
a replacement for it. In particular, don't lose `clip_actions=6.0` in
`unitree_rl_mjlab/src/tasks/velocity/config/go2/rl_cfg.py` — it is committed as
of `6faa66b` (2026-09-11) in both runner configs, so a fresh clone now has it,
but it is a real fix for an observed blow-up rather than a stylistic default
(findings.md bug #8). Don't forget `HF_TOKEN` before `sbatch`-ing
`train_pas_slurm.sh` — see `CLAUDE.md` for both.

## What you own

- Training code, env/task configs, SLURM job scripts (`a100/*.sh`),
  hyperparameters, reward shaping — anything that needs the GPU/queue.
- Submitting and monitoring `sbatch` jobs:
  - PAS: `sbatch a100/train_pas_slurm.sh` (auto-resumes via HF, see
    `a100/hf_sync.py`).
  - A specialist: `SPEC=Gaps sbatch a100/train_specialist_slurm.sh` (`SPEC`
    ∈ Flat/Rough/Stairs/Gaps/**GapsWarm**/**Generalist**; auto-resumes from
    **local disk**, see `a100/local_ckpt_resume.py`). `GapsWarm` warm-starts
    from the Rough specialist; `Generalist` is the sensing-matched stock-PPO
    arm 1 of the revised comparison, not a specialist.
  - **Specialists can now back up to HF, opt-in**: `HfSyncVelocityOnPolicyRunner`
    pushes each checkpoint when `HF_TOKEN` is set at submission, to
    `$HF_CHECKPOINT_REPO/<experiment_name>/`. findings.md bug #4 (specialists
    exist only on cluster local disk) describes the old default, not a
    permanent limitation. **Set `HF_TOKEN` from now on**: the laptop has no SSH
    key to this cluster, so rsync is dead and HF is the only route a specialist
    reaches the laptop by. Point `HF_CHECKPOINT_REPO` at the **private** repo —
    `RohanRamesh/go2-pas-saro` is public and is for PAS only (user decision,
    2026-09-12).
  - **Owed: a one-off HF backfill** for `go2_spec_flat`/`rough`/`stairs`, which
    were trained before this and exist only on local disk. Needs a small script
    over `push_checkpoint_to_hf` (`rl/hf_upload.py`) — the runner's sync only
    fires inside a live `runner.save()`. It needs no GPU, so it runs on the
    login node **while the nodes are drained**, and it unblocks both premise
    gates on the laptop. See `coordination/inbox/to-cluster.md`.
  - Debugging crashes/NaNs/OOMs/slow throughput — findings.md's "Bugs
    found" section already covers the ones hit so far (scipy, CUDA
    allocator fragmentation, tyro bool flags, reward divergence); check
    there before re-diagnosing one from scratch.
- Running `coordination/scripts/cluster_update_status.sh` whenever a
  checkpoint is saved or a job finishes/times out, so the laptop side knows
  without polling the filesystem:
  ```
  # PAS (checkpoint already on HF via hf_sync.py):
  coordination/scripts/cluster_update_status.sh go2_pas_stage2 79998 hf RohanRamesh/go2-pas-saro stage2 <job_id>

  # Specialist (local disk only):
  coordination/scripts/cluster_update_status.sh go2_spec_gaps 9999 local unitree_rl_mjlab/logs/rsl_rl/go2_spec_gaps/<run_dir>/model_9999.pt <job_id>
  ```
  Known `run_id`s: `go2_pas_stage1`, `go2_pas_stage2`, `go2_spec_flat`,
  `go2_spec_stairs`, `go2_spec_rough`, `go2_spec_gaps`, `go2_spec_gapswarm`,
  `go2_generalist`. A `run_id` outside that set now **fails fast** with a
  message telling you to set `TASK=` — the script derives the task name the
  laptop side requires, and refuses to write an entry it can't name (it used to
  omit the field silently, which broke `laptop_pull_and_eval.sh` later, after
  the entry was already committed and pushed).

  Once a specialist is on HF, register it in **`hf` mode**, not `local` — the
  laptop's rsync path no longer works:
  ```
  coordination/scripts/cluster_update_status.sh go2_spec_flat 9999 hf <private-repo> go2_spec_flat <job_id>
  ```

## What you don't decide alone

- Whether an experiment "worked" — that's a laptop-side judgment call from
  eval results, not something to conclude from training loss curves alone.
  Training loss looking healthy is necessary, not sufficient (see
  findings.md bug #1: a policy that bracing-in-place can look fine on
  reward/survival alone) — hand off to eval rather than declaring success.
- What to try next when it's a scientific/design question (reward shaping,
  terrain-mix changes, hyperparameters aimed at a hypothesis from
  `objective.md`'s 2×2 comparison of {hard, soft} × {reactive, anticipatory},
  whose arm 1 is now the sensing-matched `Unitree-Go2-Generalist` rather than
  PAS). Check
  `coordination/inbox/to-cluster.md` for requests from the laptop side
  first. If it's empty and you need direction, write the question into
  `coordination/inbox/to-laptop.md` and pause on that thread instead of
  guessing.

## Workflow

1. `git pull --rebase` at session start.
2. Check `coordination/inbox/to-cluster.md` for anything queued for you;
   check `coordination/status/laptop.json` for the last eval verdict.
3. Do the implementation/training work.
4. Note any `sbatch` submission in `coordination/log/<date>-cluster.md`
   (job id, what's being tested, which config) — this cluster is
   pure age-based FIFO (`sprio`'s FAIRSHARE is 0 for everyone), so also note
   walltime requested; a shorter request schedules faster even with more
   resubmissions, per findings.md.
5. When a checkpoint is saved, run `coordination/scripts/cluster_update_status.sh`
   (see above).
6. Commit and push. If compute nodes lack outbound internet, do this step
   from the login node / your Claude Code session there, not inside the
   SLURM job itself — the job only needs to write the checkpoint + let
   `hf_sync.py`/local disk persist it; the git step is separate and quick.
7. If `/list-agents` shows the laptop session reachable, message it by name
   to say the checkpoint is ready (e.g. "tell @laptop-eval that
   go2_spec_gaps step 9999 is ready, details in
   coordination/inbox/to-laptop.md") instead of leaving it to poll. This is
   a convenience on top of the commit, never a substitute for it — see the
   root-level coordination note in `CLAUDE.md` for why (message delivery
   isn't guaranteed the way a commit is, and cross-machine Remote Control
   needs both sides signed in via claude.ai, not a bare API key).
