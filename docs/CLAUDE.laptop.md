# Role: laptop session (evaluation & analysis)

Host: `romen` — 22GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB VRAM).
Run `hostname` if unsure which machine a given session is on.

**Model pin**: `.claude/settings.local.json` already pins this checkout to
Opus (gitignored, machine-local — set on 2026-09-11 to match the
analysis/review role split from `coordination/settings-examples/settings.local.laptop.json`;
edit or remove it directly if that's not wanted).

Read the root [`CLAUDE.md`](../CLAUDE.md), [`objective.md`](../objective.md)
and [`findings.md`](../findings.md) first if you haven't this session —
this file is only the cross-machine coordination layer on top of that.
`findings.md`'s "Bugs found" section in particular exists because past eval
numbers looked fine until a specific gotcha was found — re-read it before
trusting a new one, especially #1 (survival metrics alone can't tell walking
from bracing-in-place) and #2 (`anneal_prob` isn't stored in a PAS
checkpoint — every eval is silently oracle-mode unless `--anneal-prob 0.0`
is passed).

**Local eval environment already set up on this machine**: conda env
`unitree_rl_mjlab` has mujoco 3.5.0 / mjlab 1.2.0 / rsl_rl / torch+cuda
installed, matching the cluster's pinned versions — no fresh install needed.
`huggingface_hub` was added to it on 2026-09-11 specifically so
`eval_checkpoint.py --hf-repo` works locally.

## What you own

- Evaluation of trained checkpoints, run locally via
  `coordination/scripts/laptop_pull_and_eval.sh <run_id>` (wraps
  `unitree_rl_mjlab/scripts/eval_checkpoint.py` — see that script directly
  for the full flag set, e.g. `--anneal-prob`, `--command-name`). Defaults
  to `--num-envs 256`, empirically confirmed to fit this GPU on 2026-09-11
  (see `coordination/status/laptop.json`) — the cluster's own default is
  1024, so there may be headroom, but push it up cautiously (watch
  `nvidia-smi` in another terminal) rather than assuming. If a checkpoint's
  eval genuinely doesn't fit even at a reasonable count, flag that
  explicitly rather than silently shrinking further and calling it
  equivalent — a `--steps` cut short enough changes what the metrics mean
  (see the `time_out` vs `failed early` breakdown in the eval output).
- Reading eval output critically: sample efficiency, variance across seeds,
  whether an apparent improvement is likely noise vs. real, sanity-checking
  against what training reward curves predicted, catching reward
  hacking/degenerate policies (findings.md bug #1 is the canonical example
  here — 80.5% survival while translating ~0m).
- Deciding and writing up what to try next, with reasoning, in
  `coordination/inbox/to-cluster.md`.
- `coordination/results/` — both `laptop_pull_and_eval.sh`'s raw
  `<run_id>-step<N>-eval.log` and your own `<run_id>-step<N>-analysis.md`
  write-ups.

## What you don't do

- Don't rewrite training code, task configs, or SLURM scripts here — if a
  change is needed, describe it precisely in
  `coordination/inbox/to-cluster.md` (what to change, why, what you expect
  to see differently) and let the cluster session implement it.
- Don't submit SLURM jobs from here (this machine has no SLURM tools
  installed anyway) — keep queue interactions on the cluster side.
- Don't rsync a specialist checkpoint by hand as a substitute for updating
  `coordination/status/cluster.json` — `laptop_pull_and_eval.sh` reads that
  file to know what path to pull; if it's missing an entry, that's a
  cluster-side `cluster_update_status.sh` call that didn't happen yet, not
  something to work around locally.

## Workflow

1. `git pull --rebase` at session start.
2. Check `coordination/status/cluster.json` — is there a `runs.<run_id>`
   entry newer/further-along than what `coordination/status/laptop.json`'s
   `evaluated.<run_id>` last recorded? Also check
   `coordination/inbox/to-laptop.md` for explicit "please evaluate" notes.
3. If yes: `coordination/scripts/laptop_pull_and_eval.sh <run_id>`
   — for `go2_pas_stage1`/`go2_pas_stage2` this pulls straight from HF
   (`RohanRamesh/go2-pas-saro`, no SSH needed); for `go2_spec_*` it rsyncs
   from the cluster over SSH (`d_palmani@172.17.16.11`, needs that SSH
   access working).
4. Read the results yourself before writing anything up — don't just
   restate the numbers. Cross-check against the checkpoint/mode actually
   evaluated (which `--task` / `--anneal-prob`), not just the run_id.
5. Write findings to `coordination/results/<run_id>-step<N>-analysis.md`.
6. If there's a next step, write it to `coordination/inbox/to-cluster.md`
   clearly enough that the cluster session doesn't need to ask what you
   meant. Move the picked-up `to-laptop.md` entry to "## Done" once handled.
7. `git add coordination/ && git commit && git push`.
8. If `/list-agents` shows the cluster session reachable, message it by
   name with the headline finding and a pointer to the analysis file —
   e.g. "tell @cluster-trainer: go2_spec_gaps attempt 3 still plateaus,
   see coordination/results/go2_spec_gaps-step9999-analysis.md, next step
   is in the inbox." A nudge on top of the commit, not a replacement — see
   the coordination note in the root `CLAUDE.md`.
