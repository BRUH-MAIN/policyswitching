# Handoff / Progress Report

**As of**: 2026-10-03, end of day · **Written for**: a fresh session (human or Claude) with no
memory of how this state was reached. Where something needs more detail than fits here, a file
is named; read it rather than re-deriving it.

---

## 0. How to resume

1. `git pull --rebase`, then read `README.md` → `objective.md` → `findings.md` (per `CLAUDE.md`).
2. On the laptop (`romen`) read `docs/CLAUDE.laptop.md`; on the cluster, `docs/CLAUDE.cluster.md`.
3. Come back to §2. The first item there is the only thing still owed.

## 1. Where the project stands

The experiment the project was built for has been run, and its answer is in
`report_content/final_report.md` (the full report) and
`coordination/results/switch-follow-results.md` (the numbers and intervals).

- **Switching between terrain specialists beats any single specialist** on a mixed course:
  91.5% vs 76.6% of trials cross it (+15.0 points; +4.2 with training sensor noise on).
- **The switch has to happen at the terrain boundary.** 0.3 m late costs 16-30 points.
  Switching 0.8 m or more early, which is what the followed person's preview would allow,
  costs 7-17 points and is no better than never switching.
- **So the project's central hypothesis is not supported.** The leader's preview adds no
  useful horizon, and a classifier on the robot's own height scan times the switch as well as
  ground-truth labels do, so it adds no useful labels either.
- **Blending the specialists' actions** instead of switching hard changes nothing.
- **One planned arm is missing**: the sensing-matched generalist is training as of 2026-10-03 and has not been evaluated (§2.1).

Everything else in the project (the three specialists, gate 1, gate 2, the PAS replication,
the VLM/SARO pipeline, person-following and its two-rate perception) is unchanged from the
previous report and summarised in the final report, Section V.H.

## 2. What needs a decision or an action from you

### 2.1 The generalist is training; add its arm when it finishes

Job **12479** on `asaicomputemaster` (RTX 6000 Ada), started 2026-10-03 ~16:43 IST, 1-day
walltime, seed 42, 10k iterations, pushing every checkpoint to the private
`RohanRamesh/go2-specialists` under `go2_generalist/`. You submitted it by hand after the
cluster session's permission classifier refused three times (`coordination/log/2026-10-03-cluster.md`).
Expect ~12-13 hours if the Ada keeps an A100's pace. The cluster session is watching for a
stepping-stones plateau at ~1,500 iterations. If the walltime runs out first, resubmitting the
same command resumes from the last checkpoint.

When `go2_generalist/model_9999.pt` is on HF, the arm is one command on the laptop:

```
unitree_rl_mjlab/scripts/switch_follow_generalist.sh
```

It downloads the checkpoint, runs `fixed:generalist` next to the on-time switch and the stairs
specialist on seeds 500-502 with observation noise off and on, and prints the comparisons. The
whole path was code-checked on 2026-10-03 with the job's iteration-0 checkpoint (0% success, as
an untrained policy should score). Then: put the numbers into
`coordination/results/switch-follow-results.md`, `report_content/final_report.md` (Sections
V.B, V.F, VII and VIII.1 all say the generalist is missing) and `objective.md`'s "What the
result was". Before trusting a good-looking number, check it is walking and not bracing
(findings.md bugs #1 and #14): look at `lost %` and the tracking error, not only falls.

### 2.2 Gaps checkpoint: same block

`go2_spec_gaps/2026-09-17_08-48-35/model_9999.pt` exists on the cluster only. The push to
private HF was denied for the same reason ("Data Exfiltration"). `a100/backfill_specialist_hf.py`
does it. It has never been evaluated. Optional: nothing in the report depends on it.

### 2.3 Whether to retrain the stairs specialist

`findings.md`, "Why the specialists are weak": both non-flat specialists were demoted to
near-flat terrain at iteration 5000, when the command range widened to 2 m/s, and stayed
there. A stairs v2 is a config change (hold the command range at stage 0), not more
iterations. It would test whether the early-switching penalty survives a competent
specialist. Proposal is in `coordination/inbox/to-cluster.md`; `stairs_v2_train` stays
`not_submitted` until you say so. Not needed for the report as written.

**Update, 2026-10-03 evening (cluster).** Two things changed since the paragraph above:

- A better stairs specialist already exists: `model_4800.pt` of the same run, saved before
  the collapse, falls 2.6–3.6× less often per 100 m than `model_9999.pt` on pinned pyramid
  stairs (1.57 vs 4.10 at 5 cm risers, 5.06 vs 18.06 at 9 cm), at the same speed. It is on
  the private HF repo as `go2_spec_stairs_it4800/model_4800.pt` (`runs.go2_spec_stairs_it4800`
  in `cluster.json`). Write-up and caveats (pyramid terrain only, one eval seed, untested on
  the straight course): `coordination/results/2026-10-03-stairs-precollapse-checkpoint-eval.md`.
- The v2 task is built and config-checked, not submitted: `SPEC=StairsV2 sbatch
  a100/train_specialist_slurm.sh` (experiment `go2_spec_stairs_v2`, separate from
  `go2_spec_stairs`). Judge it by that same pinned eval, not by `terrain_levels` climbing —
  the level was already flat at ~1.9 before the collapse.

### 2.4 Report format

`report_content/final_report.md` is a complete report in Markdown with two figures. If it has
to be IEEE LaTeX or a specific template, say so. The related-work citations other than SARO
have not been checked against the sources.

## 3. What was done on 2026-10-03

- Status check with the cluster: job 12033 (stairs on pyramid stairs) had finished on 09-21
  and never been written up; now committed. Nothing else had been submitted.
- Root cause of the weak specialists found from the cluster's training logs (curriculum
  collapse at iteration 5000, both runs).
- Switching experiment: pre-registered, calibrated on seeds 400/401, confirmed on seeds
  500-502 (768 trials per arm), with robustness runs (observation noise on, orientation-only
  falls, harder level) and a scan-classifier reactive arm (pre-registered addendum).
- `findings.md`, `objective.md`, this file and the final report updated.
- **`vlm-pipeline` was fast-forwarded into `main`** (same commit on both, pushed). The linked
  worktree at `.claude/worktrees/vlm-pipeline` still exists; new work can go on `main`.

## 4. Gotchas added today (full list: `findings.md`, "Bugs found and fixed")

- **#22: the twin env has observation noise and pushes off** (it is built on the play config).
  Every `src/vlm_nav/` number was measured that way unless `--obs-noise` was passed. The fixed
  specialists do better with the noise on, and the switching advantage shrinks from +15 to +4
  points. Say which condition a number is from.
- **#23: the goal controller understates the stairs specialist.** Same course, seed and
  oracle: 52-59% under the goal controller, 89% under the follow controller.
- **What counts as a fall decides whether L1 separates anything.** All L1 failures are knee or
  calf contacts; under orientation-only falls every arm but flat-only is at 98-100%.
- **Don't edit `scripts/switch_follow.py` while a multi-seed loop is running.** Each seed is a
  new process and re-reads the file. Harmless today (only the recorded conditions differ), but
  it is the same hazard as pulling under a running eval.
- Scan-classifier weights and recorded scans live under `unitree_rl_mjlab/logs/switch_follow/`
  (gitignored). `scripts/switch_follow_clf_pipeline.sh` regenerates them in ~25 minutes.

## 5. File map

| Want to know... | Read |
|---|---|
| The whole project, as a report | `report_content/final_report.md` |
| The switching experiment's numbers and intervals | `coordination/results/switch-follow-results.md` |
| What was fixed in advance | `coordination/results/switch-follow-preregistration.md` |
| Why the project exists, the design, and the outcome | `objective.md` |
| Full experiment history, every bug | `findings.md` |
| Runner, schedules, classifier | `unitree_rl_mjlab/scripts/switch_follow*.py`, `src/vlm_nav/schedule.py`, `src/vlm_nav/scan_classifier.py` |
| Raw results | `unitree_rl_mjlab/eval_results/switch_follow/` |
| Cluster state and what is asked of it | `coordination/status/cluster.json`, `coordination/inbox/to-cluster.md` |
| VLM/SARO pipeline, person-following | `findings.md` (four sections dated 2026-09-17 to 09-22), `coordination/results/vlm-nav-*.md` |
| Older per-section report notes | `report_content/ieee_report_source.md` |
