# Handoff / Progress Report

**As of**: 2026-10-03, end of day · **Written for**: a fresh session (human or Claude) with no
memory of how this state was reached. Where something needs more detail than fits here, a file
is named; read it rather than re-deriving it.

---

## 0. How to resume

1. `git pull --rebase`, then read `README.md` → `objective.md` → `findings.md` (per `CLAUDE.md`).
2. On the laptop (`romen`) read `docs/CLAUDE.laptop.md`; on the cluster, `docs/CLAUDE.cluster.md`.
3. Come back to §2. One thing there needs a human.

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
- **One planned arm is missing**: the sensing-matched generalist was never trained (§2.1).

Everything else in the project (the three specialists, gate 1, gate 2, the PAS replication,
the VLM/SARO pipeline, person-following and its two-rate perception) is unchanged from the
previous report and summarised in the final report, Section V.H.

## 2. What needs a decision or an action from you

### 2.1 Train the generalist: needs a human at the cluster

The cluster session tried on 2026-10-03 and its own permission classifier denied the `sbatch`
("Modify Shared Resources"). Nothing is wrong with the cluster: one RTX 6000 Ada was free and
nothing was queued. Either tell the cluster session directly to submit it, or run it by hand
on `asaicomputemaster`:

```
cd /dist_home/d_palmani/c-08/policyswitching && git pull
export HF_TOKEN=$(cat ~/.hf_token)
SPEC=Generalist sbatch --gres=gpu:1 a100/train_specialist_slurm.sh
```

`--gres=gpu:1` lets it start on whichever GPU is free instead of waiting for an A100.
`HF_TOKEN` makes it push checkpoints to the private `RohanRamesh/go2-specialists` repo, which
is how the laptop gets them. About 12 hours. Check the log after ~1,500 iterations: a quarter
of its terrain is stepping stones, which the Gaps specialist never learned from a cold start.

When `go2_generalist/model_9999.pt` exists, the arm is one command on the laptop (from
`unitree_rl_mjlab/`, once per seed 500, 501, 502):

```
python scripts/switch_follow.py --seed 500 --arms fixed:generalist hard:0.3:label fixed:stairs \
    --extra-policy generalist=<path>/model_9999.pt --ckpt-root <eval_ckpts> \
    --json-out eval_results/switch_follow/generalist_s500.json
```

then `switch_follow_analyze.py confirm ... --pairs "hard:0.3:label>fixed:generalist"`, and
again with `--obs-noise`. That fills the one empty cell in the final report (Section VIII.1)
and answers the question `objective.md` says the project rests on.

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
