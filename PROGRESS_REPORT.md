# Handoff / Progress Report

**As of**: 2026-10-05, afternoon · **Written for**: a fresh session (human or Claude) with
no memory of how this state was reached. Where something needs more detail than fits here, a
file is named; read it rather than re-deriving it.

---

## 0. How to resume

1. `git pull --rebase`, then read `README.md` → `objective.md` → `findings.md` (per `CLAUDE.md`).
2. On the laptop (`romen`) read `docs/CLAUDE.laptop.md`; on the cluster, `docs/CLAUDE.cluster.md`.
3. Come back to §2 for what is still open. Nothing there blocks the report.

## 1. Where the project stands

The simulation study is finished. The answer is in `report_content/final_report.md` (Section
V.H is the evidence to rely on; Section IX is for the robot) and
`coordination/results/switch-follow-results.md`.

Across 20 randomised course layouts:

- **The central hypothesis is not supported.** Switching earlier than the terrain boundary,
  which is what the followed person's preview would allow, gains 1-2 points at most. Being
  0.3 m late loses 64.
- **Switching between specialists adds nothing** over the best single policy.
- **Training is what matters.** Stairs v2 (the stairs policy retrained without the curriculum
  collapse) alone crosses 92.7% of courses; the original stairs specialist 35.2%; the two
  generalists 14.6% and 37.5%.
- **Stairs v2 needs its height scan**: 9.2% without it.
- **Retracted**: the +15-point switching advantage and the early-switch penalty reported on
  10-03/04 came from one course layout and do not hold across layouts (`findings.md` #25).

## 2. What is still open

### 2.1 Real stairs on the Go2 (the active work)

Decided 2026-10-05: the robot has a Livox Mid-360, so the policy keeps its height scan, and
the target is real stairs. Plan: `report_content/go2_real_stairs_plan.md`.

- **Stairs v2 cannot climb real stairs** (sim): going up, 84% at 9 cm risers and 0% at 12,
  15 and 17 cm. It was trained on at most 10 cm.
- **Stairs v3 is training**: job 12563 on `asaicomputemaster`, started 2026-10-05 21:55 IST,
  about 8.6 h, checkpoints on private HF under `go2_spec_stairs_v3/`. Risers 5-20 cm at two
  tread depths, warm-started from v2, robots spread uniformly over rows (the terrain
  curriculum's promotion rule was shown to be broken:
  `coordination/results/2026-10-05-terrain-curriculum-diagnosis.md`). Expect more than one
  run. When `model_9999.pt` lands: `scripts/switch_follow_real_stairs.sh` (also with
  `STEPS=10`), `switch_follow_scan_faults.sh`, `switch_follow_stairs_ckpt.sh`. Bar: 90% up and
  down at 17 cm.
- **Scan requirement measured** (stairs v2, low steps): 200 ms of delay and 60% stale cells
  cost under 2 points; a ±6 cm height offset costs 19. Height above ground must be good to
  ~3 cm.
- **Numpy runner for the Jetson** exists and matches PyTorch
  (`deploy_numpy/policy_numpy.py`, `scripts/export_policy_numpy.py`).
- **You, on the robot, read-only**: check whether the firmware publishes a height map
  (commands in the plan, section 2.2). The Mid-360 by itself does not see the ground within
  about a metre of the robot, so this decides how the scan is produced.
- **Laptop, next**: evaluate v3 at 12 / 15 / 17 cm with `switch_follow.py --step-height`;
  add scan delay, holes and bias to the harness; numpy export of the policy for the Jetson
  (no torch or onnxruntime there).
- Robot workspace with hardware notes: `/run/media/rohan/New Volume/RL/temp`.

### 2.2 Optional

- Gaps checkpoint: on the cluster only, never evaluated.
- Scan classifier and blending were not re-tested on randomised layouts.
- Report format: Markdown with four figures; say if it needs to be IEEE LaTeX. Related-work
  citations other than SARO are unchecked.

## 3. What was done

**2026-10-03.** Status check with the cluster (job 12033 had finished on 09-21 and never been
written up). Root cause of the weak specialists found from training logs. Switching
experiment pre-registered, calibrated (seeds 400/401) and confirmed (seeds 500-502), with
robustness runs and a scan-classifier reactive arm. `vlm-pipeline` fast-forwarded into `main`.
You submitted the generalist (job 12479) by hand after the cluster session was refused.

**2026-10-04.** Generalist evaluated (two runs, seeds 500-505, noise off and on), plus its
pinned-terrain matrix and height-scan ablation (`eval_results/matrix_generalist/`) and its
iteration-4800 checkpoint. Pre-collapse stairs checkpoint and then stairs v2 tested in the
stairs slot at L1 and L2. Results, findings, objective and the final report rewritten around
these. You approved a generalist retrain (generalist v2, job 12518).

**2026-10-05.** Generalist v2 evaluated against stairs v2. A diagnostic on single staircases
showed weak policies' results depend on course layout, so the main comparisons were re-run on
20 randomised layouts, which confirmed stairs v2 and overturned two fixed-course results.

## 4. Gotchas (full list: `findings.md`, "Bugs found and fixed")

- **#22: the twin env has observation noise and pushes off** (it is built on the play config).
  Every `src/vlm_nav/` number was measured that way unless `--obs-noise` was passed. Results
  differ materially between the two; say which condition a number is from.
- **Repeat runs are not identical.** One arm on the same seeds differs by about 2 points with
  noise off and up to 3.5 with noise on. The first noise-on estimate of the switching
  advantage was +4.2; pooled over three runs it is +7.3. Pool before quoting.
- **#23: the goal controller understates the stairs specialist.** Same course, seed and
  oracle: 52-59% under the goal controller, 89% under the follow controller.
- **What counts as a fall decides whether L1 separates anything.** All L1 failures are knee or
  calf contacts; under orientation-only falls every arm but flat-only is at 98-100%.
- **#24, #25: results from one course layout are results about that layout.** Two
  pre-registered, seed-replicated findings disappeared over randomised layouts. Evaluate with
  `switch_follow.py --random-layout` and analyse with `switch_follow_layouts.py`.
- **Don't edit `scripts/switch_follow.py` while a multi-seed loop is running**: each seed is a
  new process and re-reads the file.
- **Background jobs here are killed after two hours**, counted from launch, including time
  spent waiting on another job. Chain long runs in fresh jobs; `--resume` skips finished arms.
- **The laptop GPU is shared** with other projects' sessions. Check
  `nvidia-smi --query-compute-apps` and ask the owning session before launching.
- Scan-classifier weights and recorded scans are under `unitree_rl_mjlab/logs/switch_follow/`
  in the old `vlm-pipeline` worktree (gitignored). `scripts/switch_follow_clf_pipeline.sh`
  regenerates them in about 25 minutes.

## 5. File map

| Want to know... | Read |
|---|---|
| The whole project, as a report | `report_content/final_report.md` |
| The switching experiment's numbers and intervals | `coordination/results/switch-follow-results.md` |
| What was fixed in advance | `coordination/results/switch-follow-preregistration.md` |
| Why the project exists, the design, and the outcome | `objective.md` |
| Full experiment history, every bug | `findings.md` |
| Runner, schedules, classifier | `unitree_rl_mjlab/scripts/switch_follow*.py`, `src/vlm_nav/schedule.py`, `src/vlm_nav/scan_classifier.py` |
| Raw results | `unitree_rl_mjlab/eval_results/switch_follow/`, `eval_results/matrix_generalist/` |
| Cluster state and what is asked of it | `coordination/status/cluster.json`, `coordination/inbox/to-cluster.md` |
| VLM/SARO pipeline, person-following | `findings.md` (four sections dated 2026-09-17 to 09-22), `coordination/results/vlm-nav-*.md` |
| Older per-section report notes | `report_content/ieee_report_source.md` |
