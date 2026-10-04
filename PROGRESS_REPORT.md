# Handoff / Progress Report

**As of**: 2026-10-04, late morning · **Written for**: a fresh session (human or Claude) with
no memory of how this state was reached. Where something needs more detail than fits here, a
file is named; read it rather than re-deriving it.

---

## 0. How to resume

1. `git pull --rebase`, then read `README.md` → `objective.md` → `findings.md` (per `CLAUDE.md`).
2. On the laptop (`romen`) read `docs/CLAUDE.laptop.md`; on the cluster, `docs/CLAUDE.cluster.md`.
3. Come back to §2 for what is still open. Nothing there blocks the report.

## 1. Where the project stands

Every arm the project planned has now been run. The answer is in
`report_content/final_report.md` (the full report) and
`coordination/results/switch-follow-results.md` (numbers and intervals).

- **The central hypothesis is not supported.** When following a leader over mixed terrain, the
  switch between specialists must not be late (0.3 m late costs 13-33 points of course
  success), but switching earlier, which is what the followed person's preview would allow, is
  never better. A classifier on the robot's own height scan times the switch as well as
  ground-truth labels. The leader adds neither useful horizon nor useful labels.
- **Switching vs one policy depends on conditions.** With the policies as trained it beats the
  best single specialist by 14.6 points (7.3 with sensor noise on) and beats the matched
  generalist by 8.4 points with sensor noise on, not at all with it off.
- **Blending** the specialists' actions instead of switching hard changes nothing.
- **The biggest effect is in training, not switching.** A curriculum interaction demoted both
  non-flat specialists to near-flat terrain at iteration 5000. The stairs run's own checkpoint
  from iteration 4800, used *alone*, crosses the whole course 97.7% of the time (94.9% with
  noise), better than any switching between the as-trained policies and better than the
  generalist. With it in the bank, switching adds nothing at the easy level and early
  switching costs nothing.

Everything else (gate 1, gate 2, the PAS replication, the VLM/SARO pipeline, person-following
and its two-rate perception) is unchanged and summarised in the final report, Section V.H.

## 2. What is still open

### 2.1 Stairs v2 (job 12490) is submitted; evaluate it when it finishes

You told the cluster session to go ahead on 2026-10-03. `Unitree-Go2-Spec-StairsV2` is the
Stairs task with the command range held at stage 0. It was PENDING at 20:08 IST on 10-03,
waiting for a GPU; checkpoints will appear on private HF under `go2_spec_stairs_v2/`. When
`model_9999.pt` is there, the laptop test is the same one run for the iteration-4800
checkpoint (from `unitree_rl_mjlab/`):

```
python scripts/switch_follow.py --seed 500 --level L1 --ckpt-root eval_ckpts \
    --extra-policy stairs=eval_ckpts/go2_spec_stairs_v2/model_9999.pt \
    --arms fixed:flat fixed:rough fixed:stairs hard:-0.3:label hard:0.0:label hard:0.3:label hard:0.8:label hard:1.5:label \
    --json-out eval_results/switch_follow/stairsv2_L1_clean_s500.json
```

for seeds 500-502 (L1, with and without `--obs-noise`) and 600-601 (`--level L2`), then
`switch_follow_analyze.py confirm`. Question it answers: does a properly trained stairs policy
do what the iteration-4800 checkpoint does (final report, Section V.G)? If yes, the natural
follow-up is a generalist trained the same way.

### 2.2 Gaps checkpoint: still on the cluster only

`go2_spec_gaps/2026-09-17_08-48-35/model_9999.pt` has never been evaluated. The push to private
HF was refused by the cluster session's permission classifier on 10-03; that session did push
the stairs iteration-4800 checkpoint later the same day after you told it to, so asking it
directly may now work. Nothing in the report depends on it.

### 2.3 Report format

`report_content/final_report.md` is a complete report in Markdown with four figures. If it has
to be IEEE LaTeX or a specific template, say so. The related-work citations other than SARO
have not been checked against the sources.

## 3. What was done

**2026-10-03.** Status check with the cluster (job 12033 had finished on 09-21 and never been
written up). Root cause of the weak specialists found from training logs. Switching
experiment pre-registered, calibrated (seeds 400/401) and confirmed (seeds 500-502), with
robustness runs and a scan-classifier reactive arm. `vlm-pipeline` fast-forwarded into `main`.
You submitted the generalist (job 12479) by hand after the cluster session was refused.

**2026-10-04.** Generalist evaluated (two runs, seeds 500-505, noise off and on). Pre-collapse
stairs checkpoint tested in the stairs slot at L1 and L2. Generalist run on pinned terrain
classes (`eval_results/matrix_generalist/`). Results, findings, objective and the final report
rewritten around these.

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
