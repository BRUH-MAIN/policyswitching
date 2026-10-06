# Handoff / Progress Report

**As of**: 2026-10-06, afternoon · **Written for**: a fresh session (human or Claude) with no
memory of how this state was reached. This file is the entry point; where it needs more
detail it names a file. Read that file rather than re-deriving it.

---

## 0. How to resume

1. `git pull --rebase` in `/home/rohan/rl/policyswitching` (branch `main`; everything is here).
2. Read this file, then `report_content/go2_real_stairs_plan.md` (the active work), then
   `findings.md` if you are about to trust or produce a number.
3. On the laptop (`romen`) also read `docs/CLAUDE.laptop.md`; on the cluster,
   `docs/CLAUDE.cluster.md`. `CLAUDE.md` has the repo-wide rules.
4. Go to §2. It says what is waiting on whom.

## 1. The two phases of this project

### 1.1 The simulation study: finished

Question: when a quadruped follows a person, does using that person as a preview of the
terrain, to switch between terrain-specialist policies early, help? **No.** Full report:
`report_content/final_report.md` (Section V.H is the evidence to rely on). Numbers:
`coordination/results/switch-follow-results.md`.

Across 20 randomised course layouts:

- Switching earlier than the terrain boundary gains 1-2 points of course success at most.
  Switching 0.3 m late loses 64.
- Switching between specialists adds nothing over the best single policy.
- Training is what matters: **stairs v2** (the stairs policy retrained after a curriculum bug
  was found) alone crosses 92.7% of courses; the original stairs specialist 35.2%; two
  generalists 14.6% and 37.5%.
- Stairs v2 needs its height scan: 9.2% with the scan replaced by a constant.
- Retracted: a +15-point switching advantage and an early-switch penalty reported on
  10-03/04 came from one course layout (`findings.md` #24, #25).

Consequence for everything after: **deploy one policy. No switching module, no leader
preview.**

### 1.2 The goal now: the real Go2 follows a person and climbs real stairs

Stated by Rohan on 2026-10-05. The robot carries a Livox Mid-360. Plan and all detail:
`report_content/go2_real_stairs_plan.md`. Robot workspace with hardware notes:
`/run/media/rohan/New Volume/RL/temp` (its own `CLAUDE.md` forbids editing files on the
robot; `robot_info.md` there holds credentials and should not be printed).

**No policy for real stairs exists yet.**

| | |
|---|---|
| Stairs v2 on real riser heights (sim, going up a 5-step flight) | 84% at 9 cm, 0% at 12, 15 and 17 cm. It stalls; it was trained on at most 10 cm. |
| Stairs v3, first attempt at 5-20 cm (job 12563, finished 10-06) | **Failed: it learned to stand still.** 12% of commanded speed at every riser; crosses 0 of 128 flights. Its near-zero fall rate is that, not success. Do not use the checkpoint. `coordination/results/2026-10-06-stairs-v3-result.md` |
| Run 2 (variants 4a and 4b) | **Failing.** With the normaliser kept (`go2_spec_stairs_v4a_kn` job 12586, `_v4b_kn` job 12587, started 16:40 IST on 10-06) both are at 28-30% of commanded speed on the easiest row by iteration 150 and fail the iteration-530 check. Laptop eval of their iteration-400 checkpoints: 0% of flights crossed, cannot follow on flat ground. Still running when this was written. `coordination/results/2026-10-06-stairs-v4-stop-test.md` |
| Pattern across v3, v4a, v4b | The policy gets safer by getting slower, and reward does not show it. **Hypothesis: the `foot_clearance` reward** uses the foot's height in the world frame against a fixed 0.10 m, so on a raised staircase it charges every moving foot for the staircase's height. Confirmed in the code. In the logs it is **consistent, not demonstrated**: the failed runs pay 1.5-2x stairs v2's clearance penalty, about half their tracking reward against about a fifth for v2, ordered as predicted; but the penalty falls only ~20% while speed falls from 43% to 28%. It needs an intervention: train with the term fixed or removed. Addendum of `coordination/results/2026-10-06-stairs-v4-stop-test.md`. |
| Warm start | Should keep the observation normaliser (`findings.md` #27): reset, v4a started at 28% of commanded speed against 43% kept. It did not rescue the runs. |
| Height scan on the robot | Not built. The Mid-360 does not see ground within about a metre of the robot. The firmware L1 LiDAR (`/utlidar/cloud`) may; unverified. |
| Running the policy on the Jetson | Numpy runner written and matched to PyTorch on the laptop. Never run on the robot. |
| How good the robot's scan must be | Measured on stairs v2, low steps: 200 ms delay and 60% stale cells cost under 2 points; a ±6 cm height offset costs 19. Height above ground good to ~3 cm. |

## 2. What is waiting on whom

### 2.1 Rohan

1. **Decide, in the cluster session, what to do about the real-stairs training.** State on the
   evening of 10-06: jobs 12586 and 12587 are running and failing. The laptop session's
   recommendation:
   - Cancel 12586 and 12587 (they are failing and hold both usable GPUs).
   - Say "go" to a **stairs v5 = 4b_kn with `foot_clearance` changed**: removed (a one-line
     change, the quickest test) or measured against the ground (e.g. foot height above the
     lowest foot). The log check is done and is consistent with the hypothesis without
     proving it; only training with the term changed can.
   - **Short runs first.** The failure shows within 150 iterations, about 8 minutes. Run
     300-500 iterations per variant, judge on speed fraction and mean row, and give a full
     9-hour run only to a variant that holds above ~50% speed and climbs rows.
   If v5 slows down the same way, the cluster's other options stand: a stall penalty or
   progress reward, or a lower termination cost relative to tracking
   (`coordination/results/2026-10-06-stairs-v4-stop-test.md`).
   **Be realistic about time**: this is now reward design for real stair heights, and three
   attempts have failed. It may take several more short iterations and at least one long run.
   Walking and person-following on flat ground and low steps (up to ~9 cm) do not need it:
   stairs v2 already does that in simulation.
2. **Read-only checks on the robot** (never done; they decide how the scan is produced):
   ```
   ros2 topic list | grep -i -E 'utlidar|height|odom|sportmode'
   ros2 topic hz /utlidar/cloud
   ros2 topic echo --once /utlidar/height_map_array | head -30
   ```
3. Optional: report format (the final report is Markdown with four figures; related-work
   citations other than SARO are unchecked).

### 2.2 Cluster session

- Log check of `foot_clearance` done (consistent, not conclusive). Ready to build v5 as a
  registered task and to run short trials when Rohan says go. Jobs 12586 and 12587 keep
  running until he says otherwise.
- After a run: its heights eval at 9 / 12 / 15 / 17 cm **with achieved speed and stalled
  fraction next to falls**, and an early stop if speed is under ~30% of commanded by
  iteration 1,500.

### 2.3 Laptop session

When a real-stairs checkpoint worth evaluating is on private HF
(`RohanRamesh/go2-specialists`, folder named after the experiment; intermediate checkpoints
land every 200 iterations and can be evaluated the same way), from `unitree_rl_mjlab/`:

```
# download (the token is in the repo's .env; scripts/switch_follow_generalist.sh shows the pattern)
STAIRS_CKPT=eval_ckpts/<experiment>/model_9999.pt TAG=<name> scripts/switch_follow_real_stairs.sh
STEPS=10 STAIRS_CKPT=... TAG=<name> scripts/switch_follow_real_stairs.sh
STAIRS_CKPT=... TAG=<name>_h15 EXTRA_FLAGS="--step-height 0.15" scripts/switch_follow_scan_faults.sh 900 905
STAIRS_CKPT=... TAG=<name> scripts/switch_follow_stairs_ckpt.sh        # still fine on easy ground?
```

**Acceptance**: at least 90% of flights crossed, up and down, at 17 cm, on 5- and 10-step
flights. Judge by **crossed / fell / lost** separately. A policy that never falls and never
crosses has failed (that was stairs v3).

If it passes: `scripts/export_policy_numpy.py <ckpt> --out deploy_numpy/<name>.npz` (it checks
parity with PyTorch), then the staged bring-up in the plan, section 3.

Not started, and independent of training: the scan node for the robot (waits on §2.1 item 2),
and scan faults at real riser heights (waits on a policy that climbs them).

No scheduled checks are pending. A check set for 07:12 on 10-06 was lost when the session
restarted; scheduled jobs live only inside one session.

## 3. What exists

**Policies** (private HF `RohanRamesh/go2-specialists`; copies under
`unitree_rl_mjlab/eval_ckpts/` on the laptop, gitignored):

| folder | what | use |
|---|---|---|
| `go2_spec_stairs_v2` | Stairs, command range held (job 12490) | **Best policy.** Risers up to ~9 cm. |
| `go2_spec_stairs_it4800` | original stairs run at iteration 4800 | superseded by v2 |
| `go2_spec_stairs` / `_rough` / `_flat` | the three original specialists | study only; stairs and rough were damaged by the curriculum collapse |
| `go2_generalist`, `go2_generalist_v2` | two generalists (jobs 12479, 12518) | study only; both brittle at stair edges |
| `go2_spec_stairs_v3` | first real-stairs attempt (job 12563) | **failed, do not use** |
| `go2_spec_gaps` | blended gaps run | cluster disk only, never evaluated |

**Harness** (`unitree_rl_mjlab/`): `scripts/switch_follow.py` is the runner (follow task;
`--random-layout`, `--step-height`, `--stair-steps`, `--obs-noise`, `--terminations saro`,
`--scan-delay/-dropout/-bias`, `--extra-policy name=ckpt`, arms `fixed:` / `hard:` / `soft:` /
`blind:` / `clf:`). Batch scripts: `switch_follow_real_stairs.sh`, `switch_follow_scan_faults.sh`,
`switch_follow_stairs_ckpt.sh`, `switch_follow_random_layouts.sh`, `switch_follow_generalist.sh`.
Analysis: `switch_follow_analyze.py` (per seed), `switch_follow_layouts.py` (layout as unit).
Robot side: `deploy_numpy/policy_numpy.py`, `scripts/export_policy_numpy.py`.

**Policy interface** (plan, section 2.3): 234 inputs = 47 proprioceptive + a 187-point height
scan (17 × 11 grid, 0.1 m, 1.6 × 1.0 m, heading-aligned, height of base above ground ÷ 5);
12 outputs, joint target = default + 0.25 × action at 50 Hz. The repo's C++ Go2 deploy stack
(`deploy/robots/go2`) supplies only the 47 and has no scan input.

## 4. What was done, by day

- **10-03**: cluster status check; root cause of weak specialists (curriculum collapse at
  iteration 5000); switching experiment pre-registered, calibrated and confirmed on one
  course; `vlm-pipeline` merged into `main`.
- **10-04**: generalist, pre-collapse stairs checkpoint and stairs v2 evaluated; generalist
  matrix and height-scan ablation; report rewritten.
- **10-05**: generalist v2 evaluated; diagnostic showed results depend on course layout; 20
  randomised layouts confirmed stairs v2 and overturned two fixed-course results; goal changed
  to the real Go2; stairs v2 found unable to climb real risers; real-stairs plan, numpy
  runner, scan-fault test; stairs v3 built by the cluster and submitted by Rohan.
- **10-06**: stairs v3 finished and failed (stands still), confirmed on the laptop; run 2
  (4a, 4b) built and submitted; 4a failed its early check and both were cancelled; the warm
  start's normaliser reset was found to cost speed from the first iterations (an overstated
  first account of this was corrected the same day); 4a and 4b resubmitted with the
  normaliser kept, and both failing again by iteration 150; the `foot_clearance` reward
  identified from the code as a candidate cause.

## 5. Gotchas (full list: `findings.md`, "Bugs found and fixed")

- **Judge a locomotion policy by where it gets to, not by whether it falls.** Stairs v3 has
  99% survival and near-zero falls per 100 m because it does not move (#26; earlier cases #1,
  #14). Always read achieved speed, stalled fraction, or crossing rate first.
- **A warm start should keep the observation normaliser** (#27). Reset, it cost a third of
  the speed at the start of v4a and the run never recovered. Judge the start of a run by speed
  and stalled fraction over the first few dozen iterations; the iteration-0 log line (episode
  length 17) looks the same either way and means nothing.
- **Results from one course layout are results about that layout** (#24, #25). Evaluate over
  `--random-layout` and analyse with `switch_follow_layouts.py`.
- **The twin env has observation noise and pushes off** unless `--obs-noise` (#22). Say which.
- **Repeat runs differ**: about 2 points with noise off, up to 3.5 with noise on. Pool.
- **What counts as a fall matters**: "training" terminations end a trial on any knee or shin
  contact over 10 N; `--terminations saro` only on tipping over.
- **The stock terrain curriculum does not work for stairs**: it promotes on ending more than
  4 m from the start, and the staircase is the inner 3 m of a patch
  (`coordination/results/2026-10-05-terrain-curriculum-diagnosis.md`).
- **Cluster jobs need Rohan in the cluster session.** That session's permission layer has
  refused `sbatch` and uploads on a relayed approval. It is also not always reachable by
  message; the committed `coordination/inbox/to-cluster.md` entry is the delivery that counts.
- **Laptop background jobs are killed two hours after launch**, waiting time included; chain
  fresh jobs, `--resume` skips finished arms. Scheduled checks die with the session.
- **The laptop GPU is shared** with other projects' sessions: check
  `nvidia-smi --query-compute-apps` and ask the owning session before launching.
- **Don't edit `scripts/switch_follow.py` while a multi-seed loop runs**; each seed re-reads it.
- **Set `PYTHONPATH` to this repo's `unitree_rl_mjlab/`** (see `CLAUDE.md`); the batch scripts do.

## 6. File map

| Want to know... | Read |
|---|---|
| The active plan for the real robot | `report_content/go2_real_stairs_plan.md` |
| The finished study, as a report | `report_content/final_report.md` |
| The study's numbers and intervals | `coordination/results/switch-follow-results.md` |
| What was fixed in advance | `coordination/results/switch-follow-preregistration.md` |
| Stairs v3 failure | `coordination/results/2026-10-06-stairs-v3-result.md` |
| Run 2 (4a, 4b): what was built and how its curriculum rule behaves | `coordination/results/2026-10-06-stairs-v4-built.md` |
| Why the curriculum sits on easy rows | `coordination/results/2026-10-05-terrain-curriculum-diagnosis.md` |
| Full experiment history, every bug | `findings.md` |
| The original research design and its outcome | `objective.md` |
| Cluster state and what is asked of it | `coordination/status/cluster.json`, `coordination/inbox/to-cluster.md` |
| Raw results | `unitree_rl_mjlab/eval_results/switch_follow/` (`random/`, `real_stairs/`, `scan_faults/`, `diag/`), `eval_results/stairs_heights/` (cluster) |
| Robot hardware notes | `/run/media/rohan/New Volume/RL/temp` (`project.md`, `sessions/`) |
| VLM/SARO pipeline, person-following | `findings.md` (sections dated 2026-09-17 to 09-22), `coordination/results/vlm-nav-*.md` |
