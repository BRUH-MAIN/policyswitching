# Handoff / Progress Report

**As of**: 2026-10-08, 11:30 IST · **Deadline (Rohan, 10-07)**: on the robot by Sunday 2026-10-11 · **Written for**: a fresh session (human or Claude) with no
memory of how this state was reached. This file is the entry point; where it needs more
detail it names a file. Read that file rather than re-deriving it.

---

## 0. How to resume

1. `git pull --rebase` in `/home/rohan/rl/policyswitching` (branch `main`; everything is here).
2. Read this file, then `unitree_rl_mjlab/deploy_numpy/README.md` (the robot bring-up, the
   active work), `report_content/go2_real_stairs_plan.md` (background), and `findings.md` if
   you are about to trust or produce a number.
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

**A policy now passes the simulation acceptance: 5- and 10-step flights at 9-17 cm, up and
down, at 95% or better with tipping over as the failure** (stairs v8a, laptop run,
`model_2400`). It touches the steps with its shins on most tall descents. Nothing has run on
the robot. Rohan's deadline is Sunday 2026-10-11, and on 10-07 he gave the laptop session
leave to do anything project-related here, training code included.

| | |
|---|---|
| Stairs v2 on real riser heights (sim, going up a 5-step flight) | 84% at 9 cm, 0% at 12, 15 and 17 cm. It stalls; it was trained on at most 10 cm. |
| Stairs v3, first attempt at 5-20 cm (job 12563, finished 10-06) | **Failed: it learned to stand still.** 12% of commanded speed at every riser; crosses 0 of 128 flights. Its near-zero fall rate is that, not success. Do not use the checkpoint. `coordination/results/2026-10-06-stairs-v3-result.md` |
| Runs v3, v4a, v4b (5-20 cm risers) | All failed the same way: the policy got safer by getting slower (12-30% of commanded speed), with reward unchanged. All cancelled or discarded. |
| **Cause found: the `foot_clearance` reward** | It measures each foot's height in the world frame against a fixed 0.10 m, so on a raised staircase it charges every moving foot for the staircase's height. **Shown by intervention** (cluster, 600-iteration trials of the 4b task with one change): term removed (v5a) 67% of commanded speed and mean row 3.3 and rising; term measured above the lowest foot (v5b) 66% and 2.9; unchanged (v4b) 30% and 0.04. `findings.md` #28, `coordination/results/2026-10-06-stairs-v5-clearance-trials.md` |
| Stairs v5a, final (job 12594, finished 10-07 16:06) | **Not a real-stairs policy.** Flights, 256 trials per cell, tipping-only: up 100 / 96.5 / 0 / 0% at 9 / 12 / 15 / 17 cm, down 100 / 76 / 0 / 0%. Every failure is a stall. Mean row 3.45 at the end. `coordination/results/2026-10-07-stairs-v5a-final-and-reward-diagnosis.md` |
| **Why it stalls: the reward pays for refusing** (`findings.md` #29) | In the training env a robot told to walk that trots on the spot keeps 2.6 of ~3.0 reward/s; on a 15 cm flight it earns ~1.8. Measured with `scripts/diag_reward_terms.py`, then tested by training. |
| Stairs v6a (`Unitree-Go2-Spec-StairsV6a`), finished 10-08 02:30 | v5a with the posture and gait rewards multiplied by achieved/commanded speed; 6,000 laptop iterations from v5a final (`go2_spec_stairs_v6a_lap/` on HF). **Final checkpoint, 256 trials per cell, tipping only: up 100 / 99.6 / 85 / 3%, down 100 / 100 / 100 / 99% at 9 / 12 / 15 / 17 cm.** With shin contact counted as a fall: up 83 / 95 / 57 / 1%, down 27 / 39 / 29 / 34% (it brushes steps on two descents in three). **Its 15 cm ascent swings between 0% and 98% from checkpoint to checkpoint** (`findings.md` #31), so the checkpoint for the robot has to be chosen by full evaluation; `model_1600` is the first candidate (98% on 64 trials). `coordination/results/2026-10-08-stairs-v6a-final.md` |
| **Stairs v7a** (`Unitree-Go2-Spec-StairsV7a`): v6a plus a reward for height gained on up-flights | Two runs. **Cluster, 8,192 envs (job 12613, `go2_spec_stairs_v7a/`, from the laptop's v7a `model_1600`; running to ~09:50 on 10-08): `model_1600` passes the 5-step acceptance**, 256 trials per cell, tipping only: up 100 / 100 / 100 / 98.8%, down 100 / 100 / 100 / 98.8% at 9 / 12 / 15 / 17 cm (shin contact counted: up 95 / 88 / 87 / 48, down 85 / 92 / 68 / 70). `model_2000` is 100% on the quick flights too. **On 10-step flights it fails above 12 cm**: up 100 / 100 / 93 / 45, down 100 / 100 / 72 / 41, tipping over on 28% of 15 cm and 59% of 17 cm descents. Laptop, 1,536 envs (`go2_spec_stairs_v7a_lap/`, finished): a 15 cm policy only (17 cm up 30-69%); the batch size was the difference. Exported for the robot as `deploy_numpy/stairs_v7a_c1600.npz` (not in git); through the runner and map sampler it climbs a 17 cm flight in plain MuJoCo. |
| **Stairs v8a, the current candidate** (`Unitree-Go2-Spec-StairsV8a`): v7a on 10-step flights | Laptop run from cluster v7a `model_1600`, **1,024 envs** (1,536 does not fit this terrain; the 07:00 launch died 30 times), 4,000 iterations, ends ~11:35 on 10-08, HF `go2_spec_stairs_v8a_lap/`. **`model_2400`, 256 trials per cell, tipping only, at 9 / 12 / 15 / 17 cm: 10-step flights up 100 / 100 / 99.6 / 95.3%, down 100 / 100 / 100 / 99.6%; 5-step flights 100% in all eight cells.** With shin contact counted as a failure: 10-step down 65 / 52 / 5 / 5%. Exported as `deploy_numpy/stairs_v8a_lap2400.npz`; climbs a 17 cm flight through the runner in plain MuJoCo. Scan-fault runs in progress. The cluster's 8,192-env v8a (job 12614, `go2_spec_stairs_v8a/`) is running; its `model_600` matches on the quick flights. `coordination/results/2026-10-08-stairs-v6a-final.md` section 10 |
| Stairs v5c (cluster job 12608, finished) | v5a's reward with half the robots on uniform rows. **Closed: `model_800` and the final `model_3999` cross no flight at 12-17 cm in either direction.** |
| Flight numbers depend on the commanded speed (`findings.md` #30) | v5a on a 12 cm up-flight at a constant command: 7 of 32 at 0.5 m/s, 28 of 32 at 0.8 m/s. The follow eval's controller speeds up when the robot lags, which is why it reports 96.5%. |
| Warm start | Should keep the observation normaliser (`findings.md` #27); it is now the default for the stairs specs. |
| Height scan on the robot | `deploy_numpy/go2_scan_node.py` written against the firmware's LiDAR height map (`rt/utlidar/height_map_array`) plus a pose, with the vertical offset anchored on the loaded feet. **Never run on the robot**; `--probe` is the first thing to run there. Its sampler passes the simulator test from a noisy map with holes and drift (median error 0.7 cm). The scan must be terrain only: the trained scan also sees the robot's legs on a few cells, and v5a does not care (`findings.md`, checked 10-07). |
| Running the policy on the Jetson | `deploy_numpy/go2_runner.py`: stand-up, 50 Hz policy loop, remote, safety stops, `--dry-run`. Observation builder equals the simulator's to 1e-5; the runner stands the robot up and climbs a 12 cm flight in plain CPU MuJoCo. **Never run on the robot.** Runbook: `deploy_numpy/README.md`. |
| Person-following on the robot | `deploy_numpy/follow_cmd.py` (laptop): perception hub -> velocity command over UDP, used by the runner only while R1 is held. Untested. |
| How good the robot's scan must be | Measured on stairs v2, low steps: 200 ms delay and 60% stale cells cost under 2 points; a ±6 cm height offset costs 19. Height above ground good to ~3 cm. |

## 2. What is waiting on whom

### 2.1 Rohan

1. **The robot session** (the long pole now; nothing here has touched the robot). Follow
   `unitree_rl_mjlab/deploy_numpy/README.md` stage by stage with the laptop session:
   - Stage 0, nothing moves (15 minutes): `go2_scan_node.py --probe` and
     `go2_runner.py --dry-run`. These replace the `ros2 topic` checks asked for earlier and
     decide how the height scan is produced.
   - Stage 1-3: legs free, standing, walking on flat ground with `--scan flat`. Needs no
     LiDAR and no new policy; stairs v2, v5a or v6a all do.
   - Stage 4-5: the real scan, then steps, only up to the riser the policy passes in
     simulation.
2. **Cluster**: v5c (12608) is still pending. Whether to replace it with
   `SPEC=StairsV6a BUDGET=4000` is yours to say in the cluster session (inbox entry of 10-07
   night has the evidence). The laptop run does not depend on it.
3. **What can go on the robot today**: `deploy_numpy/stairs_v8a_lap2400.npz` (laptop v8a
   `model_2400`; re-export it, weights are not in git). In simulation it crosses 5- and
   10-step flights up to 17 cm, up and down, 95-100% of the time without tipping over. It
   brushes steps with its shins on most tall descents. Tread in these tests is 0.30 m
   (training also has 0.26 m). **Measure the demo staircase: riser, tread and steps per
   flight**, and stay inside what was tested.

### 2.2 Cluster session

- v5a (12594) finished. v5c is job 12608, pending, warm start from v5a `model_4400` (Rohan's
  choice there). Its status file edits were uncommitted on the night of 10-07.
- Open suggestion from the laptop: run `SPEC=StairsV6a` at 8,192 envs (see 2.1 item 2).

### 2.3 Laptop session

**Running now**: stairs v8a on the laptop GPU (v6a and the laptop v7a are finished).
```
systemctl --user status go2-v8a-lap                       # the training unit
python3 coordination/scripts/stairs_run_status.py unitree_rl_mjlab/logs/train_StairsV8a_lap.log
grep terrain_row_mean_up unitree_rl_mjlab/logs/train_StairsV8a_lap.log | tail -3    # ascent, which the mean row hides
cd unitree_rl_mjlab && STEPS=10 STAIRS_CKPT=logs/rsl_rl/go2_spec_stairs_v8a_lap/<run>/model_<N>.pt TAG=v8alap_c<N> scripts/switch_follow_real_stairs.sh
cd unitree_rl_mjlab && scripts/quick_flights.sh <ckpt> <tag>          # 5-step, 64 trials: a look, not a result
PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/diag_reward_terms.py --task Unitree-Go2-Spec-StairsV7a \
    --checkpoint <ckpt> --step-height 0.15 --num-envs 128      # share of robots crossing, up and down
```
An eval can share the GPU with it at up to ~128 envs; at 256 the eval runs out of memory
(the training restarts itself from its last checkpoint if it is the one killed). When it
ends: the full acceptance below on **several** checkpoints, not only the last (#31), then
export the chosen one and run the README's checks. Still owed for v6a: the full acceptance
on `model_1600` (started 03:00 on 10-08) and the 10-step flights.

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
crosses has failed (that was stairs v3). For v5 policies read the `saro` rows first (only
tipping over is a fall): they are trained with thigh and calf contact penalised, not terminal,
so the "training" rows count every shin brush as a fall. Report both.

Intermediate checkpoints (every 200 iterations) can be evaluated the same way while 12594
runs: download `go2_spec_stairs_v5a/model_<N>.pt` and point `STAIRS_CKPT` at it with a `TAG`
that names the iteration, e.g. `TAG=stairsv5a_it2000`. The 10-06 results for iterations 400
and 599 are `eval_results/switch_follow/real_stairs/v5a_it400_*` and `v5ab_it599_*`.

If it passes: `scripts/export_policy_numpy.py <ckpt> --out deploy_numpy/<name>.npz` (it checks
parity with PyTorch), then the staged bring-up in the plan, section 3.

Waiting on the robot: the scan node's assumptions about the firmware's height map and pose
(README stage 0), and scan faults at real riser heights (waits on a policy that climbs them).

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
| `go2_spec_stairs_v4a`, `_v4b`, `_v4a_kn`, `_v4b_kn` | run 2 and its reruns | **failed (slow), do not use** |
| `go2_spec_stairs_v5a`, `_v5b` | 600-iteration trials with the clearance term fixed; v5a continues as job 12594 | in progress; early checkpoints only |
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
  identified from the code as the cause and confirmed by the cluster's v5 trials; full v5a
  run submitted (12594).
- **10-07**: v5a at iteration ~4,400 walks at full speed and has plateaued on 12-14 cm rows;
  laptop flights: up 100% to 12 cm, 0% at 15-17 cm; down refuses from 12 cm. v5c queued.
  v5a finished, unchanged apart from descending 12 cm. Night: per-term reward diagnosis
  (the reward pays for refusing), StairsV6a built and training on the laptop GPU, descents
  to 17 cm solved by iteration 600; robot-side runner, observation builder, scan node and
  follow bridge written and checked against the simulator; Sunday deadline set.
- **10-08, early**: v6a finished (up to 15 cm, down to 17 cm; ascent unstable across
  checkpoints; a 17 cm climb unpaid). v5c found to cross nothing. The runner passed a DDS
  loopback test against a pretend Go2. StairsV7a (reward for height gained) started.
- **10-08, morning**: laptop v7a is a 15 cm policy; the cluster's v7a at 8,192 envs passes the
  5-step acceptance at 17 cm both ways (`model_1600`) and fails on 10-step flights. StairsV8a
  (10-step training flights) started on the laptop. The cluster session dropped off the
  session list at ~07:05; the request for v8a at 8,192 envs is in its inbox.
- **10-08, late morning**: the 07:00 laptop v8a launch had died of memory (unnoticed for two
  hours); relaunched at 1,024 envs. Its `model_2400` passes the acceptance on 5- and 10-step
  flights. Cluster v8a (12614) started.

## 5. Gotchas (full list: `findings.md`, "Bugs found and fixed")

- **Judge a locomotion policy by where it gets to, not by whether it falls.** Stairs v3 has
  99% survival and near-zero falls per 100 m because it does not move (#26; earlier cases #1,
  #14). Always read achieved speed, stalled fraction, or crossing rate first.
- **Price the do-nothing policy** (#29): posture, gait and angular-tracking rewards paid a
  stalled robot 85% of a walking one's reward, so tall flights were refused. Any new reward
  term: ask what a robot standing still earns from it.
- **Batch size decided 17 cm**: StairsV7a at 1,536 envs (laptop) stayed at 30-69% going up
  17 cm for 6,000 iterations; at 8,192 envs (cluster) it was at 98-100% within 1,600. The
  laptop is for quick trials of a design, not for the final policy.
- **Test the flight length you will meet**: a policy trained on 5-step flights tips over on
  10-step ones. `STEPS=10 scripts/switch_follow_real_stairs.sh`.
- **Evaluate several checkpoints, fully, before choosing one** (#31): v6a's 15 cm ascent
  reads anywhere from 0% to 98% depending on the checkpoint, and the mean row shows nothing.
- **Quote the command with a flight number** (#30): the same policy climbs a flight at
  0.8 m/s and stops at its foot at 0.5 m/s.
- **`foot_clearance` uses world-frame foot height** (#28): on any raised terrain it punishes
  moving. Every stairs policy before v5 trained under it. Do not reuse the stock term on
  stairs.
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
| How to bring the policy up on the robot | `unitree_rl_mjlab/deploy_numpy/README.md` |
| The plan for the real robot, background | `report_content/go2_real_stairs_plan.md` |
| v5a final result and the reward diagnosis | `coordination/results/2026-10-07-stairs-v5a-final-and-reward-diagnosis.md` |
| The finished study, as a report | `report_content/final_report.md` |
| The study's numbers and intervals | `coordination/results/switch-follow-results.md` |
| What was fixed in advance | `coordination/results/switch-follow-preregistration.md` |
| Stairs v3 failure | `coordination/results/2026-10-06-stairs-v3-result.md` |
| Why v4 was slow, and the v5 trials that fixed it | `coordination/results/2026-10-06-stairs-v4-stop-test.md`, `coordination/results/2026-10-06-stairs-v5-clearance-trials.md` |
| Run 2 (4a, 4b): what was built and how its curriculum rule behaves | `coordination/results/2026-10-06-stairs-v4-built.md` |
| Why the curriculum sits on easy rows | `coordination/results/2026-10-05-terrain-curriculum-diagnosis.md` |
| Full experiment history, every bug | `findings.md` |
| The original research design and its outcome | `objective.md` |
| Cluster state and what is asked of it | `coordination/status/cluster.json`, `coordination/inbox/to-cluster.md` |
| Raw results | `unitree_rl_mjlab/eval_results/switch_follow/` (`random/`, `real_stairs/`, `scan_faults/`, `diag/`), `eval_results/stairs_heights/` (cluster) |
| Robot hardware notes | `/run/media/rohan/New Volume/RL/temp` (`project.md`, `sessions/`) |
| VLM/SARO pipeline, person-following | `findings.md` (sections dated 2026-09-17 to 09-22), `coordination/results/vlm-nav-*.md` |
