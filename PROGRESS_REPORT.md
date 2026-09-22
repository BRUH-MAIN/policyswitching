# Handoff / Progress Report

**As of**: 2026-09-22 · **Repo state**: branch `vlm-pipeline`, 20 commits ahead of `origin/main`,
**plus a large uncommitted working tree** (see §0.1 — this is the first thing to deal with).
This branch lives in its own linked worktree at `.claude/worktrees/vlm-pipeline`; the shared
checkout at the repo root stays on `main`. **If you are reading this from the `main` checkout,
`cd` into the worktree first** — none of the code or results below exist on `main`.
**Written for**: a fresh session (human or Claude) with no memory of how this state was reached.

---

## 0. How to resume

1. Confirm you are on `vlm-pipeline` (`git branch --show-current`), not `main`.
2. Read `README.md` → `objective.md` → `findings.md` (per `CLAUDE.md`), then this file.
   `findings.md` now has four sections for this branch; the last three are from 2026-09-21/22
   and are the current work.
3. Check what is actually running before assuming anything (§5). **This laptop has rebooted
   unexpectedly several times during this project**, silently killing background jobs.
4. Then go to §2 — that is the actionable part.

### 0.1 Nothing from 2026-09-21/22 is committed

`git status` shows ~17 modified/untracked paths. The work is real and tested but exists only
in this working tree, on one laptop that reboots without warning. **Committing this is the
highest-value, lowest-effort action available.** What is uncommitted:

- **New modules**: `src/vlm_nav/leader.py`, `detector.py`, `overlay.py`
- **New scripts**: `scripts/vlm_nav_follow.py`, `vlm_nav_make_detector_dataset.py`,
  `vlm_nav_train_detector.py`
- **Modified**: `controllers.py` (follow law), `perception.py` (bearing→world estimator),
  `camera.py` (`project_body`, `world_to_body`), `course.py` (`goal_marker` flag),
  `twin_env.py` (`leader=`), `prompts.py` (planner target prompt), `scripts/vlm_nav_run.py`
  (video overlay), `tests/test_vlm_nav.py` (20 tests, all passing), `findings.md`
- **Do NOT commit**: `unitree_rl_mjlab/yolo11n.pt`, `unitree_rl_mjlab/weights/` — model
  weights, per `CLAUDE.md`'s rule. Add them to `.gitignore` instead.

## 1. What this is

Two lines of work share this branch:

- **SARO replication + VLM specialist selection** (the 2026-09-16 pivot): build SARO's system
  (arXiv:2407.16412 — a VLM plans, perceives and double-checks sub-tasks to cross one terrain
  obstacle) in this simulator, modified so the VLM also picks which of the three trained
  specialists runs.
- **Person-following** (2026-09-21, new): `objective.md`'s original setting — a scripted
  leader the robot follows at a standoff — reduced to its locomotion core, plus a two-rate
  perception architecture (fast detector + slow planner) that came out of it.

## 2. What needs a decision from you — in priority order

1. **Commit and push this branch** (§0.1). Everything below is at risk until then.
2. **A 4B local VLM is not sufficient for SARO's perception step on this renderer**, and that
   is now measured at the paper's own protocol (§3.3), not inferred. The options are unchanged
   and still undecided:
   - a larger local VLM (Gemma-4-12B weights are already on the model drive, untested, likely
     will not fit in 8 GB beside the simulator),
   - a hosted API (sends frames off the laptop, costs money, `VLM` protocol is backend-agnostic
     so it is easy to wire),
   - **feed the VLM depth-derived geometry as text** instead of asking it to read the image —
     the depth estimator already locates edges to 0–8 cm, so this sidesteps the failure
     entirely, at the cost of "the VLM decides from what it sees",
   - make the rendered terrain more visually realistic (untested whether it moves the needle).
3. **The stairs specialist is still the blocker for the switching claim.** Unchanged from the
   previous report: 75–78% with a *perfect* chooser at the gentlest risers. Retraining needs
   the cluster. `coordination/status/cluster.json` was last updated 2026-09-12 and said all
   GPU nodes were drained since 2026-09-10; a request to the cluster session for current
   status went unanswered on 2026-09-20 (the message could not be delivered).
4. **Where to take person-following next.** The most valuable idea on the table is
   *anticipatory terrain switching while following*: the leader walks flat → rough → flat, the
   robot follows at a gap larger than its 0.8 m height-scan horizon, so the person is literally
   a preview of terrain the robot has not reached. That is `objective.md`'s thesis and every
   part now exists. **Prerequisite, not yet measured**: is there any cost to running the rough
   specialist on flat ground? If rough is as good everywhere, flat↔rough switching has nothing
   to show and this needs stairs (blocked by #3).

## 3. What's done and what it found

### 3.1 Person-following (2026-09-21)

`scripts/vlm_nav_follow.py`. A scripted kinematic leader (`src/vlm_nav/leader.py`) walks a flat
course at a speed that changes occasionally, including a full stop; the robot holds a standoff.

- **Ground-truth leader, 2.5 m gap: 0.159 m RMS gap error / 0.256 m max**, no falls, over 42 s
  and 7 speed changes.
- The leader is a fixed-base mjlab entity (auto-wrapped as a mocap body), **non-colliding and
  in the camera-only geom group**, so it cannot perturb the specialists' 187-dim `height_scan`.
  Same trick the goal flag uses. This is load-bearing, not cosmetic.
- **Two control defects the varying speed exposed**, both invisible to a constant-speed leader:
  proportional droop against a moving set-point (fixed with line-of-sight velocity
  feed-forward), and `FollowGains.v_max` being *below* the leader's top speed so the robot
  could not close a gap once opened. Both have regression tests.

### 3.2 Two-rate perception: YOLO tracker + VLM planner (2026-09-21)

Putting the VLM *inside* the control loop is why the robot lost the person: ~2 s/call and only
11 of 84 calls produced a usable position. The fix is structural — split *what* from *where*.

| arm | gap error (true) | wall/sim |
|---|---|---|
| VLM inside the control loop | 5.88 m | 10.4× slower than realtime |
| YOLO only | 0.488 m | 0.98× |
| **VLM planner + YOLO tracker** | **0.495 m** | **1.05×** |

- `src/vlm_nav/detector.py` is the seam (`Detector` protocol + `YoloDetector`). The VLM names
  the target class in language once (blocking, 0.7 s, before the robot moves) and re-confirms
  **asynchronously** every 5 s; YOLO localizes every control step at **3.95 ms**. 9/9 planner
  answers correct, 87.5% detector hit rate.
- **Stock COCO YOLO cannot see this project's leader** — reads the capsule legs as "baseball
  bat" (0.79 conf), the head sphere as "sports ball", and finds nothing at 6 m. Camera pitch is
  not the cause (re-tested at 0° and 8°). This is an appearance gap specific to flat-shaded
  mjlab geoms and **does not carry to hardware**, where a real person is COCO's home ground.
- **Fine-tuning fixes it and labels are free**: the leader's pose is ours and its geom extents
  fixed, so the exact 2D box is the projection of its 3D box (`CameraSpec.project_body`,
  round-trip verified to 6e-14 px). 1000 auto-labelled frames → YOLO11n, 40 epochs:
  **P 1.000, R 0.942, mAP50 0.951, mAP50-95 0.892**.
- ⚠️ **The fine-tuned weights are gitignored** (they live under a `logs/` path, which
  `unitree_rl_mjlab/.gitignore` excludes). Current location:
  `unitree_rl_mjlab/runs/detect/logs/vlm_nav/detector/leader/weights/best.pt`. A fresh clone
  will not have them — regenerate with the two scripts (~20 min total) or copy the file.

### 3.3 SARO task replication, paper protocol (2026-09-22)

SARO's own task (§III.A): goal-tracking across `{P1 → I → P2}`, **20 trials per
intermediation**, goals off-axis, full closed loop. `logs/vlm_nav/saro_protocol/`.

| intermediation | VLM Overall | VLM Across | ground-truth ceiling | planner's answer |
|---|---|---|---|---|
| stairs_up | **0%** | 0% | 75% | `none` 20/20 |
| stairs_down | **45%** | 45% | 95% | `none` 20/20 |
| rough | **100%** | 100% | 100% | `none` 17, `rough ground` 17, `stairs` 1 |

SARO's Table I (real robot, LLaVA-34B): stair 60/70/88, ramp 25/50/67, gap 45/80/94, door 30/50/63.

- **The failure is perception and it is upstream of everything.** On both stairs courses the
  planner answered `intermediation: none` on **40/40** trials — it never emits a `climb`
  sub-task because it never concedes there is anything to climb. The ground-truth arm crosses
  the same courses at 75%/95%, so the terrain is crossable and the locomotion works.
- **The rough 100% is NOT evidence of perception.** The planner still said `none` half the time
  there and the selector chose the *flat* policy on 267/391 calls — on rough ground — yet every
  trial passed, because every specialist survives rough. Same ceiling effect as gate B. Do not
  cite that column as validation.
- **Deviations from the paper, all forced**: intermediations are stairs_up/stairs_down/rough
  (no ramp or door segment exists here); low-level policies are the three specialists, not PAS
  (the PAS replication is degenerate, bug #14, and its estimator-only actor has a different
  observation width so it cannot enter the policy bank); VLM is Gemma-4-E4B on one 8 GB laptop
  GPU, not LLaVA-34B on an 8×3090 server; pose is ground truth. **Because localization is
  perfect here, compare against SARO's `Stable Loc` column, not its `Overall`.**

### 3.4 Still true from before this branch

Three specialists trained (flat/rough/stairs, 10k iters). Gate 1 (specialization matters)
passes robustly. Gate B fails: on a single-obstacle course one good fixed choice matches
perfect switching. Gemma cannot see the simulated stairs (Phase 2, 0/32) — now reconfirmed in
closed loop at 40/40.

## 4. Gotchas — read before trusting a new run

Full list in `findings.md` ("Bugs found and fixed"). Beyond the repo-wide ones in `CLAUDE.md`:

- **A follower's self-reported range cannot validate a follower.** The first VLM follow run
  scored a near-perfect 0.15 m gap error while the person walked away to 16.5 m — the robot had
  locked onto the **goal flag** and parked 6 m from it. The metric was computed from the very
  estimate under test. Only an independently logged ground-truth range exposed it. Fixed via
  `CourseSpec.goal_marker=False` for follow courses, but the lesson generalises: any real scene
  has other vertical objects.
- **The camera cannot see a person at close follow range.** Pitched 15° down with a 42.5°
  vertical FOV it sees ~6.25° above horizontal, so at 1.5–2.5 m only the leader's legs are in
  frame. Camera-driven following on this rig needs ~6 m standoff, or a re-aimed camera.
- **The VLM's horizontal localization is far better than its boxes** — 0.08° bearing error at
  6 m while the vertical extent was flatly wrong. Use the column, recover range from depth.
- **GPU contention is real on this 8 GB card.** Running the llama.cpp server and the simulator
  together has OOM'd the server mid-run (`ErrorOutOfDeviceMemory`). Use `CTX=4096 PARALLEL=1
  UBATCH=576`, or render frames first and query the VLM after. A Jupyter kernel unrelated to
  this project has also held ~5.6 GB — check `nvidia-smi --query-compute-apps` before blaming
  your own job, and do not kill processes you did not start.
- **The VLM model drive is NTFS and unmounts on reboot**: `udisksctl mount -b /dev/nvme0n1p4 -o ro`.
- **An instruction that names the answer contaminates the planner.** The follow task string is
  "follow the person ahead of you", so the planner choosing `person` demonstrates language→class
  mapping, not visual grounding. Phase 2 was bitten by the same thing.
- **`pgrep -f <pattern>` matches your own shell** when the pattern appears in your command line.
  Several "still running" readings this session were false positives. Use `pgrep -x`.

## 5. Operational state right now

- VLM server: **stopped**. Restart with `scripts/vlm_server.sh` (mount the drive first).
- GPU: free. No background jobs, no queued runs.
- Tests: **20/20 passing** (`PYTHONPATH=$PWD python3 -m pytest tests/test_vlm_nav.py -q`).
- `ultralytics` was installed into the `unitree_rl_mjlab` conda env on 2026-09-21. A dry-run
  confirmed it added packages only — torch/torchvision/numpy were untouched.
- Videos: nine, in `/home/rohan/policyswitching_videos/` (outside the repo, not committed).
- Other Claude sessions may share this laptop/GPU — announce before launching a GPU job.

## 6. File map

| Want to know... | Read |
|---|---|
| Condensed version of §3 | `findings.md`, the four `vlm-pipeline` sections |
| SARO protocol numbers + transcripts | `logs/vlm_nav/saro_protocol/` |
| Person-following runs | `logs/vlm_nav/follow_*`, `trace.json` per run |
| How to run anything | `--help` on `scripts/vlm_nav_*.py`; each has a usage example in its docstring |
| The pipeline code | `src/vlm_nav/` (camera, course, controllers, detector, executor, leader, overlay, perception, prompts, vlm_backend, oracle_vlm, policy_bank, twin_env) |
| Unit tests (no sim/VLM needed) | `tests/test_vlm_nav.py` |
| Pre-pivot state (specialists, gate 1/2) | `main`'s `PROGRESS_REPORT.md` |
