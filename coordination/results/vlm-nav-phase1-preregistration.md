# VLM navigation, Phase 1 — pre-registration (course calibration and "does the choice matter" gate)

**Written**: 2026-09-17, before any calibration or confirmation data exists · **Branch**:
`vlm-pipeline` · **Code**: `unitree_rl_mjlab/scripts/vlm_nav_baseline.py`,
`unitree_rl_mjlab/src/vlm_nav/{course,controllers,policy_bank,twin_env}.py`

## Why this phase exists

The VLM pipeline (SARO, arXiv:2407.16412, plus our modification: the VLM also picks which of the
flat / rough / stairs specialists runs) can only show a policy-choice effect if choosing matters on
the test courses. Gate 1 established that the best specialist varies by terrain, but under
different conditions (random commands up to 2 m/s, a 1200-step window, pyramid terrain patches).
Here the robot walks one straight course at 0.5 m/s toward a goal. If one fixed specialist does as
well as perfect switching, the VLM's policy choice cannot move any outcome metric and the
comparison would be uninformative however good the VLM is.

No VLM is involved. Every arm uses the same goal-seeking controller (P control toward the goal,
`controllers.goto_command`, v ≤ 0.5 m/s) and differs only in which specialist is active:

- `oracle`: the specialist for the ground-truth terrain under the robot's footprint
  (`CourseSpec.required_terrain`: base x − 0.35 m to base x + 0.30 m, non-flat wins)
- `flat`, `rough`, `stairs`: that specialist for the whole run

## Per-trial outcome definitions (fixed)

- **success**: base within 0.3 m (xy) of the goal before any termination or the time limit
- **fall**: a termination that isn't a timeout (`fell_over` or `illegal_contact`, the
  specialists' own training terminations)
- **timeout**: neither of the above within 2.5 × course length / 0.5 m/s + 10 s
- **crossed**: base passed the far edge of the last intermediation (SARO Table I's grey column)
- secondary: falls per 100 m of path, mean achieved speed (bug #1/#14 check), policy-match %

One trial per env; nothing pooled across episodes (bug #15).

## Code-check runs already done (not data, excluded from everything below)

A 4-env smoke run and three single-env traces on `stairs_up`, seed 0, used to find two code bugs.
Both are fixed. First, the lateral command was scaled by distance and saturated, walking the robot
off the goal line. Second, the point-based oracle switched back to `flat` with the hind legs still
on the stairs. The traces also showed the stairs specialist climbing 0.07 m risers slowly, with one
`illegal_contact` on the stairs, which is why the course difficulty is calibrated instead of assumed.

## Step 1 — calibration (oracle arm only)

- Courses: `stairs_up`, `stairs_down`, `rough`. Goal offset 0. Terrain/spawn seed **100**.
  32 trials each.
- Levels (`course.DIFFICULTY_LEVELS`): L1 riser 0.05 m / rough noise ≤ 0.06 m; L2 0.07 / ≤ 0.08;
  L3 0.09 / ≤ 0.10. All inside the training ranges.
- **Selection rule**: the hardest level at which oracle success ≥ 80% on all three courses.
  Harder is preferred because easy terrain makes all specialists equal by construction.
- If no level reaches 80%: stop and report to the user. The specialists can't reliably cross these
  intermediations even with perfect choice, which undermines the whole VLM comparison. Don't
  silently lower the bar.

## Step 2 — confirmation (all arms, selected level)

- Courses: `flat`, `stairs_up`, `stairs_down`, `rough`, `multi`. Goal offsets {−1.0, 0, +1.0}.
  Terrain/spawn seeds **{200, 201}** (fresh; never used for calibration). 32 trials per
  (course, offset, seed, arm), i.e. 192 per (course, arm).
- **Gate A (courses are crossable)**: oracle success ≥ 80% on every course.
- **Gate B (the choice matters)**: on `multi`, oracle success exceeds the best fixed
  specialist's success by ≥ 15 percentage points, with non-overlapping 95% Wilson intervals.
  Also reported, not gated: the same gap per single-intermediation course, and for the `flat`
  control, whether any fixed specialist beats oracle. It shouldn't. If one does, the footprint
  rule or the flat specialist is suspect.
- **If Gate B fails**: re-run the confirmation once at the next harder level, if it still passes
  Gate A. If it fails again, report it as a finding. In that case the VLM's policy choice can be
  scored only as a decision-accuracy metric, not by course outcomes, and the user decides whether
  to continue.
