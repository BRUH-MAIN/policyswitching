# VLM navigation, Phase 1: calibration result (pre-registered)

**Date**: 2026-09-17 · **Machine**: `romen` · **Branch**: `vlm-pipeline` · **Pre-registration**:
`vlm-nav-phase1-preregistration.md` (committed `acb76c0`, before any of this data existed) ·
**Raw**: `unitree_rl_mjlab/eval_results/vlm_nav/phase1_calib/*.json`

## Headline

**Calibration fails under the pre-registered rule: at no difficulty level does perfect policy choice
get ≥ 80% success on all three intermediation courses.** Rough ground is fine. The stairs are the
problem: even the ground-truth oracle, choosing the right specialist at every step, fails 22–25% of
stairs crossings at the easiest level (0.05 m risers) and most of them at 0.07 m. The pre-registration
says to stop here and hand the decision to the user rather than lower the bar, so the confirmation
run (Step 2) has **not** been done.

This is a finding about the stairs specialist, not about the VLM: no VLM is in the loop, and the
two diagnostics below rule out the policy switch as the cause.

## Oracle success rate, 32 trials per cell (95% Wilson interval)

Seed 100, goal offset 0, 0.5 m/s goal-seeking controller.

| level | stairs_up | stairs_down | rough |
|---|---|---|---|
| L1 (riser 0.05 m, noise ≤ 0.06 m) | 75% (58–87) | 78% (61–89) | **100%** (89–100) |
| L2 (0.07 m, ≤ 0.08 m) | 28% (16–45) | 19% (9–35) | **100%** (89–100) |
| L3 (0.09 m, ≤ 0.10 m) | crashed at setup* | 0% (0–11) | 75% (58–87) |

\* `nconmax` overflow at environment setup, not during the run. The robot's pre-reset keyframe sits
at the world origin, which on this course is inside the L3 staircase (44 initial contacts vs.
`nconmax=35`). Fixed afterwards (twin env `nconmax=96`). Not re-run: L3 is already the hardest level
and fails on `stairs_down`. No completed run logged a runtime contact overflow at 35 (checked every
log), so the numbers above are not affected by dropped contacts.

## How the stairs fail

- **Every recorded fall is `illegal_contact`, none is `fell_over`.** A non-foot body (knee or calf)
  touched a step with > 10 N. That is the specialists' own training termination. SARO's fall
  definition (Appendix B.3) is orientation only (roll > 0.8 rad or pitch > 1.0 rad). An
  exploratory run under SARO's definition is below.
- **`stairs_down`: at the top edge.** At L2, 24 of 26 falls happen with the base at course x 2.8–2.9,
  as the front legs step off the platform onto the first tread (at x = 3.0). At L3, 30 of 32 do.
- **`stairs_up`: stuck at the top.** At L2, 20 of 32 trials time out with the base at x ≈ 4.4:
  front feet on the platform, hind feet at the last riser, walking in place for the rest of the
  50 s. At L1 the 3 falls are all at x = 4.5, the same place.

## Diagnostics: is it the switch? No.

The oracle switches flat→stairs when the front of the footprint reaches the first riser. The
`stairs_down` falls happen 0.1–0.2 m after that switch, so the switch transient was a real
suspect. Tested at L2, seed 100, 32 trials each (`unitree_rl_mjlab/logs/vlm_nav/diag/`):

| arm | stairs_down success | stairs_up success |
|---|---|---|
| oracle (switch at the edge) | 19% | 28% |
| stairs specialist throughout (no switch ever) | **3%** | **28%** |
| oracle switching 1 m before the edge | **3%** | — |

Removing the switch does not help. Running the stairs specialist over the approach makes
`stairs_down` *worse*. So the failures belong to the stairs specialist's locomotion on these
stairs at this command. Gate 1 only ever measured it at 0.05 m risers under random commands,
and nothing in gate 1 contradicts this.

## Code-level checks done along the way (all ruled out)

- Grout-line visual geoms: compiled model shows all 92 in group 4 with `contype=conaffinity=0`;
  only the 11 structural boxes collide.
- Contact overflow: no runtime overflow warnings in any completed log.
- Controller: lateral-command saturation bug found and fixed *before* calibration (pre-registration
  "code-check runs").
- Footprint rule: point-based oracle bug found and fixed *before* calibration.

## Decision needed (user)

1. **Accept L1 with a lower bar.** The stairs courses sit at 75–78% under perfect choice, so a
   VLM arm can at best match that. The comparison is still informative but noisier.
2. **Adopt SARO's fall definition** (orientation-only terminations, exploratory numbers below).
   This matches the "follow SARO's objective" instruction, but it is a post-hoc change of outcome
   definition and must be labelled as such, then confirmed on fresh seeds.
3. **Improve the stairs specialist** (retrain / fine-tune on linear stairs at 0.05–0.10 m). It needs
   the cluster, which was last reported drained.
4. **Drop stairs from the gated comparison.** Keep rough + flat, where oracle = 100%, and report
   stairs descriptively.

## Exploratory: fall definition × specialist, all arms (not pre-registered; seed 101, 32 trials each)

Rough-course arms are still queued. Success rate (falls / timeouts):

| level | fall definition | oracle | flat | rough | stairs |
|---|---|---|---|---|---|
| L1 (0.05 m) | training (knee contact ends trial) | 84% (6/9) | 0% (6/94) | **97%** (0/3) | 81% (9/9) |
| L1 (0.05 m) | SARO (orientation only) | 91% (0/9) | 0% (3/97) | **100%** (0/0) | 84% (0/16) |
| L2 (0.07 m) | SARO (orientation only) | 22% (0/78) | 0% (31/69) | 9% (0/91) | **34%** (0/66) |

`stairs_down`:

| level | fall definition | oracle | flat | rough | stairs |
|---|---|---|---|---|---|
| L1 (0.05 m) | training | **81%** (19/0) | 50% (50/0) | 25% (75/0) | 66% (34/0) |
| L1 (0.05 m) | SARO (orientation only) | 100% | 100% | 100% | 100% |
| L2 (0.07 m) | SARO (orientation only) | 100% | 91% (9/0) | 100% | 100% |

What this suggests, pending confirmation on more seeds:

1. **The choice matters.** The flat specialist can't climb at all: 0% everywhere, mostly stuck.
2. **The specialist named after the terrain is not always the best one on it.** At 0.05 m risers
   the *rough* specialist crosses up-stairs more reliably than the stairs specialist, under both fall
   definitions. At 0.07 m it's the other way round. A fixed "terrain label → specialist" rule is
   wrong in one direction or the other. The oracle arm is exactly that rule, so it is not an upper
   bound on the best choice. This is the argument in `rough-stairs-switching-decision.md` (choose by
   expected outcome, not by class) showing up in closed loop.
3. **Which fall definition you pick decides whether the choice matters on down-stairs.** Under the
   training definition, switching (oracle) beats every fixed specialist. Under SARO's, every
   specialist gets down 0.05 m stairs 32/32 without tipping over, so there is nothing for a policy
   choice to win.
4. **SARO's fall definition does not rescue L2.** The failures there are stalls at the top of the
   stairs (timeouts), not knee contacts.
5. Seed-to-seed spread is large at 32 trials: the L1 oracle was 75% on seed 100 and 84% on seed 101
   under the same definition.
