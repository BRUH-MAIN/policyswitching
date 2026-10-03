# Switching while following: results

**Date**: 2026-10-03 · **Machine**: `romen` · **Branch**: `vlm-pipeline` ·
**Pre-registration**: `switch-follow-preregistration.md` (committed `6916211`, before any of
this data) · **Raw**: `unitree_rl_mjlab/eval_results/switch_follow/*.json` ·
**Code**: `scripts/switch_follow.py`, `src/vlm_nav/schedule.py`, `scripts/switch_follow_analyze.py`

## Headline

1. **Switching specialists by terrain beats every single specialist** on the mixed course:
   91.5% of trials cross it against 76.6% for the best fixed specialist (+15.0 points, 95% CI
   +11.4 to +18.6; 768 trials per arm, fresh seeds). This is the first course in the project
   where that holds; on single-obstacle courses one good fixed choice was always enough.
2. **The switch has to happen at the boundary.** Success peaks with the switch 0 to 0.3 m
   before a terrain boundary. Switching 0.3 m late costs about 30 points (16 with observation
   noise on). Switching 0.8 m or more early costs about 17 points (7 with noise on) and leaves
   the robot no better off than never switching.
3. **So the leader's preview has nothing to add to switch timing.** The best moment to switch
   is well inside the 0.8 m the robot's own height scan already covers. Using a longer horizon
   to switch earlier is worse in every condition tested, never better. This is the project's
   central hypothesis and it is **not supported**.
4. **Blending the two specialists' actions buys nothing** over a hard switch at the easy level
   and is worse at the harder one.
5. **How big effect 1 is depends on conditions the pre-registration did not vary.** With the
   training observation noise switched back on it shrinks to +4.2 points (CI +0.6 to +7.7).
   If only tipping over counts as a fall, every arm but the flat specialist is at 98-100% on
   this course and there is nothing to win; at the harder level the advantage reappears
   (+10.5 points). Findings 2-4 hold in every one of these conditions.

6. **The robot's own height scan is enough to time the switch.** A small classifier on the
   187-ray scan, driving the same hard switch, matches the ground-truth-label switch: 87.5% vs
   88.3% with the training observation noise on (difference -0.8 points, CI -4.1 to +2.5) and
   90.0% vs 90.6% with it off (-0.7, CI -3.6 to +2.3). It switches a median 0.17-0.28 m before
   each boundary and never after it. So on this course the leader adds neither horizon nor
   labels.

The sensing-matched generalist (arm 1) is not in any table: its training job has not run
(cluster submission blocked on a permission prompt, 2026-10-03).

## Setup, as run

Course `multi`, level L1 (0.05 m risers, rough noise 0.02-0.06 m): flat 3 m, rough 3 m, flat
2 m, stairs up 1.5 m, flat 2 m, stairs down 1.5 m, flat 3 m. The robot follows a point leader
walking the centreline at 0.5 m/s, 3.0 m ahead, with `controllers.follow_command`. One trial
per env; a trial succeeds when the base is 1.5 m past the last stair. Failures are **fall** (a
training termination: bad orientation, or a non-foot body touching with more than 10 N) or
**lost** (leader more than 6 m away). Every switching arm reads the true segment layout and
differs only in where, relative to each boundary, the incoming specialist takes over.

One condition was inherited rather than chosen: the twin env is built on the play config, so
**observation noise and pushes are off**. The specialists were trained with both on. Section
"Robustness" re-runs the confirmation with observation noise on.

## Stage 1: calibration (seeds 400, 401; 512 trials per arm)

Success %, `label` mapping. The `comp` mapping, which sends up-stairs to the rough specialist,
has the same shape: 91.8 at lead 0, 91.6 at lead 0.3, 73.8-76.2 from 0.8 m outward, and 41.2
at lead -0.3. Full table: `switch_follow_analyze.py calib`.

| hard switch, lead (m) | -0.6 | -0.3 | 0.0 | 0.3 | 0.8 | 1.5 | 2.5 |
|---|---|---|---|---|---|---|---|
| success % | 37.9 | 59.6 | 88.5 | **92.2** | 73.6 | 75.6 | 75.6 |

| cross-fade, from → to (m) | 0.3 → 0.0 | 0.8 → 0.0 | 0.8 → 0.3 | 1.5 → 0.3 | 1.5 → 0.8 | 2.5 → 0.3 | 2.5 → 0.8 | 2.5 → 1.5 |
|---|---|---|---|---|---|---|---|---|
| success % | **93.2** | 89.6 | 82.4 | 77.1 | 75.6 | 76.2 | 76.4 | 74.6 |

Fixed specialists: flat 29.7, rough 35.7, stairs 75.8. Figure:
`report_content/figures/switch_lead_sweep.png`.

Selected by the pre-registered rule: mapping `label` (92.2 vs 91.6 at lead 0.3); reactive hard
`hard:0.3`; reactive soft `soft:0.3:0.0`; anticipatory hard `hard:1.5`; anticipatory soft
`soft:1.5:0.3`.

## Stage 2: confirmation (seeds 500, 501, 502; 768 trials per arm)

| arm | success % (95% CI) | per seed | fall % | lost % |
|---|---|---|---|---|
| hard switch, 0.3 m ahead (reactive hard) | **91.5** (89.4-93.3) | 91.0 / 92.2 / 91.4 | 8.5 | 0.0 |
| cross-fade 0.3 → 0 m (reactive soft) | **90.8** (88.5-92.6) | 89.1 / 92.6 / 90.6 | 9.2 | 0.0 |
| hard switch, 1.5 m ahead (anticipatory hard) | **74.6** (71.4-77.6) | 73.0 / 72.3 / 78.5 | 25.4 | 0.0 |
| cross-fade 1.5 → 0.3 m (anticipatory soft) | **76.3** (73.2-79.2) | 74.2 / 75.8 / 78.9 | 23.7 | 0.0 |
| stairs specialist only | 76.6 (73.4-79.4) | 71.9 / 77.3 / 80.5 | 23.4 | 0.0 |
| rough specialist only | 37.4 (34.0-40.8) | 34.4 / 38.3 / 39.5 | 62.6 | 0.0 |
| flat specialist only | 25.7 (22.7-28.9) | 25.4 / 25.4 / 26.2 | 43.9 | 30.5 |

Figure: `report_content/figures/switch_confirm.png`.

Pre-registered claims:

- **H1, switching value: supported.** Hard 0.3 vs the best fixed specialist: +15.0 points
  (95% CI +11.4 to +18.6).
- **H2, blending: not supported.** Cross-fade vs hard: success -0.8 points (-3.6 to +2.1);
  action rate in the entry-boundary windows 1.021 vs 1.022 of the arm's own rollout mean
  (difference -0.002, CI -0.006 to +0.002); actuator-force rate the same (-0.001, -0.004 to
  +0.003). There is no boundary transient to smooth: a hard switch between these specialists
  raises the action rate in the +-0.5 m window by about 2% over the rollout average.
- **H3, anticipation: not supported; the effect has the opposite sign.** Hard 1.5 vs hard 0.3:
  -16.9 points (-20.6 to -13.3). Cross-fade from 1.5 m vs cross-fade from 0.3 m: -14.5 points
  (-18.1 to -10.8). The early arms are statistically indistinguishable from never switching
  (hard 1.5 vs stairs-only: -2.0, CI -6.2 to +2.3). They do show slightly lower action rate at
  boundaries (-0.006 and -0.015 relative), which the pre-registered rule counts only when
  success is within 2 points. It is not.

### Why early switching fails

Where the failed trials ended, % of 768 (confirmation seeds). "Lip" is the last 0.25 m of flat
ground before the top edge of the down-stairs.

| arm | on up-stairs | lip | on down-stairs |
|---|---|---|---|
| hard 0.3 | 0.1 | **0.0** | 7.2 |
| cross-fade 0.3 → 0 | 0.0 | **0.0** | 9.1 |
| hard 1.5 | 0.0 | **15.2** | 8.9 |
| cross-fade 1.5 → 0.3 | 0.0 | **13.5** | 9.2 |
| stairs only | 0.1 | **12.9** | 9.4 |
| rough only | 0.5 | 8.5 | 53.6 |
| flat only | 40.9 | 7.8 | 25.0 |

The whole cost of switching early is at the lip. A stairs specialist that has been walking the
flat approach touches a knee or calf to the edge in 13-15% of trials as it steps off; the flat
specialist walking the same approach and handing over 0.3 m before the edge never does. The
specialist is worse on the approach to its own terrain than the specialist it replaces, so the
earlier it takes over the more it loses. This matches the diagnostic from 2026-09-17 (switching
1 m before the edge made down-stairs worse) and is why the result does not depend on the
schedule family: any policy that hands over early inherits it. About 7-9% of trials are lost on
the down-stairs themselves under every arm that uses the stairs specialist there; no switch
timing touches that.

Late switching fails differently: the flat specialist is still in control on the up-stairs and
cannot climb (27.9% of trials end there at lead -0.3, calibration seeds).

## Robustness (same arms, labelled exploratory where not pre-registered)

Success %, difference vs on-time hard switch in brackets with its 95% CI where it is the point
of the row.

| condition | stairs only | hard 0.3 | switching value (hard 0.3 − best fixed) | hard 1.5 − hard 0.3 | cross-fade − hard 0.3 |
|---|---|---|---|---|---|
| **L1, as pre-registered** (seeds 500-502, n=768) | 76.6 | 91.5 | +15.0 (+11.4, +18.6) | −16.9 (−20.6, −13.3) | −0.8 (−3.6, +2.1) |
| L1, observation noise on (seeds 500-502, n=768) | 83.2 | 87.4 | +4.2 (+0.6, +7.7) | −6.9 (−10.6, −3.2) | +0.7 (−2.6, +4.0) |
| L1, orientation-only falls, pre-registered check (n=768) | 99.2 | 98.4 | −1.6 vs rough-only 100.0 (−2.7, −0.7) | +0.5 (−0.7, +1.8) | +1.2 (+0.2, +2.3) |
| L2 (0.07 m risers), training falls (seeds 600-601, n=512) | 33.4 | 51.0 | +17.6 (+11.6, +23.4) | −16.2 (−22.1, −10.2) | −6.4 (−12.5, −0.3) |
| L2, orientation-only falls (n=512) | 79.1 | 89.6 | +10.5 (+6.1, +15.0) | −2.3 (−6.3, +1.6) | −14.1 (−18.6, −9.5) |

Reading it:

- **Observation noise on.** Every fixed specialist does *better* with the noise it was trained
  with (stairs-only 76.6 → 83.2, rough-only 37.4 → 54.0), and the on-time switch slightly worse,
  so the switching advantage falls to +4.2 points with an interval that only just excludes 0.
  Early switching still costs 6.9 points; late switching (0.3 m) still costs 15.6.
  This is the more deployment-like condition and the more conservative number.
- **Orientation-only falls at L1.** All the L1 failures above are knee or calf contacts with a
  step edge. If those do not count, every arm but flat-only crosses 98-100% of the time and the
  course cannot separate anything. The +15 points is a statement about knee contacts, not about
  robots falling over.
- **L2.** With 0.07 m risers the pattern repeats under the training definition, and under
  orientation-only falls on-time switching is worth +10.5 points over the best fixed
  specialist. Early switching is then not significantly worse, and never better. The
  lead-0 switch, fine at L1, is 29 points worse than lead 0.3 at L2 (training falls): the
  window for an on-time switch narrows as the terrain gets harder.
- **Blending** is neutral at L1 and worse at L2 under both definitions.

## What this means for the project's claim

`objective.md` asks whether terrain preview beyond the onboard horizon improves switching, and
says the deliverable is gain as a function of horizon. Measured: the gain is zero from 0.3 m
outward and switching earlier than that is harmful, under ground-truth labels and exact
boundary positions, which is the best case for preview. The mechanism is not specific to the
schedule: these specialists are worse than their neighbours on the approach to their own
terrain, so handing over early costs more than any transient it could avoid, and there is no
measurable transient to avoid.

What the sweep does show is that **timeliness is everything**: 0.3 m late costs 16-30 points
depending on condition, 0.6 m late costs over 50. Whether the robot's own scan can deliver a
label that promptly is a separate question, tested next. That, not horizon, is where a followed
person could still have been worth something.

## Addendum: a real reactive arm (pre-registered in the addendum, before any classifier data)

`objective.md`'s arm 2a is a terrain classifier on the height scan, not ground-truth labels.
An MLP (187 -> 128 -> 64 -> 3) reads the `height_scan` slice of the actor observation and
predicts which specialist the footprint rule has active; its output passes through an
exponential average and a hold-to-switch filter and drives a hard switch. Trained on
single-obstacle courses (`rough`, `stairs_up`, `stairs_down`, L1, seeds 700/701, ~142k scans
from 768 trials); the `multi` course is never trained on. One classifier per sensing condition.

| | held-out per-step accuracy | recall flat / rough / stairs |
|---|---|---|
| noise off | 99.6% | 99.4 / 99.9 / 99.8 |
| training noise on (scan +-0.1 m) | 83.8% | 82.5 / 69.4 / 96.2 |

With noise on, a single scan confuses flat and rough about a quarter of the time (the rough
ground's 2-6 cm relief is under the +-10 cm ray noise) but almost never misses stairs. The
filter averages the rest away.

Filter chosen on seeds 400/401 by success (all four candidates were within 2 points of each
other and of the label-timed arm): alpha 0.1 / hold 5 with noise on, alpha 0.3 / hold 3 with
noise off. Confirmation, seeds 500-502, 768 trials per arm, same sensing condition in both arms:

| sensing | label-timed hard 0.3 | scan classifier | difference (95% CI) | classifier's lead at the three entries (median) | agreement with footprint rule |
|---|---|---|---|---|---|
| training noise on | 88.3% | 87.5% | -0.8 (-4.1 to +2.5) | +0.17 / +0.20 / +0.23 m | 84.4% of steps |
| noise off | 90.6% | 90.0% | -0.7 (-3.6 to +2.3) | +0.28 / +0.27 / +0.26 m | 92.7% of steps |

**H4: "onboard sensing suffices" holds in both conditions** (both intervals inside +-5 points).
The classifier is never late: in no trial did it first select a segment's class after the
boundary. Its disagreements with the footprint rule are mostly on the release side (staying on
a specialist a little longer after its segment) and, with noise on, brief flat/rough flicker
on ground where either specialist is fine.

Limit, as stated in the addendum: the stairs in `multi` are the same straight 0.05 m stairs the
classifier trained on, and the specialists it selects among are forgiving of flat/rough
confusion. This is the easy case for a classifier. It shows onboard sensing can be enough; it
does not show a scan classifier generalises to unseen stair geometry, where gate 2a's
difficulty-pooled measurement was much less favourable.

(Same seeds, same arm, different runs: hard 0.3 scored 91.5% and 90.6% with noise off, 87.4%
and 88.3% with noise on. Run-to-run spread from simulator nondeterminism is about 1 point.)

## Limits

- Single-seed specialists, and weak ones: both the stairs and rough specialists finished
  training on near-flat terrain (findings.md, "Why the specialists are weak"). A properly
  trained stairs specialist might not have the lip problem, which would flatten the early side
  of the curve; it would not make early switching *better* than on-time.
- One course family, one leader speed (0.5 m/s), one robot.
- The generalist comparison (the claim `objective.md` says the project rests on) is missing.
- Stage 1 selected the "anticipatory" arms as the best of a bad set; stage 2 confirms they are
  worse than on-time, which is the conclusion, but "best lead above 0.8 m" is not a tuned
  anticipatory controller. A controller with the long horizon is free to switch at 0.3 m; the
  result says that is what it should do.
