# Switching while following: results

**Date**: 2026-10-03, extended 2026-10-04 (generalist, better stairs checkpoint) · **Machine**: `romen` ·
**Pre-registration**: `switch-follow-preregistration.md` (committed `6916211`, before any of
this data) · **Raw**: `unitree_rl_mjlab/eval_results/switch_follow/*.json` ·
**Code**: `scripts/switch_follow.py`, `src/vlm_nav/schedule.py`, `scripts/switch_follow_analyze.py`

## Headline

Read in this order; items 7 and 8, added on 2026-10-04, change how items 1 and 3 should be read.

1. **With the specialists as trained, switching by terrain beats every single specialist** on
   the mixed course: 91.5% of trials cross it against 76.6% for the best fixed specialist
   (+15.0 points, 95% CI +11.4 to +18.6; 768 trials per arm, fresh seeds, as pre-registered).
   Pooled over every run of the two arms: +14.6 points with sensor noise off, +7.3 with the
   training sensor noise on (2,304 trials per arm each).
2. **The switch must not be late.** 0.3 m late costs 13 to 33 points at the easy level and
   43 to 65 at the harder one, depending on the stairs policy and the noise condition.
3. **Switching early is never better than switching at the boundary.** With the stairs
   specialist as trained it costs 7 to 17 points and leaves the robot no better off than never
   switching. With an undamaged stairs policy (item 8) it costs nothing and gains nothing.
   Either way, a preview beyond the 0.8 m the robot's own scan covers has nothing to add. The
   project's central hypothesis is **not supported**.
4. **Blending the two specialists' actions buys nothing** over a hard switch at the easy level
   and is worse at the harder one.
5. **Which condition a number comes from matters.** The experiment as pre-registered ran with
   observation noise off (inherited from the evaluation environment); the specialists were
   trained with it on. If only tipping over counts as a fall, every arm but the flat specialist
   is at 98-100% at the easy level.
6. **The robot's own height scan is enough to time the switch.** A small classifier on the
   187-ray scan, driving the same hard switch, matches the ground-truth-label switch: 87.5% vs
   88.3% with the training observation noise on (-0.8 points, CI -4.1 to +2.5) and 90.0% vs
   90.6% with it off (-0.7, CI -3.6 to +2.3). It switches a median 0.17-0.28 m before each
   boundary and never after it.
7. **Against the matched generalist, switching wins only under sensor noise.** The generalist
   (same observations, rewards, budget and curriculum as a specialist, trained on all terrain)
   crosses the course 90.5% of the time with noise off, the same as switching (89.3%;
   difference -1.2, CI -3.4 to +0.9), and 82.5% with noise on, where switching holds at 90.9%
   (+8.4, CI +6.0 to +10.8). 1,536 trials per arm per condition, two independent runs that agree.
8. **The largest effect in the study is how the stairs policy was trained, not how policies are
   switched.** Put the stairs run's own checkpoint from before its curriculum collapsed
   (iteration 4800 instead of 9999) in the stairs slot and that one policy, used alone, crosses
   the whole course 97.7% of the time (94.9% with noise on). Switching adds nothing to it
   (-1.0 points, CI -2.8 to +0.7), early switching costs nothing (+0.7, CI -1.1 to +2.4), and it
   beats both the switching system built on the as-trained specialists and the generalist. At
   the harder level it lifts stairs-only from 33.4% to 82.8% and switching from 51.0% to 86.5%.

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
  with (stairs-only 76.6 -> 83.2, rough-only 37.4 -> 54.0), and the on-time switch slightly worse,
  so in this run the switching advantage falls to +4.2 points with an interval that only just
  excludes 0. Early switching still costs 6.9 points; late switching (0.3 m) still costs 15.6.
  *Added 2026-10-04:* the same two arms were run twice more with noise on (as controls in the
  generalist runs, seeds 500-502 again and 503-505). On-time switching scored 87.4, 90.9 and
  90.9%, stairs-only 83.2, 82.9 and 81.1%. Pooled over all three runs (2,304 trials per arm):
  **89.7% vs 82.4%, +7.3 points (CI +5.3 to +9.3)**. The +4.2 above was the low end of that
  run-to-run spread; with noise on, repeat runs of one arm on the same seeds differ by up to
  3.5 points, because the noise draws differ. The pooled figure is the one to quote.
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
says the deliverable is gain as a function of horizon. Measured, under ground-truth labels and
exact boundary positions, which is the best case for preview: with the specialists as trained
the gain is zero from 0.3 m outward and switching earlier than that is harmful; with an
undamaged stairs policy (Addendum 2B) it is zero everywhere from 0.3 m outward and early
switching is merely harmless. In neither case is there anything for a longer horizon to buy,
and there is no measurable boundary transient for an earlier, gentler hand-over to smooth.

What the sweep does show is that **lateness is what costs**: 0.3 m late loses 13-33 points at
L1 and 43-65 at L2. Whether the robot's own scan can deliver a label that promptly is a
separate question, tested next. That, not horizon, is where a followed person could still have
been worth something.

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
boundary. Where its remaining disagreements with the footprint rule fall was not broken down.
One pattern is visible in the lead distribution: for one stairs segment in each condition the
90th-percentile lead sits at the 1.5 m window limit, so in at least a tenth of trials it was
already on the stairs specialist well before that staircase (not released since the previous
one, or selected early). Success is unaffected at this sample size.

Limit, as stated in the addendum: the stairs in `multi` are the same straight 0.05 m stairs the
classifier trained on, and the specialists it selects among are forgiving of flat/rough
confusion. This is the easy case for a classifier. It shows onboard sensing can be enough; it
does not show a scan classifier generalises to unseen stair geometry, where gate 2a's
difficulty-pooled measurement was much less favourable.

(Same seeds, same arm, different runs: hard 0.3 scored 91.5% and 90.6% with noise off, 87.4%
and 88.3% with noise on. Run-to-run spread from simulator nondeterminism is about 1 point.)

## Addendum 2A: the matched generalist (pre-registered before it was run)

`go2_generalist/model_9999.pt`, cluster job 12479: stock PPO on the union of the four terrain
classes, otherwise identical to a specialist (observations, rewards, 10k iterations, seed 42,
and the same two-stage command curriculum). Run as a fixed policy next to the on-time switch
and the stairs specialist. First run on seeds 500-502; because the two noise conditions
disagreed, a replication on fresh seeds 503-505 was registered before it was run. Both runs
are reported and pooled; they agree.

| sensing | generalist | hard switch 0.3 | stairs only | switch − generalist (95% CI) | by run |
|---|---|---|---|---|---|
| noise off (n = 1,536) | **90.5%** (88.9-91.9) | 89.3% (87.6-90.7) | 74.8% | **−1.2 (−3.4 to +0.9)** | −0.7, −1.8 |
| training noise on (n = 1,536) | **82.5%** (80.5-84.3) | 90.9% (89.3-92.2) | 82.0% | **+8.4 (+6.0 to +10.8)** | +9.2, +7.6 |

Per seed, generalist: 89.5 / 89.8 / 91.4 / 89.8 / 92.2 / 90.2 with noise off; 82.0 / 81.6 /
81.2 / 79.7 / 83.2 / 87.1 with noise on. Figure: `report_content/figures/switch_generalist.png`.

- **It is walking, not bracing** (bugs #1, #14): it never loses the leader (lost 0.0% in both
  conditions), its per-step velocity-tracking error is 0.170 m/s against 0.169 for the switching
  arm, and successful trials take the same 27.3 s over the same 13.6 m.
- **Pre-registered claim, "switching beats the matched generalist": holds with sensor noise on,
  not with it off.** The addendum said that if the conditions disagreed, that would be the
  result. It is.
- **Where the generalist loses with noise on**: at the lip of the down-stairs (0.7% of trials
  with noise off, 5.4% with it on) and on the down-stairs themselves (8.3% -> 11.2%). The
  specialists move the other way: every one of them does better with the noise on. A natural
  reading is that the generalist has to read the scan to know what terrain it is on, so scan
  noise costs it, while a specialist can afford to lean on it less. That is an inference from
  the failure locations, not something tested here.
- With noise on the generalist is no better than the stairs specialist alone (82.5% vs 82.0%,
  +0.5, CI -2.2 to +3.2). With noise off it is 15.7 points better.
- The generalist shares the specialists' handicap: it trained under the curriculum that
  collapsed at iteration 5000 for both of them. Whether its own terrain level collapsed was
  not checked.

## Addendum 2B: a stairs policy from before the curriculum collapse (pre-registered before it was run)

`go2_spec_stairs_it4800/model_4800.pt` is the Stairs training run's own checkpoint at
iteration 4800, the last one saved before the command range widened and the terrain
curriculum collapsed (findings.md, "Why the specialists are weak"). The cluster measured it at
2.6-3.6x fewer falls per 100 m than `model_9999` on pyramid stairs
(`2026-10-03-stairs-precollapse-checkpoint-eval.md`). Here it replaces `model_9999` in the
stairs slot of the bank; nothing else changes. Same course, task and seeds as the main
experiment. Every arm run is reported.

Success %, 95% CI in brackets for the comparisons:

| | L1, noise off (n = 768) | L1, noise on (n = 768) | L2, noise off (n = 512) |
|---|---|---|---|
| stairs policy alone | **97.7** | **94.9** | **82.8** |
| hard switch, 0.3 m late | 75.4 | 81.8 | 21.3 |
| hard switch at the boundary (0.0) | 97.9 | 90.8 | 39.8 |
| hard switch 0.3 m ahead | 96.6 | 94.9 | 86.5 |
| hard switch 0.8 m ahead | 97.7 | 95.3 | 84.8 |
| hard switch 1.5 m ahead | 97.3 | 95.7 | 87.5 |
| rough only / flat only | 36.7 / 30.1 | 48.7 / 40.5 | 0.0 / 0.6 |
| **H5a** early − on-time (1.5 − 0.3) | +0.7 (−1.1, +2.4) | +0.8 (−1.4, +2.9) | +1.0 (−3.2, +5.1) |
| **H5b** switch 0.3 − stairs alone | −1.0 (−2.8, +0.7) | 0.0 (−2.2, +2.2) | +3.7 (−0.7, +8.1) |
| same arms with `model_9999`: stairs alone / switch 0.3 | 76.6 / 91.5 (pooled 75.4 / 90.0) | 83.2 / 87.4 (pooled 82.4 / 89.7) | 33.4 / 51.0 |

Figure: `report_content/figures/switch_lead_two_checkpoints.png`.

- **H5a: the early-switch penalty does not survive a competent stairs policy.** It was the
  lip of the down-stairs: 13-15% of trials lost there with `model_9999` walking the approach,
  0.0% with `model_4800` walking it (stairs-alone and every switch at or ahead of the boundary;
  L1, noise off). Early switching is now free. It is still not *better*
  than on-time, so the conclusion about preview stands on firmer ground than before: the
  earlier result could be blamed on a bad specialist, this one cannot.
- **H5b: switching no longer beats the best single policy at L1**, and at L2 the +3.7 points
  has an interval that includes 0. The stairs policy from iteration 4800 crosses rough ground
  and both staircases by itself. The 15-point switching advantage of the main experiment was the
  size of the damage the curriculum did to the stairs specialist.
- **One policy beats every system built from the as-trained ones**: stairs-alone with
  `model_4800` at 97.7% / 94.9% (noise off / on) against 90.0% / 89.7% for switching between
  the as-trained specialists (pooled) and 90.5% / 82.5% for the generalist.
- **Lateness got more expensive, not less, at L2**: a switch at the boundary instead of 0.3 m
  ahead of it drops from 86.5% to 39.8%, and 0.3 m late to 21.3%. The flat specialist cannot
  put a foot on a 7 cm riser. So when a switch is used at all, the safe side is early, and
  with this stairs policy early is free.
- Not tested: the scan classifier with `model_4800`, and at L2, where its 0.2 m lead would sit
  in the steep part of the curve.
- `model_4800` is one earlier checkpoint of one single-seed run, chosen before it was
  evaluated as "the last checkpoint before iteration 5000". It is not a retrained specialist;
  that is stairs v2 (cluster job 12490).

## Limits

- Single-seed policies throughout. The as-trained stairs and rough specialists finished
  training on near-flat terrain (findings.md, "Why the specialists are weak"), and Addendum 2B
  shows how much of the main experiment's switching advantage that explains.
- One course family, one leader speed (0.5 m/s), one robot, simulation only.
- The generalist was trained under the same collapsing curriculum. A generalist trained with
  the command range held might do what the iteration-4800 stairs policy does.
- Stage 1 selected the "anticipatory" arms as the best of a bad set; stage 2 confirms they are
  worse than on-time, which is the conclusion, but "best lead above 0.8 m" is not a tuned
  anticipatory controller. A controller with the long horizon is free to switch at 0.3 m; the
  result says that is what it should do.
- With observation noise on, repeat runs of one arm on the same seeds differ by up to 3.5
  points (the noise draws differ); with noise off, by about 2. Differences of that size
  between single runs should not be read.
