# Terrain-Specialist Switching for a Person-Following Quadruped: Does Looking Further Ahead Help?

*Project report, 2026-10-03. Simulation study, Unitree Go2 in `mjlab` (MuJoCo). Every number
here traces to a file named in the Appendix. One planned arm, the matched generalist policy,
has not been trained; Section VIII says what that leaves open.*

## Abstract

A quadruped that follows a person could, in principle, use that person as a preview of the
terrain ahead and switch between terrain-specialist locomotion policies before its own sensors
see the change. We trained three specialists (flat, rough, stairs) for a Unitree Go2 and
tested that idea in a pre-registered experiment on a mixed-terrain course, with the robot
following a scripted leader. Switching between specialists by terrain is worth having: 91.5%
of trials cross the course against 76.6% for the best single specialist (+15.0 points, 95% CI
+11.4 to +18.6; +4.2 points when the training sensor noise is switched back on). But the
switch has to happen at the terrain boundary. Switching 0.3 m late costs 16 to 30 points, and
switching 1.5 m early, which is what a preview beyond the robot's 0.8 m height scan would
allow, costs 7 to 17 points and is no better than never switching. Blending the specialists'
actions instead of switching hard changes nothing. A small classifier on the robot's own
height scan times the switch as well as ground-truth labels do. The anticipation hypothesis is
therefore not supported: on this course the followed person adds neither useful horizon nor
useful labels. We also report why the specialists themselves are weak (their training
curriculum collapsed halfway through), a replication of the SARO vision-language pipeline in
which a 4B local model fails to perceive simulated stairs, and a two-rate perception design
that makes camera-based person-following run in real time.

## I. Introduction

Most learned quadruped locomotion uses one policy for every surface, often with a terrain
encoder or a short exteroceptive preview such as a height scan. The alternative studied here is
a small set of specialist policies, each trained on one terrain class, with a module that
decides which one is in control.

The project's specific hypothesis comes from the person-following setting. The robot's own
height scan reaches about 0.8 m ahead of its base. A person being followed walks a few metres
ahead, over terrain the robot will reach later, so their path is a free preview beyond that
horizon. If knowing the terrain further ahead lets the robot switch or blend policies earlier
and more smoothly, a follower should out-perform a robot that reacts only to what its own scan
sees. The claim is about *horizon extension*, so the experiment is built to measure outcome as
a function of how far ahead of a terrain boundary the switch begins.

The design is a 2×2 of {hard switch, soft blend} × {reactive, anticipatory}, against
fixed-policy baselines, with every comparison isolating one factor. A null result was
anticipated as possible and the analysis rules were fixed in advance so that it would be
reportable.

**Contributions.**

1. A closed-loop demonstration that switching between frozen terrain specialists beats any
   single one of them on a mixed course, with the conditions under which that advantage is
   large, small, or absent.
2. A measurement of how success depends on switch timing, showing the optimum lies at the
   terrain boundary and inside the onboard sensing horizon, and identifying why early switching
   fails.
3. A negative result for the anticipation hypothesis under best-case conditions for it
   (exact labels, exact boundary positions), and a demonstration that an onboard scan
   classifier is sufficient to time the switch.
4. A diagnosis of why the specialists are weak: an interaction between two curricula that
   leaves them training on near-flat ground for the second half of their budget, invisible in
   reward and episode length.
5. Two side results: a SARO-protocol replication with a small local vision-language model,
   and a two-rate perception architecture for person-following.

## II. Background and related work

*Bibliographic details of the works below other than SARO were not re-checked against the
sources and should be verified before external submission.*

Single-policy approaches include Rapid Motor Adaptation (online adaptation through a learned
latent), the ETH perceptive-locomotion line for ANYmal (policies reading an elevation map or
height scan), and massively parallel PPO training ("Learning to Walk in Minutes"), from which
this project's training stack descends. SARO (arXiv:2407.16412) uses a vision-language model
to plan, perceive and double-check sub-tasks for crossing one terrain obstacle, on top of a
low-level policy trained by Probability Annealing Selection (PAS), a two-stage distillation
from a privileged policy to a proprioceptive one. PAS produces one generalist policy; it is
not a runtime switch between policies.

Explicit switching between discrete terrain specialists is less common, and to our knowledge
using a followed person's trajectory as terrain preview has not been tested.

## III. System

### A. Robot, simulator and terrain

Unitree Go2 in `mjlab`, 50 Hz control. Terrain is geometric only: flat, rough (random and
wave heightfields), stairs (pyramid stairs, 0.3 m tread, riser 0 to 0.10 m), and
stepping-stone gaps. The actor observation is 234-dimensional and includes a 187-ray height
scan over a 1.6 m × 1.0 m grid centred on the base, so it sees 0.8 m ahead. In training each
ray carries uniform noise of ±0.1 m.

### B. Specialists

Three policies, each trained with PPO (RSL-RL) on one terrain class for 10,000 iterations at
8,192 environments, sharing one observation space, reward and network shape so they can be
swapped or blended inside one environment. A gaps specialist failed twice under its original
design; a blended-terrain retrain finished but has not been evaluated and is not used here.

### C. Follow task and course

The leader is a scripted point walking the course centreline at 0.5 m/s, 3.0 m ahead of the
robot. The robot's velocity command comes from a follow controller (line-of-sight velocity
feed-forward plus a proportional term on the gap), so the command rises when the robot falls
behind. The course `multi` is a straight strip: flat 3 m, rough 3 m, flat 2 m, stairs up
1.5 m, flat 2 m, stairs down 1.5 m, flat 3 m. Level L1 has 0.05 m risers and 0.02 to 0.06 m
rough-ground relief; L2 has 0.07 m and up to 0.08 m.

A trial ends in **success** (base 1.5 m past the last stair), **fall**, or **lost** (leader
more than 6 m away). A fall is one of the specialists' own training terminations: bad
orientation, or any non-foot body touching the ground with more than 10 N. SARO's definition
(orientation only) is used as a robustness check.

### D. Switch schedules

A switch into a non-flat segment is described by a *lead*: how many metres of base travel
before the segment's first edge the incoming specialist takes over. A hard switch has one
lead. A soft switch has a start and an end lead and blends the two specialists' actions
linearly in between. A negative lead means the switch happens only after the robot is already
on the new terrain, as with a detector that needs to be on a terrain to recognise it. Leaving
a segment is the same in every arm: the specialist releases when the rear of the footprint
has cleared it.

All schedule arms read the true segment layout. The only thing a lead encodes is horizon: a
lead of at most 0.8 m could be supplied by the robot's own scan, and a longer one needs
terrain knowledge from beyond it, which is what a followed person provides. This keeps sensor
quality out of the comparison. A hard switch with a 0.3 m lead is the "footprint rule" used
as the oracle in the project's earlier experiments (switch when the front of the footprint
reaches the edge).

### E. Scan classifier

For a reactive arm that uses no ground truth, a small MLP (187 → 128 → 64 → 3) reads the
height-scan slice of the actor observation and predicts which specialist the footprint rule
has active. Its class probabilities pass through an exponential average and a hold-to-switch
filter, and the result drives a hard switch. It is trained on single-obstacle courses and
never on `multi`.

## IV. Experimental design

**Premise gates.** Before any switching experiment, two premises were tested directly:
that no single specialist is best on every terrain (a cross-terrain evaluation matrix), and
that the height scan carries enough information to tell terrain classes apart (a classifier
on labelled scans with no policy in the loop).

**Pre-registration.** Hypotheses, arms, metrics and decision rules for the switching
experiment were committed before any timing data existed, and an addendum for the classifier
arm before any classifier data existed. Stage 1 (seeds 400, 401; 512 trials per arm) swept
seven hard leads and eight soft schedules and selected the best arm in each cell of the 2×2.
Stage 2 (fresh seeds 500, 501, 502; 768 trials per arm) evaluated only the selected arms and
the baselines.

**Metrics.** Course success with Wilson intervals; differences of proportions with Newcombe
intervals. Boundary smoothness is action rate and actuator-force rate inside ±0.5 m of base
travel around each terrain-class boundary, relative to the same arm's whole-rollout value. It
is tied to the terrain layout and never to the arm's own switch events, since a blend has
none.

**Hypotheses.** H1: reactive hard switching beats the best fixed specialist. H2: blending
improves boundary smoothness without losing success. H3: schedules that need the long horizon
beat those that do not. H4 (addendum): a scan-driven switch is within 5 points of the
label-timed one.

## V. Results

### A. Specialists differ by terrain

Falls per 100 m travelled at pinned difficulty 0.5, with training noise on:

| policy \ terrain | flat | rough | stairs |
|---|---|---|---|
| Flat specialist | 0.00 | 14.41 | 10.42 |
| Rough specialist | 0.00 | **5.36** | 11.14 |
| Stairs specialist | 0.00 | 13.02 | **7.60** |

The best policy changes with the terrain, which is the condition switching needs. The rough
margin is solid (2.4× the runner-up under three different fall metrics); the stairs margin is
1.4× on a single seed. The stairs specialist is weak in absolute terms: on its own training
terrain it survives 65%, 44% and 32% of 24-second episodes at difficulty 0.5, 0.7 and 0.9
(6.1, 12.6 and 16.9 falls per 100 m), nearly all from non-foot contact with a step edge.

### B. Switching beats every single specialist

![Course success by arm](figures/switch_confirm.png)

| arm (confirmation seeds, 768 trials) | success (95% CI) | per seed |
|---|---|---|
| hard switch, 0.3 m ahead | **91.5%** (89.4-93.3) | 91.0 / 92.2 / 91.4 |
| cross-fade over the last 0.3 m | 90.8% (88.5-92.6) | 89.1 / 92.6 / 90.6 |
| hard switch, 1.5 m ahead | 74.6% (71.4-77.6) | 73.0 / 72.3 / 78.5 |
| cross-fade from 1.5 m to 0.3 m | 76.3% (73.2-79.2) | 74.2 / 75.8 / 78.9 |
| stairs specialist only | 76.6% (73.4-79.4) | 71.9 / 77.3 / 80.5 |
| rough specialist only | 37.4% (34.0-40.8) | 34.4 / 38.3 / 39.5 |
| flat specialist only | 25.7% (22.7-28.9) | 25.4 / 25.4 / 26.2 |

**H1 is supported**: +15.0 points over the best fixed specialist (95% CI +11.4 to +18.6). On
the project's earlier single-obstacle courses one well-chosen fixed specialist always matched
perfect switching; a course on which the best specialist changes mid-run is what makes
switching pay.

### C. Switch timing: the optimum is at the boundary

![Success against switch lead](figures/switch_lead_sweep.png)

Calibration sweep of the hard-switch lead (512 trials each):

| lead (m) | −0.6 | −0.3 | 0.0 | 0.3 | 0.8 | 1.5 | 2.5 |
|---|---|---|---|---|---|---|---|
| success | 37.9% | 59.6% | 88.5% | **92.2%** | 73.6% | 75.6% | 75.6% |

Success peaks with the switch 0 to 0.3 m before the boundary. Later than that it collapses,
because the flat specialist is still in control on the up-stairs and cannot climb. Earlier
than that it drops to the level of the stairs specialist alone and stays there out to 2.5 m.

**H3 is not supported, and the effect has the opposite sign.** On the confirmation seeds a
hard switch 1.5 m ahead is 16.9 points worse than one 0.3 m ahead (CI −20.6 to −13.3), and a
cross-fade starting 1.5 m ahead is 14.5 points worse than one starting 0.3 m ahead (CI −18.1
to −10.8). Neither differs from never switching (−2.0, CI −6.2 to +2.3).

The cost of switching early is located in one place:

| arm | failures in the last 0.25 m before the down-stairs edge | failures on the down-stairs |
|---|---|---|
| hard switch 0.3 m ahead | **0.0%** | 7.2% |
| hard switch 1.5 m ahead | **15.2%** | 8.9% |
| cross-fade from 1.5 m | **13.5%** | 9.2% |
| stairs specialist only | **12.9%** | 9.4% |

A stairs specialist that has been walking the flat approach touches a knee or calf to the
edge in 13 to 15% of trials as it steps off. The flat specialist walking the same approach
and handing over 0.3 m before the edge never does. Each specialist is worse than its neighbour
on the approach to its own terrain, so the earlier it takes over, the more is lost, and this
holds for any controller that hands over early, not only for these schedules.

A controller with a long preview horizon is free to switch at 0.3 m, and the data say that is
what it should do. The gain from extending the horizon beyond what the scan already covers is
therefore zero.

### D. Blending adds nothing

**H2 is not supported.** A cross-fade over the last 0.3 m against a hard switch at 0.3 m:
success −0.8 points (CI −3.6 to +2.1); boundary action rate 1.021 against 1.022 of the arm's
rollout mean (difference −0.002, CI −0.006 to +0.002); actuator-force rate the same. There is
no boundary transient to smooth: a hard switch between these specialists raises the action
rate near a boundary by about 2% over the rollout average. At L2 blending is worse than a
hard switch (−6.4 points under training terminations, −14.1 under orientation-only falls).

### E. The robot's own scan is enough

| sensing | per-step accuracy, held out | label-timed switch | scan-classifier switch | difference (95% CI) | median lead at the three segment entries |
|---|---|---|---|---|---|
| training noise on | 83.8% | 88.3% | 87.5% | −0.8 (−4.1 to +2.5) | +0.17 / +0.20 / +0.23 m |
| noise off | 99.6% | 90.6% | 90.0% | −0.7 (−3.6 to +2.3) | +0.28 / +0.27 / +0.26 m |

**H4 holds in both conditions.** With noise on, a single scan confuses flat and rough about a
quarter of the time but recognises stairs 96% of the time, and the filter averages the rest
away. The classifier never first selected a segment's specialist after the boundary. So the
on-time switch that Section C shows to be necessary can be delivered by onboard sensing, and
the leader's path is not needed as a source of labels either.

This is the easy case for a classifier: the test course uses the same straight 0.05 m stairs
it trained on. An earlier measurement on the training terrain generator, pooled over
difficulty, found rough and stairs much harder to separate, though that measurement was later
shown to be inflated by near-flat samples labelled as rough or stairs.

### F. Robustness

Differences in points of course success, with 95% intervals.

| condition | stairs only | hard 0.3 | switching value | 1.5 m early vs 0.3 m | cross-fade vs hard |
|---|---|---|---|---|---|
| **L1, as pre-registered** (n = 768) | 76.6% | 91.5% | +15.0 (+11.4, +18.6) | −16.9 (−20.6, −13.3) | −0.8 (−3.6, +2.1) |
| L1, observation noise on (n = 768) | 83.2% | 87.4% | +4.2 (+0.6, +7.7) | −6.9 (−10.6, −3.2) | +0.7 (−2.6, +4.0) |
| L1, orientation-only falls (n = 768) | 99.2% | 98.4% | none: −1.6 vs rough only at 100% | +0.5 (−0.7, +1.8) | +1.2 (+0.2, +2.3) |
| L2, training falls (n = 512) | 33.4% | 51.0% | +17.6 (+11.6, +23.4) | −16.2 (−22.1, −10.2) | −6.4 (−12.5, −0.3) |
| L2, orientation-only falls (n = 512) | 79.1% | 89.6% | +10.5 (+6.1, +15.0) | −2.3 (−6.3, +1.6) | −14.1 (−18.6, −9.5) |

Three things change the *size* of the switching advantage. First, the experiment as
pre-registered ran without the observation noise the specialists were trained with, because
the evaluation environment was built on a configuration that turns it off. With the noise
back on every fixed specialist does better and the advantage falls to +4.2 points. Second, all
the L1 failures are knee or calf contacts; if only tipping over counts as a fall, nothing
fails at L1. Third, with harder stairs the advantage returns under either definition.

Nothing changes the *direction* of the timing result: switching early is never better than
switching at the boundary, and switching late is always much worse.

### G. Why the specialists are weak

The training logs of the stairs and rough specialists show the same event:

| | terrain level at iteration 5000 | peak, iteration 5100 | iteration 6000 | iteration 9999 |
|---|---|---|---|---|
| Stairs | 1.92 | 2.63 | 0.79 | 0.97 |
| Rough | 2.14 | 2.65 | 0.42 | 0.53 |

Training has two curricula. One widens the commanded velocity range at exactly iteration 5000
(forward speed from at most 1.0 to at most 2.0 m/s). The other moves a robot to harder terrain
when it covers enough ground and to easier terrain when it covers less than half of what its
command asked for. When the command range doubles, the policies are demoted to the easiest
terrain rows and never climb back. Terrain level runs from 0 to 9 and a stair riser is about
1 cm per level, so the stairs specialist spent its last 4,700 iterations on risers of about
1 cm and never practised much above 2 to 3 cm. Mean reward and episode length are flat across
the collapse, so it is invisible unless terrain level is plotted.

This explains the stairs specialist's failures at 5 cm, and it means a better stairs
specialist needs a configuration change (hold the command range where the follow task uses
it), not more training iterations.

### H. Side results

**PAS replication.** The SARO low-level policy was trained in full (80,000 iterations, about
8× a specialist's budget). It survives flat ground and stairs almost perfectly, but only
because it barely moves: achieved speed is 8 to 11% of commanded on every terrain, against 24
to 46% for the specialists. Two extra reward terms that penalise motion explain it. It is
reported as a replication, not as a baseline.

**Vision-language specialist selection.** SARO's plan, perceive and double-check loop was
rebuilt with a local 4B model (Gemma-4-E4B) choosing among the three specialists, and run at
SARO's own protocol of 20 trials per obstacle:

| obstacle | VLM success | success with ground-truth choices | planner's answer |
|---|---|---|---|
| stairs up | 0% | 75% | "no obstacle" 20/20 |
| stairs down | 45% | 95% | "no obstacle" 20/20 |
| rough | 100% | 100% | mixed |

The model does not perceive the simulated stairs at all, and the 100% on rough ground is not
evidence of perception, since every specialist crosses rough ground. Depth geometry, by
contrast, locates step edges to within 0 to 8 cm.

**Person-following perception.** With a ground-truth leader the robot holds a 2.5 m gap to
0.16 m RMS through seven speed changes. Putting the VLM inside the control loop fails (5.9 m
gap error, 10× slower than real time). Splitting the job works: the VLM names the target
class once, off the control path, and a detector localises it every control step in about
4 ms. Gap error is then 0.50 m at real-time speed. A stock COCO detector cannot see the
simulated leader; fine-tuning on 1,000 frames labelled automatically from the leader's known
pose gives precision 1.00 and recall 0.94.

## VI. Discussion

The hypothesis was that a followed person extends the robot's terrain preview and that the
extension improves policy switching. The experiment separates two ways that could be true and
finds neither.

*Horizon.* The best moment to switch is within 0.3 m of the boundary. The robot's own scan
reaches 0.8 m. Extra horizon can only be spent on switching earlier, and switching earlier
puts a specialist in control on ground where its neighbour is better. With these specialists
that costs between 7 and 17 points; with better ones it might cost nothing; there is no
mechanism in the data by which it would help. Nor is there a transition transient that a
longer, gentler hand-over could smooth, since the hard switch barely registers in action rate.

*Labels.* What the timing curve does show is that being late is expensive, so a reliable,
prompt terrain label matters a great deal. The leader's path could have been valuable as such
a label. But a 33,000-parameter classifier on the robot's own noisy scan already delivers the
switch 0.2 m before the boundary and matches ground truth.

The positive finding is narrower than the one hoped for but is real: on a course where the
best specialist changes, switching wins, and its value is governed almost entirely by
timeliness. How large that value is depends on details that are easy to leave implicit, such
as sensor noise in the evaluation environment and what counts as a fall. The pre-registered
headline of +15 points and the more deployment-like +4 points are the same experiment under
two settings of one flag.

## VII. Limitations

- **No generalist.** The matched generalist policy was never trained, so the comparison that
  matters most for the specialist approach as a whole, switching against one policy trained
  on everything, is missing. All "switching value" numbers are against the best fixed
  *specialist*.
- **Weak, single-seed specialists.** Both non-flat specialists finished training on near-flat
  ground (Section V.G). The early-switching penalty comes from a stairs specialist that is
  poor on the approach to stairs; a properly trained one might not show it.
- **Parametric blending, not a learned gate.** The soft arms are linear cross-fades. A
  trained gating network could in principle find a state-dependent blend these schedules
  cannot express. The results give no reason to expect that, but do not rule it out.
- **One course family, one leader speed, a ground-truth leader**, and simulation only.
- **The classifier's test is in-distribution** for stair geometry.
- **Run-to-run spread** from simulator nondeterminism is about 1 point: the same arm on the
  same seeds scored 91.5% and 90.6% in two runs.

## VIII. What remains

1. **Train the generalist** (one job, about 12 GPU-hours) and add it as a fixed arm. The
   harness accepts it with one flag. This is the only planned arm not run.
2. **Retrain the stairs specialist with the command range held** at the follow task's range,
   then repeat the lead sweep to see whether the early-switching penalty survives a competent
   specialist.
3. Evaluate the existing gaps checkpoint and add a gaps segment to the course.
4. Vary leader speed and course layout; test the classifier on unseen stair geometry.

## Appendix: where each number comes from

| Section | Source |
|---|---|
| V.A | `coordination/results/gate1-cross-terrain-matrix-analysis.md`; `coordination/results/2026-10-03-stairs-step1-pyramid-eval-results.md` |
| V.B-F | `coordination/results/switch-follow-results.md`; raw `unitree_rl_mjlab/eval_results/switch_follow/`; rules `coordination/results/switch-follow-preregistration.md` |
| V.G | `coordination/results/2026-10-03-stairs-training-curve.md`; `findings.md`, "Why the specialists are weak" |
| V.H | `findings.md` (PAS, VLM navigation, person-following, SARO protocol sections); `coordination/results/vlm-nav-*.md` |
| Gate 2 remark in V.E | `coordination/results/gate2a-height-scan-discriminability-analysis.md`; `findings.md` bug #16 |

To reproduce Section V.B-F from `unitree_rl_mjlab/` (conda env `unitree_rl_mjlab`,
`PYTHONPATH=$PWD MUJOCO_GL=egl`, specialist checkpoints under `--ckpt-root`):

```
python scripts/switch_follow.py --arm-set calib --seed 400 --ckpt-root <ckpts> --json-out eval_results/switch_follow/calib_s400.json
python scripts/switch_follow_analyze.py calib eval_results/switch_follow/calib_s40*.json
python scripts/switch_follow.py --seed 500 --arms fixed:flat fixed:rough fixed:stairs hard:0.3:label soft:0.3:0.0:label hard:1.5:label soft:1.5:0.3:label --ckpt-root <ckpts> --json-out eval_results/switch_follow/confirm_s500.json
python scripts/switch_follow_analyze.py confirm eval_results/switch_follow/confirm_s50*.json --pairs "hard:0.3:label>fixed:stairs"
bash scripts/switch_follow_clf_pipeline.sh        # scan classifier: collect, train, calibrate
```

Add `--obs-noise`, `--terminations saro` or `--level L2` for the robustness rows, and
`--extra-policy generalist=<ckpt> --arms fixed:generalist` once the generalist exists.
