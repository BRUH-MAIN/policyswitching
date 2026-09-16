# Design proposal — the rough/stairs sensing decision

**Date**: 2026-09-16 · **Machine**: `romen` (laptop) · **Status**: recommendation, not a
decision — `CLAUDE.md` reserves this call for the user. Supersedes the three-option framing
in `PROGRESS_REPORT.md` §3.2 and the "Suggested next step" of
`coordination/results/gate2a-height-scan-discriminability-analysis.md`.

**Nothing new was simulated for this.** Every number below is arithmetic on data already
committed: the gate-1 falls/100m matrix and the gate-2a confusion matrices
(`unitree_rl_mjlab/eval_results/gate2a/height_scan_discriminability.json`).

---

## The short version

The open question has been posed as *"how do we make the height scan separate rough from
stairs?"*, with three levers to choose between. That is the wrong question, and it is the
wrong question in this repo's now-familiar way: **classification accuracy is a property
adjacent to the decision.** What the switch has to get right is *which specialist to run*.
Those two come apart, and once you measure the thing that actually matters they come apart
by enough to change the answer.

Three things follow, none of which is in the current option set:

1. **Merging (option 3) isn't a concession — at today's accuracy it already beats keeping
   four classes**, by 2.1×. But it is a floor, not a solution: the entire quantity in
   dispute is 1.77 falls/100 m.
2. **The ~0.51 "measured ceiling" is largely a protocol artifact**, not a sensor property.
   ~26% of the `stairs` samples are *literally flat ground*, because the measurement
   averaged over a difficulty range whose bottom is zero relief. A ceiling measured on a
   dataset with unknowable labels is not the sensor's ceiling.
3. **The right fix is a cost-sensitive decision rule, not a taxonomy change.** It
   *derives* option 3's behaviour as its current degenerate case, costs nothing, touches no
   checkpoint, and upgrades by itself if the sensing improves.

There is also a fourth issue, unrelated to the taxonomy but surfaced by looking at it, that
threatens the project's headline claim directly — see §5. That one needs a decision more
urgently than this one does.

---

## 1. What the decision is actually worth, in the units the project is scored in

From the gate-1 matrix (falls per 100 m, difficulty 0.5,
`coordination/results/gate1-cross-terrain-matrix-analysis.md`):

| | true `rough` | true `stairs` |
|---|---|---|
| run flat specialist | 14.41 | 10.42 |
| run rough specialist | **5.36** | 11.14 |
| run stairs specialist | 13.02 | **7.60** |

**The cost of getting the call wrong is asymmetric**: calling rough "stairs" costs +7.66
falls/100 m; calling stairs "rough" costs +3.54. A 2.16× ratio. Nothing in the current
design uses this — argmax treats both errors as equal, which is already a mistake
independent of how good the classifier is.

Expected falls/100 m on a course that is half rough, half stairs:

| policy for the uneven class | falls/100 m | regret |
|---|---|---|
| oracle 2-way switch (perfect sensing) | 6.48 | — |
| **merge → always run the rough specialist** | **8.25** | **+1.77** |
| 4-way argmax switch, gate-2a CNN, *clean* | 9.24 | +2.76 |
| 4-way argmax switch, gate-2a CNN, noisy (today) | 10.28 | +3.80 |
| merge → always run the stairs specialist | 10.31 | +3.83 |
| merge → always run the flat specialist | 12.41 | +5.93 |

Read the middle two rows carefully. **The four-class switch, driven by the classifier we
actually have, is worse than merging and never distinguishing the pair at all** — and it is
*still* worse than merging when handed a perfect, zero-noise sensor. That is not an argument
for merging the taxonomy; it is a demonstration that argmax over this posterior is a bad
decision rule, which §4 fixes without merging anything.

Two numbers worth keeping:

- **The whole prize is 1.77 falls/100 m** — what perfect rough/stairs discrimination buys
  over the best single fallback, on the half of a course that is rough or stairs. For scale,
  the distinction the scan *already* makes reliably (gaps, at 0.97 accuracy) is worth
  **684–1357** falls/100 m. The sensing already delivers the distinction that matters by two
  orders of magnitude; this one is the rounding error being used to block the project.
- **Break-even ≈ 0.68.** A symmetric argmax classifier must reach 0.68 recall on *both*
  rough and stairs before it beats merging. Given the cost asymmetry, if stairs recall were
  perfect, rough recall alone would need 0.54. Today's measured pair is 0.20 / 0.64.

This is the acceptance criterion gate 2 should have had. "Is accuracy above chance?" cannot
tell you whether to build the thing; "is regret below the best fallback's 1.77?" can.

## 2. The 0.51 ceiling is mostly manufactured by the measurement protocol

The gate-2a run collected with `difficulty=None` — envs spread uniformly over all 10
difficulty rows. That was a defensible choice at the time (the comment in
`height_scan_classifier.py` says so: bug #12's fix wasn't known to be in the checkout). It
is not a neutral choice for *this* measurement, and the committed confusion matrices show
the damage.

From `ROUGH_TERRAINS_CFG` (mjlab's `terrains/config.py`):

| sub-terrain | difficulty-scaled parameter | value at the bottom row |
|---|---|---|
| `pyramid_stairs`, `..._inv` | `step_height_range=(0.0, 0.1)` | **0 cm — flat** |
| `wave_terrain` | `amplitude_range=(0.0, 0.2)` | **0 cm — flat** |
| `random_rough` | `noise_range=(0.02, 0.10)` | 2 cm — never flat |
| `flat` | — | flat by definition |
| `stepping_stones` | `stone_height=0.2`, `floor_depth=2.0` | difficulty-independent |

**The only two classes whose defining geometry degenerates to flat ground are `stairs` and
`rough` — which are exactly the two classes that scored ~0.51.**

The prediction this makes is quantitative, and the committed data confirms it. The bottom
~3 of 10 difficulty rows give ≤2.2 cm of relief, i.e. ~30% of `stairs` samples are flat or
near-flat; `rough` is half `wave_terrain`, so ~15% of it is. In the **clean** condition —
a *perfect sensor* — the CNN sends:

| true class | predicted `flat` | predicted, as % | expected from the difficulty ranges |
|---|---|---|---|
| `stairs` | 158 / 610 | **25.9%** | ~30% |
| `rough` | 83 / 530 | **15.7%** | ~15% |
| `flat` | 810 / 810 | 100% | 100% |
| `gaps` | 2 / 610 | 0.3% | ~0% |

A quarter of the stairs samples are being called flat by a model reading noiseless geometry,
and the fraction matches the fraction of the difficulty range that generates no steps, to
about a point. Those samples are not hard to classify — **they have no recoverable label**.
Their cost lands entirely on `rough` and `stairs` recall and not at all on `flat` or `gaps`,
which is precisely the pattern the write-up reported as a sensor ceiling.

Two smaller contributors, both also protocol rather than sensor:

- **`rough` is not one geometric kind.** `TERRAIN_CLASSES` defines it as `random_rough` ∪
  `wave_terrain`. `wave_terrain` is 4 waves over 8 m — a 2 m wavelength, seen through a
  1.6 m scan footprint, i.e. a smooth ramp. Geometrically that is *closer to a staircase's
  mean slope than to isotropic noise*. A class-level confusion matrix cannot say whether the
  pair collides wholesale or only through the wave/stairs half, and the current
  write-up's conclusion requires that it collides wholesale.
- **The CNN's head throws the signature away.** `fit_cnn` does
  `AdaptiveAvgPool2d(2)` on the 17×11 grid before its only linear layer — the spatial map is
  averaged down to four numbers per channel. Periodic step edges, which the write-up
  correctly identifies as what a staircase actually looks like, largely do not survive that.
  Note this is the *same mistake one level down* that the document already caught once: the
  linear probe's "stairs is undetectable" turned out to be a model limitation, and the CNN
  disproved it. Reading the CNN's number as a sensor ceiling makes the identical inference a
  second time.

**None of this says the scan can separate rough from stairs.** It says the experiment run so
far cannot tell us, and that its headline number has a large, identified, fixable bias in a
known direction. Options 1 and 2 in §3.2 are currently being weighed against a ceiling that
has not actually been measured.

## 3. Reject levers 1 and 2 (scan noise, difficulty range)

Independently of §2, and even if the ceiling turns out to be real:

- Both are **global training settings every one of the four existing checkpoints was trained
  under**. Changing either makes future runs inconsistent with all of them, on a cluster
  whose GPU nodes have been drained since 2026-09-10 and which still owes the project a
  generalist and a gaps specialist.
- The prize is bounded above by **1.77 falls/100 m** on the uneven half of a course (§1).
- Lever 1's benefit is available for free without touching the config — see §4(D).

Recommend: **rejected, on measured cost/benefit.** Not "insufficient" — disproportionate.

## 4. The recommendation

Four parts. (A)–(C) are the design change; (D) is free performance.

**(A) Keep `TERRAIN_CLASSES` at four classes.** Don't merge in the config. Merging is a
statement about the *decision rule*, not about the terrain, and baking it into the taxonomy
throws away the rough specialist's 2.4× win — gate 1's only solid result — and is awkward to
undo once eval scripts, class indices and the gating network's output dimension depend on it.

**(B) Report a "policy-equivalence" view alongside the class view.** Two classes are worth
distinguishing iff the regret from confusing them exceeds the cost of trying. That is a
measured property of the gate-1 matrix, re-derivable whenever a new specialist lands, not a
fact about the terrain generator.

**(C) Replace argmax with a cost-sensitive rule.** Pick the specialist that minimises
expected falls under the classifier posterior and the gate-1 matrix:

```
π* = argmin_π  Σ_t  P(t | scan) · M(π, t)          M = the gate-1 falls/100m matrix
```

This is the actual proposal, and it is why this isn't a pick-one-of-three:

- At today's near-uninformative rough/stairs posterior it **derives** "run the rough
  specialist on anything uneven" — option 3's behaviour, reached as a conclusion rather than
  assumed, and with the taxonomy left intact.
- It uses the **2.16× cost asymmetry** that argmax discards, extracting value from a
  posterior that looks worthless when scored by accuracy.
- It **upgrades automatically**. If the re-measurement in §6 lifts rough/stairs past the
  0.68 break-even, the same rule starts distinguishing them with no retraining, no config
  change, and no second design decision.
- It is **the object the project already needs anyway**: arms 2b and 3 consume a posterior,
  not a label. A hard switch (2a) is then the argmin of the same quantity, so 2a and 2b stop
  differing in two things at once.

**(D) Take the noise reduction in the estimator, not in the observation.** The gate-2a
measurement is single-frame, from a robot standing still with zero actions. The deployed
switch runs at 50 Hz on a moving robot, and `Unoise(±0.1 m)` is resampled i.i.d. every step.
Filtering the scan over a short window *inside the classifier* cuts the noise std by √N —
~5× over 0.5 s, which is most of the clean-vs-noisy gap (0.778 → 0.617). This buys lever 1's
benefit while changing nothing about what any policy observes, so no checkpoint is
invalidated. The window length is a real bias/variance tradeoff (a moving robot smears
terrain across a long window) and should be measured, not assumed — but the measurement is
laptop-local and cheap.

## 5. The confound this turned up, which matters more than the taxonomy

`objective.md` specifies the anticipatory signal as ground truth — *"a scripted kinematic
leader body with a known trajectory... makes the 'terrain along the leader's path' signal
ground truth during the sim ablations that carry the empirical weight of the result."*
Meanwhile arm 2a/2b's reactive input is a classifier reading the noisy height scan.

**If arm 3 reads ground-truth terrain and arm 2b reads a ~0.6-accuracy classifier, then
2b vs 3 measures sensing quality, not preview horizon** — and preview horizon is the
contribution. Arm 3 would win by construction, for the wrong reason, and the
preview-horizon sweep would not save it: a better sensor beats a worse one at every lead
distance, including the ≈0.5 m control point that is supposed to show *no* gain.

Fix, and it is cheap because it is a spec change before anything is built: **the terrain
channel must be the same function at both sample points, differing only in *where* it is
sampled.** Concretely — run the primary 2×2 with an oracle channel at both the robot's and
the leader's position (isolates horizon, which is the claim), and a secondary pass with the
same learned estimator at both (shows the result survives realistic sensing). Under that
design the rough/stairs limitation largely **cancels between arms** instead of favouring
arm 3.

One upside worth noting: a class pair the sensor genuinely cannot resolve is the textbook
case where a **soft blend beats a hard switch** — it hedges where argmax coin-flips. If the
pair stays unresolvable, that makes the 2a-vs-2b comparison *more* informative, not less.
The limitation is closer to being a result than a blocker.

## 6. What to run before deciding — all laptop-local, no cluster, no drained nodes

`unitree_rl_mjlab/scripts/height_scan_classifier.py` has been extended (uncommitted, this
session) with `--difficulty` and with sub-terrain-level classes. **Not yet run — the laptop
is on battery.** When it is plugged in:

```bash
export PYTHONPATH=unitree_rl_mjlab
P=/home/rohan/miniconda3/envs/unitree_rl_mjlab/bin/python

# 1. the headline condition: difficulty pinned to 0.5, matching every gate-1 cell.
$P unitree_rl_mjlab/scripts/height_scan_classifier.py --difficulty 0.5 \
    --json-out unitree_rl_mjlab/eval_results/gate2a/discriminability_d0.5.json

# 2. where the residual confusion actually lives, at the granularity the generator works at.
$P unitree_rl_mjlab/scripts/height_scan_classifier.py --difficulty 0.5 \
    --classes flat random_rough wave_terrain pyramid_stairs pyramid_stairs_inv stepping_stones \
    --json-out unitree_rl_mjlab/eval_results/gate2a/discriminability_subterrain.json
```

Keep the committed `difficulty=None` run as the control so the pair is comparable. Predicted,
on the argument in §2: rough and stairs recall rise materially in (1); in (2), `random_rough`
vs `pyramid_stairs` separates better than the class-level number suggests, with `wave_terrain`
carrying a disproportionate share of the collision. **If (1) and (2) come back flat, the next
suspect is the `AdaptiveAvgPool2d(2)` head, not the sensor** — widen it before concluding
anything about the hardware.

**The recommendation in §4 does not depend on the outcome.** The cost-sensitive rule is
correct whether the pair ends at 0.51 or 0.85; the re-measurement only decides whether that
rule ends up distinguishing rough from stairs or quietly falling back to the rough
specialist. That is the point of proposing a rule rather than a taxonomy: the decision stops
needing to be made twice.

## What the user is actually being asked to approve

1. Don't change scan noise or the difficulty range (§3). ← the only irreversible one
2. Don't merge `TERRAIN_CLASSES`; adopt the cost-sensitive rule instead (§4A–C).
3. Add temporal filtering inside the classifier (§4D).
4. Amend `objective.md` so the 2×2's terrain channel is matched across arms (§5).

(4) is the one with a deadline: it changes what gets built in Phase 4, and Phase 4 is the
next thing to start once the nodes come back.
