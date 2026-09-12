# Premise gate 2a — is the height scan terrain-discriminative?

**Date**: 2026-09-12 · **Machine**: `romen` (laptop) · **Script**:
`unitree_rl_mjlab/scripts/height_scan_classifier.py` · **Raw**:
`unitree_rl_mjlab/eval_results/gate2a/height_scan_discriminability.json`

## Why this exists

`objective.md`'s premise gate 2 asks "do policies use the height scan?" and tests it
by ablating the scan and watching fall rate. That measures whether the *locomotion
policy* needs the scan. Phase 4 needs something different: that the scan is
**terrain-discriminative**, since the reactive classifier and the gating network's
current-terrain input both have to tell the classes apart. Those two properties
dissociate in both directions (see `coordination/inbox/to-cluster.md`,
2026-09-12 (3), accepted by the cluster session). This measures the second one
directly — labelled scans per class, multinomial logistic regression, held-out
accuracy — with no RL and no checkpoint, so it runs while the nodes are drained.

## Headline (revised after the CNN follow-up — see that section)

**The scan supports a three-way distinction — `gaps`, `flat`, and
"rough-or-stairs" — and cannot split rough from stairs at this noise level with
either model tried.** Best held-out 4-way accuracy under realistic noise is
**0.617** (conv net) against 0.25 chance.

The linear-probe section below is kept because it is what the first verdict rested
on, and because the linear-vs-CNN gap is itself the finding: the original
"stairs is barely above chance" conclusion was a limitation of the probe, not a
property of the sensor.

## Linear probe (first pass)

Under the noise the policy actually sees, 4-way accuracy is 0.549 against 0.25
chance — but that number is carried by gaps.

| | clean | noisy (as the policy sees it) |
|---|---|---|
| overall (4-way, chance 0.25) | 0.623 | **0.543** |
| flat | 0.647 | 0.410 |
| rough | 0.532 | 0.506 |
| stairs | 0.357 | **0.330** |
| gaps | 0.936 | **0.966** |

(An earlier identical invocation gave 0.632 / 0.549 overall. Run-to-run spread is
~1 point, from simulator nondeterminism in collection — worth knowing before
reading a 1-point difference as signal. All numbers quoted in this document come
from the final run, which is the one in the committed JSON.)

Injected noise costs 8.0 accuracy points overall. `stairs` sits at 0.330 — barely
above the 0.33 you would get by guessing uniformly among the three non-gap classes.
Confusion is concentrated in the flat/rough/stairs block; gaps almost never leaks
into it or out of it.

Setup: `Unitree-Go2-Generalist` observation space, 256 envs per class, 10 samples
per env, terrain pinned per class with difficulty left unpinned (deliberately —
the difficulty-pinning path is bug #12 and its fix is stranded on the cluster).
Train/test split is **by env**, not by sample: a standing robot's consecutive scans
are near-duplicates, and a sample-wise split would put the same terrain patch on
both sides and report a meaningless near-perfect accuracy.

## Why gaps is easy and the rest is hard

Per-class scan statistics (scaled units; per-ray injected noise std is **0.0115**):

| class | mean | ray-std within a sample |
|---|---|---|
| flat | 0.0505 | 0.0059 |
| rough | 0.0507 | 0.0076 |
| stairs | 0.0505 | 0.0059 |
| gaps | 0.2797 | 0.2207 |

`stepping_stones` has holes, so rays miss and return far larger distances — gaps is
separated by a mean shift of ~0.23, twenty times the noise std, and is trivially
detectable. Everything else lives in a band whose entire spread is *below* the noise
std. That is the quantitative form of the concern already recorded in `findings.md`
("the height scan's terrain signal is near the injected noise floor"), and it
confirms it: `Unoise(±0.1 m)` against terrain relief of ≤10 cm stairs and 2–10 cm
rough is noise at or above 100% of signal.

## A methodology trap this run fell into first, and the fix

The first run reported `flat` at **0.016 accuracy — far below chance** — misread as
stairs 758/810 times, and noise *improved* it to 0.298. Below-chance on one class
with noise helping is not a weak-signal signature, so I chased it rather than
reporting it.

Cause: the task's `reset_base` randomizes spawn by only ±0.5 m around the patch
origin. Patches are 8×8 m, so that samples 1/64th of the patch, dead centre — and
the centre of a `pyramid_stairs` patch is its flat top platform, wider than the
1.6×1.0 m scan. Every stairs sample was a flat landing. Measured: `flat` and
`stairs` had **identical** mean, within-sample ray-std and across-sample std to four
decimal places. That is not "hard to separate", it is the same distribution, and it
was a property of where the robot stood, not of the sensor.

The script now takes `--spawn-spread` (default ±3 m, kept 1 m inside the patch edge
so the scan footprint stays on-patch). Every number above is from the corrected run.
**Anyone re-running this must keep the spread**; at the task default the result is an
artifact.

## What this means for the design

1. **A 4-way reactive terrain classifier on this scan is not viable as specified.**
   Gaps is reliable (0.97) and flat is workable (0.60). Rough vs stairs is the
   failure: neither model separates them, and the conv net's apparent stairs gain
   comes out of rough one-for-one.
2. **If the lever is sensing, it is the noise magnitude, not the scale.** Scaling
   multiplies signal and noise alike and cannot change SNR. `Unoise(±0.1 m)` is
   applied pre-scale against ≤10 cm of relief. Reducing it, or raising the terrain
   difficulty range so the relief clears the noise, would only need to buy
   separation for the rough/stairs pair now.
3. **Or narrow the taxonomy — and the concession is smaller than it first looked.**
   Merging rough and stairs into one "uneven" class leaves a 3-way switch
   (gaps / flat / uneven) that the sensing supports at roughly 0.6–0.97 per class,
   rather than collapsing all the way to gap-vs-non-gap.

**None of these is a config tweak, and none is mine or the cluster's to pick.** The
noise level and difficulty range are global training settings that every
already-trained policy was trained under, so changing either makes future runs
inconsistent with existing checkpoints; and the taxonomy is a change to what the
project claims. Flagged to the user by the cluster session.

## CNN follow-up — the linear caveat was load-bearing

Ran a small conv net (2 conv layers → adaptive pool → linear) over the 17×11 ray
grid, on the **same collected data, same env-wise split, same clean/noisy pair** as
the linear probe, so the comparison isn't a different draw.

| | linear clean | linear noisy | **CNN clean** | **CNN noisy** |
|---|---|---|---|---|
| overall | 0.623 | 0.543 | 0.778 | **0.617** |
| flat | 0.647 | 0.410 | 1.000 | 0.604 |
| rough | 0.532 | 0.506 | 0.511 | **0.200** |
| stairs | 0.357 | 0.330 | 0.508 | **0.643** |
| gaps | 0.936 | 0.966 | 0.985 | 0.970 |

**Stairs was not undetectable — it was not linearly separable.** Under realistic
noise it goes 0.330 → 0.643 once the model can see spatial structure, which is what
a staircase's signature actually is. Reporting the linear result as a lower bound
rather than as "gate 2 fails" was therefore the right call.

**But the headline gain is not what it looks like.** Note rough *collapses*
(0.506 → 0.200) exactly as stairs improves, with 291 of 530 rough samples predicted
as stairs. Averaging the pair: linear noisy (0.506 + 0.330)/2 = 0.418; CNN noisy
(0.200 + 0.643)/2 = 0.422. **Identical.** The conv net does not resolve rough vs
stairs — it reallocates between them, and buys its real gain on flat (0.410 →
0.604) and by keeping gaps clean. Any future model that reports a big stairs number
should be checked for the same trade before it is believed.

**Not overfitting.** Held-out accuracy exceeded train accuracy in both conditions
(clean 0.778 test vs 0.749 train; noisy 0.617 vs 0.604), so the conv net is not
memorising the 7.7k training samples — the env-wise split held.

## Caveats — read before quoting these numbers

- Two model classes were tried. A larger or better-tuned network might separate
  rough from stairs, but the fact that a conv net moved the confusion around
  without reducing it is weak evidence that the pair is genuinely not separable at
  this noise level, rather than merely awkward to model.
- A quick variance-feature probe under the *old* (centre-spawn) protocol scored
  flat 1.00 / stairs 0.00, i.e. it collapsed the two rather than separating them.
  Not repeated under the corrected spread; superseded by the CNN result.
- Zero actions throughout, so the robot holds its default pose. This measures the
  terrain, not the gait, which is what a classifier reads — but a moving robot's
  scan distribution will differ.
- Single seed (42).

## Suggested next step

The CNN follow-up is **done** (above) and it narrows the question rather than
answering it: the blocker is specifically **rough vs stairs**, not the scan as a
whole. That makes the three levers cheaper to reason about than when this was
"three of four classes are indistinguishable":

- reducing the scan noise, or raising the terrain difficulty range, only has to
  buy enough separation for one pair;
- narrowing the taxonomy is now a smaller concession than it looked — merging
  rough and stairs into one "uneven" class preserves a 3-way switch
  (gaps / flat / uneven) that the sensing demonstrably supports at ~0.6–0.97 per
  class, rather than dropping to gap-vs-non-gap.

All three remain the user's call, not a config tweak: the noise level and
difficulty range are global training settings that every already-trained policy was
trained under, and the taxonomy is a change to what the project claims.
