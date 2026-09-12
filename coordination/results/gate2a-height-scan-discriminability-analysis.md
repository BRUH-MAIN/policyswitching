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

## Headline

**Partial pass, and the part that passes is not the part the design needs.** The
scan identifies `gaps` almost perfectly and separates `flat`/`rough`/`stairs`
weakly. Under the noise the policy actually sees, 4-way accuracy is 0.549 against
0.25 chance — but that number is carried by gaps.

| | clean | noisy (as the policy sees it) |
|---|---|---|
| overall (4-way, chance 0.25) | 0.632 | **0.549** |
| flat | 0.648 | 0.419 |
| rough | 0.562 | 0.506 |
| stairs | 0.339 | **0.336** |
| gaps | 0.964 | **0.974** |

Injected noise costs 8.3 accuracy points overall. `stairs` sits at 0.336 — barely
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
   Gap/non-gap is reliable. Flat vs rough vs stairs is close to a coin toss under
   realistic noise, and stairs — the class with the most distinctive geometry, which
   ought to be the *easiest* — is the worst.
2. **The fix is the noise magnitude, not the scale.** Scaling multiplies signal and
   noise alike and cannot change SNR. `Unoise(±0.1 m)` is the problem: it is applied
   pre-scale, against ≤10 cm of relief. Either reduce it towards the terrain relief,
   or raise the terrain difficulty range so the relief clears the noise. This is a
   config decision on the cluster side.
3. **Or narrow the taxonomy.** If the noise level is realistic and must stay, the
   honest move is to let the switching module discriminate gap/non-gap from the scan
   and take the rest from proprioception — a smaller claim, but one the sensing
   actually supports.

## Caveats — read before quoting these numbers

- **This is a linear classifier, so these are a lower bound.** A CNN or MLP over the
  17×11 ray grid could do better, particularly for stairs, whose signature is
  spatial structure rather than a mean shift. The gap/non-gap-vs-rest conclusion is
  robust to that (it rests on a 20σ mean separation and a near-zero one), but
  "stairs is undetectable" is not established — only "stairs is not linearly
  separable from flat at this noise level".
- A quick variance-feature probe under the *old* (centre-spawn) protocol scored
  flat 1.00 / stairs 0.00, i.e. it collapsed the two rather than separating them.
  Worth repeating under the corrected spread before concluding anything from it.
- Zero actions throughout, so the robot holds its default pose. This measures the
  terrain, not the gait, which is what a classifier reads — but a moving robot's
  scan distribution will differ.
- Single seed (42).

## Suggested next step

Re-run with a small CNN over the ray grid before accepting "flat/rough/stairs are
indistinguishable". If a CNN also fails, gate 2 fails as specified and the noise
level has to change before Phase 4 — which is exactly the decision `objective.md`
says this gate exists to force.
