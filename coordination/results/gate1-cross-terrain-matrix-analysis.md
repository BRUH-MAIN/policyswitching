# Premise gate 1 — does specialization matter, and does it vary by terrain?

**Date**: 2026-09-12/13 · **Machine**: `romen` (laptop) · **Script**: `a100/eval_matrix.py`
(37 cells, difficulty 0.5, 128 envs, 1200 steps) · **Raw**: `unitree_rl_mjlab/eval_results/matrix/*.json`,
`summary.md` · Diagnostic side-quests: `unitree_rl_mjlab/eval_results/bug12_verify/`, `.../drift_test/`.

## Headline

**Gate 1 passes, on the two columns that are actually informative.** The rough specialist
is the least-often-falling policy on `rough` terrain (2.4× the runner-up), and the stairs
specialist wins `stairs` (1.4×, single-seed, see caveat below). `flat` is a three-way tie
at the floor (every specialist: 0 falls). `gaps` and `mixed` are catastrophic for every
specialist (no specialist has ever seen stepping-stones terrain). **Both PAS modes are
excluded from this comparison entirely** — see below, they're degenerate everywhere, not
competitive evidence either way.

This required two rounds of correction before it was trustworthy: a stale-cache bug that
silently reused pre-bug-#12-fix `pas_oracle` data from earlier in the day, and a genuine new
measurement bug (findings.md #15) that made a terrain-generation artifact look real until a
controlled test and a from-first-principles metric fix took it apart. Both are described in
full below because the corrections are as load-bearing as the result.

## The metric: falls per 100 m travelled, not `fall_pct`

`eval_checkpoint.py`'s `fall_pct` pools episodes over a whole rollout — a policy that dies
every 15 steps generates far more (shorter) episodes than one that survives the full window,
so raw `fall_pct` silently over-weights whichever policy/terrain-column dies fastest
(findings.md bug #15). Falls-per-100m is robust to that *and* to bracing (bug #1/#14): a
policy that stands still travels ~0 m, so its rate is undefined/floored rather than
artificially low. Computed post-hoc from the same JSON (`total_fails = round(fall_pct *
episodes / 100)`, `total_distance = distance_rate * num_envs * steps * dt`) — no new eval
runs needed, no change to `eval_checkpoint.py` (that's a fix proposed to the cluster session,
not made here).

## Falls per 100 m, specialists only

| policy \ terrain | flat | rough | stairs | gaps | mixed |
|---|---|---|---|---|---|
| flat spec | 0.00 | 14.41 | 10.42 | 684 | 154 |
| rough spec | 0.00 | **5.36** | 11.14 | 1357 | 272 |
| stairs spec | 0.00 | 13.02 | **7.60** | 975 | 303 |

Column winners (bold) are the same specialist that owns that home terrain, on both
informative columns. Margins: `rough` 2.4× (13.02/5.36), `stairs` 1.4× (10.42 and 11.14 vs
7.60 → 1.37×/1.47×). **The stairs margin is the one to be careful with**: single seed, ~27-47%
depending which runner-up you compare against, well within a range a different seed could
close. `rough`'s 2.4× is comfortable. `objective.md`'s statistical plan already flagged
specialists as single-seed by budget decision — this is the concrete case where it bites.
Report `rough`'s result as solid, `stairs`'s as directionally consistent but provisional.

**Verdict is robust across three metrics, which is worth stating precisely** because #15
just invalidated one of them: raw `fall_pct`, falls/robot-minute (hazard rate, ignores
distance), and falls/100m all agree that rough wins `rough` and stairs wins `stairs`. The
*winner* doesn't depend on the metric that turned out to be broken — only the *margins* do,
and the three metrics don't even agree with each other on magnitude, which is itself a reason
not to trust any single one alone for a magnitude claim. Concretely, for flat-specialist vs.
rough-specialist on `rough` (bug #15's own worked example): 2.4× by `fall_pct` (65.5/27.2),
3.3× by hazard rate (3.01/0.92 falls/robot-min), 2.7× by falls/100m (14.41/5.36) — three
different numbers for the same comparison. (This is a different ratio from the *column
margin* above, rough-specialist vs. its runner-up stairs-specialist, 2.4×/13.02/5.36 — same
`5.36` denominator, different numerator, easy to conflate; caught doing exactly that in an
earlier draft of this file.)

`flat`: 0.00 for all three (every specialist trained with `height_scan` intact handles flat
ground perfectly at this difficulty — unsurprising, and uninformative for gate 1 beyond
confirming none of them is broken). `gaps`: 684–1357 falls/100m — catastrophic and expected,
no specialist has ever trained on `stepping_stones`. `mixed`: 154–303, dominated by the 25%
of each specialist's envs that spawn on the gaps column they've never seen (see #15 for why
this isn't as extreme as `fall_pct` alone made it look, and isn't zero either).

**Sanity check — are these numbers measuring real locomotion, not bracing?** Achieved speed
as % of commanded, all cells: 23.9–54.3%, well above PAS's 8–11% (bug #14) and stalled% is
correspondingly lower (4–17% vs PAS's 21–35%). Specialists on `gaps` specifically are moving
*more*, not less (flat spec 54.3%, stairs spec 50.9%) — the catastrophic fall rate there is
falling while actually trying to walk fast on unfamiliar terrain, not bracing. Checked before
trusting the diagonal, per the lesson from bug #14.

## PAS: excluded from the comparison, not just a low scorer

Both `pas_oracle` and `pas_estimator` are locomoting at 8.0–10.5% of commanded speed across
*every* terrain, uniformly — the exact bug #14 signature. (First pass showed `pas_oracle` at
22.9% on `flat`, which briefly looked partially healthy; that number came from a stale file
left over from this morning's pre-bug-#12-fix diagnostic run that `eval_matrix.py`'s
resumable "skip if exists" logic silently reused instead of regenerating — caught by checking
file mtimes, not by the eval failing. Regenerated: 8.3% on `flat`, consistent with everywhere
else.) A policy that barely moves can't meaningfully "win" a column by falling less — it has
fewer opportunities to fall, which is exactly the bug #15 mechanism from a different angle.
PAS is a reference result (already demoted from arm 1, see `objective.md`), not a data point
for gate 1.

## Height-scan ablation cells: inconclusive as a gate-2 test, but one real positive finding

Per `objective.md`'s revised gate 2 text: ablating the scan on a *specialist* (near-constant
scan throughout training, ~0.003 of real signal per findings.md's normalizer-std table)
substitutes an already-near-constant input with a constant — a near-zero delta is close to
guaranteed regardless of whether a policy trained on varied terrain could use the scan. That
argument correctly predicts most of the 12 cells: `flat` is floored at 0 for everyone, and
`rough`-on-`rough` barely moves (−2.1%). But it does **not** hold uniformly, and recomputing
all 12 under falls/100m (not `fall_pct`, which the ablation's episode-length shift would
distort per bug #15) surfaces a real exception:

| policy | terrain | with scan | without scan | Δ | speed with → without |
|---|---|---|---|---|---|
| flat | rough | 14.41 | 17.22 | +19.5% | 35.7 → 35.1 |
| flat | stairs | 10.42 | 11.22 | +7.7% | 40.6 → 38.6 |
| rough | rough (home) | 5.36 | 5.25 | −2.1% | 29.5 → 30.5 |
| rough | stairs | 11.14 | 9.32 | −16.3% | 29.7 → 28.8 |
| stairs | rough | 13.02 | 15.30 | +17.6% | 23.9 → 25.6 |
| **stairs** | **stairs (home)** | **7.60** | **9.87** | **+29.9%** | **26.3 → 29.2** |
| all three | gaps | see below | | | (all ~50-60% slower) |

**The stairs specialist genuinely uses its scan on its home terrain.** +30% more falls per
metre without it, and — the detail that rules out "it's just being more cautious with the
scan" — its speed *rises* without the scan (26.3%→29.2% of commanded). It's moving faster
and falling more, which is a real functional dependence, not a null result dressed up by the
weak-intervention argument. This is a genuine positive finding, not "run, not evidence" —
credit to `orchestrator-session` for pushing the recompute rather than letting the blanket
argument stand once one cell contradicted it.

**Discard the `gaps` ablation cells entirely — they're artifacts, not a scan-usage signal.**
All three show large apparent *improvements* without the scan (flat −57.7%, rough −71.8%,
stairs −69.8%), but speed collapses by roughly half in every case (e.g. stairs 50.9%→21.5%).
Falls/100m being low because the policy has nearly stopped moving is bug #1's mechanism yet
again — "removing the scan improves gaps by 70%" would be exactly the wrong headline.

**Net for gate 2**: still inconclusive as originally scoped (needs `Unitree-Go2-Generalist`,
job 11918, which doesn't exist), but not uninformative — the stairs-specialist result is a
real, if narrow, positive finding worth keeping regardless of how the generalist test lands.

## `mixed` terrain: a real anomaly, fully explained, not a terrain bug

Initial numbers showed every specialist at 96–97% `fall_pct` on `mixed`, ~39–49 points worse
than an env-weighted average of its four single-class columns predicted — investigated as
three hypotheses in sequence, two rejected:

1. **Column-allocation rounding** (does `mixed`'s cumsum-based column assignment over-weight
   `stepping_stones`?) — checked directly against config, allocation is exact (5/5/5/5 of 20
   columns), rejected.
2. **Commanded lateral drift** (do robots wander from their spawn column into a `gaps`
   column?) — arithmetic initially seemed to support it (a 25%-gaps decomposition matched
   observed numbers closely), but a controlled test rejected it: zeroing commanded lateral
   velocity (`--lin-vel-y 0 0`) left both `mixed` and a single-class control terrain's fall
   rate unchanged, the opposite of the predicted dissociation (mixed should have dropped
   toward the class average if drift were the cause; it didn't move).
3. **The metric** (bug #15, above) — confirmed. `mixed` pools 4 sub-populations with wildly
   different episode-cycling rates; the 25% of envs on the never-trained `gaps` column
   generate ~93% of all counted episodes, so pooled `fall_pct` is essentially the gaps number
   wearing a `mixed` label. Episode-weighted predictions match observed values to within 1.3
   points for all four policies checked; an unbiased hazard rate matches the *env-weighted*
   prediction almost exactly and the anomaly vanishes.

Recorded as its own bug (#15) rather than folded in here, since it's a general problem with
`fall_pct`/`survival_pct`, not specific to `mixed`.

## What's still open

- Stairs' 1.4× margin needs seeds before it's a settled result (`objective.md`'s existing
  statistical-plan gap).
- Gate 2b (scan ablation) needs the generalist, which needs job 11918, which needs the nodes
  undrained.
- `fall_pct`/`survival_pct` have no unbiased replacement in `eval_checkpoint.py`'s output yet
  — proposed to the cluster session, not fixed here (cluster-owned file).
- No `Unitree-Go2-Spec-Gaps`/`GapsWarm` checkpoint exists, so gate 1's `gaps` column is
  "every existing specialist fails," not "specialists can't do gaps" — those are different
  claims and only the training log (findings.md, "Terrain specialists") speaks to the latter.
