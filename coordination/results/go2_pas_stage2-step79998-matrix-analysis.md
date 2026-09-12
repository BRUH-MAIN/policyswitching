# go2_pas_stage2 (model_79998) — laptop eval_matrix.py slice, 2026-09-12

Ran on `romen` (laptop), `unitree_rl_mjlab` conda env, 128 envs, 1200 steps.
`a100/eval_matrix.py --only pas_oracle --difficulties 0.5 --num-envs 128`.
No specialists on this machine yet (no local checkpoint, no SSH to the cluster,
HF backfill not landed) — **this slice is PAS-only.** Raw JSON + `summary.md`
under `unitree_rl_mjlab/eval_results/matrix/` (not duplicated here per
`coordination/inbox/to-laptop.md`'s convention); this file also isn't a
complete matrix archive — see the caveat below before quoting `summary.md`
on its own.

## Numbers (PAS oracle, `--anneal-prob 1.0`, difficulty 0.5, pinned conditions)

| terrain | fall % | mean ep. length | achieved/commanded speed | stalled % |
|---|---|---|---|---|
| flat | 98.9 | 30.2 | 23% | 12.4 |
| rough | 99.3 | 27.1 | 25% | 10.7 |
| stairs | 0.8 | 994.8 | 8% | 31.7 |
| gaps | 67.5 | 385.2 | 8% | 34.6 |
| mixed | 39.1 | 637.7 | 9% | 28.8 |

**Do not read this table as "PAS can't handle flat/rough ground."** See findings.md
bug #12: pinning difficulty to a single row (`--difficulty 0.5`) produces near-total
immediate failure on `flat` independent of policy capability — the same checkpoint
scores 0% fall / 100% full-length episodes on `--terrain flat` with difficulty left
at its default (spread over rows). Isolated by direct comparison, both runs otherwise
identical (same checkpoint, terrain class, envs, steps). Very likely a terrain-generator
artifact of collapsing to `num_rows=1` with one sub-terrain type at 100% proportion, not
a real capability gap. Flagged to the cluster session (`coordination/inbox/to-cluster.md`,
2026-09-12 (2)) since it also puts every difficulty-pinned cell job 11919 would produce
under suspicion, not just this one.

Given that, the only things in the table above I'd currently trust at face value are the
*relative* achieved-speed/stall pattern (consistently low achieved speed, 8-25% of
commanded, across every terrain including the "healthy-looking" stairs cell) — worth a
second look on its own, separately from the fall-rate numbers, once bug #12 is resolved.

## Correction to an in-flight suggestion: PAS can't answer either premise gate here

`orchestrator-session` suggested `pas_estimator` + the height-scan ablation as the
highest-value next step, on the reasoning that PAS-only can still speak to gate (b)
(do these policies use `height_scan` at all). Checked `a100/eval_matrix.py`: both
`pas_oracle` and `pas_estimator` are constructed with `ablatable=False` (matches
`eval_checkpoint.py --ablate-height-scan`'s own help text: "Stock-PPO policies only
(not PAS)"), and the cell-generation loop only emits ablation cells when
`p.ablatable and not args.no_ablation`. So no amount of `--ablation-difficulties` or
`--only pas_estimator` will produce an ablation cell for PAS — **gate (b) needs an
`ablatable=True` policy (a specialist or the generalist), same as gate (a)**. Both
premise gates are blocked on the HF backfill (`coordination/inbox/to-cluster.md`,
"Decision 2" thread), not just gate (a). The oracle-vs-estimator split itself is a
different, PAS-specific comparison (does the LSTM estimator approximate the
privileged latent), not a height-scan ablation.

## What's actually usable today

- The bug #12 report itself (findings.md #12) — the more valuable output of this
  slice, arguably, than the numbers.
- A clean, difficulty-uniform (not difficulty-pinned) PAS oracle vs. estimator
  comparison would be unaffected by bug #12 and would supersede the old
  bug-#10/#11-tainted `laptop.json` entry (oracle 75.5% / estimator 42.0%,
  measured with the terrain curriculum live). Not yet run — holding for the
  team's steer given GPU is shared with `orchestrator-session` and the
  premise-gate work is fully blocked regardless of what PAS-only shows.

## Next step

Pending team input (see `coordination/inbox/to-cluster.md` and messages to
`orchestrator-session`) rather than deciding unilaterally, since bug #12 changes
what's worth spending laptop GPU time on right now.
