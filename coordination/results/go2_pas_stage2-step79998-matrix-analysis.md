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

## Update 2026-09-12: bug #12 diagnosed, confirmed, and fixed (fix stuck on cluster, unpushed)

`orchestrator-session` traced the root cause to mjlab's `terrain_generator.py`:
`difficulty_range=(d, d)` alone already pins every row to difficulty `d`, so
`apply_eval_conditions`'s `num_rows=1` contributes nothing to the pin and instead
collapses terrain extent + forces every env onto row 0. I confirmed this directly
with a throwaway monkeypatch script (not a repo edit): same PAS oracle checkpoint,
`--terrain flat --difficulty 0.5`, `num_rows` left at its configured 10 with only
`difficulty_range` pinned + `max_init_terrain_level=None` → 0% fall, exactly
reproducing the difficulty-unpinned result. The cluster session independently
re-confirmed via the real checkpoint (three-way comparison: as-written 19/32 fallen
by step 60, difficulty=None 0/32, fix candidate 0/32) and applied the one-line fix
to `env_cfgs.py` — but that commit is currently stuck local-only on the cluster
checkout (its git push is still blocked pending a deploy key, same issue as the
81906b2 blocker earlier today). **Not yet on `origin/main` as of this write-up** —
verified directly (`git merge-base --is-ancestor <their-commit> origin/main` fails).
Don't treat the fix as landed until confirmed on origin/main again.

## Difficulty-uniform PAS oracle vs. estimator, all 5 terrains (unaffected by #12)

Ran per `orchestrator-session`'s steer once #12 was diagnosed (not blocked on the
fix landing, since this path was already proven safe). Full numbers now in
`findings.md`'s PAS section ("Pinned-condition PAS oracle vs. estimator, per
terrain class"). Headline: oracle and estimator track closely everywhere except
`gaps`, where the estimator degrades sharply (74.0%→93.1% fall) — plausibly because
PAS's own training gave gap terrain only 15% weight. Also flagged an unexplained
secondary pattern: achieved speed is a near-constant 8–11% of commanded across
every terrain/mode, even where fall rate is ~0% — worth a look before reading
flat/stairs survival as "solved."

## Next step

Handed off: GPU passed to `orchestrator-session` for `height_scan_classifier.py`
(premise gate 2a). Waiting on the cluster to get bug #12's fix onto `origin/main`
before any more difficulty-pinned cells run.
