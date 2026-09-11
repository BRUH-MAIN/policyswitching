# CLAUDE.md

Read [`README.md`](README.md) first for project orientation, [`objective.md`](objective.md) for what the research is actually trying to show (and how that differs from the SARO/PAS code that's currently the active training target), then [`findings.md`](findings.md) for what's actually been tried and what happened — checkpoint locations, eval results, and every infra bug found so far, several of which would silently produce a wrong-but-plausible-looking result if repeated. Update it (don't just append to conversation) whenever a training run finishes or a new gotcha turns up. `docs/` is separate: thorough, hand-maintained reference documentation for the mjlab/`unitree_rl_mjlab` framework itself, not this project's experiment history — don't re-derive what's already documented there either.

## Working in this repo

- **This is the canonical (and only) checkout.** A second clone at `/dist_home/d_palmani/policyswitching` was removed on 2026-09-05; its `logs/`, `eval_ckpts/` and `stage1/` were moved in here, and every `a100/*.sh` script now points at this path. All job outputs land here.
- **`clip_actions=6.0` in `rl_cfg.py` is still uncommitted**, so a fresh `git clone` of the GitHub repo would NOT have it and training would re-diverge the way job 11769 did. `train_pas_slurm.sh`'s clone step only fires if `REPO_DIR` is missing — keep this checkout in place, or commit the fix first.
- **`import src` collides with an unrelated project.** The conda env's editable `unitree_rl_mjlab` install can resolve `import src` to `/home/rohan/unitree_rl_mjlab` instead of this repo's `src/`. Set `PYTHONPATH` explicitly to this repo's `unitree_rl_mjlab/` before running training/eval scripts (see any `a100/*_slurm.sh` for the pattern).
- **Training runs need `HF_TOKEN` set** before `sbatch`-ing anything in `a100/` or `kaggle/` — checkpoints sync to a Hugging Face model repo (`a100/hf_sync.py`) since neither Kaggle nor SLURM ephemeral compute persists local disk across sessions/resubmissions.
- **`rsl_rl`'s `--agent.max-iterations N` is relative to the resume point, not absolute.** `a100/hf_sync.py` exists specifically to track an absolute per-stage iteration budget instead — this bit the project once already (a Kaggle run resumed at iteration 7800 and silently retargeted to 47800). Don't bypass `hf_sync.py`'s budget tracking when writing new training scripts.
- **Long PPO runs are prone to reward divergence without action clipping.** `clip_actions=6.0` in `rl_cfg.py`'s runner configs is a real fix for an observed blow-up (reward → -10M over one run), not a stylistic default — don't drop it when adding new runner configs.
- **`mjlab`'s terrain generator is geometry-only** (pyramid stairs, stepping stones, random grids, Perlin-noise/wave heightfields) — there's no material/friction-based terrain ("grass" vs. "gravel"). Friction variation is domain randomization (the `foot_friction` event term), not a terrain class.
- Use `scripts/eval_checkpoint.py` (headless, deterministic, reuses the *training* env config) for numeric comparisons across checkpoints/tasks — `scripts/play.py` uses the "play" config (near-infinite episodes, no curriculum) and is for qualitative/video inspection instead.

## Roadmap

The active phased plan (terrain-specialist policies, person-following-based anticipatory switching, evaluation harness) lives at `~/.claude/plans/the-idea-in-one-iridescent-wilkes.md`.
