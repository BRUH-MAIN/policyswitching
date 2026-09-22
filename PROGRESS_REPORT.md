# Handoff / Progress Report

**As of**: 2026-09-16, end of day · **Repo state**: `d50eabc` on `origin/main`, pushed and
in sync · **Written for**: starting a fresh session (human or Claude) with no memory of how
this state was reached. Where something needs more detail than fits here, a file path is
given — read that file rather than asking to re-derive it.

> **Update, 2026-09-22**: §4.2/§5 below ("when to start the VLM pipeline", "hasn't been given
> a go-ahead yet") are stale — you gave the go-ahead on 2026-09-16 and it's been built.
> **All of that work is on branch `vlm-pipeline`, in its own worktree at
> `.claude/worktrees/vlm-pipeline`, 22 commits ahead of this file's commit, pushed to
> `origin/vlm-pipeline` as of 2026-09-22, but still not merged.** `cd` into that worktree and
> read its own `PROGRESS_REPORT.md` for the real current state and next steps — don't keep
> reading this file expecting it to reflect the VLM work, it doesn't. Headline: the pipeline
> runs end to end with a real VLM choosing the specialist, plus a working person-following
> stack (scripted leader + a two-rate VLM-planner/YOLO-detector perception split that fixed a
> real-time-control failure), but the stairs specialist trained here (job 11849) turned out to
> be unreliable enough (75–78% success with a *perfect* chooser, at the gentlest stairs
> tested) that no course exists yet where switching can be shown to beat a fixed policy —
> that, not the VLM or its perception, is still the critical path. The rest of this file
> (§1–3, specialists, gate 1/2) is still accurate and untouched by that branch.
> **Cluster update, 2026-09-20/22** (`coordination/results/2026-09-20-stairs-step0-cluster-status.md`,
> `findings.md`): the "drained since 2026-09-10" framing below (§1) is stale — nodes resumed
> ~09-17, the constraint now is GPU queue wait (all 6 allocated by other users, ~12–19h
> projected for a fresh submission). Job 11914 (Gaps blended-terrain retrain) **completed**
> 09-17 — a Gaps specialist checkpoint exists, not yet evaluated. Jobs 11918 (generalist) and
> 11919 (eval matrix) were **cancelled by root**, never ran, and haven't been resubmitted yet
> (deferred for GPU priority on a stairs investigation: job 12033 tests whether the stairs
> specialist's unreliability is specific to pyramid-stairs geometry, queued not yet run).
> **If you want to draft an IEEE-style report from this project, read
> `report_content/ieee_report_source.md` on that branch** — it organizes everything (this
> file, `objective.md`, `findings.md`, and the `coordination/results/` write-ups) into paper
> sections with every claim traced to its source and every provisional number flagged as
> such.

---

## 0. How to resume

1. Read `README.md` → `objective.md` → `findings.md`, in that order (this is also what
   `CLAUDE.md` says to do first, every session).
2. Read `docs/CLAUDE.laptop.md` if you're on `romen` (this laptop) — this repo is worked on
   from two machines (this laptop + a SLURM cluster) and that file has the coordination
   protocol. Run `hostname` if unsure which machine you're on.
3. `git pull --rebase` before doing anything — another session may have moved things.
4. Then come back to §4 of this file ("what needs a decision") — that's the actionable part.

## 1. What this project is

Instead of one generalist walking policy for a quadruped (Unitree Go2), train a small set
of **terrain-specialist** RL policies, each expert at one terrain type, and build a system
that picks the right one at runtime — anticipating upcoming terrain from a followed person's
path rather than only reacting once the robot's own sensor sees it. Full design: `objective.md`.

## 2. What's done and solid (don't re-verify, just use)

- **Three specialists trained**: Flat (job 11852), Rough (11851), Stairs (11849), all
  10,000 iterations, checkpoints at `eval_ckpts/go2_spec_{flat,rough,stairs}/model_9999.pt`.
  No gaps specialist exists — two attempts plateaued, neither finished.
- **Gate 1 (does specialization matter?) — passes, robustly.** Cross-terrain evaluation
  (`coordination/results/gate1-cross-terrain-matrix-analysis.md`) shows no single specialist
  wins everywhere: Rough specialist best on rough terrain (2.4× fewer falls than runner-up,
  solid), Stairs specialist best on stairs (1.4×, real but single-seed/provisional). This
  survived two rounds of correction (a stale-cache bug and a metric bug, findings.md #15) and
  holds under three independent ways of measuring "falls," so it's trustworthy.
- **`report_content/` deliverable, written today**: `project_report.md` (pipeline +
  specialists + their evaluation, aimed at a reader who wants the short version) plus
  `videos/` — 9 clips, one per (specialist × terrain) combination, showing the gate-1 result
  visually. **A rendering bug was found and fixed while making these** (findings.md #17): the
  sensor-ray visualization overlay broke visually at distance from spawn, looking exactly
  like a physics crash — confirmed by instrumented rollout that the actual physics was fine
  throughout, then confirmed the overlay was the cause by disabling it. Fixed in
  `scripts/record_specialist_videos.py`; all 9 videos re-rendered clean. This never touched
  any real eval number.
- **Cluster git auth is fixed**: repo-scoped SSH deploy key set up on the cluster side, no
  longer blocked. `HF_TOKEN` is in this laptop's `.env` (gitignored) and specialists are
  backed up to a **private** HF repo (`RohanRamesh/go2-specialists` — NOT the public
  `go2-pas-saro`, which holds only PAS and must stay untouched).

## 3. What's unresolved (genuinely open, not just unverified)

- **Can the height-scan sensor tell terrain types apart? — messy, not settled.**
  - Original measurement said rough vs. stairs hit a hard ~0.51 ceiling even with a perfect
    sensor. Then (`findings.md` #16) that measurement turned out to have a real flaw: it was
    collected across a difficulty range whose bottom rows are literally flat ground labelled
    "stairs"/"rough", which was inflating the apparent confusion.
  - Re-ran at a fixed, fair difficulty (matching gate 1's own conditions) to fix that. Result
    was **not a clean confirmation**: the simple (linear) model improved on both classes as
    predicted, but a fancier model (small CNN) got *worse*, and breaking terrain down into
    its finer sub-types showed confusion spread broadly across flat/rough/stairs rather than
    concentrated in one specific pair the way it first looked. Full detail and raw numbers:
    `coordination/results/gate2a-height-scan-discriminability-analysis.md`.
  - There's also a substantive **design recommendation sitting unapproved**:
    `coordination/results/rough-stairs-switching-decision.md` argues the entire question is
    being asked wrong — the metric that matters is expected falls, not classifier accuracy —
    and proposes a specific fix (a cost-sensitive decision rule instead of merging terrain
    classes, §4 of that file). **This has not been decided on.** Note its numeric worked
    example was written before the difficulty-fix reruns above, so its exact numbers may need
    re-checking, though its core argument doesn't obviously depend on them.
  - **You (the user) already said**: if these results aren't good, stop investigating and
    move to the camera/VLM approach instead (see §5). Given time spent here already, that's
    the reasonable call unless someone specifically wants to chase this further.
- **A real, unfixed flaw in the experiment design** (found while writing the doc above, §5 of
  it): `objective.md`'s planned comparison gives one arm a perfect ground-truth terrain
  reading and another arm a noisy real classifier, for what's supposed to be a test of *how
  far ahead you can see*, not *how good your sensor is*. As written, this could make the
  "anticipatory" arm win for the wrong reason. **Not yet fixed in `objective.md` — this needs
  a decision before Phase 4 (the switching module) gets built, independent of the rough/stairs
  question above.**

## 4. What needs a decision from you specifically

1. **Rough/stairs classification**: keep investigating (§3, first bullet), approve the
   cost-sensitive-rule proposal as-is, or drop it and move straight to VLM? You've already
   leaned toward the third option.
2. **When to start the camera/VLM decision pipeline.** Discussed and agreed as the likely
   right direction (the height-scan sensor is inherently short-range — ~0.8m — which is the
   root of the classification struggle; a camera doesn't have that limit). **Nothing has been
   built yet — no code, no new sensor wiring.** Say when to start and I will.
3. **The oracle-vs-classifier confound in §3 above** — this affects whether Phase 4's core
   result would even be valid, independent of everything else. Worth deciding before Phase 4
   starts, whenever that is.
4. **Cluster compute**: the GPU nodes have been drained since 2026-09-10 (an admin issue, not
   something fixable from a session). This blocks: the sensing-matched generalist baseline
   (job 11918), the full cross-terrain matrix at multiple difficulties (job 11919), and any
   gaps-specialist retrain. Nobody has reported the nodes back up as of the last check
   (`coordination/status/cluster.json`, still stamped 2026-09-12).

## 5. The VLM/camera pivot — context if a new session picks this up

You asked whether terrain classification should use a camera + LiDAR + a vision-language
model instead of the current narrow height-scan sensor. Checked against the simulator
(`mjlab`) directly rather than assuming:

- **LiDAR-style sensing**: cheap — the current height-scan already *is* a raycast sensor,
  just configured as a small downward grid. A longer-range/wider-angle version is a config
  change, not new infrastructure.
- **Camera (RGB/depth)**: mjlab supports a real camera sensor type and it's used elsewhere in
  the same codebase, but it is **not wired into this project's robot at all yet** — genuine
  integration work.
- **Running a VLM**: the expensive part. It would need to run across every simulated
  environment repeatedly during training/eval — a different order of compute than the small
  classifier used so far, likely needing the cluster (currently down).

This is a real pivot, not a tweak, and per §4 above, hasn't been given a go-ahead yet.

## 6. Gotchas — read before trusting any new eval number

Full list with mechanisms: `findings.md`, "Bugs found and fixed" (17 entries as of this
writing). The ones most likely to bite immediately:

- Set `PYTHONPATH` to this repo's `unitree_rl_mjlab/` explicitly before running anything —
  the conda env can otherwise resolve `import src` to an unrelated project (`CLAUDE.md`).
- Never compare raw survival-percentage numbers across different policies/terrains without
  checking `findings.md` #1, #11, #15 first — several metrics silently favor whichever policy
  dies fastest or moves least. Falls-per-distance-travelled is the safe one.
- A policy that "never falls" isn't necessarily walking — check achieved speed vs. commanded
  speed before trusting a low fall-rate (#1, #14).
- A qualitative video that looks broken isn't necessarily the policy's fault (#3, #17) —
  cross-check against the numeric eval before concluding a policy regressed.
- The three trained specialists share one observation space by design (`objective.md`'s scope
  decisions) — don't change one without checking whether it breaks cross-specialist tooling.

## 7. File map

| Want to know... | Read |
|---|---|
| Why this project exists, the actual research design | `objective.md` |
| Full experiment history, every bug found | `findings.md` |
| The short version, for someone who wants pipeline + specialist eval only | `report_content/project_report.md` (+ `report_content/videos/`) |
| Full gate-1 (specialization) numbers and methodology | `coordination/results/gate1-cross-terrain-matrix-analysis.md` |
| Full gate-2 (sensor discriminability) numbers, including the open questions in §3 | `coordination/results/gate2a-height-scan-discriminability-analysis.md`, `coordination/results/rough-stairs-switching-decision.md` |
| What's queued for the cluster session, and why | `coordination/inbox/to-cluster.md` ("Open" section) |
| Cluster/laptop coordination mechanics | `docs/CLAUDE.cluster.md`, `docs/CLAUDE.laptop.md` |
| Live job/checkpoint status | `coordination/status/cluster.json`, `coordination/status/laptop.json` |
