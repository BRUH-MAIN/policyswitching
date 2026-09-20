# Handoff / Progress Report

**As of**: 2026-09-20 · **Repo state**: branch `vlm-pipeline`, 20 commits ahead of `main`
(`main` is at `5404d89`), working tree clean, **not pushed, not merged**. This branch lives
in its own linked worktree at `.claude/worktrees/vlm-pipeline` (the shared checkout at the
repo root stays on `main` — see "Two-machine coordination" in `CLAUDE.md` and the
multi-session house rules memory for why). **If you're reading this from the `main`
checkout, `cd` into the worktree (or `git worktree add`/`git switch vlm-pipeline` a copy)
before doing anything else** — none of the code, results or history below exist on `main`.
**Written for**: starting a fresh session (human or Claude) with no memory of how this
state was reached.

---

## 0. How to resume

1. Confirm you're on `vlm-pipeline` (`git branch --show-current`), not `main`.
2. Read `README.md` → `objective.md` → `findings.md` (in that order, per `CLAUDE.md`), then
   this file. `findings.md`'s "VLM navigation pipeline" section (search for that heading) is
   the condensed version of everything below; this file has the operational detail and the
   actual next steps.
3. `git pull --rebase` before doing anything — another session may have moved things. This
   branch isn't pushed yet, so a plain `git pull` on `main` won't show any of this; you need
   to already be on `vlm-pipeline` locally.
4. Check what's actually running before assuming anything is queued (§5 below) — **this
   laptop has rebooted unexpectedly several times during this work**, silently killing every
   background job each time.
5. Then go to §2 ("what needs a decision") — that's the actionable part.

## 1. What this is

`objective.md`'s person-following 2×2 (terrain specialists + anticipatory switching) is a
separate, still-open line of work — untouched by this branch. This branch is a **user-directed
pivot** (2026-09-16/17): build SARO's system (arXiv:2407.16412 — a VLM plans, perceives, and
double-checks sub-tasks in a closed loop to cross one terrain obstacle) in this project's
simulator, modified so the VLM also **chooses which of the three trained specialists
(flat/rough/stairs) runs** for each sub-task, and evaluate it against SARO's own objective
(goal-tracking across one intermediation), not `objective.md`'s.

VLM: local Gemma-4-E4B-it (Q4_K_M + F16 vision projector) served by llama.cpp
(`scripts/vlm_server.sh`) on this laptop's RTX 5060. Code: `unitree_rl_mjlab/src/vlm_nav/`
(camera, courses, controllers, executor, perception, prompts, VLM client, oracle-VLM
stand-in, policy bank). Scripts: `unitree_rl_mjlab/scripts/vlm_nav_*.py` (smoke test, frame
rendering, no-VLM baseline, offline perception eval, edge-accuracy eval, closed-loop runner,
result aggregation). Tests: `unitree_rl_mjlab/tests/test_vlm_nav.py`, 14 passing, pure logic
only (no simulator/VLM server needed to run them).

## 2. What needs a decision from you — in priority order

**#1 is the critical path.** Nothing else on this list can produce the actual result
(anticipatory/reactive switching beating a fixed policy) until it's resolved, because it's
what blocks the only course design that can test that claim at all (§3.3).

1. **The stairs specialist isn't reliable enough to build a switching demo around, at any
   tested difficulty.** With a *perfect* (ground-truth) policy chooser, up/down stairs
   succeed only 75–78% at the gentlest risers tested (0.05 m) and 0–34% at 0.07–0.09 m —
   confirmed not to be a switching-transient artifact (§3.1). Options, roughly cheapest
   first:
   - **Adopt SARO's fall definition** (orientation-only tipping, not knee/calf contact —
     Appendix B.3). Single-seed exploratory data says this alone fixes descending 0.05–0.07 m
     stairs (32/32 for every specialist) but does **not** fix climbing (stalls at the top,
     not falls — timeouts, unaffected by the fall definition). Needs re-confirming on fresh
     seeds either way; changing the outcome definition after seeing data must be labelled as
     such.
   - **Retrain the stairs specialist** on linear stairs at 0.05–0.10 m risers. Needs the
     cluster — check `coordination/status/cluster.json` for whether the drained-nodes issue
     from 2026-09-10 has cleared; it hadn't as of the last check before this branch started.
   - **Design a multi-obstacle course out of terrain the current specialists already handle**
     (flat + rough only, no stairs). Not a free win: with only flat/rough, one specialist
     (rough) already wins everywhere on *success*, so this needs a different metric where
     the tradeoff actually exists — e.g. time-to-goal, if the rough specialist turns out to
     be slower than flat on long clean stretches. **Not yet measured** whether that tradeoff
     exists at all; this is the first thing to check if you want to route around stairs
     entirely. See `coordination/results/vlm-nav-phase1-confirmation-rough.md` §"Reading it"
     for the reasoning.
   - **Drop stairs, accept no switching-value demonstration for now**, and report the
     pipeline + perception findings as the deliverable.
2. **What counts as a fall** (same knob as #1's first option, called out separately because
   it also affects how every existing stairs number should be read, independent of which
   path #1 takes): training terminations (`illegal_contact`, any non-foot contact > 10 N) vs.
   SARO's orientation-only definition.
3. **Stairs perception**: Gemma-4-E4B cannot see the simulated stairs regardless of what the
   specialist can do (§3.2) — it describes a staircase as "a flat, gridded floor" and never
   once answered "stairs" to the policy-selector question across 32 labelled frames. This is
   independent of #1: even a perfect stairs specialist is useless if the VLM never engages
   it. Options, not mutually exclusive:
   - A larger local VLM. Gemma-4-12B's weights and vision projector are already on the model
     drive (`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`) but **untested** — likely won't fit in 8 GB
     VRAM alongside the simulator; would need to check whether it fits at all, or run
     sim and VLM at different times.
   - A hosted API (Claude, GPT-4V, etc.) — sends rendered frames off the laptop, costs money
     per call, easy to wire in (the `VLM` protocol in `vlm_backend.py` is backend-agnostic).
   - Feed the VLM a **text description of the depth-derived geometry** instead of relying on
     it to read the RGB image — `depth_edge_estimate` (§3.2) already extracts an accurate
     step profile; describing that in the prompt sidesteps the vision problem entirely but
     is a bigger design change (no longer "the VLM decides from what it sees").
   - Make the rendered stairs **more visually realistic / higher-contrast** (real stairs have
     shadow lines, edge highlights, wear patterns that this project's flat-shaded mjlab
     geometry doesn't reproduce). Untested whether this would actually move the needle —
     worth a quick experiment (render a few frames with stronger directional lighting and
     re-run the description probe from Phase 2, `logs/vlm_nav/box_probe/`) before investing
     in a full re-render pipeline.
4. **How the VLM is told what each specialist is for**: by terrain label (current default —
   "the stairs policy is for stairs") or by measured competence (plumbed but off by default,
   `ExecutorConfig.policy_cards` / `prompts.POLICY_DESCRIPTIONS`). Matters because §3.1 found
   the label-implied specialist isn't always the best one on its own terrain (rough beats
   stairs at 0.05 m risers).
5. **LiDAR** (you asked 2026-09-19; not yet built). The robot has a Livox on the back and one
   on the front; mjlab's raycast sensor (already used for the specialists' height scan) can
   represent both with new ray patterns — cheap to add, no new infrastructure. It would fix
   perception cleanly (geometry instead of appearance — no blind zone, several metres of
   range instead of the camera's ~0.86 m ground blind zone and the height-scan's ~0.8 m
   range, immune to the rendering/shading problems in §3.2) but **does not touch #1** — a
   robot that sees the stairs perfectly still has a stairs specialist that can't reliably
   climb them. Three roles were proposed, still open:
   1. LiDAR as the geometry front end (replaces the depth-camera edge-finder), VLM still
      decides what/which-policy from a rendered height-map image or a text summary.
   2. Camera + LiDAR fusion (camera for appearance, LiDAR for geometry) — most faithful to
      the real hardware, most implementation work.
   3. LiDAR-only reactive switcher, no VLM at all — this is closer to `objective.md`'s
      original design with a much longer-range sensor than the height scan, and doesn't
      need SARO's language-reasoning layer.
   Say which (if any) to build, or hold this until #1–#3 are resolved, since none of them
   change what LiDAR would need to do.

## 3. What's done and what it found

### 3.1 Phase 1 — does the specialist choice matter? (no VLM; ground-truth "chooser" only)

Pre-registered *before* any data existed
(`coordination/results/vlm-nav-phase1-preregistration.md`, committed `acb76c0`) — calibrate
course difficulty, then confirm on fresh seeds, with pass/fail criteria written down first.

- **Calibration failed the pre-registered 80%-success bar on stairs at every difficulty
  tested** (`coordination/results/vlm-nav-phase1-calibration.md`): oracle (perfect
  ground-truth policy choice every step) gets 75–78% on 0.05 m risers, 0–34% at 0.07–0.09 m.
  Rough ground is 100% up to 0.08 m bump noise. Every stairs failure is a training-rule
  `illegal_contact` (knee/calf touched a step), never `fell_over`.
  - **Confirmed the failures aren't a switching artifact**: running the stairs specialist for
    the *entire* approach (never switching away from it) does as badly or worse than
    switching at the terrain boundary; switching earlier (1 m before the edge) doesn't help
    either. The failure is in the stairs specialist's locomotion on this geometry, full stop.
  - **Exploratory, single-seed** (not yet reconfirmed): the specialist *named* after a
    terrain isn't always the best one on it, and which one wins flips with step height — at
    0.05 m risers the **rough** specialist crosses up-stairs more reliably than the stairs
    specialist (97–100% vs. 81–84%); at 0.07 m it reverses (34% vs. 9%). Under SARO's
    orientation-only fall definition, essentially every specialist descends 0.05–0.07 m
    stairs without falling (stalls/timeouts at the top are unaffected, since those are
    timeouts, not falls).
- **Confirmation ran only on `rough`** (stairs stayed blocked by the calibration failure) —
  pre-registered Gate A/B, 192 trials/arm, fresh seeds 200/201, 3 goal offsets
  (`coordination/results/vlm-nav-phase1-confirmation-rough.md`):
  - **Gate A passes** (course is crossable: oracle 97.4%, CI 94.0–98.9).
  - **Gate B fails**: always running the rough specialist (98.4%, CI 95.5–99.5) matches
    perfect oracle switching (97.4%) — CIs overlap, gap is −1.0 points. **A course with one
    obstacle type cannot demonstrate switching value even with a perfect chooser.** The wrong
    fixed choice is expensive (flat 72.9%, stairs 52.6%), but *one* good fixed choice, made
    once, is exactly as good as switching. This is the reasoning behind decision #1's
    "different metric" option above.

### 3.2 Phase 2 — can Gemma-4-E4B perceive the obstacle? (offline, 720 labelled frames, no robot)

`coordination/results/vlm-nav-phase2-perception.md`.

- **Gemma does not see the stairs.** With neutral task instructions (no longer naming the
  obstacle — an earlier pilot's 8/8 planning score turned out to be reading the answer off
  the instruction text, not the image), it plans "no obstacle" on every stairs start frame
  (0/4), the policy-selector question never once answers "stairs" (0/32, always "flat"), and
  asked to freely describe a frame with a staircase ahead it says "a flat, gridded floor".
  Rough ground fares better but is still weak (selector recall 42%). Not fully disentangled
  whether this is the model or the render (0.05 m risers under a low camera do look subtle —
  a probe at 0.09 m risers still got "flat floor" in free text, though the selector did
  answer "stairs" 2/3 times there).
- **Localization (where is the obstacle) with SARO's own prompt format doesn't work at all**:
  a constant full-frame guess scores a higher IoU (0.43) than the model's actual answers
  (0.21, almost entirely `[0,0,0,0]` or the whole frame). A detection-style prompt at 560
  image tokens (needs `-ub ≥ 560` in llama.cpp — the vision tokens attend bidirectionally and
  the server otherwise aborts on the first image) helps a little (IoU 0.40) but is still
  unreliable, and live-tested edge errors were 0.4–0.75 m off, versus **0–8 cm** for a
  from-scratch depth-geometry estimator that never asks the VLM anything (`perception.py:
  depth_edge_estimate`). **Depth geometry is now the executor's default "where" source**; the
  VLM's box stays available as an ablation (`--where-source vlm_box`). The VLM still decides
  *what* the terrain is and *which* specialist to run — only the geometry localization moved
  off it.
- **The discriminator double-check ("Is there any stairs?") is 44% accurate** — not usable as
  SARO's confirmation gate; the executor treats it as advisory, not authoritative.

### 3.3 Closed loop — does it work end to end, and does the VLM's choice show up in outcomes?

`coordination/results/vlm-nav-closed-loop-smoke.md`.

- **The pipeline runs end to end with a real VLM in the loop.** Executor fed a ground-truth
  stand-in "VLM" (`oracle_vlm.py`) gets 4/4 on rough with switches landing within ~5 cm of
  the ideal point (this required fixing a real bug first: the camera's ground blind zone
  froze the near-edge estimate and the robot walked onto rough ground still on the flat
  policy — fixed by `perception.EdgeTracker`, which only trusts a near-edge sighting taken
  from beyond the blind zone).
- **Real Gemma: 2/2 on rough** (planned "rough ground", switched to the rough specialist at
  the edge) but **0/2 on 0.05 m up-stairs** (planned "no obstacle", stuck at the first step —
  the §3.2 finding showing up as a behavioural failure, not just a perception-eval number).
- **4-arm comparison on rough, 16 trials/arm, real Gemma**: VLM-nav+VLM-policy 100%,
  VLM-nav+oracle-policy 94%, VLM-nav+one-fixed-specialist 94%, ground-truth-both 100%. **All
  four indistinguishable** — exactly the ceiling effect §3.1's confirmation predicted for a
  single-obstacle course. Looking underneath the outcome: the VLM's actual decisions were
  mixed (planned "stairs" 3 times across these runs, on a course with no stairs at all; the
  policy selector split ~50/50 rough vs. flat) even though the trials still succeeded,
  because on this course either specialist usually gets across. **The 100% is not evidence
  the VLM chooses well — this course can't tell a good chooser from a mediocre one apart.**

## 4. Gotchas — read before trusting a new run or number

Full mechanisms in `findings.md` ("Bugs found and fixed", #18–#21 are this branch's). The
ones most likely to bite immediately, beyond the pre-existing repo-wide list (`CLAUDE.md`,
`findings.md` #1/#10/#11/#15):

- **The VLM model files live on an NTFS partition (`/dev/nvme0n1p4`, "New Volume") that
  unmounts on every reboot and Windows sometimes leaves flagged dirty**, refusing a
  read-write mount. Mount it read-only before starting the VLM server:
  `udisksctl mount -b /dev/nvme0n1p4 -o ro`. `scripts/vlm_server.sh` now does this itself and
  fails loudly (rather than hanging forever) if the server can't come up — but any of the
  ad-hoc `logs/vlm_nav/*/run.sh` driver scripts written during this work that predate that
  fix may not; check `curl -s localhost:8091/health` rather than trusting a "queued" job.
- **This laptop rebooted unexpectedly several times during this work**, silently killing
  every background job (training runs, the VLM server, monitors) each time with no crash log
  — `journalctl` showed clean `systemd-poweroff` sequences, not crashes. Cause not
  identified. Before assuming any run from a previous session is still going: check
  `pgrep -af "llama-server|vlm_nav_"` and `nvidia-smi`, not just a log file's last line.
- **mujoco_warp camera depth is distance along each pixel's *ray*, not optical-axis
  z-depth** — SARO's own deprojection formula (Fig. 8) assumes the latter (RealSense
  convention) and is wrong here by up to ~10% off-centre. `camera.py:CameraSpec.deproject_body`
  handles this; a round-trip unit test pins it. Don't re-derive this by hand elsewhere.
- **mujoco_warp shading has no light-intensity term** (each light adds
  `base_colour · cos(incidence)`, unboundedly) and **only textures planes/meshes, not
  boxes/heightfields** — a naive multi-light or textured-box setup saturates to white or
  renders flat. `course.py`'s two-low-oblique-lights-on-a-dark-base-colour setup and the
  grout-line tiling (geoms, not textures) work around both; don't add a third light or a box
  texture without re-checking against a rendered frame.
- **Gemma-4-E4B under a yes/no JSON schema returns empty content** (spends the whole token
  budget in hidden reasoning). Discriminator questions must be free text
  (`prompts.parse_yes_no`); the policy-selector JSON schema is fine.
- **`ExecutorConfig.where_source` defaults to `"depth"`**, not the VLM's box — see §3.2. If
  you're specifically trying to test SARO's own perception path, pass `--where-source
  vlm_box` explicitly; the default will silently use depth geometry instead.

## 5. Operational state right now

- VLM server: **stopped** (intentionally, to free the GPU between sessions). Restart with
  `scripts/vlm_server.sh` (reads `IMAGE_MAX_TOKENS`, `PARALLEL`, `CTX` from the environment;
  560 image tokens is what Phase 2's better numbers used).
- GPU: free (checked via `nvidia-smi`, ~14 MiB used) as of the last session.
- No background jobs running or queued.
- Other Claude sessions may be using this same laptop/GPU concurrently — see
  `coordination_multisession_house_rules` memory; announce before launching a GPU job.
- This worktree's `.claude/worktrees/vlm-pipeline` path is excluded via `.git/info/exclude`
  in the main checkout (not the shared `.gitignore`) so `git status` on `main` doesn't show
  it as untracked — that's local-only config, won't follow a fresh clone.

## 6. File map

| Want to know... | Read |
|---|---|
| The condensed version of everything in §3 | `findings.md`, heading "VLM navigation pipeline" |
| Exact pre-registered pass/fail criteria, written before any data existed | `coordination/results/vlm-nav-phase1-preregistration.md` |
| Full stairs-calibration numbers, diagnostics, decision options | `coordination/results/vlm-nav-phase1-calibration.md` |
| Full rough-course confirmation numbers and gate math | `coordination/results/vlm-nav-phase1-confirmation-rough.md` |
| Full perception numbers (planning/localization/discriminator/selector) | `coordination/results/vlm-nav-phase2-perception.md` |
| Closed-loop smoke-test and 4-arm comparison detail | `coordination/results/vlm-nav-closed-loop-smoke.md` |
| The pipeline code itself | `unitree_rl_mjlab/src/vlm_nav/` (one module per concern — camera, course, controllers, executor, perception, prompts, vlm_backend, oracle_vlm, policy_bank, twin_env) |
| How to run anything (baseline, closed loop, perception eval, aggregation) | `--help` on the matching `unitree_rl_mjlab/scripts/vlm_nav_*.py`; each has a usage example in its module docstring |
| Unit tests (pure logic, no sim/VLM needed) | `unitree_rl_mjlab/tests/test_vlm_nav.py` |
| The pre-pivot specialist/gate-1/gate-2 project state (still valid, untouched by this branch) | `main`'s `PROGRESS_REPORT.md` (as of `5404d89`) |
