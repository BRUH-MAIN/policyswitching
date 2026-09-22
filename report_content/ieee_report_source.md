# Report source material — terrain-specialist switching for a person-following quadruped

**Purpose of this file**: source material for drafting an IEEE-style report/paper, organized
by the sections such a paper would use, with every empirical claim traceable to a script, a
committed result file, or a `findings.md` bug number. It is not itself a paper — figures
aren't laid out, prose isn't polished, and every section flags what's solid vs. provisional
so a draft doesn't accidentally overclaim what a single-seed, laptop-scale study supports.
**As of**: 2026-09-22, branch `vlm-pipeline` (commit `6d411fe`), not yet merged to `main`.

Citations below to external papers (RMA, "Learning to Walk in Minutes", the ETH perceptive-
locomotion line) are given from memory and should be verified (exact venue/year/author list)
before submission — only the SARO citation (arXiv:2407.16412, PDF vendored at
`2407.16412v3.pdf`) has been checked against a document actually in this repo.

---

## I. Suggested title / framing

*"Terrain-Specialist Locomotion Policies for a Person-Following Quadruped: A Premise-Gated
Study of When Switching Helps"* — "premise-gated" because the honest framing of this project
at its current state is two confirmed premise checks (Section VI.A–B) plus a working
perception/control pipeline (Section VI.D–E), with the actual switching-vs-generalist result
(the paper's real claim, per `objective.md`) **still unbuilt**, blocked on cluster access and
one unresolved specialist. A paper written today should be scoped as a systems paper on the
pipeline plus the premise-gate findings, not as the full 2×2 comparison. See Section VIII.

## II. Abstract material

- **Setting**: quadruped (Unitree Go2) locomotion across four geometry-defined terrain
  classes (flat, rough, stairs, stepping-stone gaps) in a MuJoCo-based simulator (`mjlab`),
  trained with PPO (RSL-RL).
- **Core idea being tested** (`objective.md`): rather than one generalist policy, train
  terrain-specialist policies and switch between them at runtime; in a person-following
  setting, use the followed person's trajectory as a preview of terrain beyond the robot's
  own ~0.8 m sensing horizon, extending how far ahead the switch can anticipate.
- **What's established**: specialization measurably matters (Section VI.A) — no single
  policy is best on every terrain; the onboard height-scan sensor discriminates gaps and flat
  ground but not rough-vs-stairs at the noise level tested (Section VI.B); a full VLM-driven
  perception-and-control pipeline for person-following and terrain-aware sub-task execution
  runs end-to-end in closed loop at real time (Section VI.D–E).
- **What's not yet established**: whether switching beats a sensing-matched generalist
  policy (the paper's actual thesis, per `objective.md`) — the generalist was never trained
  (cluster nodes drained since 2026-09-10), and the stairs specialist is not reliable enough
  (75–78% success with a *perfect* oracle chooser) to build a demonstrative multi-terrain
  course around yet.
- **Honest one-line summary for an abstract**: this work reports the infrastructure, the
  premise-gate measurements, and a working anticipatory-perception pipeline for the switching
  hypothesis; the head-to-head switching-vs-generalist result is future work pending
  additional compute.

## III. Introduction / motivation material

Use `objective.md`'s "The gap this targets" section nearly verbatim; the argument is already
report-ready:

- Prior single-generalist-policy approaches (RMA — Rapid Motor Adaptation; the ETH
  perceptive-locomotion line; "Learning to Walk in Minutes") train one policy with a terrain
  encoder to generalize across surfaces. Perceptive variants already read a forward height
  scan — a short preview, in this project's config reaching **≈0.8 m ahead** (`terrain_scan`,
  a 1.6 m × 1.0 m grid, 187 rays, centred on the base).
- What none of this line exploits: in a *person-following* task specifically, the tracked
  person is already walking ahead of the robot, past the onboard scan horizon — a free,
  causal preview of terrain the robot hasn't reached yet.
- **The claimed contribution is horizon extension, not preview vs. no preview.** The
  experiment design (`objective.md`, "Preview-horizon sweep") is built to show gain as a
  function of how far the preview reaches, specifically so "no gain" and "leader too close to
  matter" are distinguishable outcomes.
- Explicitly scope this as **a minor-but-real contribution**: the specialists, the switching
  mechanism, and the baselines are things one would build for a clean ablation regardless; the
  anticipatory-preview idea is the one non-standard piece.

## IV. Related work material

- **RMA** (Kumar, Fu, Pathak, Malik) — rapid online adaptation via a learned latent, single
  policy, no explicit terrain classes.
- **ETH perceptive locomotion line** (Miki et al. and related ANYmal work) — perceptive
  policies with elevation-map/height-scan input, single generalist policy.
- **"Learning to Walk in Minutes"** (Rudin, Hoeller, Reist, Hutter) — massively parallel PPO
  training methodology; this project's PPO/RSL-RL training pipeline descends from the same
  lineage (`unitree_rl_mjlab` is a vendored fork built on that toolchain).
- **SARO** (arXiv:2407.16412) — the paper this project's PAS baseline and VLM-planning
  pipeline replicate. SARO uses a VLM to plan, perceive, and double-check sub-tasks
  ("intermediations") to cross a single terrain obstacle, with a low-level policy trained via
  **Probability Annealing Selection (PAS)**, a two-stage privileged-to-unprivileged
  distillation — *not* a runtime switch between frozen policies (contrast with this
  project's own specialist-switching design; see Section V.D for why PAS is a reference
  result here, not the arm-1 baseline).
- **Positioning**: explicit multi-specialist switching between discrete terrain policies is
  less common than single-generalist approaches, and (per this project's related-work read)
  nobody has combined it with using a followed person as a terrain preview specific to the
  person-following setting.

## V. Method

### A. Simulation environment and terrain taxonomy

- Simulator: `mjlab` (MuJoCo-based), Unitree Go2 quadruped.
- Terrain is **geometry-only** — pyramid stairs, stepping stones, random/wave-noise
  heightfields — there is no material/friction-based terrain (grass vs. gravel); friction
  variation is domain randomization (`foot_friction` event term), not a terrain class.
- Four terrain classes, defined once in `TERRAIN_CLASSES` (`env_cfgs.py`), shared by training
  and eval: **flat**, **rough** (`random_rough` ∪ `wave_terrain`), **stairs**
  (`pyramid_stairs`/`pyramid_stairs_inv`), **gaps** (`stepping_stones`).
- Observation includes a forward-facing height scan: 187 rays, 1.6 m × 1.0 m footprint,
  reaching ≈0.8 m ahead, with injected observation noise `Unoise(±0.1 m)` uniform per ray.
- Action clipping (`clip_actions=6.0`) is load-bearing for training stability, not a
  stylistic default — a real reward-divergence blow-up (reward → −10M over one run) was
  observed and fixed by this.

### B. Terrain-specialist policy training

- PPO via RSL-RL, one specialist per terrain class, trained only on that class.
- All specialists **share one observation space** (including raw `height_scan`), reward
  structure, and a 10,000-iteration budget (8192 envs × 24 steps) — deliberately, so
  differences between specialists are attributable to training terrain alone, and so a later
  switching/gating module can blend them.
- Three of four specialists trained successfully: **Flat** (job 11852), **Rough** (11851),
  **Stairs** (11849), all at iteration 9999.
- **Gaps failed twice** under a 100%-`stepping_stones` design (reward plateaued at ≈−6 to −8,
  episodes ending in 10–14 steps — immediate falls, diagnosed as a cold-start problem: no
  easy terrain to learn basic locomotion on before also solving gap-crossing). A
  blended-terrain retrain (job 11914) and a warm-started 100%-gaps redesign
  (`Unitree-Go2-Spec-GapsWarm`, initialized from the Rough specialist via
  `a100/warm_start_ckpt.py`) are both built; neither has completed (cluster nodes drained
  since 2026-09-10). **No usable gaps specialist exists as of this writing.**
- Absolute-iteration-budget tracking (`a100/local_ckpt_resume.py`) and warm-start hygiene
  (resetting iteration count, optimizer moments, and observation-normalizer stats) are
  infrastructure points worth a methods-section footnote if the paper describes resumable
  training, since a naive resume silently retargets `--agent.max-iterations` and corrupts the
  reported iteration budget.

### C. Premise gates — methodology

Per `objective.md`, two premise gates must pass before a switching module is worth building;
both are measured directly rather than assumed (Section VI.A–B has results):

1. **Gate 1 — no single policy is best on every terrain**, tested by a full cross-terrain
   eval matrix (`a100/eval_matrix.py`): every specialist × every terrain class, at pinned
   difficulty and command range, terrain curriculum off. This is a **column-wise** condition
   (does the per-terrain best policy vary?), not row-wise (each specialist doing worse
   off-terrain than on it) — the two are logically independent, and only the column-wise one
   is sufficient for the switching claim.
2. **Gate 2 — the height scan actually discriminates the terrain classes**, tested directly
   via a labelled-scan classifier (`height_scan_classifier.py`, multinomial logistic
   regression and a small CNN over the raw ray grid), independent of any policy.

Evaluation methodology notes worth stating in a methods section: numeric cross-policy
comparisons always use `eval_checkpoint.py` under pinned conditions (curriculum off,
explicit terrain/difficulty, fixed command range) — a live curriculum lets difficulty adapt
to the policy under test, making cross-policy numbers meaningless. `Metrics/twist/
error_vel_xy` is never compared across policies because it accumulates per episode and so
scales with survival time, not tracking quality.

### D. PAS baseline (SARO replication)

- Implements SARO's low-level training technique only (not its VLM task planner) — a
  **two-stage distillation**: Stage 1 trains a privileged actor conditioned on a 36-dim
  latent (encoded height scan + base linear velocity + foot friction) while an LSTM+MLP
  estimator learns to predict that latent from proprioception alone; Stage 2 anneals from the
  privileged latent to the estimator's prediction (`P_t = 0.9998^iteration`) so the final
  policy runs on proprioception history only.
- Fully trained: Stage 1 to iteration 39999, Stage 2 (anneal) to iteration 79998 — **≈8× a
  specialist's training compute** (80k vs. 10k iterations at the same env/step count), and it
  adds two reward terms (`energy`, `joint_vel_l2`) no specialist has.
- **Explicitly not arm 1 of the comparison.** Its estimator-only deployment mode reads no
  height scan at all (an unfair sensing comparison against specialists that do); its oracle
  mode is closer to sensing-matched but adds privileged base velocity and foot friction the
  specialists never see.
- **A measured failure mode, worth reporting as a cautionary methods point**: PAS's
  near-perfect survival on flat/stairs is an artifact of the `energy`/`joint_vel_l2` reward
  terms rewarding minimal motion — achieved speed is 8–11% of commanded across every terrain
  and both modes, versus 24–46% for the specialists. Two independent measures (base-frame
  velocity magnitude and per-step XY displacement) agree to within 1–5%, ruling out a
  measurement artifact; the mechanism is the reward shaping, not a bug. **Report PAS as a
  separate replication result, not as competitive evidence for or against specialization.**
  (This is the same "measurement answers an adjacent question" failure mode as
  `error_vel_xy` looking *better*, not worse, for a slower policy — worth one sentence in a
  limitations section about designing metrics that can't be gamed by standing still.)

### E. VLM-driven specialist selection and SARO task replication

- Architecture (`unitree_rl_mjlab/src/vlm_nav/`): a VLM (local Gemma-4-E4B-it, Q4_K_M
  weights + F16 vision projector, served by llama.cpp on a single 8 GB laptop GPU) plans a
  sub-task ("intermediation": which terrain feature is ahead), perceives it (bounding box or
  depth-derived edge), selects which trained specialist runs, and double-checks sub-task
  completion — the same plan→perceive→discriminate loop as SARO §III.
- Deviations from SARO's own setup, all forced by what this repo has: intermediations are
  {stairs_up, stairs_down, rough} (no ramp or door terrain exists in this sim; gaps has
  terrain geometry but no course segment built for it); low-level policies are this project's
  three specialists, not PAS (PAS's estimator-only actor has a different observation width
  and cannot enter the policy bank; also see the near-stationary caveat above); the VLM is a
  4B local model on one 8 GB GPU, not LLaVA-34B on an 8×3090 server; robot pose is ground
  truth, not estimated.

### F. Person-following and two-rate perception

- A scripted kinematic leader (`src/vlm_nav/leader.py`) — a fixed-base mjlab entity,
  non-colliding and in a camera-only geom group so it cannot perturb the specialists'
  height-scan observation — walks ahead of the robot at a varying speed; the robot holds a
  standoff distance via a follow-distance controller (`controllers.py`).
- **Two-rate perception architecture** (the paper-worthy systems contribution here): a VLM
  names the target class in language, once, off the control path (~0.2 Hz); a fine-tuned
  YOLO detector localizes that class at full control rate (every step). This separation is
  the fix for the naive design (VLM directly in the control loop), which is unusable for
  real-time control (Section VI.E has the numbers).
- **Labels for the fine-tuned detector are free, not hand-annotated**: the leader's pose is
  set by the simulator and its geometry is fixed, so its exact 2D bounding box is the
  projection of its known 3D box (`CameraSpec.project_body`, the algebraic inverse of the
  camera's existing deprojection — verified by a round-trip test to 6e-14 px). This is worth
  stating explicitly in a methods section as a sim-to-label pipeline, distinct from the
  detector training itself.

## VI. Results

### A. Gate 1 — cross-terrain specialization matrix

Source: `coordination/results/gate1-cross-terrain-matrix-analysis.md`, `a100/eval_matrix.py`
(37 cells, difficulty 0.5, 128 envs, 1200 steps).

Falls per 100 m travelled (chosen over raw fall-percentage, which over-weights whichever
policy dies fastest per episode — see `findings.md` bug #15):

| policy \\ terrain | flat | rough | stairs |
|---|---|---|---|
| Flat specialist | 0.00 | 14.41 | 10.42 |
| Rough specialist | 0.00 | **5.36** | 11.14 |
| Stairs specialist | 0.00 | 13.02 | **7.60** |

- **Gate 1 passes**: the per-terrain-column best specialist is not the same policy in both
  informative columns (rough → rough specialist, stairs → stairs specialist).
- **Rough's margin is solid**: 2.4× fewer falls than the runner-up, robust across three
  different fall-rate metrics tried (raw `fall_pct`, hazard rate, falls/100 m) — they disagree
  on magnitude (2.4×/3.3×/2.7×) but agree on the winner.
- **Stairs's margin is real but weaker evidence**: 1.4×, single-seed, closer to the run-to-run
  noise floor (~1 point, per the gate-2 calibration). Report as "directionally consistent, not
  yet confirmed with repeat seeds."
- Flat is a three-way tie at the floor (0 falls for every specialist) — expected, and
  uninformative for the gate beyond confirming none of the three is broken.
- Gaps and mixed-terrain columns are excluded from the headline table: every specialist is
  catastrophic on gaps (684–1357 falls/100 m — none has ever trained on stepping-stones), and
  mixed needs separate normalization; neither changes the ranking above.
- **Locomotion-quality sanity check, not just fall counts**: specialists achieve 24–46% of
  commanded speed in these cells, well above PAS's 8–11% bracing artifact (Section V.D) — the
  fall-rate differences reflect real locomotion attempts, not one policy gaming the metric by
  moving less.

### B. Gate 2 — height-scan terrain discriminability

Source: `coordination/results/gate2a-height-scan-discriminability-analysis.md`,
`height_scan_classifier.py` (labelled scans, multinomial logistic regression + small CNN, no
policy in the loop).

| class | linear probe, noisy (policy-realistic) | CNN, noisy |
|---|---|---|
| flat | 0.410 | — |
| rough | 0.506 | — |
| stairs | 0.330 | — |
| gaps | 0.966 | — |
| **overall (4-way, chance 0.25)** | 0.543 | **0.617** |

- **Gaps is trivially separable** (mean scan value ~0.28 vs. ~0.05 for the other three — a
  20× the injected-noise-std mean shift, from stepping-stones' missed ray returns).
- **Rough and stairs collide.** Per-ray terrain relief for both is at or below the injected
  `Unoise(±0.1 m)` noise floor (within-sample ray-std 0.0059–0.0076 against noise std 0.0115).
  Neither model tried separates the pair.
- **Caveat that must be stated in any write-up (bug #16, diagnosed 2026-09-16, confirmation
  run pending as of this writing)**: the originally reported "measured zero-noise ceiling"
  for rough-vs-stairs (~0.51 each, described in an earlier draft as "provably insufficient"
  sensing) is now understood to be **substantially a labelling artifact**, not a sensor
  property. The data-collection protocol samples difficulty uniformly including `difficulty=
  0`, and both `pyramid_stairs`/`pyramid_stairs_inv` (`step_height_range=(0.0, 0.1)`) and
  `wave_terrain` (`amplitude_range=(0.0, 0.2)`) **start at zero relief** — meaning a real
  fraction of "stairs" and "rough" labelled samples are flat ground carrying the wrong label,
  independent of what any sensor could measure. Quantitatively, the CNN's clean-condition
  confusion into "flat" for stairs/rough samples (25.9%/15.7%) closely tracks the fraction of
  difficulty rows that degenerate to flat by construction (~30%/~15%). **A paper drawing on
  this gate should report the corrected, difficulty-pinned re-measurement (not yet run at
  writing time — see `coordination/results/rough-stairs-switching-decision.md` §6 for the
  exact commands) rather than the original ~0.51 ceiling.**
- **Design implication, independent of the above correction**: rough and stairs are close
  enough in this sensor that a 4-way reactive classifier is not viable as specified; the
  clean option (if the discrimination question resolves against separability) is merging them
  into one "uneven" class, since gaps/flat/uneven is well-supported (0.6–0.97 per class) and
  requires no retraining risk.

### C. PAS replication result

See Section V.D for the mechanism. Headline number for a results table: PAS achieves 8–11%
of commanded speed across all terrains and both oracle/estimator modes, against a specialist's
24–46% — reported as a separate replication result, explicitly excluded from the gate-1
comparison and from any "generalist" role, since the sensing-matched generalist baseline
(`Unitree-Go2-Generalist`) was never trained (Section VIII).

### D. SARO task replication at the paper's protocol

Source: `findings.md`, "SARO task replication" (2026-09-22); run via
`scripts/vlm_nav_run.py --arms vlm gt+oracle --num-envs 20 --seed 300 --goal-y-offset 0.8
--level L1`; artefacts in `logs/vlm_nav/saro_protocol/`. 20 trials per intermediation, goals
off-axis, full closed loop (plan → perceive → discriminator check).

| intermediation | VLM Overall | VLM Across | ground-truth ceiling | planner's answer |
|---|---|---|---|---|
| stairs_up | **0%** | 0% | 75% | `none` 20/20 |
| stairs_down | **45%** | 45% | 95% | `none` 20/20 |
| rough | **100%** | 100% | 100% | `none` 17, `rough ground` 17, `stairs` 1 |

For comparison, SARO's own Table I (real robot, LLaVA-34B): stair 60% overall / 70% across /
88% stable-loc; ramp 25/50/67; gap 45/80/94; door 30/50/63. **Because this replication's pose
is ground truth (no localization error), the fair comparison column is SARO's `Stable Loc`,
not `Overall`.**

- **The failure is perception, entirely, and it is upstream of everything else in the loop.**
  The planner answers `intermediation: none` on 40/40 stairs trials (both directions) — it
  never emits a climb sub-task because it never concedes there is anything to climb. The
  ground-truth arm crosses the same courses at 75–95%, so the terrain is crossable and the
  locomotion is not the bottleneck.
- **The rough column's 100% must not be cited as evidence the VLM perceives rough terrain.**
  The planner still answered `none` on roughly half of rough trials, and the specialist
  selector chose the flat policy on 267/391 calls made *on rough ground* — every trial still
  succeeded only because every specialist survives rough terrain (the same ceiling effect
  that makes gate 1's flat column uninformative). This is a methods point worth stating
  plainly in any results section: **a course where the wrong choice is survivable cannot
  measure choice quality**, regardless of how good the headline success rate looks.
- **This is a small local VLM's specific failure**, not necessarily a fundamental limit of the
  approach — Section VIII lists the untested mitigations (larger local model, hosted API,
  text-described depth geometry instead of asking the VLM to read the image, more realistic
  rendering).

### E. Person-following and two-rate perception

Source: `findings.md`, "Person-following" and "Two-rate perception" sections
(2026-09-21/22).

- **Ground-truth localization, ideal case**: 2.5 m standoff, leader speed varying with a full
  stop: **0.159 m RMS gap error, 0.256 m max**, zero falls, over 42 s and 7 speed changes.
  Establishes the follow-controller and leader-kinematics groundwork is solid independent of
  perception.
- **Two control defects were found and fixed by varying the leader's speed** (both invisible
  under a constant-speed leader, so worth a sentence on why the eval protocol matters):
  proportional-control droop against a moving setpoint (fixed with line-of-sight velocity
  feed-forward) and a follow-controller speed cap below the leader's own top speed (the robot
  could not close a gap once opened).
- **VLM-in-the-loop control is not viable in real time**: 5.88 m gap error, 10.4× slower than
  real time, because only 11 of 84 VLM calls (~2 s each) produced a usable position in time
  to be useful.
- **Splitting perception into a slow VLM planner + a fast detector fixes this**:

  | arm | gap error (ground truth) | wall-clock / sim-time |
  |---|---|---|
  | VLM inside the control loop | 5.88 m | 10.4× slower than real time |
  | YOLO detector only | 0.488 m | 0.98× |
  | **VLM planner + YOLO detector** | **0.495 m** | **1.05×** |

  The combined system matches the detector-only arm's accuracy while retaining the VLM's
  language-level target selection — the detector runs at 3.5–3.9 ms/frame (YOLO11n/s on an
  RTX 5060), three orders of magnitude faster than a VLM call.
- **Stock COCO YOLO cannot see this project's leader geometry** (reads it as "baseball bat" /
  "sports ball", detects nothing at the follow distance) — an appearance gap specific to
  flat-shaded mjlab geoms that does not carry to hardware, where a real person is COCO's home
  ground. **Fine-tuning on 1000 auto-labelled frames** (free labels via the projection method,
  Section V.F) for 40 epochs reaches **precision 1.000, recall 0.942, mAP50 0.951, mAP50-95
  0.892** — resolves the appearance gap entirely within this simulator.
- **Methods-relevant negative result, worth a limitations sentence**: a follower's
  self-reported range cannot validate itself. The first VLM-driven follow run scored a
  near-perfect 0.159 m "error" while the leader had actually walked away to 16.5 m — the
  perception estimator had locked onto a static goal marker in the same camera-visible object
  class, and the metric was computed from the very estimate under test. Only an independently
  logged ground-truth range exposed it. Generalizes beyond this project: any real scene has
  other vertical objects a person-detector could confuse with the target, and evaluation needs
  a channel the system under test cannot influence.

## VII. Discussion / limitations (draft material)

- **Every specialist is single-seed.** The rough-terrain gate-1 margin (2.4×) is comfortably
  above the ~1-point run-to-run noise floor measured elsewhere in this project; the stairs
  margin (1.4×) is not, and should be reported as provisional pending repeat seeds.
  `objective.md`'s statistical plan already commits to ≥3 seeds for the switching/gating
  networks specifically (not the frozen specialists) for this reason.
- **The gaps specialist does not exist.** Two training attempts plateaued; two redesigns are
  built but neither has run to completion (blocked on cluster availability). Any table
  including a "gaps" row/column before one of these finishes should say "unanswerable," not
  report a null/zero result.
- **The stairs specialist is the critical blocker for the paper's actual claim.** At 75–78%
  success with a *perfect* (ground-truth) specialist chooser, at the gentlest riser height
  tested, no course design yet exists where switching could be shown to beat a fixed policy —
  a course needs the wrong choice to be meaningfully worse than the right one, and stairs is
  currently unreliable regardless of which policy runs it.
- **The sensing-matched generalist baseline (arm 1 of the actual comparison) was never
  trained.** Every "gate 1 passes" claim in this report establishes that switching beats
  *some* fixed specialist, not that it beats the generalist — `objective.md` is explicit that
  these are different claims and the second is the one the paper's thesis rests on.
- **A small local VLM is the perception bottleneck for stairs specifically**, not a
  fundamental property of the plan-perceive-select architecture — the rough-terrain success
  of the pipeline (mechanically, if not evidentially per the ceiling-effect caveat) shows the
  architecture itself functions.
- **The person-following leader is a scripted kinematic body, not a simulated human or a real
  camera-tracked person** — deliberate, per `objective.md`'s scope decisions, so the terrain
  along the leader's path is ground truth during the sim ablations that carry the empirical
  weight of the eventual result; real LiDAR/vision person-tracking is deferred to an optional
  hardware demo.
- **Everything here ran on one 8 GB laptop GPU.** This bounds both the VLM size tested (4B)
  and the practicality of long training runs locally — all specialist/generalist training
  happens on a separate SLURM cluster, whose availability (drained since 2026-09-10 as of the
  last confirmed heartbeat) is the practical bottleneck on the whole project's pace right now.

## VIII. What's not done — future work section material

Directly from `objective.md` and the branch's own `PROGRESS_REPORT.md` priority list:

1. Retrain the stairs specialist at wider/harder riser configurations, or resolve the
   fall-definition question (training terminations vs. SARO's orientation-only tipping
   definition — single-seed exploratory data suggests the latter fixes descending stairs but
   not climbing, which stalls rather than falls).
2. Train `Unitree-Go2-Generalist` (the sensing-matched arm-1 baseline) and re-run the
   cross-terrain matrix against it.
3. Resolve one of the two gaps-specialist redesigns.
4. Re-run gate 2's discriminability measurement with difficulty pinned away from zero
   (commands already written, `coordination/results/rough-stairs-switching-decision.md` §6),
   and decide the taxonomy question it answers (keep 4-way pending sensing fixes, or merge
   rough+stairs into one "uneven" class).
5. Build the actual switching/gating module and the 2×2 arm comparison (reactive/anticipatory
   × hard-switch/soft-blend) plus the preview-horizon sweep — none of this exists yet.
6. Resolve stairs perception for the VLM pipeline: larger local model, hosted API, feeding
   depth-derived geometry as text instead of relying on VLM vision, or more realistic
   rendering — none tried yet.
7. Decide whether the person-following stack (leader + two-rate perception, already built and
   measured) becomes the delivery vehicle for the preview-horizon experiment, or stays a
   separate systems contribution alongside the SARO replication.

## IX. Figure/table candidates already available without new experiments

- Gate 1 cross-terrain matrix (Section VI.A table) — ready to use as-is.
- Gate 2 discriminability numbers (Section VI.B) — use once the difficulty-pinned
  re-measurement lands; the current numbers need the bug-#16 caveat attached if used sooner.
- SARO protocol comparison table (Section VI.D) — ready, with the ceiling-effect caveat on
  the rough row stated in the caption, not just the prose.
- Person-following gap-error-vs-arm table (Section VI.E) — ready as-is; a time-series plot of
  gap error over the speed-change run (`logs/vlm_nav/follow_*/trace.json`) would make a good
  qualitative figure.
- Nine qualitative videos of each specialist on each terrain (`report_content/videos/`,
  referenced from the earlier `report_content/project_report.md`) — usable for a
  demo/supplementary-material link, not a print figure.
- YOLO fine-tuning precision/recall/mAP numbers (Section VI.E) — ready as a one-row table or
  a single sentence.

## X. Cross-reference map

| This section | Primary source file(s) |
|---|---|
| II–IV | `objective.md` |
| V.A–C | `objective.md`, `env_cfgs.py`, `a100/eval_matrix.py`, `CLAUDE.md` (action-clip note) |
| V.D | `docs/07-pas-implementation.md`, `findings.md` bug #14 |
| V.E–F | `PROGRESS_REPORT.md` §1/§3, `unitree_rl_mjlab/src/vlm_nav/` |
| VI.A | `coordination/results/gate1-cross-terrain-matrix-analysis.md` |
| VI.B | `coordination/results/gate2a-height-scan-discriminability-analysis.md`, `findings.md` bug #16 |
| VI.C | `findings.md` bug #14, `coordination/results/go2_pas_stage2-step79998-matrix-analysis.md` |
| VI.D | `findings.md` "SARO task replication", `coordination/results/vlm-nav-phase2-perception.md` |
| VI.E | `findings.md` "Person-following", "Two-rate perception" |
| VII–VIII | `PROGRESS_REPORT.md` §2, `objective.md` "Premise gates" and "What a good result looks like" |
