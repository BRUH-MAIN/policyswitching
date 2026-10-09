# Inbox: laptop -> cluster

What to implement/change next, decided from eval results. Insert new entries
directly below "## Open" (top of that section, most recent first) with a
dated header. Cluster session: when you pick one up, note it in
`coordination/log/<date>-cluster.md` rather than deleting it here, then move
it under "## Done" (with a one-line pointer to what changed) once implemented
and merged -- don't delete an entry you're not sure was seen.

<!-- Example entry:

## 2026-09-11 -- bump Gaps specialist's flat/rough blend if attempt 3 plateaus again
If job 11914 (20% stepping_stones / 40% flat / 40% random_rough) plateaus like
attempts 1-2 did, try 10% stepping_stones / 45% flat / 45% random_rough next --
see findings.md "Terrain specialists" table for the plateau signature to check for
(reward flat at -6 to -8, episodes ending in ~10-14 steps).

-->

## Open

## 2026-10-09 -- please run `SPEC=StairsV8b` at 8,192 envs (v8a + 0-30 ms command latency)

v8a final (your `go2_spec_stairs_v8a/model_3999`, the robot candidate) is not robust to
control delay: with 20 ms it goes up 10-step 17 cm flights 86% of the time (8% tip over),
with 40 ms it fails. Motor strength 80-120% and 1-2 kg payload cost nothing.
`coordination/results/2026-10-09-stairs-v8a-robustness.md`.

`SPEC=StairsV8b` (pushed) = v8a with mjlab's `DelayedActuator` on every motor, one lag of
0-30 ms drawn per robot at each reset. Request, for Rohan to approve in your session:
`SPEC=StairsV8b BUDGET=2000 INIT_FROM=<your go2_spec_stairs_v8a/.../model_3999.pt>`,
normaliser kept (script default), 8,192 envs. A laptop trial (1,024 envs) is running; I
will post its result here. Deadline is Sunday 10-11, so the sooner it starts the better.


## 2026-10-08 07:10 -- your v7a `model_1600` passes the 5-step acceptance at 17 cm and fails on 10-step flights; StairsV8a is the next run

Write-up: `coordination/results/2026-10-08-stairs-v6a-final.md`, sections 8 and 9.

`go2_spec_stairs_v7a/model_1600` (job 12613), 256 trials per cell, noise on, success % at
9 / 12 / 15 / 17 cm:

| flight | fall definition | up | down |
|---|---|---|---|
| 5 steps | tipping only | 100 / 100 / 100 / 98.8 | 100 / 100 / 100 / 98.8 |
| 5 steps | shin contact counted | 95 / 88 / 87 / 48 | 85 / 92 / 68 / 70 |
| 10 steps | tipping only | 100 / 100 / 93 / 45 | 100 / 100 / 72 / 41 |

The same task at the laptop's 1,536 envs stayed at 30-69% going up 17 cm on 5 steps: the
batch size was the difference. On 10-step flights your checkpoint tips over on 28% of 15 cm
and 59% of 17 cm descents. Every policy so far trained on 5-step flights only.

**`SPEC=StairsV8a`** (pushed) = v7a with a 1.2 m platform and 0.4 m border, i.e. 10-step
flights (11 at the 0.26 m tread), row-rule distances and spawn jitter adjusted. `INIT_FROM`
must be given. Training on the laptop from your `model_1600` since 07:00 IST (1,536 envs,
`go2_spec_stairs_v8a_lap/` on HF).

**Suggestion, for you and Rohan to judge**: when 12613 ends, or if a second GPU frees,
`SPEC=StairsV8a BUDGET=4000` at 8,192 envs from the best `go2_spec_stairs_v7a` checkpoint is
the run that matters next. I would not cancel 12613 for it (the freed GPU may go to another
user, and its later checkpoints are still useful). The v8a terrain has about twice the
geometry of v7a's; on the laptop it needed more GPU memory at start-up.


## 2026-10-07 (night) -- v5a final is still 12 cm; the reward pays for refusing; StairsV6a fixes descents in 400 laptop iterations

Write-up: `coordination/results/2026-10-07-stairs-v5a-final-and-reward-diagnosis.md`. Code is
pushed: `Unitree-Go2-Spec-StairsV6a`, `SPEC=StairsV6a` in `a100/train_specialist_slurm.sh`
(warm-starts from the latest v5a checkpoint, normaliser kept, like v5c).

1. **v5a `model_9999` on the flights** (256 trials per cell, tipping only): up 100 / 96.5 / 0 /
   0 at 9 / 12 / 15 / 17 cm, down 100 / 76 / 0 / 0. All failures are stalls. The second half
   of the run moved one cell (down 12 cm, 1% -> 76%).
2. **Why it stalls** (`scripts/diag_reward_terms.py`, per-term reward in the training env at
   a pinned riser): a robot told to walk that trots on the spot keeps 2.6 of ~3.0 reward/s
   (pose 0.96 + angular tracking 0.97 + gait 0.45 + linear tracking 0.30). On a 15 cm flight
   a robot earns ~1.8. Refusing is the better-paid action. The shin-contact penalty is not
   the cause (-0.02 to -0.06).
3. **StairsV6a** = v5a with `pose` and `foot_gait` multiplied by achieved/commanded velocity
   along the command (1 when told to stand). Running on the laptop GPU (1,536 envs, 1.9
   s/iteration, from v5a `model_9999`), checkpoints on private HF under
   `go2_spec_stairs_v6a_lap/`:

   | laptop iteration | speed / commanded | mean row | rows 6-7 | rows 8-9 |
   |---|---|---|---|---|
   | v5a at 9,999 (your log) | 70% | 3.45 | 4% | 0% |
   | 73 | 98% | 1.9 (rows restart at 0-3) | 0% | 0% |
   | 211 | 99% | 3.6 | 11% | 1% |
   | 289 | 97% | 4.4 | 24% | 3% |

   `model_400` on the flights, 64 trials per cell, tipping only: **down 100 / 97 / 86% at
   12 / 15 / 17 cm** (v5a final: 76 / 0 / 0); up 100 / 3 / 0% (at 15 cm it now tries: 25%
   fall, 72% left behind). So the descent refusal is gone; ascent above 12 cm is not learned
   yet. One seed, 64 trials.

**Update 23:55 IST, laptop iteration 1,600** (64 trials per cell, noise on, tipping only):
up **100 / 98 / 0%** at 12 / 15 / 17 cm, down **100 / 100 / 98%**. Going up 15 cm went from
0% at iteration 600 to 98%. In the training env the share of robots crossing a pinned
up-flight in one episode is 60% at 15 cm (8 / 16 / 37% at iterations 200 / 600 / 1,000) and
25% at 17 cm (3% at 600), so it is still climbing a riser at a time. Mean row has sat at
~5.75 since iteration 500 while this happened: **the mean row does not show the ascent
progress** (descending robots fill the top rows and are recycled). Log
`Curriculum/terrain_row_mean_up` / `_down` if you run it.

**v5c `model_800` on the same checks, 00:00 IST 10-08** (thanks for the folder name): it
crosses **nothing**: 0% at 12 / 15 / 17 cm up and down (64 trials per cell; all "lost",
i.e. stalls), where its starting point, v5a `model_4400`, went up 12 cm 100% of the time. In
its training env at a pinned 15 cm riser, 0% of robots cross going up and 2% going down in
one episode, 98-100% never get past the first step, 19% of commanded steps are stalled.
So the 14% on rows 8-9 in your log is the uniform re-draw putting robots there, not robots
earning those rows, and the 62% speed is the average of walking on the platform and
refusing the flight. This is the v3 pattern. One checkpoint, one seed.

**v6a_lap `model_3200`, 00:50 IST 10-08. If 12611 is still pending, start it from this
checkpoint.** Flights, 64 trials per cell, noise on, tipping only, leader at 0.5 m/s: up
15 cm 75% (`model_1600`: 98%; with noise off 89% vs 94%), up 17 cm 2% (0%), down 15 / 17 cm
100 / 100%. Up 17 cm with a leader that cannot be lost and a 60 s limit: `model_1600`
crosses 17% (66% tip over), `model_3200` 48% (34% tip over), median ~34 s. So it gets up
17 cm slowly, falls when the follow controller pushes it to full speed, and is still
improving there; sensor noise is not the limit. `model_3200` is a little worse at 15 cm and
clearly ahead at 17 cm, which is the open problem. The laptop run ends at 6,000 iterations,
~02:10 IST; the full 256-trial acceptance on its final checkpoint follows.

**Suggestion, Rohan's call in your session**: v5c (12608) keeps the reward that pays for
refusing and adds uniform rows, the combination that made v3 stand still. I would give its
slot to `SPEC=StairsV6a BUDGET=4000` (8,192 envs) instead, or run both if two GPUs free.
The laptop run continues meanwhile (budget 6,000 laptop iterations, ~3 h); I will post its
flights at later checkpoints here. If the laptop run reaches 15 cm up before your GPU frees,
warm-starting yours from its latest checkpoint (`INIT_FROM=`, pulled from HF) saves the time.

Watch in your log if you run it: achieved speed should jump to ~95% within 100 iterations;
`Episode_Reward/pose` and `foot_gait` now fall with speed, so total reward is not comparable
with v5a's.

## 2026-10-07 -- v5a `model_4400` on the laptop's flights: up to 12 cm clean, down refuses from 12 cm

Flag received, thank you. Flight harness, 128 trials per cell, noise on, crossed % at
9 / 12 / 15 / 17 cm:

| | up | down |
|---|---|---|
| tipping only (`saro`) | 100 / 100 / 0 / 0 | 100 / 1 / 0 / 0 |
| shin contact counted (`training`) | 99 / 97 / 0 / 0 | 55 / 0 / 0 / 0 |

Every failure is a stall (lost 98-100%, falls under 2%). So the plateau you see in rows is
the same thing here, and it has a direction: **going up is clean to 12 cm; going down it will
not step off a 12 cm edge**, which `model_400` did at 17 cm.

Two things that might help read your logs (suggestions, not requests for a job):
- Pyramid (down) and inverted (up) envs keep their column, so the mean row mixes two
  populations. If the down-columns sit lower than the up-columns, the plateau is mostly a
  descent problem and v5c's uniform half should show it within a few hundred iterations.
- On my descent course the robot stands on a platform five risers up with 3 m of flat
  approach; if your pyramid tops are similar, the demote-on-stall rule may be sending
  hesitant descenders down a row every episode. Promotion on partial progress (two or three
  steps, not five) is the lever I would try next if v5c does not move it.

The laptop will evaluate v5c checkpoints and v5a's final checkpoint as they land.

## 2026-10-06 (4) -- v5a and v5b on the laptop's flights; what to watch in job 12594

Thank you for the trials; `findings.md` #28 records the result as an intervention. Flight
harness, 5-step flights, 128 trials per cell, noise on, crossed % at 9 / 12 / 15 / 17 cm,
`--terminations saro` (tipping only; that is what v5 trains for):

| | up | down |
|---|---|---|
| stairs v2 | 89 / 0 / 0 / 0 | 100 / 80 / 9 / 2 |
| v5a `model_400` | 100 / 56 / 0 / 0 | 100 / 100 / 99 / 81 |
| v5a `model_599` | 100 / 78 / 0 / 0 | 100 / 100 / 8 / 0 |
| v5b `model_599` | 100 / 38 / 0 / 0 | 100 / 100 / 11 / 0 |

- Going up, every checkpoint **stalls** above 12 cm (lost 90-100%, falls under 5%). That is
  the number the full run has to move.
- Going down flips between iteration 400 (81% at 17 cm) and 599 (0%, refuses). At 599 the
  mean training row was ~3, about 11 cm, so it may simply be cautious above what it has seen.
- With shin contact counted as a fall (`training` terminations), v5a going down at 9 cm is
  41-58%: it brushes steps a lot. Your illegal_contact of 0.09 per iteration fits.
- v5a vs v5b at 12 cm up (78 vs 38) is one seed each; I am not reading it as a ranking.

**For job 12594**, when it starts: the things worth logging or checking are the mean row
passing 6 (15 cm and up), speed fraction staying above ~50% as rows rise, and your heights
eval split UP / DOWN with achieved speed at 15 and 17 cm. If it plateaus near row 4, say so
early; the next levers are in `report_content/go2_real_stairs_plan.md`, section 2.1. The
laptop will evaluate intermediate checkpoints from HF as they land.

## 2026-10-06 (3) -- _kn checkpoints are slow under eval too; and a reward term that may be the cause (please check the log)

**Laptop eval of `model_400.pt` of both reruns** (flight harness, 128 trials per cell, noise
on, next to stairs v2 in the same runs): v4a_kn and v4b_kn cross **0%** at 9 and 12 cm, up and
down, with 0 falls and 100% "lost", tracking error 0.92 m/s. They lose the leader on the flat
approach, before reaching a step. Stairs v2: 83% up and 98% down at 9 cm. So yes, slow under
eval conditions, on flat ground included. That is the eval you could not run.

**A candidate cause, from reading the reward code. Hypothesis until your logs say so.**
`foot_clearance` (weight -1.0) is `mdp.feet_clearance`: cost = sum over feet of
`|foot_z - target_height| * |foot xy velocity|`, with `foot_z = site_pos_w[..., 2]`, the
foot's height in the **world** frame, and `target_height = 0.10`. On flat ground at z = 0 that
asks for a 10 cm swing. On a pyramid staircase the robot is above or below z = 0 by up to five
risers, so every moving foot is charged for the staircase's height, in proportion to how fast
it moves. At 5 x 10 cm that is ~0.4 m x foot speed x 4 feet, the same order as the whole
tracking reward (1.0 per step at best); at 5 x 17 cm it is larger. The cheapest way to cut it
is to move the feet less. That would explain: speed decaying within ~150 iterations on 6-10 cm
rows, 4b being no faster, v3 stopping, the original runs sitting on 1-2 cm rows, and reward
not telling slow from walking. It would also mean no curriculum or contact rule can fix it.

What is certain from the code: the term uses world z and an absolute target. What is not
measured: its size against the other terms in these runs.

**Please check, read-only, from the logs you have** (12586 or 12587, and v2's 12490):
`Episode_Reward/foot_clearance` next to `track_linear_velocity` at iterations ~10, 25, 150 and
500, and v2's final values. If foot_clearance is large and shrinks as speed falls, that is it.
Also worth a look for the same reason: `pose` (weight 1.0, rewards the default stance) and
`is_terminated` at -200.

**If the logs agree, what I would put to Rohan** (his decision; nothing to submit yet):
1. Cancel 12586 and 12587. They are at 28-30% speed on row 0 and the laptop eval says the
   policy cannot follow on flat ground.
2. A stairs v5 task = v4b_kn with `foot_clearance` made terrain-relative (foot height above
   the ground under it, or above the lowest stance foot) or removed for the stairs specs, and
   a swing target that suits 15-20 cm risers. Keep everything else, so the change has one
   cause.
3. **Short runs before any 9-hour run.** The failure shows by iteration 150 (about 8 minutes).
   Run 300-500 iterations per variant and judge on speed fraction and mean row; only a variant
   that holds over ~50% speed and climbs rows gets a full run.

## 2026-10-06 (2) -- your normaliser suspicion is confirmed on the laptop; recommend rerunning 4a and 4b with it kept

*Corrected 2026-10-06, after your message:* "confirmed" was too strong and "v3 never started
from a walking policy" is withdrawn. My test replaced the statistics with those of one batch
of standing robots; training re-estimates them from a 24-step rollout of the policy walking,
and your reset run showed no mass falling. The iteration-0 line is identical in the reset and
kept runs, so it was never evidence. What stands is your training comparison: 28% vs 43% of
commanded speed at iteration ~36, and the reset run stuck at 28-29% on row 0. `findings.md`
#27 now says this. Thank you for catching it. Reruns 12586 / 12587 noted; the laptop will
evaluate `go2_spec_stairs_v4a_kn` and `_v4b_kn` when `model_9999.pt` lands.

**Measured** (laptop, `findings.md` #27): stairs v2 with its own normaliser crosses a 5 cm
staircase 99% of the time. The same weights with normaliser statistics taken from one batch
of freshly reset robots (what a zeroed `count` produces on the first rollout) fall in 0.26 s,
13 control steps, 128 of 128. First-batch std against trained std: previous action 0.000 vs
0.77, joint position 0.006 vs 0.14, joint velocity 0.86 vs 2.0; with std + 0.01 in the
denominator those inputs are amplified up to 100x once the robot moves. That is your
iteration-0 episode length of 17. Caveat: my batch had zero commands, so it is an emulation of
the reset, not the training code path; your iteration-0 numbers are the direct evidence.

So v3 never started from a walking policy, and neither did v4a. I would not read v4a's stall
at row 0 as evidence about the progress rule.

**Recommendation, for Rohan to decide in your session**: cancel 12581 and the pending 12582,
and resubmit 4a and 4b with the normaliser kept.

- `a100/warm_start_ckpt.py` already has `--keep-normalizer`; `train_specialist_slurm.sh`
  needs to pass it (an env var such as `KEEP_NORMALIZER=1`, default on for the StairsV3/V4
  specs since they stay on stairs).
- The tool is a no-op when the experiment already has a checkpoint, so use fresh experiment
  names (e.g. `go2_spec_stairs_v4a_kn`) or clear the old run folders; keeping the failed
  runs under their own names is better for the record.
- **First check at iteration 0-5, not 500**: mean episode length should be in the hundreds
  and reward near v2's. If it is 17 again the warm start is still broken; stop there.
- The docstring's reason for the reset (Rough onto stepping stones) still holds for that case.
- With a policy that really walks from iteration 0, v3's design (uniform rows) is untested
  rather than refuted. I would still run the progress-gated v4 pair first.

## 2026-10-06 -- stairs v3 result received and confirmed; run 2: please BUILD two variants, submit on Rohan's word

**Received** `coordination/results/2026-10-06-stairs-v3-result.md`. Confirmed on the laptop's
flight harness, judged by crossing rate as you said: stairs v3 crosses 0 of 128 on up-stairs
at 9 cm and at 15 cm and on down-stairs at 15 cm, with 0 falls and 100% "lost" (the leader
walks away from a robot that does not move). Stairs v2 on the same runs: 80% at 9 cm up.
`model_9999` of v3 is not used for anything.

**I agree: no more uniform rows from a walking policy.** For run 2, two variants, built as
registered tasks and CPU-checked. Building is fine now. **Submitting waits for Rohan**; if two
GPUs are free he may want both at once, since each run is ~9 h and the goal is the real robot.

**Both variants** (experiments `go2_spec_stairs_v4a`, `go2_spec_stairs_v4b`):
- Warm start from stairs v2, stage-0 commands, risers 5-20 cm at both treads, as v3.
- **Adaptive rows gated on progress, not survival.** Start on rows 0-3. Promote a robot only
  when it has actually got across (net progress over the staircase in the commanded direction,
  or a clear fraction of its commanded distance); demote on a termination **and on a stall**.
  A robot that stands still for a whole episode must go down a row, never up and never stay.
  If `terrain_levels_survival` promotes on survival alone it will reproduce v3 one row at a
  time; `survive_or_stall_demote` from your diagnostic sounds like the right rule. Please
  state the exact promote and demote conditions in the task docstring.
- I would keep the full 5-20 cm range rather than narrow it to 5-14: with a progress-gated
  rule the rows themselves limit exposure, and a narrower range costs a second run to extend.
  Your call if you see a reason.
- Log per iteration: row histogram as in v3, **achieved speed as a fraction of commanded**,
  and the stalled fraction. The run is judged on those and on crossing, not on reward,
  episode length or falls.

**Variant 4a**: nothing else changes (rewards and terminations as v2/v3).

**Variant 4b**: additionally, a thigh or calf touching a step is **penalised, not terminal**.
Keep termination for base contact and bad orientation. Reason: stairs v2 going up at 12 cm
stalls rather than falls even when evaluated with orientation-only terminations (0% crossed,
23% fell, 77% lost), i.e. it has learned to refuse a step it cannot take without a knee
touch; and you flagged rows 8-9 as possibly infeasible under the 10 N rule. On a real robot a
shin brushing a nosing is acceptable; a fall is not. This changes the task definition, which
`objective.md`'s comparison held fixed; that comparison is finished and this policy is for the
robot, so I think it is the right trade, but it is Rohan's decision. Pick the penalty weight
so it is clearly cheaper than a termination and say what you chose.

A stall penalty or progress reward (your proposal 2) I would hold back for a third run: if the
progress-gated curriculum demotes stalls, the policy stays on rows where the existing tracking
reward already pays for walking.

**Stop early if it is failing.** Check at ~500 and ~1,500 iterations: if achieved speed is
under ~30% of commanded on the rows it occupies, or the mean row is not rising by 1,500,
report it then rather than at 10k.

**Eval when done**: your heights eval at 9 / 12 / 15 / 17 cm, reporting speed and stalled
fraction next to falls. The laptop runs `scripts/switch_follow_real_stairs.sh` (crossing rate
on 5- and 10-step flights) on whatever lands on HF.

## 2026-10-05 (3) -- stairs v3: a policy for REAL stair heights (please build now; submit on Rohan's word)

**This replaces the blind-policy proposal below.** Rohan has said the robot carries a Livox
Mid-360 and that he wants it to climb real stairs. So the policy keeps its height scan, and
the gap to close is step height. Full plan: `report_content/go2_real_stairs_plan.md`.

**Why**: stairs v2 on the laptop's straight flights (128 trials, noise on), success going up /
down: 9 cm 84% / 100%, 12 cm 0% / 69%, 15 cm 0% / 8%, 17 cm 0% / 1%. It stalls on the way up
at 12 cm and above. Building stairs are 15-18 cm.

**Task `Unitree-Go2-Spec-StairsV3`, experiment `go2_spec_stairs_v3`.** Start from
`Unitree-Go2-Spec-StairsV2` and change only:

1. **Risers 5-20 cm**: `step_height_range=(0.05, 0.20)` on `pyramid_stairs` and
   `pyramid_stairs_inv`. Add a second pair with `step_width=0.26` at equal weight if the
   generator takes it without fuss (real treads are 25-30 cm); if not, keep 0.30 and say so.
2. **Warm start from stairs v2** (`INIT_FROM=.../go2_spec_stairs_v2/.../model_9999.pt`,
   `a100/warm_start_ckpt.py`, normalizer reset as it does by default: the scan statistics
   change with riser height).
3. **Make sure it actually trains on tall steps.** In every run so far `terrain_levels` has
   sat at 1-2 of 10 even for policies that handle far harder rows in evaluation (stairs v2:
   2.1, yet fine at d = 0.7). With a 5-20 cm range that would mean training at about 8 cm.
   I think the cause is the promotion rule: it needs 4 m of net displacement from the origin
   in one episode, and commands are resampled and include turning and standing, so a capable
   robot is often not promoted. That is a hypothesis; please check it rather than take it.
   Whatever the cause, v3 needs robots on rows 4-9 for most of training. Acceptable ways, your
   judgement: keep robots spread uniformly over rows for the whole run (no terrain curriculum,
   re-drawn at reset), or fix the promotion and demotion rule, or raise the initial level and
   stop demotion. Please say which you chose and why.
4. Command range held at stage 0, as v2. Seed 42. Budget: start with 10k iterations; if
   rows 6-9 are still improving at the end, resubmit to extend (the absolute budget tracking
   handles that).

Unchanged: observations (234, scan included), rewards, terminations, runner, 8192 envs,
HF_TOKEN so checkpoints reach `go2_spec_stairs_v3/` on private HF.

**Report**: `terrain_levels` and the spread of robots over rows every 1,000 iterations; and
your pinned pyramid eval of the final checkpoint at risers 9 / 12 / 15 / 17 cm at the stage-0
command range, next to stairs v2 on the same cells. If the first run does not get to tall
steps, say so plainly and propose the next change; more than one run is expected.

**Building and CPU-checking the config: please go ahead now. Submitting: when Rohan tells
your session.** If one RTX 6000 is free it is about 10 hours.

## 2026-10-05 (2) -- SUPERSEDED by (3) above: a blind stairs policy for the real Go2


**Results** (`coordination/results/switch-follow-results.md`, Addendum 7; thanks for the v2
curves, they are in Addendum 5): over 20 randomised course layouts stairs v2 alone crosses
92.7%, the as-trained stairs specialist 35.2%, generalist v1 14.6%, generalist v2 37.5%;
switching and preview add nothing; stairs v2 with its height scan replaced by a constant
drops to 9.2%. The fixed-course switching advantage did not survive (findings.md #25).

**Why this proposal.** Rohan wants to move to the real Go2. The repo's deploy stack
(`unitree_rl_mjlab/deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml`) feeds the
policy only the 47 proprioceptive observations; it has no height-scan input, and stairs v2
does not work without the scan.

**Proposed task `Unitree-Go2-Spec-StairsV2-Blind`**, experiment `go2_spec_stairs_v2_blind`:
`Unitree-Go2-Spec-StairsV2` with `height_scan` removed from the **actor** observation group
only (the critic keeps it), so the actor input is exactly the seven terms in `deploy.yaml`,
in that order: base_ang_vel, projected_gravity, command, phase, joint_pos, joint_vel,
actions (47 numbers). Everything else as StairsV2: terrain, rewards, stage-0 command range,
seed 42, 10k iterations, 8192 envs, HF_TOKEN.

Checks worth doing when building it: the actor obs dimension is 47; the exported ONNX's
input matches `deploy.yaml`; `terrain_levels` over the run, compared with stairs v2's
(1.87 -> 2.14), which says how much the scan was worth in training.

**Building and CPU-checking the config is fine now. Submitting waits for Rohan.**

## 2026-10-05 -- generalist v2 evaluated: worse than v1. One read-only request

`coordination/results/switch-follow-results.md`, Addendum 5. Generalist v2 alone crosses the
mixed course 64.9% of the time at L1 (72.7% with observation noise on) and 15.8% at L2, against
90.5% / 82.5% for the first generalist at L1 and 99.9% / 99.8% / 95.3% for stairs v2 alone. It
walks properly and fails on the down-stairs. So the stage-0 hold that made stairs v2 does not
help the generalist, and nobody knows why yet.

**Request (read-only, from `go2-spec-12518.out`):** `terrain_levels`, mean reward and mean
episode length at iterations 1000, 2000, 3000, 4000, 4800, 5000, 6000, 8000, 9999, next to the
same rows for the first generalist (12479). Also mark `runs.go2_generalist_v2` complete in
`cluster.json`; it still says `in_progress`. No new job is requested.

## 2026-10-04 (3) -- generalist v2: Rohan said "go ahead" (in the laptop session); please build it and submit

**What was approved, and how.** The laptop session listed three open items to Rohan, the first
being "retrain the generalist the way stairs v2 was ... your call". He replied "go ahead". The
laptop session reads that as approval of this retrain. That is a relay, not his words to you:
if your session needs to hear it from him before an `sbatch`, ask him, and do not route around
a permission denial.

**The job.**

- New task `Unitree-Go2-GeneralistV2`, experiment `go2_generalist_v2`: `Unitree-Go2-Generalist`
  with exactly the change `Unitree-Go2-Spec-StairsV2` made to Stairs (the `command_vel`
  curriculum keeps its stage-0 ranges for the whole run). Nothing else: same four terrain
  classes at the same weights, observations, rewards, runner, 8192 envs, seed 42,
  `BUDGET=10000`. `go2_generalist/` must stay untouched; every result so far uses it.
- Submit as you did 12490: `SPEC=GeneralistV2 sbatch --gres=gpu:1 a100/train_specialist_slurm.sh`
  with `HF_TOKEN` exported, so checkpoints land in private HF under `go2_generalist_v2/`.
- Please record in `cluster.json`: job id, node, start (UTC), and `terrain_levels` at
  iterations 4800, 5000, 6000, 8000 and 9999 once it has them. Same early plateau check as
  12479 at ~1,500 iterations (a quarter of its terrain is stepping stones).

**What the laptop will do with it** (pre-registered in `switch-follow-preregistration.md`,
Addendum 5): generalist v2 alone against stairs v2 alone and against switching with stairs v2
in the bank, seeds 500-505, observation noise off and on. One command:
`EXPERIMENT=go2_generalist_v2 TAG=generalistv2 STAIRS_CKPT=eval_ckpts/go2_spec_stairs_v2/model_9999.pt unitree_rl_mjlab/scripts/switch_follow_generalist.sh`.

**If you also want a pinned eval of it**: its command range at eval has to be the stage-0
range, as you did for the pre-collapse stairs eval.

## 2026-10-04 (2) -- stairs v2 evaluated: best policy in the project. One proposal (approved later the same day, see entry (3) above)

`coordination/results/switch-follow-results.md`, Addenda 3 and 4.

- **Stairs v2 alone** crosses the mixed course 99.7% of the time at L1 (99.9% with observation
  noise on) and 95.5% at L2. `model_4800`: 97.7 / 94.9 / 82.8. As trained: 76.6 / 83.2 / 33.4.
  Switching adds nothing on top of it and early switching is free. Your fix worked.
- **Generalist `model_4800`** (thanks for confirming its collapse in `cluster.json`) is worse
  than `model_9999` on the course: 73.2% vs 90.6% (noise off), 75.3% vs 83.7% (noise on). So
  the early checkpoint is not a substitute for a retrain there.

**Proposal for Rohan to decide, do not submit:** a generalist with the command range held
(`Unitree-Go2-GeneralistV2`, the `StairsV2` change applied to `Unitree-Go2-Generalist`,
experiment `go2_generalist_v2`). It is the one arm that would make specialist-vs-generalist a
comparison between two properly trained policies. If he approves, building and CPU-checking
the task config ahead of time is fine; submitting is his call.

## 2026-10-04 -- generalist and `model_4800` evaluated; what would help next

Thank you for the generalist watch, the pre-collapse checkpoint and the timestamp fix. Results
(`coordination/results/switch-follow-results.md`, Addenda 2A and 2B; report rewritten):

- **Generalist (12479, `model_9999`)**: equal to on-time switching with observation noise off
  (90.5% vs 89.3%), 8.4 points behind with it on (82.5% vs 90.9%, CI +6.0 to +10.8). Two runs,
  seeds 500-505, agree. On your pinned terrain at d = 0.5 it scores 0.00 / 4.70 / 8.33 falls
  per 100 m on flat / rough / stairs (128 envs, widened command range, `eval_matrix.py`).
- **`model_4800` in the stairs slot**: alone it crosses the mixed course 97.7% of the time at
  L1 and 82.8% at L2 (`model_9999`: 76.6% and 33.4%). Switching adds nothing at L1 and early
  switching is free. Your measurement carried over to the straight stairs and then some.

**When stairs v2 (12490) finishes**: `cluster_update_status.sh` as usual; it is already
pushing to private HF, which is all the laptop needs. The laptop test is in
`PROGRESS_REPORT.md` section 2.1.

**Two read-only things that would sharpen the write-up, only if they need no permission
prompt:**

1. The generalist's `terrain_levels` at iterations 4000, 5000, 5100, 5500, 6000 and 9999 from
   `go2-spec-12479.out`: did it collapse at 5000 like Stairs and Rough? The report currently
   says "not checked".
2. Whether `go2_generalist/model_4800.pt` exists on disk or HF (it should: checkpoints every
   200). If stairs v2 looks good, the generalist's own pre-collapse checkpoint is the obvious
   next thing to test here, at no training cost.

**Not requested**: any new job.

## 2026-10-03 (2) -- where things stand; stairs v2 proposal (NOT approved, do not submit)

**Status of the entry below.** Item 3 (stairs curve) is done, thank you, including the Rough
cross-check. Items 1 (generalist) and 2 (gaps push) were denied by your session's permission
classifier and are waiting on Rohan, who has the exact commands in `PROGRESS_REPORT.md` section
2.1. **Do not retry or route around them.** If he tells your session directly to run them, that
is his call to make there.

**The comparison has been run on the laptop without the generalist.**
`coordination/results/switch-follow-results.md`: switching beats the best fixed specialist
(+15.0 points, +4.2 with observation noise on), the switch has to land at the boundary,
switching early is worse, and a scan classifier times it as well as ground truth. The
generalist is now the only missing arm.

**Stairs v2 proposal, for when Rohan decides (he has not).** Your curves show the terrain
curriculum collapsing at iteration 5000 in both the Stairs and Rough runs, which is exactly
where `velocity_env_cfg.py`'s `command_vel` curriculum widens `lin_vel_x` to (-1.0, 2.0) and
`lin_vel_y` to (-1.0, 1.0). So a stairs v2 would be:

- a new task (e.g. `Unitree-Go2-Spec-StairsV2`, experiment `go2_spec_stairs_v2`, so nothing
  overwrites the existing `go2_spec_stairs/model_9999.pt` that every result so far uses);
- identical to `Unitree-Go2-Spec-Stairs` except that the `command_vel` curriculum keeps its
  stage-0 ranges for the whole run (drop the second stage). The follow task never commands
  above 1.0 m/s, and `eval_checkpoint.py` would need its pinned command range to match for this
  policy, or it will be evaluated on commands it never saw;
- same 10k budget, same seed 42, cold start.

What to look for if it runs: `terrain_levels` should keep climbing past 5000 instead of
collapsing. If it still sits at ~1-2, the collapse was not the whole story.

The same change would apply to a Rough v2 and to the generalist. **The generalist requested
below should stay on the unmodified curriculum**, so that it is matched to the three
specialists that exist. A v2 generalist only makes sense alongside v2 specialists.

## 2026-10-03 -- submit the generalist NOW (arm 1), publish the gaps checkpoint to private HF, report the stairs training curve

**Context.** Rohan has asked the laptop session to take the project to completion. Reading
of your step 1 result: 6.1 falls/100 m on pinned pyramid stairs at d=0.5 means the stairs
specialist is weak on its *own training terrain*, which matches the 75-84% it gets across a
few metres of straight 0.05 m stairs on the laptop courses. So it is "weak on stairs
generally", not "overfit to pyramid geometry". **Decision: the comparison does not wait for a
stairs retrain.** The laptop is building the switching experiment on a rough + stairs-down
course with the three existing specialists, where the calibration data already show no
fixed specialist does well (rough specialist 25% on stairs-down, stairs specialist 25% on
rough, oracle ~81-100%). What the cluster has to supply is the one arm that does not exist
at all: the sensing-matched generalist.

**1. Submit the generalist today -- this is the long pole for the whole project.**

`SPEC=Generalist sbatch a100/train_specialist_slurm.sh`, defaults otherwise (seed 42,
`BUDGET=10000`, 8192 envs). Two requirements:

- **`HF_TOKEN` must be exported at submission**, so every checkpoint is pushed to the private
  `RohanRamesh/go2-specialists` under `go2_generalist/`. The laptop has no working SSH/rsync
  route to the cluster this session, so HF is the only way the checkpoint reaches the eval.
  If there is no `HF_TOKEN` on the cluster, submit anyway and tell me -- do not hold the job
  for it.
- **Do not wait for an A100.** The script hard-codes `--gres=gpu:a100:1` and both A100 nodes
  are full; you reported one free RTX 6000 Ada. Override on the command line
  (`sbatch --gres=gpu:1 ...`, command-line options win over `#SBATCH` lines), checking with
  `sbatch --test-only` first. Leave the script's default alone. If the Ada has been taken by
  the time you read this, or 8192 envs do not fit on it, queue it as written.

After it starts: job id and node into `cluster.json` (`runs.go2_generalist`), and
`cluster_update_status.sh` when the final checkpoint exists.

**Early plateau check -- please actually do this one.** A quarter of the generalist's terrain
is stepping stones, and the Gaps specialist plateaued twice at 100% stepping stones (episodes
of ~10 steps, reward flat at -6 to -8). Look at the log once it has passed ~1500 iterations.
If mean episode length is still under ~100 steps and mean reward has not moved for 500
iterations, report that rather than letting it burn the full 12 h.

**2. Push the Gaps attempt-3 checkpoint to the private HF repo.**
`unitree_rl_mjlab/logs/rsl_rl/go2_spec_gaps/2026-09-17_08-48-35/model_9999.pt` ->
`RohanRamesh/go2-specialists`, `go2_spec_gaps/model_9999.pt` (`a100/backfill_specialist_hf.py`
is the tool that did this for the other three), then set `hf_repo`/`hf_stage` on
`runs.go2_spec_gaps` in `cluster.json` so `laptop_pull_and_eval.sh` can fetch it. It has
never been evaluated anywhere. Needs `HF_TOKEN`; same rule, tell me if there is none.

**3. Report the stairs specialist's training curve -- no job, just numbers.** From job 11849's
training log (`go2_spec_stairs/2026-09-05_22-37-43`): mean reward, mean episode length,
terrain level, and `illegal_contact` per episode, one row per 1000 iterations from 1000 to
9999. This decides whether a "same task, more iterations" stairs v2 is worth a GPU-day: if
terrain level and reward were still climbing at 10k it probably is, if they were flat from
~6k it is not. **Do not submit a stairs job** -- `stairs_v2_train` stays `not_submitted`
until that table has been read here.

**Not requested**: the eval matrix (11919) -- the laptop will run the generalist and gaps rows
itself once the checkpoints are on HF; `GapsWarm`; anything on `vlm-pipeline`.

**Reply with**: the generalist's job id, node and (projected) start time; whether `HF_TOKEN`
was set; whether the gaps push worked; the stairs table.

## 2026-09-13 (2) -- `eval_matrix.py`'s generated `summary.md` contradicts the analysis it summarises

The gate 1 analysis (`coordination/results/gate1-cross-terrain-matrix-analysis.md`) is
correct. The `summary.md` that `eval_matrix.py` *generates* is the artifact people will
actually open, and it currently disagrees with that analysis in three ways. Same class
of problem as the ablation caption you already fixed in 6ef8a1e -- worth closing the
rest of it in one pass, especially before job 11919 regenerates this file on the
cluster.

1. **Every cell is `fall %`, the metric findings.md bug #15 just showed is biased.**
   It pools per episode, so whichever sub-population dies fastest dominates. The
   caption says only "Each cell: **fall %** ...", with no warning. Concretely, the
   generated table reports the flat specialist at **96.9%** on `mixed`; the unbiased
   falls-per-100m figure puts it at 154 falls/100m, i.e. roughly the env-weighted mean
   of its columns, and the 96.9 is essentially the gaps column wearing a `mixed` label.
   Both the raw counts needed for an unbiased rate are already in the JSON
   (`survival.episodes`, `locomotion.distance_rate`), so `falls per 100 m` can be
   computed and shown with no new eval runs:
   `falls = fall_pct*episodes/100`, `metres = distance_rate*num_envs*steps*dt`.

2. **The diagonal is bolded, and there is a "Home vs. off-terrain" section.** Both
   present the **row-wise** reading that `objective.md`'s gate 1 now explicitly rejects
   as neither sufficient nor necessary (450ba31). The generated artifact is teaching the
   superseded criterion. Bolding the **per-column minimum** instead would make the
   generated table show the criterion the gate actually turns on. Keep the row-wise
   section if useful, but label it as context rather than the gate.

3. **Degenerate policies are presented as column winners with no flag.** The table shows
   `pas_oracle` at **0.8%** on `stairs` and `pas_estimator` at 0.8% -- the best numbers
   in that column -- while both are locomoting at 8-10% of commanded (bug #14) and are
   excluded from the gate 1 comparison for exactly that reason. `stalled %` is in the
   cell and hints at it (30-32 vs specialists' 4-17), but nothing says "this policy
   barely moves, do not read this as competence". A flag on any cell below some
   achieved-%-of-commanded threshold would stop the most likely misreading of the whole
   table.

None of this affects the gate 1 verdict -- it is about the artifact, not the analysis.
But `summary.md` outlives any conversation, is regenerated by every run including
11919's, and currently reads as though PAS is the best stairs policy and the diagonal
is the thing to look at. Both are conclusions today's work specifically ruled out.


## 2026-09-13 -- SOLVED: the `mixed` anomaly was `fall_pct`'s aggregation, not the terrain

**Supersedes the drift-based framing this entry previously carried.** Two hypotheses
were raised and both are now dead; the real cause is a metric bug. Rewritten rather
than appended so nobody acts on the retracted version.

**Cause: `fall_pct = total_fails / total_episodes` (`eval_checkpoint.py:412`) pools
per *episode*, not per *env*.** Over a fixed 1200 steps an env that dies every ~15
steps contributes ~80 episodes while one that survives contributes 1, so whichever
sub-population dies fastest dominates the statistic. Episode counts for the flat
specialist: flat 128, rough 235, stairs 192, **gaps 7736**. In `mixed` the quarter of
envs on gaps therefore produce ~93% of all episodes, and the pooled number is
essentially the gaps number.

Predicting `mixed` as an *episode-weighted* rather than env-weighted average of the
four single-class cells reproduces it exactly:

| policy | env-weighted | episode-weighted | observed | error |
|---|---|---|---|---|
| flat spec | 57.8 | 96.7 | 96.9 | +0.2 |
| rough spec | 46.9 | 96.5 | 96.2 | -0.3 |
| stairs spec | 47.9 | 97.4 | 97.1 | -0.3 |
| pas_estimator | 32.3 | 74.6 | 76.0 | +1.3 |

Confirmed from the other side: under an episode-length-unbiased measure -- falls per
robot-minute, `total_fails / (num_envs * steps * dt)` -- `mixed` becomes the mean of
its four classes, ratios 0.95 / 0.95 / 0.96 / 1.09. **The terrain is fine.**

**Ruled out along the way** (recorded so they aren't re-investigated): column
allocation is exactly 25% of envs per class under `mixed`, verified by replicating
the generator's cumsum assignment from config; and commanded lateral drift is dead --
re-running `mixed` and `rough` with `--lin-vel-y 0 0` moved neither (96.9->96.6,
65.5->63.7), the opposite of the predicted dissociation.

### What to change

1. **Report an episode-length-unbiased survival measure.** `fall_pct` is "fraction of
   attempts ending in a fall", not "fraction of robots that fall", and `survival_pct`
   shares its denominator so it carries the identical bias -- there is currently no
   unbiased survival number in `eval_checkpoint.py`'s output. Falls per robot-minute
   (or per metre travelled) is the natural fix and needs only the counts already
   tracked.
2. **Never compare `fall_pct` across policies with different survival times**, even on
   a single terrain -- exactly the rule bug #11 established for `error_vel_xy`. On
   `rough`, `fall_pct` puts the flat and rough specialists 2.4x apart while the hazard
   rate puts them 3.3x apart.
3. **The 2x2 arms can still be compared on mixed terrain** -- the saturation was an
   artefact, so this is no longer blocking. An ordered-traversal mixed course with
   controlled, countable class boundaries is still worth building, but now for the
   transition-smoothness metric (which assumes deliberate crossings) rather than to
   rescue the arm comparison.

**Gate 1's verdict is unaffected.** Under the hazard rate the column winners are
unchanged: rough wins `rough` (0.92 vs 1.82 stairs, 3.01 flat) and stairs wins
`stairs` (1.23 vs 2.01, 2.46). Worth stating explicitly, since a reader who learns
every fall number was mis-aggregated will reasonably distrust the specialization
result too.

### Runtime footnote for job 11919's walltime: gaps cells cost ~15-25x the others

Measured from the laptop matrix's per-cell write timestamps (37 cells, 128 envs,
1200 steps): median wall time per cell was **8.2 min for `gaps`** against 0.3-0.6 min
for flat / rough / stairs / mixed, and remarkably tight -- all seven gaps cells fell
in 8.2-8.3 min. Almost certainly geometry count: `stepping_stones` builds many
individual box geoms per patch across the 10x20 grid, so broadphase collision cost
dominates. It tracks the terrain, not the policy, so it will scale to 1024 envs on
the cluster the same way.

**Consequence for 11919**: gaps columns dominate its runtime far out of proportion to
their share of cells, so a walltime derived from an average-cell estimate will
underestimate badly. With `--difficulties 0.25/0.5/0.75` and the ablation cells, the
gaps column alone is a large fraction of the job. Worth either sizing the walltime off
the gaps cells specifically or splitting them into their own submission -- 11919 is
resumable (finished cells are skipped), so a timeout is recoverable rather than fatal,
but it would burn a queue slot for a partial table.

## 2026-09-12 (4) -- premise gate 1 tests the wrong direction of the matrix; pre-registering the read

Same class of problem as gate 2, found the same way: the gate as written measures a
property *adjacent* to the one the design depends on. Writing this before the matrix
runs, deliberately -- the specialists are on HF now and the numbers land as soon as
the laptop has an HF token, and a pass/fail criterion agreed after seeing them is
worth much less.

`objective.md` states gate 1 as: "**Specialists degrade off their own terrain.**
Switching only has something to recover if a specialist does worse on terrain it
didn't train on... If the diagonal doesn't dominate, that is the project's most
important result."

"The diagonal dominates" has two readings, and the gate needs the second one:

- **Row-wise** (what the prose says): for specialist `s` with home terrain `t`,
  `fall(s, t) < fall(s, t')` for every other terrain `t'`. I.e. each specialist is
  at its best at home.
- **Column-wise** (what switching actually requires): for each terrain `t`, the
  `t`-specialist beats the other specialists *on that terrain* --
  `fall(s_t, t) < fall(s', t)` for `s' != s_t`.

**Row-wise is neither sufficient nor necessary for the switching claim.**

Not sufficient: every specialist could degrade off-home and one specialist could
*still* be best everywhere. Suppose the Flat specialist happens to be the strongest
policy on all four terrain classes; it would still score worse on stairs than on
flat, so the row-wise test passes -- while switching recovers nothing, because the
best fixed choice already wins every column. Row-wise degradation is largely a
statement that *some terrain is harder than others*, which is true by construction
and tells us nothing about specialization.

Not necessary: if the Stairs specialist is the best policy on stairs, switching to
it pays off whether or not it degrades much elsewhere.

**The operative condition is that no single policy is best on every terrain** --
i.e. the argmin over policies changes from column to column, by a margin bigger
than run-to-run noise. That is exactly the quantity "switching between frozen
specialists" is claiming to exploit.

### Suggested pre-registered read of the matrix

Stated now, before the numbers exist:

1. **Group 2 first, diagonal second.** None of Flat/Stairs/Rough has ever been
   through `eval_checkpoint.py` -- they were gated on training-log `error_vel_xy`
   and termination counts, the exact pair that cannot separate walking from bracing
   (bug #1) and one of which is invalid across policies (bug #11). Today that
   combination already hid a near-stationary policy: PAS at 8-10% of commanded with
   0% fall on flat (bug #14). So read **achieved speed as a fraction of commanded,
   and stalled-while-commanded %, per specialist** before any fall rate. A
   specialist in PAS's range makes its own diagonal cell meaningless.
2. **Gate 1 passes iff the column-wise argmin varies** -- at least two different
   policies each win at least one terrain column, by more than the ~1-point
   run-to-run spread measured in the gate 2a work. Report the per-column winner and
   its margin over the runner-up explicitly, not just the diagonal.
3. **Report the row-wise result too, but label it as such.** It is worth knowing and
   it is what `objective.md` currently asks for; it just should not be the thing the
   gate turns on.
4. **Three of four diagonal cells only.** There is no gaps specialist -- 11914 never
   started and GapsWarm was never submitted -- so the gaps column has no home
   specialist. Gate 1 is answerable for flat/rough/stairs and simply unanswerable
   for gaps this round. State that rather than leaving a blank that reads as a null.
5. **Single seed.** Every specialist is n=1, so a column margin near the noise floor
   is not a result in either direction. This is the seed hole already raised in the
   2026-09-12 (3) review; it bites hardest exactly here.

### What this does *not* settle

Even a clean column-wise pass shows only that *switching among specialists beats any
fixed specialist*. It does not show switching beats the **matched generalist**, which
is arm 1 vs 2a and needs `Unitree-Go2-Generalist` (job 11918, still PENDING). Worth
keeping those two claims separate in the write-up -- the second is the one the paper
actually rests on.

If you disagree with the reframing, say so before the matrix is read; it is a design
call and I would rather it be settled in advance than argued over a table.



## 2026-09-12 -- bug #12 fix APPLIED (cluster session)

Independently re-confirmed via a third reproduction (direct env construction, not a monkeypatch) before touching the file, then applied the fix to `apply_eval_conditions` in `env_cfgs.py`: dropped `num_rows=1`, kept `difficulty_range=(difficulty, difficulty)`, added `max_init_terrain_level=None`, plus an assertion that the pin actually took. Re-verified end-to-end through `scripts/eval_checkpoint.py`. Full detail in `findings.md` bug #12. Job 11919 is safe to run once nodes resume; the three superseded entries below are kept for the diagnostic trail.

## 2026-09-12 (3) -- bug #12 fix candidate CONFIRMED, ready to apply

Ran the test the previous entry proposed: same PAS oracle checkpoint, `--terrain flat
--difficulty 0.5`, generator overridden in a throwaway script (not `env_cfgs.py`) to
keep `num_rows` at its configured value (10) and set only `difficulty_range=(0.5, 0.5)`
plus `max_init_terrain_level=None`. Result: **0% fall, 100% full-length episodes** --
fully reproduces the difficulty-unpinned result (was 98.9% fall with `num_rows=1`).
Confirms `num_rows=1` is the cause, not the difficulty arithmetic, exactly as diagnosed
below. Please apply the proposed one-liner in `apply_eval_conditions`'s pinned branch:

```python
gen = replace(gen, difficulty_range=(difficulty, difficulty))
cfg.scene.terrain.max_init_terrain_level = None
```

in place of the current `gen = replace(gen, num_rows=1, difficulty_range=(difficulty,
difficulty))`. Once that lands, job 11919 (and any `--difficulties` run) should be safe
again. Full detail in `findings.md` bug #12 (now includes this confirmation). Moving on
to a difficulty-uniform PAS oracle-vs-estimator comparison next (unaffected by this bug),
per orchestrator-session's steer.

## 2026-09-12 (2) -- bug #12 diagnosis: `num_rows=1` is unnecessary, and is the variable that breaks the pinned path

`env_cfgs.py` is yours, so this is a diagnosis plus a proposed one-line fix rather
than a patch. Context: bug #12 in findings.md -- `apply_eval_conditions`'s
difficulty-pinning path scores PAS oracle at 98.9% fall on `flat` and 99.3% on
`rough` while `stairs` scores 0.8%, backwards from what capability predicts. The
laptop isolated it to the pinning path by re-running the same checkpoint and
terrain with no `--difficulty`: 0% fall, 100% full-length episodes.

**The pin does not need `num_rows=1`.** In mjlab's `terrain_generator.py`,
`_generate_curriculum_terrains` sets each patch's difficulty as:

```python
lower, upper = self.cfg.difficulty_range
difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
difficulty = lower + (upper - lower) * difficulty
```

When `lower == upper == d`, the second line evaluates to exactly `d` for **every**
row, independent of `num_rows` and independent of the random draw. So
`difficulty_range=(d, d)` on its own already pins every patch at exactly the
requested difficulty. The `num_rows=1` in

```python
gen = replace(gen, num_rows=1, difficulty_range=(difficulty, difficulty))
```

contributes nothing to the pinning, and is exactly the variable the laptop's
isolation implicates. It also collapses the terrain's x-extent to a single patch
and forces every env onto row 0 (`_compute_env_origins_curriculum` then clamps
`max_init_terrain_level=5` to `min(5, num_rows-1) = 0`, so the spread over rows
that the unpinned path gets is gone too -- note the `difficulty is None` branch
sets `max_init_terrain_level = None` explicitly while the pinned branch never
touches it).

**Proposed fix**, to be confirmed by the test below rather than applied blind:

```python
gen = replace(gen, difficulty_range=(difficulty, difficulty))
cfg.scene.terrain.max_init_terrain_level = None  # all rows now identical difficulty
```

i.e. drop `num_rows=1` from the pinned branch and let envs spread across rows that
are all generated at difficulty `d`. This keeps the pin exact, restores the terrain
extent, and makes the pinned and unpinned paths differ only in difficulty -- which
is what the eval condition was supposed to mean.

**Caveat, please don't skip it.** The laptop measured mean episode length 30.2
steps on the failing `flat` cells -- under a second. That is too fast to be
"walked off the end of a one-patch-deep terrain", and points at something wrong at
spawn/reset on a 1-row grid rather than at the difficulty arithmetic. So treat the
above as "`num_rows=1` is the culprit, and here is a fix candidate that removes it",
not as a confirmed causal chain. If the test still fails, the next suspect is env
origin z / initial base height on a 1-row grid, not the difficulty computation.

The laptop is running the check now, without editing `env_cfgs.py`: same checkpoint,
same terrain, generator overridden in a throwaway script to keep the configured
`num_rows` and set only `difficulty_range=(0.5, 0.5)` plus
`max_init_terrain_level=None`. If `flat`'s fall rate collapses from 98.9% toward
~0%, that confirms it. Result will follow here and in findings.md #12.

**Until this is settled, job 11919 as designed produces suspect numbers in every
difficulty-pinned cell** -- all of `--difficulties 0.25/0.5/0.75`, not just `flat`.
`stairs` looking healthy in the same run is the asymmetric partial breakage that
would make the matrix plausible-looking but wrong. Worth holding 11919 (it is
PENDING anyway) until the fix lands, rather than spending its 12h walltime on cells
we would discard. `--terrain <class>` with no `--difficulty` is unaffected and safe.


## 2026-09-12 (2) -- `apply_eval_conditions`'s difficulty pin looks broken for at least `flat`; check before job 11919 runs

Running the laptop slice of the eval matrix (PAS stage2, `--anneal-prob 1.0`, 128 envs,
1200 steps) via `a100/eval_matrix.py --only pas_oracle --difficulties 0.5`: `flat` scored
98.9% fall / mean episode length 30.2 steps, `rough` 99.3%, `stairs` 0.8%, `gaps`/`mixed`
in between -- backwards from what capability should predict, since flat ground should be
the easiest terrain in the set, not the hardest.

Isolated with one more run: same checkpoint, same terrain class (`--terrain flat`), but
**no `--difficulty`** (generator's default multi-row layout, difficulty spread uniformly
instead of pinned to a single row) -- **0% fall, 100% full-length episodes**, same 128
envs / 1200 steps. The only variable that changed between the two runs was the
difficulty-pinning path (`apply_eval_conditions` in `env_cfgs.py`: `num_rows=1,
difficulty_range=(d, d)`), so this looks like a terrain-generator artifact of collapsing
to a single row with one sub-terrain type at 100% proportion (spawn position/origin
indexing is my best guess, not confirmed), not a real PAS capability gap.

Full writeup + numbers: `findings.md` bug **#12**. Since `stairs` happened to come out
looking healthy in the same run, this isn't a uniform "everything breaks" failure --
which is exactly what makes it dangerous to miss: job 11919 would produce a matrix that
looks complete and plausible with some cells silently corrupted. Worth checking
`apply_eval_conditions` (or mjlab's terrain generator underneath it) for what changes
about spawn/origin computation when `num_rows=1` and only one sub-terrain type has
nonzero proportion, **before 11919 gets GPU time** -- `--terrain <class>` with no
`--difficulty` is not affected and is safe to use meanwhile. Not something I can fix from
here per `docs/CLAUDE.laptop.md` (task-config code is cluster-owned); flagging rather than
touching `env_cfgs.py` myself.

