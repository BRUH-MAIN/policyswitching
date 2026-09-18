# VLM navigation, Phase 1 confirmation — rough course only (2026-09-18)

**Scope**: the pre-registered confirmation (Step 2 of `vlm-nav-phase1-preregistration.md`) run on the
one course the specialists handle reliably. The stairs courses are held pending a user decision
(`vlm-nav-phase1-calibration.md`). · **Conditions**: `rough` L2 (bumps 0.02–0.08 m), fresh seeds
200/201, goal offsets −1/0/+1 m, 32 trials per cell, training fall definition, 0.5 m/s goal-seeking
controller, no VLM. · **Raw**: `unitree_rl_mjlab/eval_results/vlm_nav/phase1_confirm/*.json` ·
**Pooling**: `scripts/vlm_nav_summarize.py` (one trial per env; nothing pooled per episode)

## Result: 192 trials per arm

| arm | success (95% Wilson) | fall | timeout | policy match |
|---|---|---|---|---|
| oracle (ground-truth specialist per footprint) | 97.4% (94.0–98.9) | 2.6% | 0% | 100% |
| **rough specialist throughout** | **98.4% (95.5–99.5)** | 1.6% | 0% | 61% |
| flat specialist throughout | 72.9% (66.2–78.7) | 17.2% | 9.9% | 26% |
| stairs specialist throughout | 52.6% (45.6–59.5) | 10.9% | 36.5% | 0% |

- **Gate A (course is crossable): PASS** — oracle 97.4%.
- **Gate B (switching beats the best fixed specialist by ≥ 15 points, disjoint CIs): FAIL** — the gap
  is −1.0 points. Always running the rough specialist matches perfect switching.

## Reading it

**Choosing wrongly is expensive; choosing dynamically is not worth anything here.** The spread across
fixed specialists is large (98% → 73% → 53%), so the choice matters. But a course with a single
obstacle type needs only one good choice, made once. Switching can only pay where the best specialist
*changes along the course*, which is exactly why Gate B was pre-registered on the `multi` course.

`multi` contains stairs, and the stairs specialist is unreliable at every level tested
(`vlm-nav-phase1-calibration.md`), so the gate that would decide the project's claim cannot currently
be run. Options are the ones already listed in the calibration write-up, plus:

- **Build a multi-obstacle course out of terrain the specialists can handle.** With three
  specialists, two of which (flat, rough) are only distinguished on rough ground, a course of
  flat + rough segments still has a single best specialist (rough), so this does not by itself
  create a switching case. It would need a terrain class where `rough` is clearly *worse* than
  `flat` — e.g. speed on long flat stretches, which makes the metric time-to-goal rather than
  success. Not yet measured.
- **Note for the VLM comparison**: on this course, a VLM that always answers "rough" scores as well
  as a perfect switcher. Any VLM-arm result here measures whether the VLM avoids a *bad* choice, not
  whether it switches well.
