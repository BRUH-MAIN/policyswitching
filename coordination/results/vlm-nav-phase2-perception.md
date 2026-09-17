# VLM navigation, Phase 2: offline perception gate (Gemma-4-E4B)

**Date**: 2026-09-17 · **Machine**: `romen` · **Branch**: `vlm-pipeline` · **Model**: Gemma-4-E4B-it
Q4_K_M + F16 vision projector, llama.cpp b10686 (Vulkan), temperature 0, thinking off ·
**Frames**: 720 labeled ego-camera frames at L1 (0.05 m risers). Covers `flat`, `stairs_up`,
`stairs_down`, `rough`, `multi`. Poses are x in 0.5 m steps × lateral offset {−1, 0, +1} m × yaw
{−0.25, 0, +0.25} rad; terrain seed 300; tiled visuals. Each frame has depth, pose and labels:
footprint ground-truth policy, and a depth-checked visible box of the next intermediation ·
**Scripts**: `scripts/vlm_nav_perception_eval.py`, `scripts/vlm_nav_edge_eval.py`,
`logs/vlm_nav/box_probe/probe.py`

## Headline

- **Planning works.** From the start pose, Gemma names the right intermediation and assigns the
  right crossing policy in every sampled frame (8/8, pilot). This is the part of SARO the policy
  modification leans on hardest.
- **Localization with SARO's box prompt does not work with this model.** Most answers are
  `[0,0,0,0]` or the whole frame, and a constant full-frame box scores *higher* IoU than the model.
  A detection-style prompt at 560 image tokens helps but stays unreliable.
- **Depth geometry finds the intermediation's edges accurately without the VLM**, so the executor
  uses the VLM's box when it is usable and depth geometry otherwise, logging which. The VLM still
  decides *what* the terrain is and *which* specialist runs.
- **The policy selector question is weak at default resolution**: 61% accuracy, almost always
  "flat", never "stairs". The executor therefore uses the plan's policy and lets the selector
  override only when it agrees with itself twice in a row.

## 1. Planning (SARO prompt + policy field, JSON schema)

Pilot, default image budget, 8 start-pose frames: valid JSON 8/8, intermediation named correctly
8/8, crossing sub-task assigned the matching specialist 8/8.

## 2. Localization

### SARO's prompt ("Where is the <I>? Answer in [x0,y0,x1,y1] format")

77 frames with a visible intermediation, default image budget:

| | value |
|---|---|
| answers `[0,0,0,0]` | 34 |
| answers the whole frame (`[0,0,1000,1000]` / `[0,0,100,100]`) | 40 |
| mean IoU of the model (best convention, 0–1000 grid) | 0.21 |
| mean IoU of a **constant full-frame box** | **0.43** |
| mean IoU of the 37 non-degenerate answers | 0.02 |

The 0.21 IoU is almost entirely the full-frame answers overlapping wide ground-truth boxes, so it
measures "the box is large", not localization. Reading it as a detector score would be this
repo's familiar adjacent-metric trap.

### Prompt format and image budget (30-frame probe)

| prompt | image tokens | mean IoU | IoU ≥ 0.5 | degenerate / not found |
|---|---|---|---|---|
| SARO `[x0,y0,x1,y1]` | default | 0.21 | 23% | 26/30 |
| SARO `[x0,y0,x1,y1]` | 560 | — | — | 27/30 |
| detection `box_2d` JSON | default | 0.28 | 30% | 17/30 |
| detection `box_2d` JSON | 560 | 0.40 | 43% | 12/30 |
| (constant full-frame box) | — | 0.46 | — | — |

560 image tokens needs `-ub ≥ 560` in llama.cpp. Gemma-4 vision tokens attend bidirectionally,
and at the default micro-batch of 512 the server aborts on the first image (`scripts/vlm_server.sh`).

### What the executor actually needs: edge positions in metres

The specialist is engaged at the near edge and released past the far edge, so edge error decides
outcomes, not IoU. Error along the heading, edges within 4 m; the near edge is scored only beyond
the camera's 0.86 m ground blind zone:

| course | source | near within 0.2 m | near MAE | far MAE | far within 0.3 m |
|---|---|---|---|---|---|
| stairs_up | depth only | 100% | 0.00 m | 0.65 m | 21% |
| stairs_up | ground-truth box + depth | 100% | 0.04 m | 0.24 m | 57% |
| stairs_down | depth only | 100% | 0.05 m | 0.11 m | 100% |
| stairs_down | ground-truth box + depth | 88% | 0.15 m | 0.17 m | 73% |
| rough | depth only | 100% | 0.08 m | 0.29 m | 61% |
| rough | ground-truth box + depth | 100% | 0.05 m | 0.42 m | 44% |
| multi | depth only | 100% | 0.04 m | 0.60 m | 54% |
| flat (control) | depth only | 1/126 false detections | | | |

`stairs_up`'s depth-only far error is almost entirely a constant −0.31 m, exactly one tread. The
last tread is flush with the platform, so the geometry ends at the last riser, 0.3 m before the
labeled region end. Releasing on it still leaves the hind feet ~0.15 m past that riser. At close
range, far edges overshoot, which delays the release: the safe direction.

**Gemma box + depth**: *pending (live run in progress)*.

## 3. Discriminator ("Is there any <I>? Just answer yes or no.")

Under a yes/no JSON schema, Gemma-4-E4B spends the whole token budget and returns empty content in
100% of pilot queries (93/93). Free text works. The pipeline now asks in SARO's free-text format.
Accuracy at 560 tokens: *pending*.

## 4. Policy selector (this project's addition)

Pilot, default image budget, 150 frames, truth = footprint ground-truth policy:

| truth \ answer | flat | rough | stairs |
|---|---|---|---|
| flat (92) | 83 | 9 | 0 |
| rough (26) | 18 | 8 | 0 |
| stairs (32) | 30 | 2 | 0 |

Accuracy 61%, stairs recall 0%. At 560 tokens: *pending*.

## Consequences for the pipeline (implemented)

- Perception prompt: detection `box_2d` format at 560 image tokens, with depth-geometry fallback
  when the box is degenerate. Every perception event logs `source` (`vlm_box` / `depth_fallback`).
- Discriminators in free text.
- Policy: the plan's per-sub-task policy, with the selector as a consensus override.
