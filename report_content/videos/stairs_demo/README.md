# Stairs policy demo clips (simulation)

Recorded 2026-10-08 with `unitree_rl_mjlab/scripts/record_stairs_demo.py`. Each clip is one
robot on one trial, following a scripted walking leader (0.5 m/s, 2 m ahead) with the
evaluation's follow controller; training sensor noise on; tread 0.30 m. **A clip shows what a
crossing looks like. How often the policy crosses is in
`coordination/results/2026-10-08-stairs-v6a-final.md`** (256 trials per cell).

| clip | policy | course | what happens |
|---|---|---|---|
| `1_v8a_17cm_10steps_up_down.mp4` | stairs v8a, cluster run, `model_3999` (the robot candidate) | 17 cm risers, 10 steps up, landing, 10 steps down | crosses |
| `2_v8a_15cm_10steps_up_down.mp4` | same | 15 cm, 10 steps up and down | crosses |
| `3_v8a_17cm_5steps_up_down.mp4` | same | 17 cm, 5 steps up and down | crosses |
| `4_v8a_12cm_10steps_up_down.mp4` | same | 12 cm, 10 steps up and down | crosses |
| `5_before_v5a_15cm_refuses.mp4` | stairs v5a final (10-07) | 15 cm, 5 steps up | stops at the foot of the flight; the leader walks on |
| `6_before_v7a_17cm_10steps.mp4` | stairs v7a, cluster `model_1600` (trained on 5-step flights) | 17 cm, 10 steps up | tips over near the top |

Clips 5 and 6 are the two failures the later versions fixed: v5a's reward paid more for
refusing a tall flight than for crossing it (`findings.md` #29), and policies trained on
5-step flights did not hold on 10-step ones (#34). Clip 6 is one of the trials that fail;
that checkpoint goes up this flight 45% of the time.

Nothing here has run on the real robot.
