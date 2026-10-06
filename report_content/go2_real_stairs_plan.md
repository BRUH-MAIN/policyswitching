# Plan: real stairs on the physical Go2

**Date**: 2026-10-05, updated 2026-10-06 · **Goal (Rohan)**: the real Go2, which carries a Livox Mid-360, follows a
person and climbs real stairs. · **Status**: plan. Nothing here has been run on the robot.
Robot facts are from the robot workspace (`/run/media/rohan/New Volume/RL/temp`, its
`sessions/` logs and `project.md`); simulation facts from this repository.

## 1. Where things stand

| | |
|---|---|
| Best policy in simulation | Stairs v2 (`go2_spec_stairs_v2/model_9999.pt`): 93% of randomised courses with 5-7 cm risers. Needs its height scan (9% without). Still the best; stairs v3 failed (section 2.1). |
| On real riser heights (sim, 128 trials per cell, 5-step straight flight, 0.3 m tread) | see table below: **it cannot climb real stairs** |
| Switching, leader preview | Not needed: no benefit in any test. Deploy one policy. |
| Robot compute | Jetson, L4T R35.3.1, Ubuntu 20.04, Python 3.8, ROS 2 foxy. **No torch, no onnxruntime.** |
| Robot sensing | Mid-360 at 20 Hz on `/livox/lidar`; firmware L1 LiDAR on `/utlidar/cloud`; `/lowstate` at 450 Hz; front camera at 14 fps through `unitree_sdk2py` `VideoClient`. |
| Already on the robot | camera frame server, LiDAR obstacle monitor (fail-closed), person tracker and terrain classifier on the laptop. Nothing commands motion yet. |

Stairs v2 on taller steps (success %, training sensor noise on, leader 0.5 m/s):

| riser | up, knee contact counts as failure | down, same | up, only tipping over counts | down, same |
|---|---|---|---|---|
| 9 cm | 84.4 | 100.0 | 89.1 | 100.0 |
| 12 cm | 0.0 | 68.8 | 0.0 | 78.9 |
| 15 cm | 0.0 | 7.8 | 0.0 | 12.5 |
| 17 cm | 0.0 | 0.8 | 0.0 | 0.8 |

It was trained on risers of at most 10 cm and its curriculum sat near 2 cm. Going up it stalls
at 12 cm and above; it does not fall, it just cannot get up. A building stair is 15-18 cm.

## 2. Three pieces of work

### 2.1 A policy trained on real step heights (cluster; the long pole)

**Status, 2026-10-06: the first attempt failed. No policy for real stairs exists yet.**

**Run 1, "stairs v3"** (job 12563; risers 5-20 cm at treads 0.30 and 0.26 m, warm start from
stairs v2, robots spread uniformly over all rows, 10k iterations): **it learned to stand
still.** Achieved speed is 12% of commanded at every riser height (v2: 54-61%), and 53% of
steps are stalled. On the laptop's stair flights it crosses 0 of 128 at 9 cm and at 15 cm,
with no falls: the leader simply walks away from it. Its near-zero falls per 100 m and 99%
survival are that, not success. Do not use `go2_spec_stairs_v3/model_9999.pt`.
Write-up: `coordination/results/2026-10-06-stairs-v3-result.md`.

Why, as far as known: with 60% of robots on steps it could not climb, standing still avoided
the termination for touching a step with a knee or shin, and a time-out costs nothing. The
row spread itself behaved as designed. Why it stopped walking even on 5-9 cm steps is not
isolated.

Along the way the cluster confirmed why every earlier run sat on the easiest rows: the
terrain curriculum promotes a robot only when it ends more than 4 m from where it started,
and the staircase fills only the inner 3 m of a patch
(`coordination/results/2026-10-05-terrain-curriculum-diagnosis.md`).

**Run 2, built and CPU-checked on the cluster, not submitted**
(`coordination/results/2026-10-06-stairs-v4-built.md`; tasks `Unitree-Go2-Spec-StairsV4a` and
`V4b`). Replayed offline, its curriculum rule sends stairs v2 to a mean row of about 5.5 and
stairs v3 to row 0, so it does tell a walker from a stander.
Two variants, both warm-started from stairs v2 with risers 5-20 cm:

- **4a**: rows adapt to the robot, starting easy. A robot moves up only when it has actually
  crossed, and moves down when it falls **or stalls**. Nothing else changes.
- **4b**: the same, plus a knee or shin touching a step is penalised and no longer ends the
  episode (base contact and tipping over still do). Reason: stairs v2 going up at 12 cm
  refuses the step rather than falling, and on a real robot a shin brushing a step is
  acceptable where a fall is not. This changes the task the earlier comparison held fixed;
  that comparison is finished.

Each is about 9 GPU-hours. They are judged on crossing rate and achieved speed, never on
falls, survival or reward alone. Whether a policy of this size learns 17 cm in this simulator
is still unknown; more runs may be needed.

**Acceptance in simulation before the robot** (laptop):
`scripts/switch_follow_real_stairs.sh` on single flights at 9 / 12 / 15 / 17 cm, up and down,
5 and 10 steps; target at least 90% crossed up and down at 17 cm, reported with knee contact
counted and with only tipping over counted. Then `switch_follow_scan_faults.sh` at real riser
heights and `switch_follow_stairs_ckpt.sh` to check it still handles easy ground.

### 2.2 A height scan on the robot

The policy needs, 50 times a second, the height of the base above the ground at 187 points: a
17 × 11 grid at 0.1 m spacing covering 0.8 m ahead and behind and 0.5 m to each side, aligned
with the robot's heading, each value divided by 5.

What the hardware log says about the Mid-360: standing, its forward view spans −9.2° to
+48.5° of elevation, the floor shows up about 0.25 m below the sensor, and the nearest forward
return was at 1.22 m. So **the Mid-360 does not see the ground the policy needs**: not under
the robot, and not within roughly the first metre ahead. Two ways to get the scan:

- **A. The firmware L1 LiDAR.** It sits in the head and looks down and forward, and the robot
  already publishes `/utlidar/cloud`. Unitree's firmware may also publish a ready-made
  robot-centric height map and odometry on `/utlidar/...` topics. **To check on the robot**
  (read-only):
  ```
  ros2 topic list | grep -i -E 'utlidar|height|odom|sportmode'
  ros2 topic hz /utlidar/cloud
  ros2 interface show unitree_go/msg/HeightMap        # if the type exists
  ros2 topic echo --once /utlidar/height_map_array | head -30
  ```
  If a height map is there, resampling it to the 17 × 11 grid is a small numpy node.
- **B. Build the map from the Mid-360.** Accumulate ground points seen 1-2.5 m ahead into a
  robot-centric 2.5-D grid, carried forward by odometry until the robot is standing on them.
  Needs odometry good to a few centimetres over 2 m of travel (LiDAR-inertial odometry, or the
  robot's own leg odometry if it is published), and it never sees what is directly underfoot
  on the first step after standing up. More work and more to go wrong than A.

Either way the scan on the robot will be late, patchy and biased in ways the simulation's
uniform ±10 cm per-ray noise does not model. The harness can now inject those
(`switch_follow.py --scan-delay / --scan-dropout / --scan-bias`,
`scripts/switch_follow_scan_faults.sh`). **Baseline, stairs v2 alone, six randomised layouts
with 7 cm risers, 128 trials each, mean success:**

| scan fault | success | change from clean |
|---|---|---|
| none | 93.9% | |
| 40 ms late | 95.2% | +1.3 |
| 100 ms late | 93.9% | 0.0 |
| 200 ms late | 93.4% | −0.5 |
| 30% of cells stale each step | 92.2% | −1.7 |
| 60% of cells stale each step | 91.9% | −2.0 |
| whole scan off by up to ±3 cm (constant per trial) | 90.8% | −3.1 |
| whole scan off by up to ±6 cm | 74.7% | −19.1 |
| 100 ms late + 30% stale + ±3 cm | 90.9% | −3.0 |

So on low steps the policy shrugs off latency and missing cells, and what it cannot take is a
**height offset**: the robot's estimate of its own height above the ground has to be good to
about 3 cm. That is the number to design the scan node around (base height from leg
kinematics or from the map itself, not from drifting odometry). These are 7 cm risers and
stairs v2; repeat on stairs v3 at real riser heights before trusting it, where a late or
wrong scan has less margin.

### 2.3 Running the policy on the Jetson

- The policy is a small MLP plus an observation normaliser. With no torch or onnxruntime on
  the robot, it runs in numpy. **Done on the laptop (2026-10-05)**:
  `unitree_rl_mjlab/deploy_numpy/policy_numpy.py` is a numpy-only runner written for the
  Jetson's Python 3.8, and `scripts/export_policy_numpy.py <checkpoint> --out <file>.npz`
  exports a checkpoint and checks the numpy output against the PyTorch policy on simulator
  observations. Stairs v2 exported and matched to 1.4e-6 over 200 steps × 16 robots. The
  `.npz` holds weights and is not committed; re-export it. It has not been run on the robot.
- Observation, in order: base angular velocity (3), projected gravity (3), velocity command
  (3), gait phase sin/cos with a 0.6 s period, zero when the command is under 0.1 (2), joint
  positions minus default (12), joint velocities (12), last action (12), height scan (187).
- Action: joint position target = default pose + 0.25 × action, at 50 Hz, gains 20 / 20 / 40
  N·m/rad and 1 / 1 / 2 N·m·s/rad (hip, thigh, calf); default pose hip / thigh / calf = (−0.1, 0.9, −1.8) for the
  left legs and (0.1, 0.9, −1.8) for the right, in the simulator's joint order FL, FR, RL, RR. Clip actions to ±6 before scaling; training did.
- Two known traps from the robot log: joint order on the robot is FR / FL / RR / RL (the
  deploy config's `joint_ids_map` is `[3,4,5,0,1,2,9,10,11,6,7,8]`), and the IMU quaternion is
  (w, x, y, z) and was observed with negative w.
- `/lowstate` arrives at 450 Hz; run the policy at 50 Hz on the latest sample.
- Commands: at most 1.0 m/s forward, 0.5 m/s backward and sideways, 1.0 rad/s. The follow
  controller in `src/vlm_nav/controllers.py` stays inside that.
- The alternative is the repository's C++ deploy stack (`unitree_rl_mjlab/deploy/robots/go2`),
  which needs onnxruntime built for the Jetson and a new height-scan observation.

## 3. Order of work

1. **Rohan, in the cluster session**: submit run 2 (4a and 4b are built), and have that
   session check each run at about 500 and 1,500 iterations so a failing one is stopped early.
2. **Rohan, on the robot (read-only, 10 minutes)**: the topic checks in 2.2. Still not done;
   they decide between scan options A and B.
3. **Laptop, when a run-2 checkpoint lands on HF**: the acceptance runs in 2.1.
4. **Laptop, meanwhile**: a scan node for whichever of A or B the robot checks support.
5. **When a policy passes acceptance**: export with `scripts/export_policy_numpy.py`, then
   bring-up on the robot in stages, each gated on the last: suspended with legs free (joint
   order and signs) → standing on flat ground → walking on flat ground with the real scan →
   one 9 cm step → one 15-17 cm step → a short flight with a spotter and a tether. Stop and go
   back at the first surprise.
6. **Person-following** rides on top once walking is trusted: the person tracker already on
   the laptop gives direction, depth or LiDAR gives range, the follow controller gives the
   velocity command, and the LiDAR obstacle monitor keeps its veto.

## 4. What is not known

- Whether any run reaches 17 cm at all, and in how many attempts. Run 1 failed.
- Whether the firmware publishes a usable height map (2.2 A).
- How the policy tolerates a real scan's delay and holes.
- Real friction, real stair nosings and open risers, payload, and battery sag: none of it is
  in the simulation.
- The teammate's Isaac Gym policy mentioned in the robot workspace is not on this laptop; how
  it compares with stairs v2 or v3 is unknown.
