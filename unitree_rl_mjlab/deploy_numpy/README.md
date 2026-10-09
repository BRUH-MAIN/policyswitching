# Running the policy on the real Go2

Everything in this folder is for the robot's Jetson (Python 3.8, numpy, `unitree_sdk2py`; no
torch). **None of it has run on the robot yet.** It has run against the simulator in two
ways, described under "What has been checked". Work through the stages in order and stop at
the first surprise.

## Files

| file | what it does | sends motor commands? |
|---|---|---|
| `go2_runner.py` | the control program: stand up, run the policy at 50 Hz, safety stops | **yes**, unless `--dry-run` |
| `go2_scan_node.py` | height scan from the robot's own LiDAR map, to the runner over local UDP | no |
| `go2_obs.py`, `go2_scan.py`, `policy_numpy.py` | observation, scan maths, the network | no |
| `follow_cmd.py` (laptop) | person-follow velocity commands from the perception hub, over UDP | no |
| `*.npz` | exported policy weights; not in git, export on the laptop (below) | |

Export a policy on the laptop (it checks the numpy network against PyTorch):

```
cd unitree_rl_mjlab
PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/export_policy_numpy.py <checkpoint.pt> --out deploy_numpy/<name>.npz
PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/verify_deploy_obs.py <checkpoint.pt> --npz deploy_numpy/<name>.npz
PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/test_go2_runner_sim.py --npz deploy_numpy/<name>.npz --scan map
```

Copy to the robot as **new files in a new folder** (the robot workspace's rule: nothing
existing on the robot is edited), and add them to the list of files created on the robot:

```
scp go2_runner.py go2_scan_node.py go2_obs.py go2_scan.py policy_numpy.py <name>.npz unitree@<robot>:~/go2_policy/
```

Check the copy arrived whole (compare with `md5sum` of the same files on the laptop):

```
ssh unitree@<robot> 'cd ~/go2_policy && md5sum *.py *.npz'
```

Start the runner inside `tmux` or `screen` on the robot, so a dropped SSH session does not end
a run. If the session does drop without them, the runner treats the hang-up like Ctrl-C and
goes limp (it used to stop without damping).

## The remote

| buttons | effect |
|---|---|
| **L2 + B** | motors limp (PASSIVE), from any state. The stop button. |
| L2 + Up | from PASSIVE: crouch, then stand (2 s) |
| R2 + A | from standing: the policy takes over |
| left stick / right stick | forward-back and sideways / turn |
| hold R1 | with `--cmd udp`: follow the commands coming from the laptop; release = stop |

Ctrl-C in the terminal also goes limp. Speeds are deliberately low by default (0.3 m/s
forward at full stick, 0.15 m/s back and sideways, 0.5 rad/s turning, ramped over about a
second); raise them with `--max-vx` etc. only after the slow runs are clean.

**Motor temperature.** The program prints the hottest motor once a second and refuses to stand
up or start the policy if any motor is above 60 C. On 2026-10-09 the robot's own controller
went limp four times in 20 minutes with the rear hip motors at 65-69 C (others 36-42 C), and
the Go2 lying powered loads its rear hips. Cool it with the robot switched off. The program goes limp by itself if the robot state
stops arriving, the body tilts past about 55 degrees, or a joint moves faster than 30 rad/s.

## Stages

**0. Nothing moves: what does the robot publish?** Robot standing in its normal mode.

```
cd ~/go2_policy
python3 go2_scan_node.py --probe
python3 go2_runner.py --npz <name>.npz --dry-run
```

`--dry-run` never creates a command publisher. Check in its printout:
- gravity about `(0, 0, -1)`; lift the robot's nose and gravity x goes **negative**, push the
  nose down and it goes **positive**; lower the robot's right side and gravity y goes
  **negative**;
- "joints - default" small (under ~0.3) and mirrored between left and right legs;
- "base height (legs)" about 0.30 m standing, and it follows the body if the robot is made
  to stand lower;
- move the sticks: `cmd` follows (forward stick = positive first number).

Copy `scan_probe_*.npz` back to the laptop. It decides how the scan node is finished.

**1. Legs free.** Robot hung from its handle or on a stand, feet off the ground.

```
python3 go2_runner.py --npz <name>.npz --scan flat
```

Type `yes`; the robot's own controller is switched off (it lies down or goes slack first).
L2+Up, then R2+A. With the sticks centred the legs should hold still or step lightly; with
the stick forward all four legs should trot, feet moving **backward** during stance. L2+B.
Anything violent or one-sided: L2+B and stop here.

**2. Standing on the floor**, clear flat area, someone at the remote. Same command. L2+Up
(it stands), wait, R2+A. It should stand in place. L2+B to finish; it sinks down.

**3. Walking on flat ground** with `--scan flat`: forward, back, turn, slowly first. This
scan assumes level ground, so no steps yet.

**4. The real scan, still on flat ground.** Second terminal first:

```
python3 go2_scan_node.py            # prints scan range, empty cells, anchor, ages
python3 go2_runner.py --npz <name>.npz --scan udp
```

On flat ground the scan should read the same height everywhere, equal to "legs say", within
2-3 cm. Walk around. Then stand the robot facing a single step and look at the printed scan
range before walking at it.

**5. One low step (under 10 cm), then one 12 cm step, then a short flight**, with a spotter
and a tether or a hand on the handle. Use the riser heights **and flight lengths** the policy
passes in simulation (table in `PROGRESS_REPORT.md`), not more. As of 2026-10-08 18:30 the
policy to use is `stairs_v8a_final.npz` (cluster v8a `model_3999`): in simulation, 5- and
10-step flights to 17 cm, up and down, without a tip-over in 4,096 trials, at a 0.30 m tread.
It touches a step with a shin on about one tall flight in three; watch for that on real
nosings. On the
robot go up first, and bring it down by hand until descents are trusted.

**6. Following a person**: `--cmd udp` on the robot, `follow_cmd.py` on the laptop, R1 held.

Afterwards, to give the robot back to its own controller:

```
python3 go2_runner.py --restore-motion-service
```

## What has been checked, and what has not

Checked on the laptop:
- `scripts/verify_deploy_obs.py`: the observation built by `go2_obs.py` from robot-order
  joint data, quaternion, gyro and an independently cast scan equals the simulator's own
  observation to 1e-5 on every term over a climb of a 12 cm flight, and the numpy policy's
  action equals PyTorch's to 4e-5.
- `scripts/test_go2_runner_sim.py`: `go2_runner.Runner` itself, driven by a scripted remote
  against plain CPU MuJoCo (not the training simulator), stands the robot up from the floor,
  starts the policy, walks and climbs a 12 cm flight, with the scan from exact ray casts and
  through `go2_scan.HeightMapScan` from a noisy gridded map with holes and vertical drift
  (median scan error 0.7 cm).
- `scripts/sim_dds_robot.py`: the unmodified `go2_runner.py` process, talking DDS through
  `unitree_sdk2py` to a pretend Go2 (MuJoCo behind `rt/lowstate` / `rt/lowcmd` on DDS domain 1,
  scan over UDP), stood up, ran the policy, walked 6.4 m and climbed a 12 cm flight; the
  pretend robot received 424 commands a second with no CRC errors, and the runner went
  PASSIVE by itself 120 ms after the robot state stopped. This used the current SDK from
  GitHub with cyclonedds 11 on the laptop; the robot has an older checkout with cyclonedds
  0.10.2, so imports and the motion-service calls are still to be seen there.
- Base height from leg kinematics: within 2 mm on average (worst 8 mm) of the truth while
  walking on flat ground in that test.

Not checked, and only the robot can say:
- that motor index, joint sign, IMU axes and quaternion order on the robot are what
  `go2_obs.py` assumes (stage 0 and stage 1 exist for this);
- that the LiDAR height map and a pose are published, in which frame, and whether they keep
  coming once the robot's own controller is off (stage 0, and again during stage 2);
- the foot-force scale used to decide which feet carry load (`--stance-force`);
- everything physical: motor response, friction, real stair nosings, battery sag.

Known behaviour to expect: under a steady moderate command the stairs policies can stop at
the foot of a flight they would climb at a higher command (v5a, 12 cm: 7 of 32 at 0.5 m/s,
28 of 32 at 0.8 m/s in simulation). Open-loop the heading also drifts; steer with the stick.
