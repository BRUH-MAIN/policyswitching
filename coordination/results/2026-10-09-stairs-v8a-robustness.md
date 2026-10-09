# Stairs v8a final: robustness to control delay, motor strength and payload

**Date**: 2026-10-09 (laptop) · **Checkpoint**: `go2_spec_stairs_v8a/model_3999` (the robot
candidate) · `scripts/stairs_robustness.sh`: 10-step flights, 128 trials per cell (seed 800),
training sensor noise on, tipping only, leader at 0.5 m/s. New `switch_follow.py` options:
`--action-delay N` (each action applied N control steps = N x 20 ms late; the policy still
sees its own latest action), `--motor-strength s` (every motor's stiffness, damping and
torque limit x s), `--payload kg` (point mass added at the trunk's COM). Checked that they
take effect: stiffness 20 -> 16, torque limit 23.5 -> 18.8 N m, trunk 6.92 -> 7.92 kg.

Success / fall / lost %:

| condition | up 15 cm | down 15 cm | up 17 cm | down 17 cm |
|---|---|---|---|---|
| nominal | 100 / 0 / 0 | 100 / 0 / 0 | 99.2 / 0 / 0.8 | 100 / 0 / 0 |
| 20 ms delay | 94.5 / 3.9 / 1.6 | 99.2 / 0.8 / 0 | **85.9 / 7.8 / 6.2** | 98.4 / 1.6 / 0 |
| 40 ms delay | **7.0 / 61.7 / 31.2** | **53.1 / 45.3 / 1.6** | **5.5 / 67.2 / 27.3** | **32.0 / 68.0 / 0** |
| motors 80% | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 99.2 / 0.8 / 0 |
| motors 120% | 100 / 0 / 0 | 98.4 / 0 / 1.6 | 100 / 0 / 0 | 96.9 / 0 / 3.1 |
| +1 kg | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| +2 kg | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 | 100 / 0 / 0 |
| 20 ms + motors 85% + 1 kg | 88.3 / 7.8 / 3.9 | 99.2 / 0.8 / 0 | **78.1 / 14.1 / 7.8** | 95.3 / 4.7 / 0 |

**Motor strength and payload do not matter; latency does.** 20 ms costs 14 points going up
17 cm, mostly tip-overs; 40 ms breaks it. Training never varied latency (only friction,
encoder offset, COM and pushes). The robot's latency (DDS, a Python loop, the Jetson) is not
known; the runner's printout reports the state's age, and stage 0 on the robot should
measure the loop. The delay here is whole control steps; real latency is likely a few to
~15 ms, between the nominal row and the 20 ms row.

**Fix under test: StairsV8b** = v8a with every motor command delayed 0-30 ms (mjlab's
`DelayedActuator`, 5 ms steps, one lag per robot drawn at each reset). Laptop trial since
11:40 IST 10-09: 1,024 envs, 1,600 iterations from v8a final, HF `go2_spec_stairs_v8b_lap/`.
Pass mark, set now: the robustness table above with the 20 ms and combined rows at 95% or
better at 17 cm, and the nominal row not worse. The 8,192-env version is requested of the
cluster (inbox, 10-09).


## Update 13:15 IST: corrected delay test, v8a rows remeasured; laptop v8b fails

The first delay test applied the delay to the action fed to the environment, so the policy's
"last action" input was the delayed one, which will not happen on the robot. `--action-delay`
now delays inside the actuators (mjlab `DelayedActuator`, fixed lag) as on the robot and as
v8b trains; motor strength now scales the actuator settings directly (mjlab's
`effort_limits` event rejects delayed actuators). Same table, 10-step flights, 128 trials,
tipping only, success / fall / lost %:

| condition | v8a up 17 | v8a down 17 | v8b laptop up 17 | v8b laptop down 17 | v8a up 15 | v8b laptop up 15 |
|---|---|---|---|---|---|---|
| nominal | 99.2 / 0 / 0.8 | 100 / 0 / 0 | 94.5 / 3.9 / 1.6 | **68.8 / 1.6 / 29.7** | 100 | 100 |
| 20 ms | 87.5 / 7.8 / 4.7 | 100 / 0 / 0 | 88.3 / 7.0 / 4.7 | 94.5 / 0 / 5.5 | 96.1 | 97.7 |
| 40 ms | 2.3 / 89.1 / 8.6 | 9.4 / 90.6 / 0 | 41.4 / 40.6 / 18.0 | 79.7 / 19.5 / 0.8 | 1.6 | 63.3 |
| motors 120% | 100 / 0 / 0 | 96.9 / 0 / 3.1 | 96.1 / 3.1 / 0.8 | **5.5 / 0 / 94.5** | 100 | 100 |
| 20 ms + 85% + 1 kg | 87.5 / 8.6 / 3.9 | 96.9 / 3.1 / 0 | 85.2 / 10.2 / 4.7 | 96.9 / 3.1 / 0 | 89.1 | 96.9 |

(v8b down 15 cm with motors at 120%: 16.4% success, 83.6% lost.) v8a with the corrected
test: 20 ms costs 12 points going up 17 cm and nothing going down; 40 ms makes it fall
everywhere (89-91%).

**Laptop v8b (1,024 envs, 1,600 iterations) fails the pass mark.** It is more tolerant of
40 ms, no better at 20 ms, and it has **started refusing tall descents**: down 17 cm with no
delay 69% (30% stopped at the top), with stiffer motors 5%. Small-batch laptop runs have
under-trained before (findings #33); the 8,192-env v8b (cluster job 12747, ends ~17:00) is
the one that decides. Watch its descents, not only its delay tolerance.

**Single steps** (one riser, the demo Rohan asked about), v8a final, 128 trials: up and down,
15 and 17 cm, with and without 20 ms delay: 100% in all eight cells, no falls
(`eval_results/switch_follow/single_step/`).


## Update 15:50 IST: cluster v8b (8,192 envs, job 12747) passes at iteration 600

Same table (10-step flights, 128 trials, tipping only, success %; corrected delay test):

| condition | v8a final up 17 / down 17 | **v8b `model_600`** up 15 / down 15 / up 17 / down 17 | v8b `model_1200` up 15 / down 15 / up 17 / down 17 |
|---|---|---|---|
| nominal | 99.2 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| 20 ms | 87.5 / 100 | 100 / 100 / **98.4** / 100 | 100 / 100 / 98.4 / 100 |
| 40 ms | 2.3 / 9.4 | 81.2 / 94.5 / **64.1** / 85.2 | 87.5 / 97.7 / 72.7 / 95.3 |
| motors 80% | 100 / 99.2 | 100 / 100 / 100 / 95.3 | 100 / 98.4 / 99.2 / 100 |
| motors 120% | 100 / 96.9 | 100 / 100 / 100 / 100 | 100 / **83.6** / 96.1 / **68.0** (stops at the top) |
| +1 kg / +2 kg | 100 / 100 | 100-98.4 | 96.9-100 |
| 20 ms + 85% + 1 kg | 87.5 / 96.9 | 99.2 / 100 / **97.7** / 96.9 | 97.7 / 100 / 96.1 / 100 |

**`model_600` meets the pass mark** set before the run (20 ms and combined rows at 95% or
better at 17 cm, nominal not worse) and has no weak cell below 95% outside the 40 ms row.
`model_1200` is better at 40 ms but has begun refusing tall descents when the motors are
stiffer than simulated (the laptop v8b's failure mode, at a smaller scale), so it is not
preferred. Single steps (one riser) with 20 ms delay, `model_600`: up and down, 15 and 17 cm,
100% (128 trials each). Exported as `deploy_numpy/stairs_v8b_c600.npz` (md5
f0be1404569cd331be59368ba0c26a3c); numpy/PyTorch parity and robot-side observation parity
pass. **It replaces v8a final as the robot candidate.** The run continues to 2,000
iterations (~17:00); its last checkpoints get the same table before a final choice.


## Update 17:30 IST: cluster v8b final (`model_1999`) is the robot candidate

10-step flights, 128 trials, tipping only, success %, up 15 / down 15 / up 17 / down 17 cm:

| condition | `model_600` | `model_1800` | **`model_1999`** |
|---|---|---|---|
| nominal | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |
| 20 ms | 100 / 100 / 98.4 / 100 | 100 / 100 / 97.7 / 100 | 100 / 100 / 98.4 / 100 |
| 40 ms | 81.2 / 94.5 / 64.1 / 85.2 | 92.2 / 99.2 / 80.5 / 93.0 | 89.1 / 97.7 / 75.8 / 95.3 |
| motors 80% | 100 / 100 / 100 / 95.3 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 95.3 |
| motors 120% | 100 / 100 / 100 / 100 | 99.2 / 96.9 / 100 / 90.6 | 100 / 100 / 100 / 99.2 |
| +1 kg | 100 / 100 / 99.2 / 100 | 100 / 100 / 100 / 99.2 | 100 / 100 / 100 / 97.7 |
| +2 kg | 100 / 99.2 / 98.4 / 97.7 | 100 / 99.2 / 100 / 100 | 100 / 94.5 / 100 / 96.1 |
| 20 ms + 85% + 1 kg | 99.2 / 100 / 97.7 / 96.9 | 97.7 / 100 / 97.7 / 99.2 | 100 / 100 / 100 / 96.9 |

`model_1999` passes the mark and is clearly better than `model_600` at 40 ms (up 17 cm 76% vs
64%); its one weaker cell is +2 kg going down 15 cm (94.5%, all tip-overs counted). The
descent refusal seen at `model_1200` (stiffer motors) is gone at the end. Single steps with
20 ms delay and a 0.3 m/s leader: 100% up and down at 15 and 17 cm. Exported as
`deploy_numpy/stairs_v8b_final.npz` (md5 ddfb6633715ce6758d37a7b9a20cc90e); parity checks pass.
