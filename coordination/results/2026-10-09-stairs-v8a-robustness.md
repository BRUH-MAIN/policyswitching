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
