"""Build the policy's observation from what the robot reports. Numpy only, Python 3.8.

Everything here mirrors the simulator's observation terms exactly
(src/tasks/velocity/velocity_env_cfg.py); scripts/verify_deploy_obs.py checks it against the
simulator's own observation, term by term, on a stairs course. Change nothing here without
re-running that check.

Conventions
  * "robot order" is the Go2's motor order in LowState/LowCmd: FR, FL, RR, RL, each
    hip, thigh, calf. "sim order" is the policy's: FL, FR, RL, RR. ROBOT_TO_SIM swaps the
    pairs; it is its own inverse, so the same index array converts either way.
  * The IMU quaternion is (w, x, y, z), body to world. Its overall sign is irrelevant.
  * The gyro is in the body frame, rad/s.
  * The height scan is the height of the base_link origin above the ground at 187 points:
    x = -0.8 .. 0.8 m (17), y = -0.5 .. 0.5 m (11), in a frame at the base that shares its
    yaw but stays level (no roll or pitch). Flattened with y as the slow index:
    index = iy * 17 + ix. Divided by 5. A cell with no ground reads 1.0 (5 m).
    TERRAIN ONLY: in training the rays also hit the robot's own legs on a few cells; final
    stairs v5a crosses 9 and 12 cm flights equally well without those hits
    (eval_results/switch_follow/self_hits/), so do not try to reproduce them.
"""

import numpy as np

ROBOT_TO_SIM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
# Sim order (FL, FR, RL, RR): left hips -0.1, right hips +0.1.
DEFAULT_JOINT_POS = np.array([-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8], dtype=np.float32)
ACTION_SCALE = 0.25
CONTROL_DT = 0.02
GAIT_PERIOD = 0.6
STAND_COMMAND_NORM = 0.1  # the gait clock reads zero below this |(vx, vy, wz)|
SCAN_NX, SCAN_NY, SCAN_RES = 17, 11, 0.1
SCAN_SCALE = 1.0 / 5.0
SCAN_MISS = 5.0
OBS_DIM = 47 + SCAN_NX * SCAN_NY


def scan_points_xy():
    """(187, 2) scan points in the level, yaw-aligned base frame, in observation order."""
    xs = (np.arange(SCAN_NX) - (SCAN_NX - 1) / 2.0) * SCAN_RES
    ys = (np.arange(SCAN_NY) - (SCAN_NY - 1) / 2.0) * SCAN_RES
    gx, gy = np.meshgrid(xs, ys, indexing="xy")  # shape (11, 17): rows are y
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def projected_gravity(quat_wxyz):
    """Unit gravity direction in the body frame: (0, 0, -1) when level."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    # Third row of the body-to-world rotation matrix, negated: R^T (0, 0, -1).
    return np.array([-2.0 * (x * z - w * y), -2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y))], dtype=np.float32)


def yaw_of(quat_wxyz):
    """Heading of the body x axis projected on the ground (rad), as the scan frame uses it."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def gait_phase(step_count, command):
    """sin, cos of the 0.6 s gait clock at control step `step_count`; zeros when told to stand."""
    if float(np.linalg.norm(np.asarray(command, dtype=np.float64))) < STAND_COMMAND_NORM:
        return np.zeros(2, dtype=np.float32)
    phase = ((step_count * CONTROL_DT) % GAIT_PERIOD) / GAIT_PERIOD
    return np.array([np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)], dtype=np.float32)


def flat_scan(base_height):
    """The scan a level robot sees on flat ground with its base `base_height` m up (metres)."""
    return np.full(SCAN_NX * SCAN_NY, base_height, dtype=np.float32)


def build_obs(gyro, quat_wxyz, command, step_count, q_robot, dq_robot, last_action, scan_m):
    """One 234-entry observation.

    gyro (3) body rad/s; quat_wxyz (4); command (vx, vy, wz); step_count: control steps since
    the policy started; q_robot, dq_robot (12) in ROBOT order; last_action (12) in sim order,
    as returned by NumpyPolicy.act; scan_m (187) heights in METRES (see module docstring).
    """
    q_sim = np.asarray(q_robot, dtype=np.float32)[ROBOT_TO_SIM]
    dq_sim = np.asarray(dq_robot, dtype=np.float32)[ROBOT_TO_SIM]
    scan = np.asarray(scan_m, dtype=np.float32)
    if scan.shape != (SCAN_NX * SCAN_NY,):
        raise ValueError("scan has shape %s, expected (%d,)" % (scan.shape, SCAN_NX * SCAN_NY))
    return np.concatenate([
        np.asarray(gyro, dtype=np.float32),
        projected_gravity(quat_wxyz),
        np.asarray(command, dtype=np.float32),
        gait_phase(step_count, command),
        q_sim - DEFAULT_JOINT_POS,
        dq_sim,
        np.asarray(last_action, dtype=np.float32),
        np.minimum(scan, SCAN_MISS) * SCAN_SCALE,
    ]).astype(np.float32)


def joint_targets_robot(action):
    """Joint position targets in ROBOT order for one (already clipped) action in sim order."""
    target_sim = DEFAULT_JOINT_POS + ACTION_SCALE * np.asarray(action, dtype=np.float32)
    return target_sim[ROBOT_TO_SIM]
