#!/usr/bin/env python3
# Created by Claude (policyswitching repo, deploy_numpy/go2_runner.py).
# Purpose: run a trained locomotion policy on the Go2 with low-level joint control.
# This file COMMANDS THE MOTORS unless started with --dry-run. Read the docstring first.
"""Run a numpy locomotion policy on the Unitree Go2 (Jetson: Python 3.8, numpy, unitree_sdk2py).

    python3 go2_runner.py --npz stairs_v5a.npz --dry-run            # sends nothing at all
    python3 go2_runner.py --npz stairs_v5a.npz --scan flat          # walks; flat ground only
    python3 go2_runner.py --npz stairs_v5a.npz --scan udp           # needs go2_scan_node.py

States and the remote's buttons (same layout as Unitree's own deploy code):

    PASSIVE   motors damped, no stiffness. Entered at start, on L2+B, on Ctrl-C, on any fault.
    STAND     L2+Up from PASSIVE: crouch, then stand, over 3 s, and hold.
    POLICY    R2+A from STAND: the policy drives the joints at 50 Hz.
              Left stick: forward/back and sideways. Right stick: turn.
              With --cmd udp: velocity commands from the network are used only WHILE R1 IS
              HELD; releasing R1 commands a stop. The sticks always win when moved.
    L2+B      back to PASSIVE from anywhere (the robot sinks down on damping).

--dry-run never creates a command publisher and never touches the motion service. The
robot stays in whatever mode it is in. It prints what the policy would see and do, once a
second, so the conventions can be checked with the robot standing in its normal mode:
gravity should read about (0, 0, -1); rolling the body right side down makes gravity y
negative; nose down makes gravity x positive; joint offsets from the default pose should
be small and mirror left/right; the base height from the legs should be about 0.3 m.

Without --dry-run it first asks on the terminal, then releases the robot's own motion
service (the robot lies down first), because two controllers must never drive the motors
together. `--restore-motion-service` alone gives the robot back to its own controller.

Faults that drop to PASSIVE on their own: no robot state for 50 ms; body tilted more than
about 55 degrees; any joint faster than 30 rad/s; the control loop more than 100 ms late.
If the height scan (--scan udp) is older than 0.3 s the command is forced to zero and the
robot stands on the last scan it had (it does not go limp: that is worse on a staircase).

It never edits anything on the robot. It writes one log file per run next to itself.
"""

from __future__ import print_function

import argparse
import os
import signal
import socket
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import go2_obs  # noqa: E402
from policy_numpy import NumpyPolicy  # noqa: E402

# ---------------------------------------------------------------------------------------
# Constants, all in ROBOT motor order (FR, FL, RR, RL; hip, thigh, calf).
POLICY_KP = np.array([20.0, 20.0, 40.0] * 4)
POLICY_KD = np.array([1.0, 1.0, 2.0] * 4)
STAND_KP = np.array([60.0, 80.0, 80.0] * 4)
STAND_KD = np.array([5.0, 4.0, 4.0] * 4)
PASSIVE_KD = 3.0
CROUCH_POSE = np.array([0.0, 1.36, -2.65] * 4)
STAND_POSE = np.array([0.0, 0.8, -1.5] * 4)
TORQUE_LIMIT = np.array([23.5, 23.5, 45.0] * 4)  # what the simulator's actuators clip to
# Joint limits with a margin, robot order (front thighs and rear thighs differ).
Q_MIN = np.array([-1.0, -1.5, -2.70, -1.0, -1.5, -2.70, -1.0, -0.5, -2.70, -1.0, -0.5, -2.70])
Q_MAX = np.array([1.0, 3.4, -0.86, 1.0, 3.4, -0.86, 1.0, 4.4, -0.86, 1.0, 4.4, -0.86])
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

DT = go2_obs.CONTROL_DT
STATE_TIMEOUT_S = 0.05
MAX_TILT_GZ = -0.57  # projected gravity z above this means more than ~55 degrees of tilt
MAX_JOINT_SPEED = 30.0
# Motor temperature guard (deg C). On 2026-10-09 the robot went limp by itself four times in
# 20 minutes, its own controller protecting the rear hip motors, read at 65 and 69 C while the
# others were at 36-42 C. Standing up or starting the policy is refused above START_MAX_TEMP_C;
# above WARN_TEMP_C every status line warns. It never goes limp on temperature by itself
# (going limp mid-step is worse than finishing it): L2+B stays the operator's decision.
START_MAX_TEMP_C = 60
WARN_TEMP_C = 70
LOOP_LATE_S = 0.10
SCAN_STALE_S = 0.3
CMD_STALE_S = 0.3
GAIN_BLEND_STEPS = 25  # 0.5 s from stand gains to policy gains
STAND_RAMP_S = 1.5  # each half of the stand-up (fold, then stand) takes this long
WRITE_PERIOD_S = 0.002  # the motor board is fed at 500 Hz, as in Unitree's own low-level example
WRITER_STALE_S = 0.10  # if the control loop stops updating the target, the writer damps

PASSIVE, STAND, POLICY = "PASSIVE", "STAND", "POLICY"
MOTOR_NAMES = [leg + " " + j for leg in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]  # robot order

# Bit numbers in the remote's key word.
KEY = dict(R1=0, L1=1, start=2, select=3, R2=4, L2=5, F1=6, F2=7, A=8, B=9, X=10, Y=11,
           up=12, right=13, down=14, left=15)

# Leg geometry (metres) for the base height estimate: hip position in the base frame,
# sideways thigh offset, thigh and calf length, foot radius. Robot order FR, FL, RR, RL.
HIP_X = np.array([0.1934, 0.1934, -0.1934, -0.1934])
HIP_Y = np.array([-0.0465, 0.0465, -0.0465, 0.0465])
THIGH_Y = np.array([-0.0955, 0.0955, -0.0955, 0.0955])
LEG_L = 0.213
FOOT_R = 0.022


class Remote(object):
    def __init__(self):
        self.keys = 0
        self.lx = self.ly = self.rx = self.ry = 0.0

    def update(self, raw):
        raw = bytes(bytearray(raw))
        if len(raw) < 24:
            return
        self.keys = struct.unpack("<H", raw[2:4])[0]
        self.lx = struct.unpack("<f", raw[4:8])[0]
        self.rx = struct.unpack("<f", raw[8:12])[0]
        self.ry = struct.unpack("<f", raw[12:16])[0]
        self.ly = struct.unpack("<f", raw[20:24])[0]

    def down(self, *names):
        return all((self.keys >> KEY[n]) & 1 for n in names)


def foot_positions_base(q_robot):
    """(4, 3) foot centre positions in the base frame from joint angles, robot order."""
    q = np.asarray(q_robot, dtype=np.float64).reshape(4, 3)
    a, b, c = q[:, 0], q[:, 1], q[:, 2]
    x = HIP_X - LEG_L * np.sin(b) - LEG_L * np.sin(b + c)
    # Leg plane coordinates: sideways offset l and downward reach d, then rotate by the hip.
    d = LEG_L * np.cos(b) + LEG_L * np.cos(b + c)
    y = HIP_Y + THIGH_Y * np.cos(a) + d * np.sin(a)
    z = THIGH_Y * np.sin(a) - d * np.cos(a)
    return np.stack([x, y, z], axis=1)


def base_height_from_legs(q_robot, quat_wxyz):
    """Height of the base origin above the lowest foot's contact point, gravity-aligned (m)."""
    feet = foot_positions_base(q_robot)
    up_b = -go2_obs.projected_gravity(quat_wxyz).astype(np.float64)  # world up, in the base frame
    heights = -feet.dot(up_b)  # how far below the base origin each foot centre is
    return float(heights.max() + FOOT_R)


# ---------------------------------------------------------------------------------------
# Robot backend: unitree_sdk2py. Imported lazily so the file loads on a laptop without it.
class Go2Backend(object):
    def __init__(self, iface, dry_run, domain=0, write_hz=500.0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

        self.dry_run = dry_run
        self._state = None
        self._state_t = 0.0
        self._latest = None  # (q_des, time) of the command the writer thread keeps sending
        self._damped = False
        self._lock = threading.Lock()
        self._write_period = 1.0 / write_hz if write_hz > 0 else None
        self._stop = False
        ChannelFactoryInitialize(domain, iface)
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        self._pub = None
        self._cmd = None
        self._crc = None
        if not dry_run:
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            from unitree_sdk2py.utils.crc import CRC

            self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self._pub.Init()
            self._crc = CRC()
            cmd = unitree_go_msg_dds__LowCmd_()
            cmd.head[0], cmd.head[1] = 0xFE, 0xEF
            cmd.level_flag = 0xFF
            cmd.gpio = 0
            for i in range(20):
                cmd.motor_cmd[i].mode = 0x01
                cmd.motor_cmd[i].q = POS_STOP_F
                cmd.motor_cmd[i].dq = VEL_STOP_F
                cmd.motor_cmd[i].kp = 0.0
                cmd.motor_cmd[i].kd = 0.0
                cmd.motor_cmd[i].tau = 0.0
            self._cmd = cmd
            if self._write_period is not None:
                self._writer = threading.Thread(target=self._write_loop, name="lowcmd-writer")
                self._writer.daemon = True
                self._writer.start()

    def _fill(self, q_des, kp, kd):
        for i in range(12):
            m = self._cmd.motor_cmd[i]
            m.q = float(q_des[i])
            m.dq = 0.0
            m.kp = float(kp[i])
            m.kd = float(kd[i])
            m.tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)

    def _write_loop(self):
        """Re-send the current command at 500 Hz (the checksum is only recomputed when the
        command changes, 50 times a second). If the control loop has not refreshed it for
        WRITER_STALE_S (it crashed or hung), switch to damping instead of holding a stiff pose."""
        while not self._stop:
            with self._lock:
                if self._latest is not None:
                    q_des, t = self._latest
                    if not self._damped and time.time() - t > WRITER_STALE_S:
                        self._fill(q_des, np.zeros(12), np.full(12, PASSIVE_KD))
                        self._damped = True
                    self._pub.Write(self._cmd)
            time.sleep(self._write_period)

    def close(self):
        self._stop = True

    def _on_state(self, msg):
        self._state = msg
        self._state_t = time.time()

    def read(self):
        """dict(q, dq, quat, gyro, remote_raw, age) or None before the first message."""
        msg = self._state
        if msg is None:
            return None
        return dict(
            q=np.array([msg.motor_state[i].q for i in range(12)], dtype=np.float64),
            dq=np.array([msg.motor_state[i].dq for i in range(12)], dtype=np.float64),
            quat=np.array(msg.imu_state.quaternion, dtype=np.float64),
            gyro=np.array(msg.imu_state.gyroscope, dtype=np.float64),
            remote_raw=msg.wireless_remote,
            temp=np.array([msg.motor_state[i].temperature for i in range(12)], dtype=np.float64),
            age=time.time() - self._state_t,
        )

    def send(self, q_des, kp, kd):
        if self._pub is None:
            return
        with self._lock:
            self._fill(q_des, kp, kd)
            self._latest = (np.array(q_des, dtype=np.float64), time.time())
            self._damped = False
            if self._write_period is None:
                self._pub.Write(self._cmd)

    def release_motion_service(self):
        """Make the robot lie down and switch off its own controller. True if confirmed off."""
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        sport = SportClient()
        sport.SetTimeout(5.0)
        sport.Init()
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        except ImportError:
            MotionSwitcherClient = None
        if MotionSwitcherClient is not None:
            switcher = MotionSwitcherClient()
            switcher.SetTimeout(5.0)
            switcher.Init()
            for _ in range(10):
                code, result = switcher.CheckMode()
                if code != 0 or result is None:
                    print("  could not ask which motion service is active (code %s); retrying" % code)
                    time.sleep(1.0)
                    continue
                if not result.get("name"):
                    return True
                print("  motion service '%s' is active: lying down and releasing it" % result["name"])
                sport.StandDown()
                time.sleep(3.0)
                switcher.ReleaseMode()
                time.sleep(1.0)
            return False
        # Older SDK: no mode switcher. Switch the service off by name.
        from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

        rsc = RobotStateClient()
        rsc.SetTimeout(5.0)
        rsc.Init()
        sport.StandDown()
        time.sleep(3.0)
        ok = False
        for name in ("mcf", "sport_mode"):
            try:
                code = rsc.ServiceSwitch(name, False)
                print("  ServiceSwitch(%s, off) -> %s" % (name, code))
                ok = ok or code == 0
            except Exception as exc:  # noqa: BLE001
                print("  ServiceSwitch(%s) failed: %s" % (name, exc))
        return ok

    @staticmethod
    def restore_motion_service(iface, domain=0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(domain, iface)
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

            switcher = MotionSwitcherClient()
            switcher.SetTimeout(5.0)
            switcher.Init()
            for name in ("normal", "mcf"):
                code, _ = switcher.SelectMode(name)
                print("SelectMode(%s) -> %s" % (name, code))
                if code == 0:
                    return
        except ImportError:
            from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

            rsc = RobotStateClient()
            rsc.SetTimeout(5.0)
            rsc.Init()
            for name in ("mcf", "sport_mode"):
                print("ServiceSwitch(%s, on) -> %s" % (name, rsc.ServiceSwitch(name, True)))


# ---------------------------------------------------------------------------------------
class UdpLatest(object):
    """Newest datagram on a local UDP port, with its age. Non-blocking."""

    def __init__(self, port, size):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.size = size
        self.value = None
        self.t = 0.0

    def poll(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except (BlockingIOError, socket.error):
                break
            arr = np.frombuffer(data, dtype="<f4")
            if arr.shape[0] == self.size and np.all(np.isfinite(arr)):
                self.value = arr.astype(np.float32)
                self.t = time.time()
        return self.value, (time.time() - self.t if self.value is not None else 1e9)


class Runner(object):
    def __init__(self, backend, policy, args, scan_udp=None, cmd_udp=None, log=None, clock=time.time):
        self.b = backend
        self.policy = policy
        self.args = args
        self.scan_udp = scan_udp
        self.cmd_udp = cmd_udp
        self.log = log
        self.clock = clock
        self.remote = Remote()
        self.mode = PASSIVE
        self.mode_steps = 0
        self.stand_from = None
        self.last_action = np.zeros(12, dtype=np.float32)
        self.policy_steps = 0
        self.fault = None
        self.last_print = 0.0
        self.max_cmd = np.array([args.max_vx, args.max_vy, args.max_wz])
        # Ramp limits for the velocity command (per control step): speeding up is gentle,
        # slowing down is twice as quick, so letting go of the stick still stops it promptly.
        self.cmd_now = np.zeros(3)
        self.acc_step = np.array([args.max_acc, args.max_acc, args.max_yaw_acc]) * DT
        self.dec_step = 2.0 * self.acc_step

    # -- inputs ---------------------------------------------------------------------------
    def command(self):
        r = self.remote
        stick = np.array([r.ly * self.args.max_vx, -r.lx * self.args.max_vy, -r.rx * self.args.max_wz])
        if np.abs([r.lx, r.ly, r.rx]).max() > 0.08:
            return self._limit(stick)
        if self.cmd_udp is not None and r.down("R1"):
            value, age = self.cmd_udp.poll()
            if value is not None and age < CMD_STALE_S:
                return self._limit(value.astype(np.float64))
        elif self.cmd_udp is not None:
            self.cmd_udp.poll()  # keep the socket drained
        return np.zeros(3)

    def _ramp(self, target):
        """Move the command toward `target` no faster than the acceleration limits."""
        delta = target - self.cmd_now
        speeding_up = np.abs(target) > np.abs(self.cmd_now)
        step = np.where(speeding_up, self.acc_step, self.dec_step)
        self.cmd_now = self.cmd_now + np.clip(delta, -step, step)
        return self.cmd_now.copy()

    def _limit(self, cmd):
        cmd = np.clip(cmd, -self.max_cmd, self.max_cmd)
        cmd[0] = max(cmd[0], -self.args.max_vx_back)
        cmd[np.abs(cmd) < 0.03] = 0.0
        return cmd

    def scan(self, state):
        """(scan in metres, age in seconds)."""
        if self.scan_udp is None:
            return go2_obs.flat_scan(base_height_from_legs(state["q"], state["quat"])), 0.0
        value, age = self.scan_udp.poll()
        if value is None:
            return go2_obs.flat_scan(base_height_from_legs(state["q"], state["quat"])), age
        return value, age

    # -- one control step ------------------------------------------------------------------
    def step(self):
        state = self.b.read()
        if state is None:
            return
        self.remote.update(state["remote_raw"])
        grav = go2_obs.projected_gravity(state["quat"])

        if self.mode != PASSIVE:
            if state["age"] > STATE_TIMEOUT_S:
                self._to_passive("robot state is %.0f ms old" % (1000 * state["age"]))
            elif grav[2] > MAX_TILT_GZ:
                self._to_passive("tilted too far (gravity z %.2f)" % grav[2])
            elif np.abs(state["dq"]).max() > MAX_JOINT_SPEED:
                self._to_passive("joint speed %.0f rad/s" % np.abs(state["dq"]).max())
        if self.remote.down("L2", "B") and self.mode != PASSIVE:
            self._to_passive("L2+B")

        cmd = np.zeros(3)
        scan_m, scan_age = self.scan(state)
        action = self.last_action
        if self.mode == PASSIVE:
            self.b.send(state["q"], np.zeros(12), np.full(12, PASSIVE_KD))
            if self.remote.down("L2", "up") and not self.args.dry_run and self._too_hot(state, "stand up"):
                pass
            elif self.remote.down("L2", "up") and not self.args.dry_run:
                self.mode, self.mode_steps, self.stand_from = STAND, 0, state["q"].copy()
                self.fault = None
                print("[STAND] crouch, then stand")
        elif self.mode == STAND:
            t = self.mode_steps * DT
            ramp = STAND_RAMP_S
            if t < ramp:
                target = self.stand_from + (CROUCH_POSE - self.stand_from) * (t / ramp)
            elif t < 2 * ramp:
                target = CROUCH_POSE + (STAND_POSE - CROUCH_POSE) * ((t - ramp) / ramp)
            else:
                target = STAND_POSE
            self.b.send(target, STAND_KP, STAND_KD)
            if t >= 2 * ramp + 0.5 and self.remote.down("R2", "A") and self._too_hot(state, "start the policy"):
                pass
            elif t >= 2 * ramp + 0.5 and self.remote.down("R2", "A"):
                self.mode, self.mode_steps = POLICY, 0
                self.cmd_now = np.zeros(3)
                self.last_action = np.zeros(12, dtype=np.float32)
                self.policy_steps = 0
                print("[POLICY] running")

        if self.mode == POLICY or self.args.dry_run:
            cmd = self.command()
            if self.scan_udp is not None and scan_age > SCAN_STALE_S:
                # Stand where it is on the last scan it had. Not PASSIVE: going limp on a
                # staircase is worse than standing on one, and the ground has not moved.
                cmd = np.zeros(3)
            cmd = self._ramp(cmd)
            obs = go2_obs.build_obs(state["gyro"], state["quat"], cmd, self.policy_steps, state["q"],
                                    state["dq"], self.last_action, scan_m)
            action = self.policy.act(obs)
            if self.mode == POLICY:
                target = go2_obs.joint_targets_robot(action).astype(np.float64)
                blend = min(1.0, self.mode_steps / float(GAIN_BLEND_STEPS))
                kp = STAND_KP + (POLICY_KP - STAND_KP) * blend
                kd = STAND_KD + (POLICY_KD - STAND_KD) * blend
                # Keep the commanded torque inside what the simulator's actuators could give.
                reach = TORQUE_LIMIT / kp
                target = np.clip(target, state["q"] - reach, state["q"] + reach)
                target = np.clip(target, Q_MIN, Q_MAX)
                self.b.send(target, kp, kd)
            self.last_action = action
            self.policy_steps += 1

        self.mode_steps += 1
        now = self.clock()
        if self.log is not None:
            row = np.concatenate([[now, {PASSIVE: 0, STAND: 1, POLICY: 2}[self.mode], state["age"], scan_age],
                                  cmd, state["quat"], state["gyro"], state["q"], state["dq"], action,
                                  [float(np.min(scan_m)), float(np.max(scan_m))]])
            self.log.write(row.astype("<f8").tobytes())
        if now - self.last_print >= 1.0:
            self.last_print = now
            self._print(state, grav, cmd, action, scan_m, scan_age)

    def _to_passive(self, why):
        self.mode, self.mode_steps, self.fault = PASSIVE, 0, why
        print("[PASSIVE] %s" % why)

    def _too_hot(self, state, what):
        """True (and say so, at most once a second) if a motor is too hot to `what`."""
        temp = state.get("temp")
        if temp is None or float(np.max(temp)) <= START_MAX_TEMP_C:
            return False
        now = self.clock()
        if now - getattr(self, "_hot_said", -1e9) >= 1.0:
            self._hot_said = now
            print("[REFUSED] will not %s: %s at %.0f C (limit %d C). Let it cool, with the robot off." % (
                what, MOTOR_NAMES[int(np.argmax(temp))], float(np.max(temp)), START_MAX_TEMP_C))
        return True

    def _print(self, state, grav, cmd, action, scan_m, scan_age):
        np.set_printoptions(precision=2, suppress=True, linewidth=160)
        temp = state.get("temp")
        if temp is not None:
            hot = int(np.argmax(temp))
            warn = "  << HOT: consider L2+B and cooling" if temp[hot] > WARN_TEMP_C else ""
            print("    hottest motor: %s %.0f C%s" % (MOTOR_NAMES[hot], temp[hot], warn))
        off = (state["q"][go2_obs.ROBOT_TO_SIM] - go2_obs.DEFAULT_JOINT_POS)
        print("%s cmd %s | gravity %s | gyro %s | base height (legs) %.3f m | scan %.2f..%.2f m age %.2f s | "
              "state age %.0f ms" % (self.mode, cmd, grav, state["gyro"], base_height_from_legs(state["q"], state["quat"]),
                                     float(np.min(scan_m)), float(np.max(scan_m)), scan_age, 1000 * state["age"]))
        if self.args.dry_run or self.args.verbose:
            print("    joints - default, FL FR RL RR: %s" % off.reshape(4, 3).round(2).tolist())
            print("    action, FL FR RL RR:           %s" % np.asarray(action).reshape(4, 3).round(2).tolist())


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def run(runner, backend):
    """50 Hz loop until Ctrl-C. Always ends damped.

    A dropped SSH session sends SIGHUP and `kill` sends SIGTERM; both would otherwise end the
    process without the damping below, leaving the motors on their last stiff command. Both
    are treated as Ctrl-C. (Still start it inside tmux or screen on the robot, so a dropped
    cable does not stop a run that is going well.)"""
    signal.signal(signal.SIGHUP, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    next_t = time.time()
    try:
        while True:
            runner.step()
            next_t += DT
            late = time.time() - next_t
            if late > LOOP_LATE_S:
                if runner.mode != PASSIVE:
                    runner._to_passive("control loop %.0f ms late" % (1000 * late))
                next_t = time.time()
            elif late < 0:
                time.sleep(-late)
    except KeyboardInterrupt:
        print("\nCtrl-C: damping")
    finally:
        end = time.time() + 2.0
        while time.time() < end:
            state = backend.read()
            if state is not None:
                backend.send(state["q"], np.zeros(12), np.full(12, PASSIVE_KD))
            time.sleep(DT)
        if hasattr(backend, "close"):
            backend.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", help="Policy exported by scripts/export_policy_numpy.py.")
    ap.add_argument("--iface", default="eth0", help="Network interface that reaches the robot's motor board.")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain: 0 is the robot. Simulators use 1.")
    ap.add_argument("--write-hz", type=float, default=500.0,
                    help="How often the current command is re-sent to the motors. 0: only once per control "
                         "step (50 Hz), with no watchdog thread; for a CPU that cannot keep up.")
    ap.add_argument("--no-motion-service", action="store_true",
                    help="Simulator only (refused on domain 0): there is no robot controller to release.")
    ap.add_argument("--dry-run", action="store_true", help="Send nothing. Print what the policy sees and would do.")
    ap.add_argument("--scan", choices=("flat", "udp"), default="flat",
                    help="flat: assume level ground at the height the legs report. udp: from go2_scan_node.py.")
    ap.add_argument("--scan-port", type=int, default=9871)
    ap.add_argument("--cmd", choices=("remote", "udp"), default="remote",
                    help="udp: also accept (vx, vy, wz) float32 datagrams, used only while R1 is held.")
    ap.add_argument("--cmd-port", type=int, default=9870)
    # Deliberately slow defaults (2026-10-09, first runs on the robot). In simulation stairs v8b
    # crosses single 15/17 cm steps and 10-step flights at 0.3 m/s, and stalls on some 10-step
    # 17 cm descents at 0.2 m/s; the policies were trained up to 1.0 m/s.
    ap.add_argument("--max-vx", type=float, default=0.3, help="Forward speed limit (m/s) at full stick.")
    ap.add_argument("--max-vx-back", type=float, default=0.15)
    ap.add_argument("--max-vy", type=float, default=0.15)
    ap.add_argument("--max-wz", type=float, default=0.5, help="Turn rate limit (rad/s).")
    ap.add_argument("--max-acc", type=float, default=0.3, help="Speeding up, m/s per second (slowing down: twice this).")
    ap.add_argument("--max-yaw-acc", type=float, default=0.6, help="Turn-rate change, rad/s per second (slowing: twice).")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--yes", action="store_true", help="Skip the typed confirmation (not for first runs).")
    ap.add_argument("--restore-motion-service", action="store_true",
                    help="Only switch the robot's own controller back on, then exit.")
    args = ap.parse_args()

    if args.restore_motion_service:
        Go2Backend.restore_motion_service(args.iface, args.domain)
        return
    if args.no_motion_service and args.domain == 0:
        ap.error("--no-motion-service is for a simulator on another DDS domain, never the robot (domain 0)")
    if not args.npz:
        ap.error("--npz is required")
    policy = NumpyPolicy(args.npz)
    if policy.obs_dim != go2_obs.OBS_DIM:
        raise SystemExit("policy expects %d inputs, this runner builds %d" % (policy.obs_dim, go2_obs.OBS_DIM))

    backend = Go2Backend(args.iface, args.dry_run, args.domain, args.write_hz)
    print("waiting for robot state on %s ..." % args.iface)
    t0 = time.time()
    while backend.read() is None:
        if time.time() - t0 > 10.0:
            raise SystemExit("no rt/lowstate in 10 s: wrong --iface, or the robot is off")
        time.sleep(0.05)

    if args.dry_run:
        print("DRY RUN: nothing is sent to the robot.")
    else:
        print("\nThis will switch OFF the robot's own controller and drive the motors from this program.")
        print("The robot will lie down first. Keep it on the floor or hung with its legs free,")
        print("clear of people, with someone holding the remote (L2+B = motors limp).")
        if not args.yes and input("Type 'yes' to continue: ").strip().lower() != "yes":
            raise SystemExit("not confirmed")
        if not args.no_motion_service and not backend.release_motion_service():
            raise SystemExit("could not confirm the robot's own controller is off: not sending anything")
        print("Robot's controller is off. PASSIVE. L2+Up: stand. Then R2+A: policy. L2+B: limp.")

    scan_udp = UdpLatest(args.scan_port, go2_obs.SCAN_NX * go2_obs.SCAN_NY) if args.scan == "udp" else None
    cmd_udp = UdpLatest(args.cmd_port, 3) if args.cmd == "udp" else None
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_%s.f8" % time.strftime("%Y%m%d_%H%M%S"))
    with open(log_path, "wb") as log:
        print("logging to %s (float64 rows of %d)" % (log_path, 4 + 3 + 4 + 3 + 12 + 12 + 12 + 2))
        run(Runner(backend, policy, args, scan_udp, cmd_udp, log), backend)


if __name__ == "__main__":
    main()
