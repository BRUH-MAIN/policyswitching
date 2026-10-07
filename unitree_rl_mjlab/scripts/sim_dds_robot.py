"""A pretend Go2 on the network, for running deploy_numpy/go2_runner.py unchanged on the laptop.

Publishes `rt/lowstate` and listens to `rt/lowcmd` over DDS (unitree_sdk2py) on DDS domain 1,
interface `lo`, the way the robot's motor board does on domain 0, with a CPU MuJoCo Go2 on a
staircase course behind it, stepped in real time. The remote is scripted (L2+Up at 3 s,
R2+A at 8 s, stick forward from 10 s) and the height scan is sent to the runner's UDP port
as go2_scan_node.py would. So the runner process under test is exactly the program that
will run on the Jetson, DDS, CRC and the 500 Hz writer included:

  # terminal 1 (needs unitree_sdk2py + cyclonedds in the environment)
  python scripts/sim_dds_robot.py --seconds 40
  # terminal 2
  python deploy_numpy/go2_runner.py --npz deploy_numpy/<name>.npz --iface lo --domain 1 \
      --no-motion-service --yes --scan udp

It does not imitate the robot's motion service, so the runner needs --no-motion-service,
which the runner refuses on domain 0.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import test_go2_runner_sim as T  # noqa: E402  (also puts deploy_numpy on the path)
import go2_obs  # noqa: E402
import go2_runner  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--step-height", type=float, default=0.12)
  ap.add_argument("--speed", type=float, default=0.5)
  ap.add_argument("--max-vx", type=float, default=0.6, help="Must match the runner's --max-vx.")
  ap.add_argument("--seconds", type=float, default=40.0)
  ap.add_argument("--scan-port", type=int, default=9871)
  args = ap.parse_args()

  from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
  from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
  from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
  from unitree_sdk2py.utils.crc import CRC

  model, qpos0 = T.build_model(args.step_height)
  robot = T.SimRobot(model, qpos0)
  # Start as the real robot is after it has lain down: legs folded, belly near the floor,
  # motors damped. (Left standing with no command it would topple onto its nose, which is
  # not a pose the robot is ever started from.)
  robot.d.qpos[robot.qadr] = go2_runner.CROUCH_POSE  # the same three angles for every leg
  robot.d.qpos[robot.fq + 2] -= 0.20
  mujoco.mj_forward(model, robot.d)
  lying_damped = (np.zeros(12), np.zeros(12), np.full(12, go2_runner.PASSIVE_KD))
  ray = T.RayScan(robot)
  feet_geoms = [model.geom(("robot/" if not T._exists(model, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_foot_collision") else "")
                           + f"{f}_foot_collision").id for f in T.ROBOT_FEET]

  ChannelFactoryInitialize(1, "lo")
  pub = ChannelPublisher("rt/lowstate", LowState_)
  pub.Init()
  crc = CRC()
  got = {"n": 0, "bad_crc": 0, "cmd": None}

  def on_cmd(msg):
    got["n"] += 1
    if crc.Crc(msg) != msg.crc:
      got["bad_crc"] += 1
      return
    got["cmd"] = msg

  sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
  sub.Init(on_cmd, 10)
  scan_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  state = unitree_go_msg_dds__LowState_()

  r2s = go2_obs.ROBOT_TO_SIM
  start = robot.base_pos()
  top_z, k = start[2], 0
  dt = model.opt.timestep
  t0 = time.time()
  print("pretend Go2 up on DDS domain 1 (lo). Start the runner now.", flush=True)
  while True:
    t = k * dt
    if t >= args.seconds:
      break
    if 3.0 <= t < 3.3:
      robot.set_remote(keys=("L2", "up"))
    elif 8.0 <= t < 8.3:
      robot.set_remote(keys=("R2", "A"))
    elif t >= 10.0:
      robot.set_remote(ly=args.speed / args.max_vx)
    else:
      robot.set_remote()

    cmd = got["cmd"]
    if cmd is None:
      robot.send(*lying_damped)
    else:
      q = np.array([cmd.motor_cmd[i].q for i in range(12)])
      kp = np.array([cmd.motor_cmd[i].kp for i in range(12)])
      kd = np.array([cmd.motor_cmd[i].kd for i in range(12)])
      robot.send(np.where(np.abs(q) < 100.0, q, 0.0), kp, kd)  # 2.146e9 means "no position target"
    mujoco.mj_step(model, robot.d)

    s = robot.read()
    for i in range(12):
      state.motor_state[i].q = float(s["q"][i])
      state.motor_state[i].dq = float(s["dq"][i])
    for i in range(4):
      state.imu_state.quaternion[i] = float(s["quat"][i])
    for i in range(3):
      state.imu_state.gyroscope[i] = float(s["gyro"][i])
    touching = set()
    for c in robot.d.contact[:robot.d.ncon]:
      touching.update((c.geom1, c.geom2))
    for i in range(4):
      state.foot_force[i] = 60 if feet_geoms[i] in touching else 0
    state.wireless_remote = list(bytes(robot.remote))
    pub.Write(state)
    if k % 4 == 0:
      scan_sock.sendto(ray.poll()[0].astype("<f4").tobytes(), ("127.0.0.1", args.scan_port))

    top_z = max(top_z, robot.base_pos()[2])
    k += 1
    if k % int(2.0 / dt) == 0:
      p = robot.base_pos()
      print(f"t {t:5.1f} s | x {p[0] - start[0]:+.2f} m, height above start {p[2] - start[2]:+.2f} m | "
            f"lowcmd received {got['n']} ({got['bad_crc']} bad CRC)", flush=True)
    ahead = t0 + k * dt - time.time()
    if ahead > 0:
      time.sleep(ahead)

  end = robot.base_pos()
  print(f"\nreceived {got['n']} lowcmd messages, {got['bad_crc']} with a bad CRC, "
        f"{got['n'] / args.seconds:.0f} per second")
  print(f"walked {end[0] - start[0]:.2f} m forward, climbed {top_z - start[2]:.2f} m "
        f"(the flight is {5 * args.step_height:.2f} m)")


if __name__ == "__main__":
  main()
