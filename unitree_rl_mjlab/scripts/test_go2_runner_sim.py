"""Run the robot's control program (deploy_numpy/go2_runner.py) against plain CPU MuJoCo.

The same Runner class that will run on the Jetson is driven here by a simulated robot that
speaks its interface: joint state in the ROBOT's motor order, an IMU quaternion and gyro,
the remote's raw bytes, and position targets with gains coming back. Nothing from mjlab is
in the loop except the compiled model (robot + a staircase course). The remote is scripted:

    0.5 s   L2+Up          PASSIVE -> STAND (crouch, stand)
    4.0 s   R2+A           STAND -> POLICY
    5.0 s   left stick     walk forward at --speed until the end

and the height scan is cast against the terrain with mj_ray, as the scan node will supply it
on the robot (--scan flat instead feeds the leg-kinematics flat-ground scan, which is what
the robot's first walking test will use, and should fail at the stairs).

It checks the leg forward kinematics used for the base height against the model's foot
sites, then reports how far the robot got and how high it climbed. This is a different
physics backend (CPU MuJoCo, nominal friction, no pushes) from the one the policy trained
in, so it also says something about transfer, though only a little.

  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/test_go2_runner_sim.py --npz deploy_numpy/stairs_v5a.npz
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy_numpy"))

import go2_obs  # noqa: E402
import go2_runner  # noqa: E402
from policy_numpy import NumpyPolicy  # noqa: E402

SIM_JOINTS = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]
ROBOT_FEET = ("FR", "FL", "RR", "RL")
SUBSTEPS = 4  # 4 x 5 ms = one 20 ms control step, as in training


def build_model(step_height: float):
  """The compiled MuJoCo model of the follow-task course (robot + flat approach + 5-step flight)."""
  from mjlab.envs import ManagerBasedRlEnv

  from src.vlm_nav.course import saro_courses
  from src.vlm_nav.twin_env import make_twin_env_cfg

  course = saro_courses(visual="plain", level="L2")["stairs_up"]
  course = replace(course, segments=tuple(
    replace(g, step_height=step_height) if g.kind.startswith("stairs") else g for g in course.segments))
  cfg = make_twin_env_cfg(None, terrain_generator=course.generator_cfg(seed=0), num_envs=1, seed=0, terminations="saro")
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  env.reset()
  model = env.sim.mj_model
  qpos0 = env.sim.data.qpos[0].cpu().numpy().copy()
  env.close()
  # Distance from the spawn point to the far end of the course, where the floor stops.
  build_model.run_out_m = sum(g.length for g in course.segments) - course.start_x
  return model, qpos0


class SimRobot:
  """A Go2 in CPU MuJoCo behind the interface go2_runner.Runner expects of a backend."""

  def __init__(self, model, qpos0):
    self.m, self.d = model, mujoco.MjData(model)
    self.d.qpos[:] = qpos0
    name = lambda n: n if _exists(model, mujoco.mjtObj.mjOBJ_JOINT, n) else "robot/" + n  # noqa: E731
    self.qadr = np.array([model.joint(name(j)).qposadr[0] for j in SIM_JOINTS])
    self.dadr = np.array([model.joint(name(j)).dofadr[0] for j in SIM_JOINTS])
    base = "base_link" if _exists(model, mujoco.mjtObj.mjOBJ_BODY, "base_link") else "robot/base_link"
    self.base_id = model.body(base).id
    free = model.body(base).jntadr[0]
    self.fq, self.fv = model.jnt_qposadr[free], model.jnt_dofadr[free]
    # The model's own position actuators do the PD, as in training; their gains are rewritten
    # on every command so that STAND and POLICY can use different ones, as on the robot.
    joint_ids = [model.joint(name(j)).id for j in SIM_JOINTS]
    self.act = np.array([int(np.where(model.actuator_trnid[:, 0] == j)[0][0]) for j in joint_ids])
    self.remote = bytearray(40)
    self.sent = 0
    mujoco.mj_forward(model, self.d)

  def set_remote(self, keys=(), lx=0.0, ly=0.0, rx=0.0):
    word = 0
    for k in keys:
      word |= 1 << go2_runner.KEY[k]
    self.remote[2:4] = struct.pack("<H", word)
    self.remote[4:8] = struct.pack("<f", lx)
    self.remote[8:12] = struct.pack("<f", rx)
    self.remote[20:24] = struct.pack("<f", ly)

  def read(self):
    r2s = go2_obs.ROBOT_TO_SIM
    return dict(q=self.d.qpos[self.qadr][r2s].copy(), dq=self.d.qvel[self.dadr][r2s].copy(),
                quat=self.d.qpos[self.fq + 3:self.fq + 7].copy(), gyro=self.d.qvel[self.fv + 3:self.fv + 6].copy(),
                remote_raw=bytes(self.remote), age=0.0)

  def send(self, q_des, kp, kd):
    r2s = go2_obs.ROBOT_TO_SIM  # its own inverse
    kp, kd = np.asarray(kp, dtype=np.float64)[r2s], np.asarray(kd, dtype=np.float64)[r2s]
    self.m.actuator_gainprm[self.act, 0] = kp
    self.m.actuator_biasprm[self.act, 1] = -kp
    self.m.actuator_biasprm[self.act, 2] = -kd
    self.d.ctrl[self.act] = np.asarray(q_des, dtype=np.float64)[r2s]
    self.sent += 1

  def advance(self):
    for _ in range(SUBSTEPS):
      mujoco.mj_step(self.m, self.d)

  def base_pos(self):
    return self.d.xpos[self.base_id].copy()


class RayScan:
  """What go2_scan_node.py will send: terrain height under the 187 scan points, by ray cast."""

  def __init__(self, robot: SimRobot):
    self.r = robot
    self.pts = go2_obs.scan_points_xy()
    self.groups = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    self.geomid = np.zeros(1, dtype=np.int32)

  def poll(self):
    d, m = self.r.d, self.r.m
    base = self.r.base_pos()
    yaw = go2_obs.yaw_of(d.qpos[self.r.fq + 3:self.r.fq + 7])
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.empty(len(self.pts), dtype=np.float32)
    down = np.array([0.0, 0.0, -1.0])
    for k, (px, py) in enumerate(self.pts):
      origin = base + np.array([c * px - s * py, s * px + c * py, 0.0])
      dist = mujoco.mj_ray(m, d, origin, down, self.groups, 1, -1, self.geomid)
      out[k] = dist if self.geomid[0] >= 0 else go2_obs.SCAN_MISS
    return out, 0.0


class MapScan:
  """The scan as the robot will build it: a gridded height map of the whole course (what a
  LiDAR mapper would hold), 6 cm cells, with noise, holes, a vertical offset between the
  map and the pose that drifts, and the pose given for a point 7 cm above base_link. It goes
  through deploy_numpy/go2_scan.HeightMapScan, the code that will run on the robot, with the
  message layout that code assumes."""

  RES, HALF = 0.06, 12.0

  def __init__(self, robot: SimRobot, noise=0.01, holes=0.15, drift=0.002, seed=0):
    import go2_scan

    self.r = robot
    self.rng = np.random.default_rng(seed)
    m, d = robot.m, robot.d
    n = int(2 * self.HALF / self.RES)
    base = robot.base_pos()
    self.origin = np.array([base[0] - 4.0, base[1] - self.HALF])
    groups = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    geomid = np.zeros(1, dtype=np.int32)
    grid = np.full((n, n), 1.0e9, dtype=np.float32)  # [iy, ix]
    top = 5.0
    for iy in range(n):
      for ix in range(n):
        x, y = self.origin[0] + ix * self.RES, self.origin[1] + iy * self.RES
        dist = mujoco.mj_ray(m, d, np.array([x, y, top]), np.array([0.0, 0.0, -1.0]), groups, 1, -1, geomid)
        if geomid[0] >= 0:
          grid[iy, ix] = top - dist
    self.map_offset = 0.37  # the map frame's z is not the world's
    valid = grid < 1e8
    grid[valid] += self.map_offset + self.rng.normal(0.0, noise, size=int(valid.sum())).astype(np.float32)
    grid[self.rng.random(grid.shape) < holes] = 1.0e9
    self.sampler = go2_scan.HeightMapScan()
    self.sampler.set_map(grid.T.ravel(order="F"), n, n, self.RES, self.origin)  # data[ix + width * iy]
    self.pose_up = 0.07
    self.drift, self.z_err = drift, 0.0
    self.feet_geoms = [m.geom(("robot/" if not _exists(m, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_foot_collision") else "")
                              + f"{f}_foot_collision").id for f in ROBOT_FEET]

  def poll(self):
    d = self.r.d
    state = self.r.read()
    self.z_err += self.drift * go2_runner.DT  # the pose's z creeps: 2 mm/s by default
    base = self.r.base_pos()
    pose = np.array([base[0], base[1], base[2] + self.map_offset + self.pose_up + self.z_err])
    yaw = go2_obs.yaw_of(state["quat"])
    touching = set()
    for c in d.contact[:d.ncon]:
      touching.update((c.geom1, c.geom2))
    stance = np.array([g in touching for g in self.feet_geoms])
    up_b = -go2_obs.projected_gravity(state["quat"]).astype(np.float64)
    self.sampler.update_anchor(pose, yaw, go2_runner.foot_positions_base(state["q"]), stance, up_b)
    scan, _ = self.sampler.scan(pose, yaw, go2_runner.base_height_from_legs(state["q"], state["quat"]))
    return scan, 0.0


class CloudScan:
  """The scan as go2_scan_node.py --source cloud builds it on this robot: a head-mounted LiDAR
  looking forward and down returns points (here ray casts against the terrain, with range
  noise, ~16 scans a second, everything in a drifting "odom" frame), go2_scan.CloudGrid grids
  them, and HeightMapScan reads the grid, anchored on the loaded feet."""

  def __init__(self, robot: SimRobot, noise=0.015, drift=0.002, seed=0):
    import go2_scan

    self.r = robot
    self.rng = np.random.default_rng(seed)
    az = np.radians(np.linspace(-70, 70, 29))
    el = np.radians(np.linspace(-12, -80, 18))
    A, E = np.meshgrid(az, el)
    self.dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1).reshape(-1, 3)
    self.noise, self.drift, self.z_err = noise, drift, 0.0
    self.odom_off = np.array([1.7, -0.6, 0.37])  # the odom frame is not the world frame
    self.grid = go2_scan.CloudGrid()
    self.sampler = go2_scan.HeightMapScan()
    self.groups = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    self.geomid = np.zeros(1, dtype=np.int32)
    self.k = 0
    m = robot.m
    self.feet_geoms = [m.geom(("robot/" if not _exists(m, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_foot_collision") else "")
                              + f"{f}_foot_collision").id for f in ROBOT_FEET]

  def poll(self):
    d, m = self.r.d, self.r.m
    state = self.r.read()
    self.z_err += self.drift * go2_runner.DT
    base = self.r.base_pos()
    rot = d.xmat[self.r.base_id].reshape(3, 3)
    pose = base + self.odom_off + np.array([0.0, 0.0, self.z_err])
    if self.k % 3 == 0:
      head = base + rot.dot(np.array([0.28, 0.0, 0.05]))
      pts = []
      for v in self.dirs:
        dw = rot.dot(v)
        dist = mujoco.mj_ray(m, d, head, dw, self.groups, 1, -1, self.geomid)
        if self.geomid[0] >= 0 and dist < 4.0:
          pts.append(head + dw * (dist + self.rng.normal(0.0, self.noise)))
      if pts:
        self.grid.add_points(np.array(pts) + self.odom_off + np.array([0.0, 0.0, self.z_err]), pose)
        self.sampler.set_map(*self.grid.as_map())
    self.k += 1
    yaw = go2_obs.yaw_of(state["quat"])
    touching = set()
    for c in d.contact[:d.ncon]:
      touching.update((c.geom1, c.geom2))
    stance = np.array([g in touching for g in self.feet_geoms])
    up_b = -go2_obs.projected_gravity(state["quat"]).astype(np.float64)
    self.sampler.update_anchor(pose, yaw, go2_runner.foot_positions_base(state["q"]), stance, up_b)
    scan, _ = self.sampler.scan(pose, yaw, go2_runner.base_height_from_legs(state["q"], state["quat"]))
    return scan, 0.0


def _exists(model, kind, name) -> bool:
  return mujoco.mj_name2id(model, kind, name) >= 0


def check_leg_kinematics(robot: SimRobot) -> float:
  """Largest error (m) of go2_runner.foot_positions_base against the model's foot sites."""
  d, m = robot.d, robot.m
  mujoco.mj_kinematics(m, d)  # mj_step leaves body positions one substep behind qpos
  rot = d.xmat[robot.base_id].reshape(3, 3)
  ours = go2_runner.foot_positions_base(robot.read()["q"])
  worst = 0.0
  for i, foot in enumerate(ROBOT_FEET):
    site = foot if _exists(m, mujoco.mjtObj.mjOBJ_SITE, foot) else "robot/" + foot
    truth = rot.T.dot(d.site_xpos[m.site(site).id] - d.xpos[robot.base_id])
    worst = max(worst, float(np.abs(ours[i] - truth).max()))
  return worst


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--npz", required=True)
  ap.add_argument("--step-height", type=float, default=0.12)
  ap.add_argument("--speed", type=float, default=0.5)
  ap.add_argument("--max-vx", type=float, default=0.3, help="Runner's forward speed limit (full stick).")
  ap.add_argument("--max-acc", type=float, default=0.3, help="Runner's acceleration limit (m/s per second).")
  ap.add_argument("--seconds", type=float, default=30.0)
  ap.add_argument("--scan", choices=("ray", "map", "cloud", "flat"), default="ray",
                  help="ray: exact terrain heights. map: through go2_scan.HeightMapScan from a noisy gridded "
                       "map with holes and vertical drift. flat: the leg-kinematics flat-ground scan.")
  args = ap.parse_args()

  model, qpos0 = build_model(args.step_height)
  robot = SimRobot(model, qpos0)
  policy = NumpyPolicy(args.npz)
  run_args = SimpleNamespace(dry_run=False, verbose=False, max_acc=args.max_acc, max_yaw_acc=0.6, max_vx=args.max_vx, max_vx_back=0.3, max_vy=0.3, max_wz=0.8)
  sim_t = [0.0]
  ray = RayScan(robot)
  runner = go2_runner.Runner(robot, policy, run_args, scan_udp={"ray": RayScan, "map": MapScan, "cloud": CloudScan, "flat": lambda _: None}[args.scan](robot),
                             clock=lambda: sim_t[0])

  fk_worst = 0.0
  height_err = []
  start = robot.base_pos()
  top_z, modes, fell = start[2], [], False
  scan_err = []
  for k in range(int(args.seconds / go2_runner.DT)):
    t = k * go2_runner.DT
    sim_t[0] = t
    if 0.5 <= t < 0.7:
      robot.set_remote(keys=("L2", "up"))
    elif 5.0 <= t < 5.2:
      robot.set_remote(keys=("R2", "A"))
    elif t >= 6.0:
      robot.set_remote(ly=min(1.0, args.speed / run_args.max_vx))
    else:
      robot.set_remote()
    runner.step()
    if args.scan in ("map", "cloud") and runner.mode == go2_runner.POLICY and k % 5 == 0:
      scan_err.append(np.abs(runner.scan(robot.read())[0] - ray.poll()[0]))
    robot.advance()
    if k % 10 == 0:
      fk_worst = max(fk_worst, check_leg_kinematics(robot))
    pos = robot.base_pos()
    if runner.mode == go2_runner.POLICY and pos[0] - start[0] < 1.0 and t > 6.0:
      # Still on the flat approach: how good is the leg-kinematics base height?
      state = robot.read()
      true_height = ray.poll()[0][(go2_obs.SCAN_NY // 2) * go2_obs.SCAN_NX + go2_obs.SCAN_NX // 2]
      height_err.append(go2_runner.base_height_from_legs(state["q"], state["quat"]) - true_height)
    top_z = max(top_z, pos[2])
    if not modes or modes[-1][1] != runner.mode:
      modes.append((round(t, 2), runner.mode))
    if pos[0] - start[0] > build_model.run_out_m - 0.8:
      break  # close to the end of the course: stop before it walks off the edge
    if runner.fault and runner.mode == go2_runner.PASSIVE and t > 5.0:
      fell = True
      break

  end = robot.base_pos()
  print(f"\nmode history: {modes}")
  print(f"leg kinematics vs the model's foot sites: worst {1000 * fk_worst:.2f} mm")
  if height_err:
    e = 1000 * np.array(height_err)
    print(f"base height from the legs minus true height, walking on the flat approach: "
          f"mean {e.mean():+.1f} mm, sd {e.std():.1f} mm, worst {np.abs(e).max():.1f} mm ({len(e)} steps)")
  if scan_err:
    e = 100 * np.array(scan_err)
    print(f"map scan vs exact terrain height: median error {np.median(e):.1f} cm, 90th percentile "
          f"{np.percentile(e, 90):.1f} cm, cells off by more than 6 cm {100 * (e > 6).mean():.1f}%; "
          f"sampler {runner.scan_udp.sampler.stats}")
  print(f"walked {end[0] - start[0]:.2f} m forward, {abs(end[1] - start[1]):.2f} m sideways; "
        f"climbed {top_z - start[2]:.2f} m (the flight is {5 * args.step_height:.2f} m); "
        f"{'FAULT: ' + str(runner.fault) if fell else 'no fault'}")
  if fk_worst > 1e-3:
    raise SystemExit("leg kinematics do not match the model")


if __name__ == "__main__":
  main()
