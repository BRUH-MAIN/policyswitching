"""Find where the Go2's firmware LiDAR sits on the body: R_bl, t_bl (LiDAR frame -> base frame).

Input: a recording made on the robot WITH its own motion service running and the robot
standing still, holding `rt/utlidar/cloud` (raw points, LiDAR frame), `rt/utlidar/cloud_deskewed`
(the same scene in the odometry frame) and `rt/utlidar/robot_odom` (base pose in that frame);
see `eval_results/robot_probe/lidar_calib_rec.pkl` and how it was recorded (session notes,
2026-10-09). The deskewed points are moved into the base frame with the odometry pose, and the
raw points are aligned to them by trimmed point-to-point ICP, started from several headings
around "upside down, at the front" (the LiDAR hangs under the nose). Writes
`deploy_numpy/go2_lidar_calib.npz` for `go2_scan_node.py --source raw`.

  python scripts/calib_go2_lidar.py eval_results/robot_probe/lidar_calib_rec.pkl --out deploy_numpy/go2_lidar_calib.npz
  python scripts/calib_go2_lidar.py --selftest
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
from scipy.spatial import cKDTree


def cloud_xyz(c) -> np.ndarray:
  p = c["data"][:, :12].copy().view("<f4").reshape(-1, 3).astype(np.float64)
  return p[np.all(np.isfinite(p), axis=1) & np.any(p != 0, axis=1)]


def voxel(p: np.ndarray, size: float) -> np.ndarray:
  k = np.floor(p / size).astype(np.int64)
  _, idx = np.unique(k, axis=0, return_index=True)
  return p[idx]


def quat_to_rot(w, x, y, z) -> np.ndarray:
  n = np.sqrt(w * w + x * x + y * y + z * z)
  w, x, y, z = w / n, x / n, y / n, z / n
  return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot_z(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def icp(src: np.ndarray, dst: np.ndarray, R: np.ndarray, t: np.ndarray, iters: int = 60, keep: float = 0.7):
  """Align src to dst (dst ~ R src + t). Returns R, t, median residual of the kept pairs."""
  tree = cKDTree(dst)
  for _ in range(iters):
    moved = src.dot(R.T) + t
    d, j = tree.query(moved)
    sel = d <= np.quantile(d, keep)
    a, b = src[sel], dst[j[sel]]
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T.dot(b - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T.dot(U.T)))])
    R = Vt.T.dot(D).dot(U.T)
    t = cb - R.dot(ca)
  d, _ = tree.query(src.dot(R.T) + t)
  return R, t, float(np.median(d))


def solve(raw_l: np.ndarray, desk_b: np.ndarray):
  """Best of several starts; raw_l in the LiDAR frame, desk_b in the base frame."""
  upside_down = np.diag([1.0, -1.0, -1.0])
  best = None
  for yaw in np.radians(np.arange(-180, 180, 30)):
    for flip in (upside_down, np.eye(3)):
      R0 = rot_z(yaw).dot(flip)
      R, t, res = icp(raw_l, desk_b, R0, np.array([0.28, 0.0, -0.05]))
      if best is None or res < best[2]:
        best = (R, t, res)
  return best


def selftest() -> None:
  rng = np.random.default_rng(0)
  floor = np.c_[rng.uniform(-1, 3, 4000), rng.uniform(-2, 2, 4000), np.full(4000, -0.32)]
  box = np.c_[rng.uniform(1.0, 1.4, 1500), rng.uniform(0.3, 0.8, 1500), rng.uniform(-0.32, 0.1, 1500)]
  wall = np.c_[np.full(1500, 2.8), rng.uniform(-2, 2, 1500), rng.uniform(-0.32, 0.6, 1500)]
  scene_b = np.vstack([floor, box, wall])
  R_true = rot_z(np.radians(3.0)).dot(np.diag([1.0, -1.0, -1.0]))
  R_true = R_true.dot(quat_to_rot(np.cos(0.05), np.sin(0.05), 0, 0))
  t_true = np.array([0.29, 0.01, -0.04])
  raw_l = (scene_b - t_true).dot(R_true) + rng.normal(0, 0.01, scene_b.shape)
  R, t, res = solve(voxel(raw_l, 0.02), scene_b)
  ang = np.degrees(np.arccos(np.clip((np.trace(R.T.dot(R_true)) - 1) / 2, -1, 1)))
  print(f"selftest: rotation error {ang:.2f} deg, offset error {np.round((t - t_true) * 100, 1).tolist()} cm, residual {res * 100:.1f} cm")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("recording", nargs="?")
  ap.add_argument("--out", default="deploy_numpy/go2_lidar_calib.npz")
  ap.add_argument("--selftest", action="store_true")
  args = ap.parse_args()
  if args.selftest:
    selftest()
    return
  rec = pickle.load(open(args.recording, "rb"))
  if not rec["desk"] or not rec["odom"]:
    raise SystemExit("no corrected cloud / odometry in this recording: record with the robot's own controller on")
  o = rec["odom"][len(rec["odom"]) // 2]
  t_ob = np.array(o[2:5]); R_ob = quat_to_rot(*o[5:9])
  poses = np.array([r[2:5] for r in rec["odom"]])
  print(f"odometry moved {np.ptp(poses, axis=0).round(3).tolist()} m during the recording (should be ~0)")
  desk_w = np.vstack([cloud_xyz(c) for c in rec["desk"]])
  desk_b = voxel((desk_w - t_ob).dot(R_ob), 0.02)
  raw_l = voxel(np.vstack([cloud_xyz(c) for c in rec["raw"]]), 0.02)
  raw_l = raw_l[np.linalg.norm(raw_l, axis=1) > 0.12]
  print(f"points after 2 cm voxels: raw {len(raw_l)}, corrected {len(desk_b)}")
  R, t, res = solve(raw_l, desk_b)
  rpy = np.degrees([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(R[2, 0]), np.arctan2(R[1, 0], R[0, 0])])
  print(f"R_bl (LiDAR -> base): roll/pitch/yaw {np.round(rpy, 2).tolist()} deg; t_bl {np.round(t, 3).tolist()} m; "
        f"median residual {res * 100:.1f} cm")
  np.savez(args.out, R_bl=R, t_bl=t, residual_m=res)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
