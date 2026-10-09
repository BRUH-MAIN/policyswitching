#!/usr/bin/env python3
# Created by Claude (policyswitching repo, deploy_numpy/go2_scan_node.py).
# Purpose: read the Go2's own LiDAR height map and pose (read-only) and send the locomotion
# policy's 187-point height scan to go2_runner.py over local UDP. Sends nothing to the robot.
"""Height scan for the policy from the Go2's built-in LiDAR map. Read-only on the robot.

    python3 go2_scan_node.py --probe            # what is published, how fast, what it holds
    python3 go2_scan_node.py                    # run: scan to 127.0.0.1:9871 at 50 Hz

Probe on the robot, 2026-10-09: `rt/utlidar/height_map_array` is silent, so the default source
is now `cloud`: the firmware LiDAR's `rt/utlidar/cloud_deskewed` (odom frame, ~15 Hz) gridded
by go2_scan.CloudGrid, with the pose from `rt/utlidar/robot_odom` (150 Hz, same frame).

What the map source assumed, kept for reference:
  * rt/utlidar/height_map_array carries a HeightMap (128 x 128 cells of 6 cm, heights in a
    fixed odometry frame, 1e9 where empty), and
  * a pose in the same frame is published on rt/utlidar/robot_pose (or robot_odom, or
    rt/sportmodestate), and
  * both keep coming after the robot's own motion service is released for low-level control.
The third is the one most likely to fail; run --probe once in normal mode and once while
go2_runner.py holds the robot in PASSIVE.

The vertical offset between pose and map is not assumed: go2_scan.HeightMapScan estimates
it from the feet that carry load (foot force above --stance-force). --probe prints the
foot forces so that threshold can be set from what the robot actually reports.
"""

from __future__ import print_function

import argparse
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import go2_obs  # noqa: E402
import go2_scan  # noqa: E402
from go2_runner import base_height_from_legs, foot_positions_base  # noqa: E402


class Latest(object):
    def __init__(self):
        self.msg = None
        self.t = 0.0
        self.count = 0

    def __call__(self, msg):
        self.msg = msg
        self.t = time.time()
        self.count += 1

    def age(self):
        return time.time() - self.t if self.msg is not None else 1e9


def subscribe(topic, msg_type):
    from unitree_sdk2py.core.channel import ChannelSubscriber

    latest = Latest()
    sub = ChannelSubscriber(topic, msg_type)
    sub.Init(latest, 10)
    latest.sub = sub  # keep it alive
    return latest


def pose_of(kind, msg):
    """(xyz, quat_wxyz) from a pose-like message."""
    if kind == "sport":
        return np.array(msg.position, dtype=np.float64), np.array(msg.imu_state.quaternion, dtype=np.float64)
    p = msg.pose.pose if kind == "odom" else msg.pose
    o = p.orientation
    return np.array([p.position.x, p.position.y, p.position.z], dtype=np.float64), np.array([o.w, o.x, o.y, o.z])


def probe(args):
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_, LowState_, SportModeState_

    topics = [
        ("rt/lowstate", LowState_), ("rt/sportmodestate", SportModeState_),
        ("rt/utlidar/height_map_array", HeightMap_), ("rt/utlidar/robot_pose", PoseStamped_),
        ("rt/utlidar/robot_odom", Odometry_), ("rt/utlidar/cloud", PointCloud2_),
        ("rt/utlidar/cloud_deskewed", PointCloud2_),
    ]
    subs = {}
    for name, typ in topics:
        try:
            subs[name] = subscribe(name, typ)
        except Exception as exc:  # noqa: BLE001
            print("%-32s could not subscribe: %s" % (name, exc))
    print("listening for %.0f s ..." % args.probe_seconds)
    time.sleep(args.probe_seconds)
    print("\n%-32s %8s" % ("topic", "Hz"))
    for name, _ in topics:
        if name in subs:
            print("%-32s %8.1f" % (name, subs[name].count / args.probe_seconds))
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    save = {}

    low = subs.get("rt/lowstate")
    if low is not None and low.msg is not None:
        m = low.msg
        q = np.array([m.motor_state[i].q for i in range(12)])
        quat = np.array(m.imu_state.quaternion)
        print("\nlowstate: foot_force %s  (order FR FL RR RL)" % list(m.foot_force))
        print("  imu quaternion (w x y z) %s  rpy %s" % (quat, np.array(m.imu_state.rpy)))
        print("  projected gravity %s   base height from the legs %.3f m" % (
            go2_obs.projected_gravity(quat), base_height_from_legs(q, quat)))
        print("  joints FR FL RR RL (hip thigh calf): %s" % q.reshape(4, 3).round(2).tolist())
        save.update(q=q, quat=quat, foot_force=np.array(m.foot_force))

    for name, kind in (("rt/utlidar/robot_pose", "pose"), ("rt/utlidar/robot_odom", "odom"), ("rt/sportmodestate", "sport")):
        s = subs.get(name)
        if s is not None and s.msg is not None:
            xyz, quat = pose_of(kind, s.msg)
            print("\n%s: position %s  yaw %.2f rad  quat %s" % (name, xyz, go2_obs.yaw_of(quat), quat))
            save[kind + "_xyz"], save[kind + "_quat"] = xyz, quat

    hm = subs.get("rt/utlidar/height_map_array")
    if hm is not None and hm.msg is not None:
        m = hm.msg
        data = np.array(m.data, dtype=np.float32)
        valid = np.abs(data) < go2_scan.EMPTY_ABOVE
        print("\nheight map: frame '%s' resolution %.3f width %d height %d origin %s" % (
            m.frame_id, m.resolution, m.width, m.height, list(m.origin)))
        print("  %d values, %.0f%% valid, z %.3f .. %.3f, median %.3f" % (
            len(data), 100 * valid.mean(), data[valid].min() if valid.any() else float("nan"),
            data[valid].max() if valid.any() else float("nan"), float(np.median(data[valid])) if valid.any() else float("nan")))
        print("  map spans x %.2f..%.2f, y %.2f..%.2f" % (
            m.origin[0], m.origin[0] + m.width * m.resolution, m.origin[1], m.origin[1] + m.height * m.resolution))
        save.update(map_data=data, map_width=m.width, map_height=m.height, map_resolution=m.resolution,
                    map_origin=np.array(m.origin), map_frame=str(m.frame_id))
    else:
        print("\nNO HEIGHT MAP on rt/utlidar/height_map_array")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan_probe_%s.npz" % time.strftime("%Y%m%d_%H%M%S"))
    np.savez(out, **save)
    print("\nsaved a sample to %s (copy it to the laptop)" % out)


def cloud_xyz(msg):
    """(N, 3) float32 x, y, z of a PointCloud2 whose first three fields are float32 x, y, z."""
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    return raw[:, :12].copy().view("<f4").reshape(-1, 3)


def run(args):
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_, LowState_, SportModeState_

    pose_type = {"pose": PoseStamped_, "odom": Odometry_, "sport": SportModeState_}[args.pose_kind]
    low = subscribe("rt/lowstate", LowState_)
    if args.source == "cloud":
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

        hm = subscribe(args.cloud_topic, PointCloud2_)  # same interface: .msg, .count, .age()
        cloud = go2_scan.CloudGrid()
    else:
        hm = subscribe(args.map_topic, HeightMap_)
        cloud = None
    pose = subscribe(args.pose_topic, pose_type)
    sampler = go2_scan.HeightMapScan()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ("127.0.0.1", args.port)
    seen_map = 0
    next_t, last_print, sent = time.time(), 0.0, 0
    print("scan -> udp %s:%d; waiting for lowstate, map and pose ..." % target)
    while True:
        next_t += go2_obs.CONTROL_DT
        delay = next_t - time.time()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.time()
        if low.msg is None or hm.msg is None or pose.msg is None:
            continue
        if hm.count != seen_map and (cloud is None or pose.msg is not None):
            seen_map = hm.count
            m = hm.msg
            if cloud is None:
                sampler.set_map(m.data, m.width, m.height, m.resolution, m.origin)
            else:
                cloud.add_points(cloud_xyz(m), pose_of(args.pose_kind, pose.msg)[0])
                sampler.set_map(*cloud.as_map())
        lm = low.msg
        q = np.array([lm.motor_state[i].q for i in range(12)])
        quat_imu = np.array(lm.imu_state.quaternion, dtype=np.float64)
        xyz, quat_pose = pose_of(args.pose_kind, pose.msg)
        yaw = go2_obs.yaw_of(quat_pose)
        force = np.array(lm.foot_force, dtype=np.float64)
        stance = force > args.stance_force
        up_b = -go2_obs.projected_gravity(quat_imu).astype(np.float64)
        fallback = base_height_from_legs(q, quat_imu)
        fresh = pose.age() < args.max_pose_age and hm.age() < args.max_map_age and low.age() < 0.1
        if fresh:
            sampler.update_anchor(xyz, yaw, foot_positions_base(q), stance, up_b)
            scan, empty = sampler.scan(xyz, yaw, fallback)
            sock.sendto(scan.astype("<f4").tobytes(), target)
            sent += 1
        now = time.time()
        if now - last_print >= 1.0:
            last_print = now
            if fresh:
                print("scan %.2f..%.2f m (legs say %.2f) | empty %.0f%% | anchor %s | pose age %.0f ms, map age %.0f ms "
                      "| foot force %s | sent %d" % (
                          float(scan.min()), float(scan.max()), fallback, 100 * empty,
                          "%.3f" % sampler.anchor if sampler.anchor is not None else "none (no loaded foot yet)",
                          1000 * pose.age(), 1000 * hm.age(), force.astype(int).tolist(), sent))
            else:
                print("NOT SENDING: pose age %.2f s, map age %.2f s, lowstate age %.2f s" % (pose.age(), hm.age(), low.age()))


def run_raw(args):
    """--source raw: the LiDAR's raw cloud plus our own leg odometry. For low-level control,
    when the robot's motion service (and with it its odometry and corrected cloud) is off."""
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    cal = np.load(args.calib)
    raw = go2_scan.RawCloud(cal["R_bl"], cal["t_bl"])
    print("LiDAR calibration %s: t_bl %s" % (args.calib, np.round(cal["t_bl"], 3).tolist()))
    low = subscribe("rt/lowstate", LowState_)
    cl = subscribe(args.raw_topic, PointCloud2_)
    grid, sampler, odo = go2_scan.CloudGrid(), go2_scan.HeightMapScan(), go2_scan.LegOdometry()
    # LegOdometry puts the floor under the feet at z = 0 when it starts, and the LiDAR points go
    # into the same frame, so the map and the pose start aligned: anchor 0. Without a start
    # value the anchor waited for mapped ground under a loaded foot, which a robot standing still
    # never has (the cells under the body are hidden from the LiDAR and filtered as self-hits),
    # and the scan stayed "flat" (robot, 2026-10-09). It is still refined once the feet walk
    # onto mapped ground.
    sampler.anchor = 0.0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ("127.0.0.1", args.port)
    seen, next_t, last_print, sent = 0, time.time(), 0.0, 0
    print("scan -> udp %s:%d; waiting for lowstate and the raw LiDAR cloud ..." % target)
    while True:
        next_t += go2_obs.CONTROL_DT
        delay = next_t - time.time()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.time()
        if low.msg is None:
            continue
        lm = low.msg
        q = np.array([lm.motor_state[i].q for i in range(12)])
        quat = np.array(lm.imu_state.quaternion, dtype=np.float64)
        force = np.array(lm.foot_force, dtype=np.float64)
        stance = force > args.stance_force
        feet_b = foot_positions_base(q)
        pos, R_wb = odo.update(feet_b, quat, stance)
        if cl.msg is not None and cl.count != seen:
            seen = cl.count
            grid.add_points(raw.to_world(cloud_xyz(cl.msg), pos, R_wb), pos)
            sampler.set_map(*grid.as_map())
        yaw = go2_obs.yaw_of(quat)
        fallback = base_height_from_legs(q, quat)
        fresh = cl.age() < args.max_map_age and low.age() < 0.1
        if fresh:
            sampler.update_anchor(pos, yaw, feet_b, stance, -go2_obs.projected_gravity(quat).astype(np.float64))
            scan, empty = sampler.scan(pos, yaw, fallback)
            sock.sendto(scan.astype("<f4").tobytes(), target)
            sent += 1
        now = time.time()
        if now - last_print >= 1.0:
            last_print = now
            if fresh:
                print("scan %.2f..%.2f m (legs say %.2f) | empty %.0f%% | anchor %s | odometry %s | cloud age %.0f ms "
                      "| foot force %s | sent %d" % (
                          float(scan.min()), float(scan.max()), fallback, 100 * empty,
                          "%.3f" % sampler.anchor if sampler.anchor is not None else "none (no loaded foot yet)",
                          np.round(pos, 2).tolist(), 1000 * cl.age(), force.astype(int).tolist(), sent))
            else:
                print("NOT SENDING: cloud age %.2f s, lowstate age %.2f s" % (cl.age(), low.age()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-seconds", type=float, default=5.0)
    ap.add_argument("--source", choices=("raw", "cloud", "map"), default="raw",
                    help="raw (default): the LiDAR's raw cloud and our own leg odometry; works with the "
                         "robot's motion service off, as it is under go2_runner.py. cloud: the robot's "
                         "corrected cloud and odometry (only while its own controller runs). map: "
                         "rt/utlidar/height_map_array (not published on this robot).")
    ap.add_argument("--raw-topic", default="rt/utlidar/cloud", help="PointCloud2 in the LiDAR's own frame.")
    ap.add_argument("--calib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "go2_lidar_calib.npz"),
                    help="LiDAR pose on the body (R_bl, t_bl), from scripts/calib_go2_lidar.py on the laptop.")
    ap.add_argument("--cloud-topic", default="rt/utlidar/cloud_deskewed", help="PointCloud2 in the pose's frame.")
    ap.add_argument("--map-topic", default="rt/utlidar/height_map_array")
    ap.add_argument("--pose-topic", default="rt/utlidar/robot_odom")
    ap.add_argument("--pose-kind", choices=("pose", "odom", "sport"), default="odom")
    ap.add_argument("--stance-force", type=float, default=20.0, help="Foot force above this means the foot carries load.")
    ap.add_argument("--max-pose-age", type=float, default=0.3)
    ap.add_argument("--max-map-age", type=float, default=2.0)
    ap.add_argument("--port", type=int, default=9871)
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(0, args.iface)
    if args.probe:
        probe(args)
    else:
        try:
            run_raw(args) if args.source == "raw" else run(args)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
