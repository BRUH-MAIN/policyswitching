#!/usr/bin/env python3
"""Person-following velocity commands for go2_runner.py. Runs on the LAPTOP.

Reads the fused perception state (person bearing and distance, LiDAR obstacle flag) from
the perception hub in the robot workspace (perception/perception_hub.py, GET /state) and
sends (vx, vy, wz) as three little-endian float32 over UDP to the robot, 20 times a second.
go2_runner.py --cmd udp uses them only while R1 is held on the remote, and stops by itself
if they stop arriving for 0.3 s, so a dropped Wi-Fi link stops the robot.

    python3 follow_cmd.py --robot <robot ip> [--gap 1.5] [--no-obstacle-veto]

Control law: the one evaluated in simulation (src/vlm_nav/controllers.py follow_command),
without the leader-velocity feed-forward: turn to face the person, regulate range with
signed forward speed, hold station inside a deadband, do not translate while the person is
more than about 57 degrees off the nose.

The obstacle veto zeroes forward motion while the hub says "blocked". A staircase is an
obstacle to that monitor (the second riser is inside its height band at about 0.6 m), so
following someone up stairs needs --no-obstacle-veto, a spotter, and R1 as the dead-man.

Sends nothing but those UDP datagrams. `--print-only` sends nothing at all.
"""

import argparse
import json
import math
import socket
import struct
import time
from urllib.request import urlopen


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def command(state, args):
    """(vx, vy, wz), reason."""
    if not state.get("person_present"):
        return (0.0, 0.0, 0.0), "no person"
    bearing, dist = state.get("person_bearing_deg"), state.get("person_distance_m")
    if bearing is None:
        return (0.0, 0.0, 0.0), "no bearing"
    err = math.radians(bearing)  # positive = person to the left = turn left (positive wz)
    wz = clamp(args.k_yaw * err, -args.wz_max, args.wz_max)
    if dist is None:
        return (0.0, 0.0, wz), "no distance: turning only"
    range_err = dist - args.gap
    if abs(range_err) < args.deadband:
        range_err = 0.0
    v = clamp(args.k_range * range_err, -args.v_back_max, args.v_max)
    if abs(err) >= args.turn_in_place_rad:
        return (0.0, 0.0, wz), "turning to face"
    vx, vy = v * math.cos(err), clamp(v * math.sin(err), -args.vy_max, args.vy_max)
    if args.obstacle_veto and state.get("blocked", True) and vx > 0.0:
        return (0.0, 0.0, wz), "blocked: no forward motion"
    return (vx, vy, wz), "following at %.1f m" % dist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, help="Robot IP (where go2_runner.py listens).")
    ap.add_argument("--port", type=int, default=9870)
    ap.add_argument("--hub", default="http://127.0.0.1:8103/state")
    ap.add_argument("--gap", type=float, default=1.5, help="Distance to keep from the person (m).")
    ap.add_argument("--v-max", type=float, default=0.8)
    ap.add_argument("--v-back-max", type=float, default=0.3)
    ap.add_argument("--vy-max", type=float, default=0.3)
    ap.add_argument("--wz-max", type=float, default=0.8)
    ap.add_argument("--k-range", type=float, default=0.9)
    ap.add_argument("--k-yaw", type=float, default=1.2)
    ap.add_argument("--deadband", type=float, default=0.15)
    ap.add_argument("--turn-in-place-rad", type=float, default=1.0)
    ap.add_argument("--no-obstacle-veto", dest="obstacle_veto", action="store_false")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd, reason, hub_t, last_print = (0.0, 0.0, 0.0), "starting", 0.0, 0.0
    next_poll = 0.0
    while True:
        now = time.time()
        if now >= next_poll:
            next_poll = now + 0.1
            try:
                with urlopen(args.hub, timeout=0.5) as resp:
                    state = json.loads(resp.read().decode("utf-8"))
                state = state.get("summary", state)
                cmd, reason = command(state, args)
                hub_t = time.time()
            except Exception as exc:  # noqa: BLE001
                reason = "hub unreachable: %s" % exc
        if time.time() - hub_t > 0.5:
            cmd = (0.0, 0.0, 0.0)
            # Send nothing rather than zeros: the runner's own 0.3 s timeout then stops the robot.
        elif not args.print_only:
            sock.sendto(struct.pack("<3f", *cmd), (args.robot, args.port))
        if now - last_print >= 0.5:
            last_print = now
            print("vx %+.2f vy %+.2f wz %+.2f  (%s)" % (cmd[0], cmd[1], cmd[2], reason))
        time.sleep(0.05)


if __name__ == "__main__":
    main()
