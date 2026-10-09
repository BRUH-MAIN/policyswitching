#!/usr/bin/env python3
# Created by Claude (policyswitching repo, deploy_numpy/go2_temps.py).
# Purpose: show the Go2's motor temperatures and battery once a second. Read-only: it only
# listens to rt/lowstate and sends nothing to the robot.
"""Live motor temperatures and battery for the Go2. Ctrl-C to stop.

    python3 go2_temps.py            # on the robot
    python3 go2_temps.py --once     # one reading and exit
"""

from __future__ import print_function

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

NAMES = [leg + " " + j for leg in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]
START_MAX_TEMP_C = 60  # go2_runner.py refuses to stand or start the policy above this


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    ChannelFactoryInitialize(0, args.iface)
    latest = {}
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda m: latest.__setitem__("m", m), 10)
    time.sleep(1.0)
    try:
        while True:
            m = latest.get("m")
            if m is None:
                print("no robot state yet (is the robot on, and --iface right?)")
            else:
                t = [m.motor_state[i].temperature for i in range(12)]
                hot = max(range(12), key=lambda i: t[i])
                legs = "  ".join("%s %2d/%2d/%2d" % (leg, t[3 * k], t[3 * k + 1], t[3 * k + 2])
                                 for k, leg in enumerate(("FR", "FL", "RR", "RL")))
                flag = "OK to start" if t[hot] <= START_MAX_TEMP_C else "TOO HOT to start (> %d C)" % START_MAX_TEMP_C
                print("%s | hip/thigh/calf C: %s | hottest %s %d C: %s | battery %s %%, %.1f V" % (
                    time.strftime("%H:%M:%S"), legs, NAMES[hot], t[hot], flag, m.bms_state.soc, m.power_v))
            if args.once:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
