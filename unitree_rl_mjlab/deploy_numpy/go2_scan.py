# Created by Claude (policyswitching repo, deploy_numpy/go2_scan.py).
# Purpose: turn a robot-centred height map plus the robot's pose into the 187-point height
# scan the locomotion policy reads. Pure numpy (Python 3.8); no robot I/O in this file.
"""Height scan from a gridded height map.

The policy wants, at 187 points around the base (go2_obs.scan_points_xy), the height of the
base_link origin above the ground. A height map gives ground height z_map(x, y) in some
fixed "map" frame; the pose gives where the base is in that frame. So

    scan[k] = (z_base + anchor) - z_map(point k)

`anchor` absorbs the constant offset between the pose's reference point and base_link and
any slow vertical drift between pose and map. It is estimated here from the feet: a foot
that is carrying load is standing on the ground, so the map under it should read exactly
the height of that foot's sole. The policy tolerates about 3 cm of height error and loses
a fifth of its success at 6 cm (report_content/go2_real_stairs_plan.md, section 2.2), so
this number matters more than latency or holes.

Map layout assumed (Unitree HeightMap message, to be confirmed on the robot with
go2_scan_node.py --probe): cell (ix, iy) is at x = origin[0] + ix * resolution,
y = origin[1] + iy * resolution, value data[ix + width * iy], and a cell with no data
holds a huge number (1e9).
"""

import numpy as np

import go2_obs

EMPTY_ABOVE = 1.0e6  # anything this large is "no data"
FOOT_R = 0.022


class HeightMapScan(object):
    def __init__(self, anchor_tau_s=1.0, anchor_reject_m=0.06, dt=0.02):
        self.pts = go2_obs.scan_points_xy().astype(np.float64)
        self.anchor = None
        self.alpha = dt / max(anchor_tau_s, dt)
        self.reject = anchor_reject_m
        self.map = None
        self.res = self.ox = self.oy = 0.0
        self.stats = dict(empty=0, anchor_used=0, anchor_rejected=0)

    def set_map(self, data, width, height, resolution, origin_xy):
        grid = np.asarray(data, dtype=np.float32).reshape(int(height), int(width))  # [iy, ix]
        self.map = grid
        self.res = float(resolution)
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])

    def _lookup(self, xy):
        """Map height at world points xy (N, 2); NaN where the map has nothing nearby."""
        ix = np.rint((xy[:, 0] - self.ox) / self.res).astype(np.int64)
        iy = np.rint((xy[:, 1] - self.oy) / self.res).astype(np.int64)
        h, w = self.map.shape
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        out = np.full(len(xy), np.nan)
        z = self.map[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)].astype(np.float64)
        good = inside & (np.abs(z) < EMPTY_ABOVE)
        out[good] = z[good]
        # One ring of neighbours for cells that are empty themselves.
        todo = np.where(inside & ~good)[0]
        for k in todo:
            y0, y1 = max(iy[k] - 1, 0), min(iy[k] + 2, h)
            x0, x1 = max(ix[k] - 1, 0), min(ix[k] + 2, w)
            patch = self.map[y0:y1, x0:x1]
            ok = np.abs(patch) < EMPTY_ABOVE
            if ok.any():
                out[k] = float(np.median(patch[ok]))
        return out

    def update_anchor(self, base_xyz, yaw, feet_base, stance, up_base):
        """Refine the vertical anchor from loaded feet.

        base_xyz: pose position in the map frame. yaw: heading in the map frame.
        feet_base (4, 3): foot centres in the base frame. stance (4,) bool: carrying load.
        up_base (3,): world up in the base frame (minus projected gravity).
        """
        if self.map is None or not np.any(stance):
            return
        feet = np.asarray(feet_base, dtype=np.float64)
        c, s = np.cos(yaw), np.sin(yaw)
        # Level-frame offsets of each foot: horizontal part rotated by yaw, vertical along up.
        up = np.asarray(up_base, dtype=np.float64)
        height_below = -feet.dot(up) + FOOT_R  # sole depth below the base origin (m, positive)
        # Level frame in base coordinates: x is the base x axis flattened onto the ground.
        x_level = np.array([1.0, 0.0, 0.0]) - up[0] * up
        x_level /= np.linalg.norm(x_level)
        y_level = np.cross(up, x_level)
        lx, ly = feet.dot(x_level), feet.dot(y_level)
        fx = base_xyz[0] + c * lx - s * ly
        fy = base_xyz[1] + s * lx + c * ly
        z_map = self._lookup(np.stack([fx, fy], axis=1))
        samples = []
        for i in range(4):
            if stance[i] and np.isfinite(z_map[i]):
                # sole height in the map frame = z_base + anchor - height_below = z_map
                samples.append(z_map[i] + height_below[i] - base_xyz[2])
        if not samples:
            return
        sample = float(np.median(samples))
        if self.anchor is None:
            self.anchor = sample
            return
        if abs(sample - self.anchor) > self.reject:
            self.stats["anchor_rejected"] += 1  # a foot beside a riser reads the other level
            return
        self.anchor += self.alpha * (sample - self.anchor)
        self.stats["anchor_used"] += 1

    def scan(self, base_xyz, yaw, fallback_height):
        """(187,) heights in metres. Cells the map cannot answer get `fallback_height`."""
        if self.map is None or self.anchor is None:
            return go2_obs.flat_scan(fallback_height), 1.0
        c, s = np.cos(yaw), np.sin(yaw)
        wx = base_xyz[0] + c * self.pts[:, 0] - s * self.pts[:, 1]
        wy = base_xyz[1] + s * self.pts[:, 0] + c * self.pts[:, 1]
        z = self._lookup(np.stack([wx, wy], axis=1))
        out = (base_xyz[2] + self.anchor) - z
        empty = ~np.isfinite(out)
        self.stats["empty"] = int(empty.sum())
        out[empty] = fallback_height
        return np.clip(out, -1.0, go2_obs.SCAN_MISS).astype(np.float32), float(empty.mean())
