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


class CloudGrid(object):
    """A robot-centred 2.5-D height grid built from LiDAR points in a fixed (odometry) frame.

    For when the robot publishes a point cloud but no height map (the Go2 here: its firmware
    LiDAR publishes `rt/utlidar/cloud_deskewed` in the `odom` frame and `height_map_array`
    stays silent). The grid scrolls with the robot. `HeightMapScan` reads it exactly as it
    would read a firmware height map (`set_map`).

    Each cell tracks a low quantile (`quantile`, default 0.2) of the heights of all the points
    that have fallen in it, by stochastic quantile tracking: every point moves the estimate
    up by `step_m * quantile` if it is above it and down by `step_m * (1 - quantile)` if
    below. A cell starts at the lowest point of the first scan that reaches it. When a scan
    has a point more than `drop_m` below the estimate, the estimate drops at once to that
    point plus `drop_margin_m` (phantoms are never below the floor, so a low point is real;
    the margin allows for range noise); it is reset to a scan's lowest point when that is more
    than `reset_m` above the estimate (the terrain there is not what the cell says).

    Why a low quantile and not a mean: on the robot (2026-10-09, standing on a flat, clear
    floor, confirmed by its camera) the firmware cloud returns, 0.4-0.8 m ahead, real floor
    points and almost as many phantom points 6-11 cm above them, with the same intensity and
    range. A per-cell mean put a 4-5 cm bump on clear floor. The phantoms are always above
    the floor, never below, so a low quantile reads through them. A scan carries one or two
    points per cell, so the quantile has to be tracked across scans, not taken within one.
    """

    def __init__(self, resolution=0.05, size_m=6.0, quantile=0.2, step_m=0.004, reset_m=0.12,
                 drop_m=0.03, drop_margin_m=0.015, max_above_base_m=0.3):
        self.res = float(resolution)
        self.n = int(round(size_m / resolution))
        self.q = quantile
        self.step = step_m
        self.reset = reset_m
        self.drop, self.drop_margin = drop_m, drop_margin_m
        self.max_above = max_above_base_m
        self.grid = np.full((self.n, self.n), np.nan, dtype=np.float64)  # [iy, ix]
        self.origin = None  # world xy of cell (0, 0)

    def _recentre(self, base_xy):
        half = self.n * self.res / 2.0
        if self.origin is None:
            self.origin = np.floor((np.asarray(base_xy) - half) / self.res) * self.res
            return
        centre = self.origin + half
        shift = np.round((np.asarray(base_xy) - centre) / self.res).astype(int)
        if np.all(np.abs(shift) * self.res < 1.0):
            return
        sx, sy = int(shift[0]), int(shift[1])
        g = np.full_like(self.grid, np.nan)
        src_x = slice(max(sx, 0), self.n + min(sx, 0))
        dst_x = slice(max(-sx, 0), self.n + min(-sx, 0))
        src_y = slice(max(sy, 0), self.n + min(sy, 0))
        dst_y = slice(max(-sy, 0), self.n + min(-sy, 0))
        if src_x.start < src_x.stop and src_y.start < src_y.stop:
            g[dst_y, dst_x] = self.grid[src_y, src_x]
        self.grid = g
        self.origin = self.origin + np.array([sx, sy]) * self.res

    def add_points(self, xyz, base_xyz):
        """xyz (N, 3) in the odometry frame; (0, 0, 0) rows (no return) are ignored."""
        self._recentre(base_xyz[:2])
        pts = np.asarray(xyz, dtype=np.float64)
        keep = np.all(np.isfinite(pts), axis=1) & np.any(pts != 0.0, axis=1)
        keep &= pts[:, 2] < base_xyz[2] + self.max_above  # nothing above the body is ground
        pts = pts[keep]
        if len(pts) == 0:
            return 0
        ix = np.floor((pts[:, 0] - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((pts[:, 1] - self.origin[1]) / self.res).astype(np.int64)
        inside = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        flat, z = (iy * self.n + ix)[inside], pts[inside, 2]
        if len(flat) == 0:
            return 0
        g = self.grid.reshape(-1)
        size = self.n * self.n
        count = np.bincount(flat, minlength=size)
        zmin = np.full(size, np.inf)
        np.minimum.at(zmin, flat, z)
        hit = count > 0
        est = g.copy()
        with np.errstate(invalid="ignore"):
            # New cells, and cells whose terrain is clearly higher than they say: the scan's lowest point.
            restart = hit & (~np.isfinite(est) | (zmin > est + self.reset))
            # A point well below the estimate is real ground: drop to it (plus the noise margin).
            drop = hit & ~restart & (zmin < est - self.drop)
        est[restart] = zmin[restart]
        est[drop] = zmin[drop] + self.drop_margin
        restart = restart | drop
        # Quantile tracking against the (possibly restarted) estimate.
        below = np.bincount(flat, weights=(z < est[flat]).astype(np.float64), minlength=size)
        move = hit & ~restart
        est[move] += self.step * (self.q * count[move] - below[move])
        g[hit] = est[hit]
        return int(hit.sum())

    def z_offset(self, pts_world, max_abs_m=0.06, min_points=150, flat_m=0.03):
        """Median height of new points above the mapped cells they fall in (cells with an estimate,
        points within `max_abs_m` of it: edges, phantoms and new terrain excluded), or None.

        A static world cannot move, so this is the vertical drift of the pose the points were
        placed with since those cells were mapped."""
        if self.origin is None or len(pts_world) == 0:
            return None
        pts = np.asarray(pts_world, dtype=np.float64)
        ix = np.floor((pts[:, 0] - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((pts[:, 1] - self.origin[1]) / self.res).astype(np.int64)
        inside = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        # Only cells inside flat patches (3 x 3 neighbourhood spans < flat_m): at a step edge a cell
        # first mapped from below gets points from the step's top, which is terrain, not drift.
        g = self.grid
        pad = np.pad(g, 1, mode="constant", constant_values=np.nan)
        stack = np.stack([pad[1 + a:self.n + 1 + a, 1 + b:self.n + 1 + b] for a in (-1, 0, 1) for b in (-1, 0, 1)])
        with np.errstate(invalid="ignore"):
            span = np.max(stack, axis=0) - np.min(stack, axis=0)
        flat = np.isfinite(span) & (span < flat_m)
        cell = np.where(flat, g, np.nan)[iy[inside], ix[inside]]
        d = pts[inside, 2] - cell
        ok = np.isfinite(d) & (np.abs(d) < max_abs_m)
        if ok.sum() < min_points:
            return None
        return float(np.median(d[ok]))

    def as_map(self):
        """(data, width, height, resolution, origin) in the layout HeightMapScan.set_map takes."""
        data = np.where(np.isfinite(self.grid), self.grid, 1.0e9).astype(np.float32)
        return data.reshape(-1), self.n, self.n, self.res, self.origin


def quat_to_rot(quat_wxyz):
    """Body-to-world rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class LegOdometry(object):
    """Base position from the legs: a foot that carries load is assumed not to slide.

    For when the robot's own odometry is not published (it stops with its motion service,
    which low-level control has to switch off). Orientation comes from the IMU quaternion;
    position from the stance feet: each loaded foot is pinned where it touched down, and the
    base is where those pins say it must be. Drifts by a few per cent of the distance walked,
    which is fine for a height scan that only needs the last couple of metres, and whose
    vertical offset is re-anchored on the feet anyway (HeightMapScan.update_anchor).
    """

    def __init__(self):
        self.pos = np.zeros(3)
        self.pins = [None] * 4  # world position of each stance foot, or None in swing
        self.started = False

    def update(self, feet_base, quat_wxyz, stance):
        """feet_base (4, 3) foot centres in the base frame; stance (4,) bool. Returns (pos, R)."""
        R = quat_to_rot(quat_wxyz)
        feet_w = np.asarray(feet_base, dtype=np.float64).dot(R.T)  # base -> world-aligned, relative to the base
        if not self.started:
            # Start with the base above the origin at its leg height, so z = 0 is the floor.
            self.pos = np.array([0.0, 0.0, -float(np.min(feet_w[:, 2])) + FOOT_R])
            self.started = True
        est = []
        for i in range(4):
            if stance[i]:
                if self.pins[i] is None:
                    self.pins[i] = self.pos + feet_w[i]
                else:
                    est.append(self.pins[i] - feet_w[i])
            else:
                self.pins[i] = None
        if est:
            self.pos = np.mean(est, axis=0)
        return self.pos.copy(), R

    def shift_z(self, dz):
        """Move the whole odometry frame's estimate up by dz (pins too), e.g. to remove drift."""
        self.pos[2] += dz
        self.pins = [None if p is None else p + np.array([0.0, 0.0, dz]) for p in self.pins]


class RawCloud(object):
    """Points of the firmware LiDAR's raw cloud (its own frame) moved into the odometry frame.

    `R_bl`, `t_bl`: the LiDAR's pose in the base frame (calibrated: go2_lidar_calib.npz).
    Points on the robot itself, and within `min_range_m` of the sensor, are dropped: a box of
    `body_half_x` by `body_half_y` around the base, at every height (the floor under the body is
    hidden from the LiDAR anyway; the grid remembers it from when it was ahead).
    """

    def __init__(self, R_bl, t_bl, body_half_x=0.45, body_half_y=0.28, min_range_m=0.12):
        self.R_bl = np.asarray(R_bl, dtype=np.float64)
        self.t_bl = np.asarray(t_bl, dtype=np.float64)
        self.hx, self.hy, self.rmin = body_half_x, body_half_y, min_range_m

    def to_world(self, pts_lidar, base_pos, R_wb):
        p = np.asarray(pts_lidar, dtype=np.float64)
        p = p[np.all(np.isfinite(p), axis=1) & (np.linalg.norm(p, axis=1) > self.rmin)]
        b = p.dot(self.R_bl.T) + self.t_bl  # base frame
        b = b[~((np.abs(b[:, 0]) < self.hx) & (np.abs(b[:, 1]) < self.hy))]
        return b.dot(R_wb.T) + base_pos
