"""Offline check of *where* perception: near/far edge of the intermediation, in metres.

The executor switches specialists on the near edge and releases on the far edge,
so this -- not box IoU -- is the perception quantity that decides outcomes. For
each labelled frame (rendered with depth + pose by vlm_nav_render_frames.py):
  depth_only  -- perception.depth_edge_estimate (no VLM)
  gt_box      -- the depth-checked ground-truth box through estimate_from_box, i.e.
                 SARO's box+depth path with a perfect box (upper bound for it)
  vlm_box     -- (--vlm) the live VLM's detection box through the same path; frames
                 where the box is degenerate count as not detected
Ground truth is along the robot's heading: (edge_x - x) / cos(yaw). A near edge
inside the camera's blind zone can't be observed, so near error is scored only
where the true edge is beyond it. On the `flat` course any detection within
range is a false positive.

Usage (from unitree_rl_mjlab/):
  PYTHONPATH=$PWD python scripts/vlm_nav_edge_eval.py --frames logs/vlm_nav/frames_L1/*_s300 \
      --out eval_results/vlm_nav/phase2/edges_L1.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.vlm_nav.camera import CameraSpec
from src.vlm_nav.course import saro_courses
from src.vlm_nav.perception import depth_edge_estimate, estimate_from_box, ground_blind_distance


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--frames", nargs="+", required=True)
  ap.add_argument("--max-range", type=float, default=4.0, help="Score only edges within this distance.")
  ap.add_argument("--vlm", action="store_true", help="Also score the live VLM's boxes (server must be up).")
  ap.add_argument("--vlm-max-frames", type=int, default=120)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  cam = CameraSpec()
  vlm = None
  if args.vlm:
    from PIL import Image  # noqa: PLC0415

    from src.vlm_nav import prompts as P  # noqa: PLC0415
    from src.vlm_nav.vlm_backend import OpenAICompatVLM  # noqa: PLC0415
    vlm = OpenAICompatVLM()
    rng = np.random.default_rng(0)
  sources = ("depth_only", "gt_box", "vlm_box") if args.vlm else ("depth_only", "gt_box")
  blind = ground_blind_distance(cam)
  rows = []
  for d in args.frames:
    d = Path(d)
    for line in open(d / "labels.jsonl"):
      f = json.loads(line)
      if "depth" not in f:
        raise SystemExit(f"[ERROR] {d} has no depth/pose: re-render with the current vlm_nav_render_frames.py")
      course = saro_courses(visual=f["visual"], level=f["level"])[f["course"]]
      depth = np.load(d / f["depth"]).astype(np.float32)
      pos, quat = np.array(f["base_pos"]), np.array(f["base_quat"])
      x, yaw = f["x"], f["yaw"]
      region = next((r for r in course.regions() if r.terrain != "flat" and r.x1 > x), None)
      gt_near = gt_far = None
      if region is not None:
        gt_near = max(0.0, region.x0 - x) / math.cos(yaw)
        gt_far = (region.x1 - x) / math.cos(yaw)
      row = dict(course=f["course"], terrain=None if region is None else region.terrain, x=x, yaw=yaw,
                 gt_near=gt_near, gt_far=gt_far)
      for src in sources:
        if src == "vlm_box":
          # Sample: querying every frame is slow on this laptop.
          if region is None or rng.random() > args.vlm_max_frames / 700:
            row[src] = "skipped"
            continue
          img = np.asarray(Image.open(d / f["image"]).convert("RGB"))
          name = {"stairs": "stairs", "rough": "rough ground"}[region.terrain]
          rep_ = vlm.ask(img, P.perception_detect(name), None, max_tokens=96)
          px = P.parse_detect_box(rep_.text, depth.shape[1], depth.shape[0])
          est = None if P.box_is_degenerate(px, depth.shape[1], depth.shape[0]) else estimate_from_box(px, depth, cam, pos, quat)
          row["vlm_text"] = rep_.text[:120]
        elif src == "depth_only":
          est = depth_edge_estimate(depth, cam, pos, quat)
        elif f["intermediation_bbox"] is not None:
          est = estimate_from_box(f["intermediation_bbox"], depth, cam, pos, quat)
        else:
          est = None
        row[src] = None if est is None or not est.valid else dict(near=est.near_along, far=est.far_along)
      rows.append(row)

  def summarize(src: str) -> dict:
    out: dict = {}
    by = defaultdict(list)
    for r in rows:
      if r.get(src) == "skipped":
        continue
      by[r["course"]].append(r)
    for course, rs in by.items():
      if course == "flat":
        fp = [r for r in rs if r[src] is not None and r[src]["near"] < args.max_range]
        out[course] = dict(n=len(rs), false_detections=len(fp))
        continue
      in_range = [r for r in rs if r["gt_near"] is not None and r["gt_near"] < args.max_range]
      det = [r for r in in_range if r[src] is not None]
      near_obs = [r for r in det if r["gt_near"] > blind + 0.15]
      near_err = [r[src]["near"] - r["gt_near"] for r in near_obs]
      far_vis = [r for r in det if r["gt_far"] < args.max_range]
      far_err = [r[src]["far"] - r["gt_far"] for r in far_vis]
      out[course] = dict(
        n_in_range=len(in_range), detection_rate=len(det) / max(1, len(in_range)),
        near_n=len(near_err), near_mae=float(np.mean(np.abs(near_err))) if near_err else None,
        near_bias=float(np.mean(near_err)) if near_err else None,
        near_within_0p2=float(np.mean(np.abs(near_err) < 0.2)) if near_err else None,
        far_n=len(far_err), far_mae=float(np.mean(np.abs(far_err))) if far_err else None,
        far_bias=float(np.mean(far_err)) if far_err else None,
        far_within_0p3=float(np.mean(np.abs(far_err) < 0.3)) if far_err else None,
      )
    return out

  result = dict(blind_distance=blind, max_range=args.max_range, n_frames=len(rows),
                **{src: summarize(src) for src in sources})
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text(json.dumps(dict(result=result, rows=rows), indent=1))
  print(json.dumps(result, indent=1))


if __name__ == "__main__":
  main()
