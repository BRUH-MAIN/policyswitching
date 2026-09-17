"""Phase-2 offline perception gate: score the VLM on labelled frames, no robot in the loop.

Reads frame dirs written by scripts/vlm_nav_render_frames.py (PNG + labels.jsonl)
and asks the running VLM server, per frame:
  selector   -- which specialist for the ground ahead (our modification), vs the
                footprint ground truth `required_policy`
  present    -- SARO discriminator "Is there any <I>?", vs whether the next
                intermediation is visible (label has a bbox)
  box        -- SARO perception prompt, only on frames where the intermediation
                is visible; IoU vs the depth-checked GT box under every box
                convention, since the model's convention is an unknown
  planning   -- SARO planning prompt, on frames at the course start pose only
Each question kind can run free-text (SARO's format) or JSON-schema-constrained
(--json); both matter, because constrained decoding can change what a small
model answers, not just how it formats it.

Writes a per-query JSONL and a summary JSON (accuracy, confusion matrices,
box IoU by convention, latency, parse-failure rate).

Usage (from unitree_rl_mjlab/, server running):
  PYTHONPATH=$PWD python scripts/vlm_nav_perception_eval.py \
      --frames logs/vlm_nav/frames/* --tag e4b_default --out eval_results/vlm_nav/phase2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from src.vlm_nav import prompts as P
from src.vlm_nav.policy_bank import POLICY_NAMES
from src.vlm_nav.vlm_backend import OpenAICompatVLM, health

INTERMEDIATION_NAME = {"stairs": "stairs", "rough": "rough ground"}


def load_frames(dirs: list[str]) -> list[dict]:
  frames = []
  for d in dirs:
    d = Path(d)
    for line in open(d / "labels.jsonl"):
      rec = json.loads(line)
      rec["path"] = str(d / rec["image"])
      frames.append(rec)
  return frames


def build_jobs(frames: list[dict], kinds: set[str], start_x: float) -> list[tuple[str, dict]]:
  jobs = []
  for f in frames:
    if "selector" in kinds:
      jobs.append(("selector", f))
    if f["next_intermediation"] is not None:
      if "present" in kinds:
        jobs.append(("present", f))
      if "box" in kinds and f["intermediation_bbox"] is not None:
        jobs.append(("box", f))
    if "planning" in kinds and abs(f["x"] - start_x) < 1e-6:
      jobs.append(("planning", f))
  return jobs


def run_job(vlm: OpenAICompatVLM, kind: str, f: dict, use_json: bool, instructions: dict) -> dict:
  image = np.asarray(Image.open(f["path"]).convert("RGB"))
  out = {"kind": kind, "image": f["path"], "course": f["course"], "x": f["x"], "y": f["y"], "yaw": f["yaw"]}
  if kind == "selector":
    r = vlm.ask(image, P.policy_selector(), P.POLICY_SCHEMA if use_json else None, max_tokens=32, tag=kind)
    pred = (r.parsed or {}).get("policy") if use_json else P.parse_policy(r.text)
    out.update(truth=f["required_policy"], truth_ahead_1m=f["terrain_ahead_1m"], pred=pred)
  elif kind == "present":
    name = INTERMEDIATION_NAME[f["next_intermediation"]]
    # Always free text: the yes/no JSON schema makes Gemma-4-E4B return empty content.
    r = vlm.ask(image, P.discriminator_present(name), None, max_tokens=8, tag=kind)
    pred = P.parse_yes_no(r.text)
    out.update(truth=f["intermediation_bbox"] is not None, pred=pred, intermediation=name,
               dist=f["dist_to_intermediation"])
  elif kind == "box":
    name = INTERMEDIATION_NAME[f["next_intermediation"]]
    r = vlm.ask(image, P.perception(name), None, max_tokens=48, tag=kind)  # SARO asks for raw [x0,y0,x1,y1]
    box = P.parse_box(r.text)
    h, w = image.shape[:2]
    ious = {c: (P.iou(P.box_to_pixels(box, c, w, h), f["intermediation_bbox"]) if box else None) for c in P.BOX_CONVENTIONS}
    out.update(truth=f["intermediation_bbox"], raw_box=box, iou=ious, intermediation=name)
  elif kind == "planning":
    task = instructions[f["course"]]
    r = vlm.ask(image, P.planning(task), P.PLANNING_SCHEMA if use_json else None, max_tokens=256, tag=kind)
    plan = r.parsed if use_json else None
    out.update(truth=f["next_intermediation"], plan=plan)
  out.update(text=r.text, latency_s=r.latency_s, error=r.error, prompt_tokens=r.prompt_tokens)
  return out


def summarize(results: list[dict]) -> dict:
  by_kind = defaultdict(list)
  for r in results:
    by_kind[r["kind"]].append(r)
  s: dict = {}
  if sel := by_kind["selector"]:
    conf = defaultdict(Counter)
    for r in sel:
      conf[r["truth"]][r["pred"] or "unparsed"] += 1
    s["selector"] = dict(
      n=len(sel),
      accuracy=float(np.mean([r["pred"] == r["truth"] for r in sel])),
      accuracy_vs_ahead_1m=float(np.mean([r["pred"] == r["truth_ahead_1m"] for r in sel])),
      per_class_recall={t: c[t] / sum(c.values()) for t, c in conf.items()},
      confusion={t: dict(c) for t, c in conf.items()},
      unparsed_rate=float(np.mean([r["pred"] is None for r in sel])),
      pred_distribution=dict(Counter(r["pred"] or "unparsed" for r in sel)),
    )
  if pres := by_kind["present"]:
    tp = sum(r["pred"] is True and r["truth"] for r in pres)
    tn = sum(r["pred"] is False and not r["truth"] for r in pres)
    pos = sum(bool(r["truth"]) for r in pres)
    s["present"] = dict(
      n=len(pres), accuracy=float(np.mean([r["pred"] == r["truth"] for r in pres])),
      recall_visible=tp / pos if pos else None, specificity_not_visible=tn / (len(pres) - pos) if len(pres) > pos else None,
      said_yes_rate=float(np.mean([r["pred"] is True for r in pres])), unparsed_rate=float(np.mean([r["pred"] is None for r in pres])),
    )
  if box := by_kind["box"]:
    s["box"] = dict(
      n=len(box), parsed_rate=float(np.mean([r["raw_box"] is not None for r in box])),
      mean_iou_by_convention={c: float(np.mean([(r["iou"][c] or 0.0) for r in box])) for c in P.BOX_CONVENTIONS},
      iou50_rate_by_convention={c: float(np.mean([(r["iou"][c] or 0.0) >= 0.5 for r in box])) for c in P.BOX_CONVENTIONS},
    )
  if plan := by_kind["planning"]:
    def named(r):
      p = r.get("plan") or {}
      want = INTERMEDIATION_NAME.get(r["truth"], "none") if r["truth"] else "none"
      return p.get("intermediation") == want
    def climb_policy_ok(r):
      p = r.get("plan") or {}
      subs = p.get("subtasks") or []
      want = r["truth"] or "flat"
      crossing = [t for t in subs if t.get("ending") == "across intermediation"]
      return bool(crossing) and all(t.get("policy") == want for t in crossing) if r["truth"] else not crossing
    s["planning"] = dict(
      n=len(plan), valid_rate=float(np.mean([r.get("plan") is not None for r in plan])),
      intermediation_named_rate=float(np.mean([named(r) for r in plan])),
      crossing_policy_correct_rate=float(np.mean([climb_policy_ok(r) for r in plan])),
    )
  lat = [r["latency_s"] for r in results]
  s["latency_s"] = dict(median=float(np.median(lat)), p95=float(np.percentile(lat, 95)), n=len(lat))
  s["errors"] = int(sum(r["error"] is not None for r in results))
  return s


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--frames", nargs="+", required=True)
  ap.add_argument("--kinds", nargs="+", default=["selector", "present", "box", "planning"])
  ap.add_argument("--json", action="store_true", help="Schema-constrained answers (not used for box).")
  ap.add_argument("--base-url", default="http://127.0.0.1:8091")
  ap.add_argument("--workers", type=int, default=2)
  ap.add_argument("--max-frames", type=int, default=None)
  ap.add_argument("--tag", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  if not health(args.base_url):
    raise SystemExit(f"[ERROR] VLM server not healthy at {args.base_url} (scripts/vlm_server.sh)")
  from src.vlm_nav.course import saro_courses
  instructions = {name: c.instruction for name, c in saro_courses().items()}
  frames = load_frames(args.frames)
  if args.max_frames:
    rng = np.random.default_rng(0)
    frames = [frames[i] for i in sorted(rng.choice(len(frames), size=min(args.max_frames, len(frames)), replace=False))]
  start_x = min(f["x"] for f in frames)
  jobs = build_jobs(frames, set(args.kinds), start_x)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  vlm = OpenAICompatVLM(base_url=f"{args.base_url}/v1")
  print(f"[INFO] {len(frames)} frames, {len(jobs)} queries, tag={args.tag}")
  with ThreadPoolExecutor(args.workers) as pool:
    results = list(pool.map(lambda j: run_job(vlm, j[0], j[1], args.json, instructions), jobs))
  with open(out / f"{args.tag}_queries.jsonl", "w") as f:
    for r in results:
      f.write(json.dumps(r) + "\n")
  summary = dict(tag=args.tag, json=args.json, frames=args.frames, summary=summarize(results))
  (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=1))
  print(json.dumps(summary["summary"], indent=1))


if __name__ == "__main__":
  main()
