"""Person-following on the digital twin: hold a fixed gap behind a scripted leader.

`objective.md`'s setting, reduced to its locomotion core: a kinematic leader walks
the course at a speed that changes occasionally (including a full stop) and the
robot holds a standoff `--gap`. No VLM and no terrain switching -- the leader pose
is ground truth (objective.md's own scope decision) and the surface is smooth, so
what is measured is the follow controller and the specialist's ability to track a
command that reverses sign.

The leader is non-colliding and lives in the camera-only geom group, so it cannot
perturb the specialists' height_scan -- see `src/vlm_nav/leader.py`.

Usage (from unitree_rl_mjlab/):
  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/vlm_nav_follow.py \
      --ckpt-root ../unitree_rl_mjlab/eval_ckpts --gap 2.5 --video --out logs/vlm_nav/follow
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

import src.tasks  # noqa: F401,E402
from src.vlm_nav.camera import CameraSpec
from src.vlm_nav.controllers import FollowGains, follow_command
from src.vlm_nav.course import CourseSpec, Segment
from src.vlm_nav.leader import DEFAULT_SCHEDULE, LeaderPath, write_leader_pose
from src.vlm_nav.overlay import VlmEvent, annotate
from src.vlm_nav.detector import YoloDetector
from src.vlm_nav.perception import target_from_bearing_column
from src.vlm_nav import prompts as P
from src.vlm_nav.vlm_backend import OpenAICompatVLM
from src.vlm_nav.policy_bank import PolicyBank, default_checkpoints
from src.vlm_nav.twin_env import BASE_TASK, CAMERA_SENSOR_NAME, make_twin_env_cfg, set_velocity_command



def build_course(length: float, width: float) -> CourseSpec:
  return CourseSpec(
    name="follow_flat",
    segments=(Segment("flat", length),),
    intermediation="none",
    instruction="follow the person ahead of you",
    width=width,
    visual="tiled",
    goal_marker=False,  # no goal in a following task; see CourseSpec.goal_marker
  )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt-root", required=True)
  ap.add_argument("--gap", type=float, default=2.5,
                  help="Standoff distance (m). Default 2.5: past the 0.8 m height-scan "
                       "horizon, close enough that the leader's terrain is the robot's soon.")
  ap.add_argument("--policy", default="flat", help="Specialist to run throughout (smooth surface).")
  ap.add_argument("--duration", type=float, default=42.0)
  ap.add_argument("--course-length", type=float, default=28.0)
  ap.add_argument("--course-width", type=float, default=6.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--vlm", action="store_true",
                  help="Let the VLM find the person instead of reading its true pose. It supplies "
                       "the image COLUMN only; range comes from depth (see perception."
                       "person_from_bearing_column and the 0.08 deg bearing / wrong-box finding).")
  ap.add_argument("--vlm-period", type=int, default=25, help="Control steps between VLM queries (25 = 0.5 s).")
  ap.add_argument("--yolo", metavar="WEIGHTS",
                  help="Fast detector for the per-step 'where'. With --vlm the VLM is demoted to "
                       "naming the target class once; without it the class is --target. Runs every "
                       "--yolo-period steps (~4 ms), so the control loop never waits on the VLM.")
  ap.add_argument("--yolo-period", type=int, default=1, help="Control steps between detector calls.")
  ap.add_argument("--yolo-conf", type=float, default=0.25)
  ap.add_argument("--target", default="person", help="Class the detector looks for.")
  ap.add_argument("--base-url", default="http://127.0.0.1:8091")
  ap.add_argument("--video", action="store_true")
  ap.add_argument("--no-overlay", action="store_true",
                  help="Record the raw ego view instead of drawing the detector box and planner banner.")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  course = build_course(args.course_length, args.course_width)
  camera = CameraSpec()
  cfg = make_twin_env_cfg(
    camera, terrain_generator=course.generator_cfg(seed=args.seed), num_envs=1,
    seed=args.seed, leader=True,
  )
  configure_torch_backends()
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=load_rl_cfg(BASE_TASK).clip_actions
  )
  u = env.unwrapped
  robot = u.scene["robot"]
  dt = u.step_dt
  bank = PolicyBank(env, default_checkpoints(args.ckpt_root), device)
  assert args.policy in bank.names, f"{args.policy!r} not in {bank.names}"

  start = course.start_course
  path = LeaderPath(start_x=float(start[0]) + args.gap, centre_y=float(start[1]))

  torch.manual_seed(args.seed)
  obs, _ = env.reset()
  # Place the leader before the first step so frame 0 already shows it.
  write_leader_pose(env, course.course_to_world(path.pos_at(0.0)), path.yaw_at(0.0))

  gains = FollowGains()
  pidx = torch.full((1,), bank.names.index(args.policy), dtype=torch.long, device=device)
  cam = u.scene[CAMERA_SENSOR_NAME]
  tm = u.termination_manager

  vlm = OpenAICompatVLM(base_url=f"{args.base_url}/v1", transcript=out / "vlm_transcript.jsonl") if args.vlm else None
  det = YoloDetector(args.yolo, conf=args.yolo_conf) if args.yolo else None
  target_label = args.target
  det_calls = det_hits = 0
  cur_box = None
  cur_conf = None
  overlay_events: list[VlmEvent] = []
  est_xy: np.ndarray | None = None  # last VLM+depth estimate of the person, held between queries
  vlm_calls = vlm_hits = 0
  vlm_plan_log: list[dict] = []
  vlm_block_s = 0.0
  pool = ThreadPoolExecutor(1)
  pending = None

  # Two-rate mode: the VLM is the planner. It names the class ONCE before the robot
  # moves (SARO plans, then executes), then re-confirms asynchronously -- the control
  # loop reads the last answer and never waits on a ~2 s inference.
  if vlm is not None and det is not None:
    # Distractors the detector CANNOT serve, so the planner's answer is testable:
    # pick wrong and the robot visibly stalls with nothing to follow. A single-class
    # choice would make "the VLM chose correctly" unfalsifiable.
    servable = {n for n in det._names.values()}
    classes = tuple(sorted(servable | {"chair", "dog", "bicycle", "potted plant"}))
    for _ in range(5):  # settle into a stand so the first frame is representative
      set_velocity_command(env, torch.zeros(1, 3, device=device))
      obs, _, _, _ = env.step(bank.act_per_env(obs, pidx))
    rgb0 = cam.data.rgb[0].cpu().numpy()
    _t0 = time.time()
    reply0 = vlm.ask(rgb0, P.follow_target(course.instruction, classes),
                     schema=P.follow_target_schema(classes), max_tokens=64, tag="plan")
    vlm_block_s += time.time() - _t0
    vlm_calls += 1
    named = P.parse_target(reply0.text or "", classes)
    if named:
      target_label = named
      vlm_hits += 1
    vlm_plan_log.append(dict(t=0.0, kind="plan", answer=named, raw=(reply0.text or "").strip()[:120],
                             latency_s=round(reply0.latency_s, 2), blocking=True))
    overlay_events.append(VlmEvent(0.0, "plan", named, reply0.latency_s, True))
    print(f"[PLAN] VLM chose target={named!r} from {classes} in {reply0.latency_s:.1f}s", flush=True)
  frames: list = []
  trace: list[dict] = []
  prev_leader_xy: np.ndarray | None = None
  outcome = "completed"
  wall0 = time.time()
  steps = int(args.duration / dt)

  for step in range(steps):
    t = step * dt
    leader_course = path.pos_at(t)
    leader_w = course.course_to_world(leader_course)
    write_leader_pose(env, leader_w, path.yaw_at(t))

    pos = robot.data.root_link_pos_w
    quat = robot.data.root_link_quat_w

    # Planner re-confirmation, off the control path: submit and keep stepping.
    if vlm is not None and det is not None:
      if pending is not None and pending.done():
        try:
          rep = pending.result()
          vlm_calls += 1
          ans = P.parse_target(rep.text or "", classes)
          if ans:
            target_label = ans
            vlm_hits += 1
          vlm_plan_log.append(dict(t=round(t, 2), kind="reconfirm", answer=ans,
                                   raw=(rep.text or "").strip()[:120],
                                   latency_s=round(rep.latency_s, 2), blocking=False))
          overlay_events.append(VlmEvent(t, "reconfirm", ans, rep.latency_s, False))
        finally:
          pending = None
      if pending is None and step > 0 and step % args.vlm_period == 0:
        _rgb = cam.data.rgb[0].cpu().numpy()
        pending = pool.submit(
          vlm.ask, _rgb, P.follow_target(course.instruction, classes),
          P.follow_target_schema(classes), 64, f"reconfirm_t{t:.1f}",
        )

    if det is not None and step % args.yolo_period == 0:
      rgb = cam.data.rgb[0].cpu().numpy()
      if det_calls == 0:
        det.warmup(rgb)
      hits = det.detect(rgb, target_label)
      det_calls += 1
      cur_box = tuple(hits[0].box_px) if hits else None
      cur_conf = hits[0].conf if hits else None
      if hits:
        depth = cam.data.depth[0, ..., 0].cpu().numpy()
        got = target_from_bearing_column(
          depth, camera, hits[0].centre_u, pos[0].cpu().numpy(), quat[0].cpu().numpy(),
        )
        if got is not None:
          est_xy = got
          det_hits += 1

    if det is None and vlm is not None and step % args.vlm_period == 0:
      rgb = cam.data.rgb[0].cpu().numpy()
      depth = cam.data.depth[0, ..., 0].cpu().numpy()
      reply = vlm.ask(rgb, P.perception_detect("person"), max_tokens=128, tag=f"t{t:.1f}")
      vlm_calls += 1
      box = P.parse_detect_box(reply.text or "", camera.width, camera.height)
      if box is not None and not P.box_is_degenerate(box, camera.width, camera.height):
        got = person_from_bearing_column(
          depth, camera, 0.5 * (box[0] + box[2]),
          pos[0].cpu().numpy(), quat[0].cpu().numpy(),
        )
        if got is not None:
          est_xy, _ = got, None
          vlm_hits += 1

    if vlm is not None or det is not None:
      # Hold the last good estimate between queries; before the first hit there is
      # nothing to follow, so stand still rather than guess.
      if est_xy is None:
        set_velocity_command(env, torch.zeros(1, 3, device=device))
        obs, _, dones, extras = env.step(bank.act_per_env(obs, pidx))
        continue
      target_np = est_xy
    else:
      target_np = leader_w[:2]
    leader_xy = torch.tensor(target_np, dtype=pos.dtype, device=device).unsqueeze(0)
    # Velocity from differencing observed positions -- what a tracker would give,
    # rather than reading the leader's script.
    if prev_leader_xy is None:
      leader_vel_np = np.zeros(2)
    else:
      _per = args.yolo_period if det else (args.vlm_period if vlm else 1)
      leader_vel_np = (target_np - prev_leader_xy) / (dt * max(1, _per))
    prev_leader_xy = np.asarray(target_np, dtype=np.float64).copy()
    leader_vel = torch.tensor(leader_vel_np, dtype=pos.dtype, device=device).unsqueeze(0)
    cmd, rng = follow_command(pos, quat, leader_xy, args.gap, gains, leader_vel_w=leader_vel)
    set_velocity_command(env, cmd)

    trace.append(dict(
      t=round(t, 3), range_m=float(rng[0]), gap_err_m=float(rng[0]) - args.gap,
      leader_speed=path.speed_at(t), leader_speed_est=float(np.linalg.norm(leader_vel_np)),
      cmd_vx=float(cmd[0, 0]), cmd_wz=float(cmd[0, 2]),
      robot_x=float(pos[0, 0]), leader_x=float(leader_w[0]),
      true_range_m=float(np.linalg.norm(leader_w[:2] - pos[0, :2].cpu().numpy())),
    ))
    if args.video and step % 2 == 0:
      _f = cam.data.rgb[0].cpu().numpy()
      if not args.no_overlay:
        _f = annotate(
          _f, t=t, target=target_label, box_px=cur_box, conf=cur_conf,
          det_ms=(det.latencies_ms[-1] if det and det.latencies_ms else None),
          range_m=float(rng[0]), gap_m=args.gap, events=overlay_events,
        )
      frames.append(_f)

    obs, _, dones, extras = env.step(bank.act_per_env(obs, pidx))
    timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
    if bool(dones[0]) and not bool(timeouts[0]):
      outcome = "fall:" + ",".join(name for name in tm.active_terms if bool(tm.get_term(name)[0]))
      break

  pool.shutdown(wait=False)
  if frames:
    import imageio.v2 as imageio  # noqa: PLC0415

    imageio.mimsave(out / "follow_ego.mp4", frames, fps=int(round(1 / (2 * dt))))

  # Skip the first second: the robot starts from a standing keyframe and the
  # initial range error is the spawn offset, not tracking performance.
  settled = [r for r in trace if r["t"] >= 1.0]
  errs = np.array([r["gap_err_m"] for r in settled]) if settled else np.zeros(1)
  summary = dict(
    outcome=outcome, gap_setpoint_m=args.gap, policy=args.policy,
    duration_s=round(trace[-1]["t"], 2) if trace else 0.0,
    range_mean_m=float(np.mean([r["range_m"] for r in settled])) if settled else 0.0,
    range_min_m=float(np.min([r["range_m"] for r in settled])) if settled else 0.0,
    range_max_m=float(np.max([r["range_m"] for r in settled])) if settled else 0.0,
    gap_err_rms_m=float(np.sqrt(np.mean(errs**2))),
    gap_err_abs_max_m=float(np.max(np.abs(errs))),
    leader_distance_m=round(path.distance_at(trace[-1]["t"]), 2) if trace else 0.0,
    source="yolo" if args.yolo else ("vlm" if args.vlm else "ground_truth"),
    vlm_calls=vlm_calls, vlm_detect_hits=vlm_hits,
    det_calls=det_calls, det_hits=det_hits,
    det_hit_rate=round(det_hits / max(1, det_calls), 3),
    det_median_latency_ms=round(det.median_latency_ms, 2) if det else None,
    vlm_target_label=target_label,
    vlm_blocking_s=round(vlm_block_s, 2),
    vlm_plan=vlm_plan_log,
    true_range_mean_m=float(np.mean([r["true_range_m"] for r in settled])) if settled else 0.0,
    true_gap_err_rms_m=float(np.sqrt(np.mean((np.array([r["true_range_m"] for r in settled]) - args.gap) ** 2))) if settled else 0.0,
    speed_changes=[asdict(c) for c in DEFAULT_SCHEDULE],
    wall_s=round(time.time() - wall0, 1),
  )
  (out / "trace.json").write_text(json.dumps(trace, indent=1))
  (out / "result.json").write_text(json.dumps(dict(summary=summary, course=asdict(course)), indent=1, default=str))
  print("[FOLLOW] " + " ".join(f"{k}={v}" for k, v in summary.items() if not isinstance(v, list)))
  env.close()


if __name__ == "__main__":
  main()
