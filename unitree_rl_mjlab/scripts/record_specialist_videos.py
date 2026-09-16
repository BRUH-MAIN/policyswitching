"""Record a qualitative video of one specialist checkpoint on one terrain class.

Built for the report_content/ deliverable rather than as a permanent tool: it
answers "what does specialist X actually look like walking on terrain Y", which
`play.py` cannot do (its task argument bakes together the checkpoint's own task
AND the terrain -- there's no way to run one specialist's weights against a
different terrain), and it fixes a documented gotcha (findings.md bug #3) that
`play.py` still has: with the task's own command distribution, ~5% of episodes
draw a "stand still" command, and the offscreen camera always follows env 0, so
an unlucky draw silently records a healthy policy standing motionless for the
whole clip. This forces a fixed forward-walk command instead.

Loading and env-construction pattern lifted directly from eval_checkpoint.py
(apply_eval_conditions for terrain control, the runner/policy loading sequence)
and play.py (VideoRecorder wiring) -- fused because neither script alone does
both "control the terrain independent of the checkpoint" and "record video
headlessly, no interactive viewer".

Usage (from unitree_rl_mjlab/, PYTHONPATH set to it, MUJOCO_GL=egl):

  python3 scripts/record_specialist_videos.py \
      --task Unitree-Go2-Spec-Flat --checkpoint eval_ckpts/go2_spec_flat/model_9999.pt \
      --terrain stairs --out-dir ../report_content/videos --name flat_on_stairs
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402

from src.tasks.velocity.config.go2.env_cfgs import EVAL_TERRAINS, apply_eval_conditions  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", required=True, help="e.g. Unitree-Go2-Spec-Flat")
  ap.add_argument("--checkpoint", required=True, help="Path to model_*.pt, independent of --task.")
  ap.add_argument("--terrain", required=True, choices=EVAL_TERRAINS)
  ap.add_argument("--difficulty", type=float, default=0.5)
  ap.add_argument("--command-name", default="twist")
  ap.add_argument("--forward-speed", type=float, default=1.0,
                  help="Fixed forward command (m/s). Fully deterministic, not "
                       "sampled, so the clip never shows a standing robot (bug #3).")
  ap.add_argument("--video-length", type=int, default=300, help="Steps; dt=0.02s, so 300 -> 6s.")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--show-scan-rays", action="store_true",
                   help="Keep the height-scan sensor's ray debug-visualization. Off by "
                        "default -- confirmed (2026-09-16) to visually break down into "
                        "what looks like a physics glitch (contorted legs, then a sudden "
                        "camera-framing jump) once the robot has walked several metres "
                        "from its spawn point. Root-caused by disabling it and comparing: "
                        "the robot's actual state (position/orientation/joint angles) is "
                        "provably identical whether or not this is on -- turning it off "
                        "does not change the walk, only removes the broken overlay.")
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--name", required=True, help="Video filename prefix (no extension).")
  ap.add_argument("--device", default=None)
  args = ap.parse_args()

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  ckpt = Path(args.checkpoint)
  if not ckpt.exists():
    raise SystemExit(f"[ERROR] checkpoint not found: {ckpt}")

  print(f"[INFO] task={args.task} checkpoint={ckpt} terrain={args.terrain} device={device}")

  # play=True: near-infinite episodes, no curriculum -- CLAUDE.md's own guidance
  # for qualitative inspection, so the clip isn't cut short by a training-time
  # termination or curriculum artifact.
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  apply_eval_conditions(env_cfg, terrain=args.terrain, difficulty=args.difficulty, seed=args.seed)
  env_cfg.curriculum.pop("command_vel", None)

  # Force a fixed nonzero forward command. Two changes, both needed: pinning the
  # range alone isn't sufficient because `rel_standing_envs` overrides the
  # sampled range with a hard zero command for a fraction of resets regardless
  # of how narrow the range is.
  cmd = env_cfg.commands[args.command_name]
  cmd.rel_standing_envs = 0.0
  cmd.ranges.lin_vel_x = (args.forward_speed, args.forward_speed)
  cmd.ranges.lin_vel_y = (0.0, 0.0)
  cmd.ranges.ang_vel_z = (0.0, 0.0)

  if not args.show_scan_rays:
    sensors = list(env_cfg.scene.sensors)
    for i, s in enumerate(sensors):
      if getattr(s, "name", None) == "terrain_scan":
        sensors[i] = replace(s, debug_vis=False)
    env_cfg.scene.sensors = tuple(sensors)

  agent_cfg = load_rl_cfg(args.task)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  video_env = VideoRecorder(
    raw_env,
    video_folder=args.out_dir,
    step_trigger=lambda step: step == 0,
    video_length=args.video_length,
    name_prefix=args.name,
    disable_logger=False,
  )
  env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task)
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  obs, _ = env.reset()
  with torch.inference_mode():
    for _ in range(args.video_length):
      action = policy(obs)
      obs, _, _, _ = env.step(action)

  # Call the VideoRecorder's own close() directly rather than relying on
  # RslRlVecEnvWrapper.close() to propagate -- guarantees the mp4 is finalized
  # regardless of what the outer wrapper's close() does.
  video_env.close()
  print(f"[INFO] done: {args.name}")


if __name__ == "__main__":
  main()
