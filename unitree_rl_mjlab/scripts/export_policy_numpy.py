"""Export a policy checkpoint to .npz for deploy_numpy/policy_numpy.py, and check it.

The check runs the real PyTorch policy and the numpy one on observations from a
simulator rollout on a stairs course and reports the largest difference in
action. An export that has not passed this check should not go near the robot.

Usage (from unitree_rl_mjlab/):
  PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/export_policy_numpy.py \
      eval_ckpts/go2_spec_stairs_v2/model_9999.pt --out deploy_numpy/stairs_v2.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy_numpy.policy_numpy import NumpyPolicy  # noqa: E402

NORMALIZER_EPS = 1e-2  # rsl_rl EmpiricalNormalization default: (x - mean) / (std + eps)


def export(checkpoint: str, out: str, clip_actions: float) -> None:
  actor = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
  layers = sorted({int(k.split(".")[1]) for k in actor if k.startswith("mlp.") and k.endswith(".weight")})
  arrays = dict(
    obs_mean=actor["obs_normalizer._mean"].numpy().reshape(-1),
    obs_std=actor["obs_normalizer._std"].numpy().reshape(-1),
    obs_eps=np.float32(NORMALIZER_EPS), clip_actions=np.float32(clip_actions), n_layers=np.int32(len(layers)),
  )
  for i, layer in enumerate(layers):
    arrays[f"w{i}"] = actor[f"mlp.{layer}.weight"].numpy()
    arrays[f"b{i}"] = actor[f"mlp.{layer}.bias"].numpy()
  Path(out).parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, **arrays)
  print(f"wrote {out}: obs {arrays['obs_mean'].shape[0]}, layers {[arrays[f'w{i}'].shape for i in range(len(layers))]}")


def verify(checkpoint: str, npz: str, steps: int) -> float:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  from src.vlm_nav.course import saro_courses
  from src.vlm_nav.policy_bank import PolicyBank, default_checkpoints
  from src.vlm_nav.twin_env import BASE_TASK, make_twin_env_cfg, set_velocity_command

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  clip = load_rl_cfg(BASE_TASK).clip_actions
  course = saro_courses(visual="plain", level="L2")["stairs_up"]
  cfg = make_twin_env_cfg(None, terrain_generator=course.generator_cfg(seed=0), num_envs=16, seed=0)
  cfg.observations["actor"].enable_corruption = True
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=clip)
  bank = PolicyBank(env, {"p": Path(checkpoint)}, device)
  policy = NumpyPolicy(npz)
  obs, _ = env.reset()
  worst = 0.0
  for _ in range(steps):
    set_velocity_command(env, [0.5, 0.0, 0.0])
    with torch.inference_mode():
      reference = bank.policies["p"](obs)
    ours = policy.act(obs["actor"].cpu().numpy())
    # The env wrapper clips what it receives; compare after the same clip.
    worst = max(worst, float(np.abs(np.clip(reference.cpu().numpy(), -clip, clip) - ours).max()))
    obs, _, _, _ = env.step(reference)
  return worst


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("checkpoint")
  ap.add_argument("--out", required=True)
  ap.add_argument("--clip-actions", type=float, default=6.0)
  ap.add_argument("--steps", type=int, default=200)
  ap.add_argument("--no-verify", action="store_true")
  args = ap.parse_args()
  export(args.checkpoint, args.out, args.clip_actions)
  if not args.no_verify:
    worst = verify(args.checkpoint, args.out, args.steps)
    print(f"largest |torch - numpy| action difference over {args.steps} steps x 16 robots: {worst:.2e}")
    if worst > 1e-3:
      raise SystemExit("MISMATCH: do not use this export")
    print("PARITY OK")


if __name__ == "__main__":
  main()
