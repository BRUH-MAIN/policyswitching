"""Headless, deterministic numeric eval for a PAS checkpoint.

Unlike play.py (which uses the "play" env config with effectively-infinite
episode length, no curriculum, and a live viewer loop that never exits on
its own), this reuses the *training* env config -- same finite episode
length, curriculum, and domain randomization the checkpoint was actually
trained/logged under -- so the printed stats are directly comparable to a
training-log "Learning iteration" block. No viewer, no video: it runs N
steps and prints aggregate stats, then exits.

Reports two groups of metrics:

  1. Survival -- episode completions, timeout vs. early-failure split, mean
     episode length/return. These answer "does it stay upright?"
  2. Locomotion -- commanded vs. achieved base velocity, tracking error, and
     distance actually travelled. These answer "does it *go anywhere*?"

Group 2 exists because group 1 cannot distinguish a policy that walks from
one that braces in place and never falls: stage1_model_31800.pt scored 80.5%
survival / +39.5 return while translating a measured zero metres over an
8-second rollout. Bracing in place actively maximises the energy,
joint_vel_l2 and action_rate_l2 penalty terms, so a degenerate policy can
look healthy on survival alone. Always read the two groups together.
"""

import argparse
import os
from dataclasses import asdict

import torch

os.environ.setdefault("MUJOCO_GL", "egl")

import src.tasks  # noqa: F401  (registers Unitree-Go2-PAS-* tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def resolve_hf_checkpoint(repo_id: str, stage: str, cache_dir: str) -> str:
  """Download the highest-iteration model_*.pt under {stage}/ and return its local path.

  Mirrors a100/hf_sync.py's `latest_checkpoint` selection so eval always runs
  against the same checkpoint the training job most recently pushed, rather
  than a stale path baked into a shell script.
  """
  from huggingface_hub import HfApi, hf_hub_download

  files = HfApi().list_repo_files(repo_id, repo_type="model")
  candidates = []
  for f in files:
    if f.startswith(f"{stage}/model_") and f.endswith(".pt"):
      try:
        candidates.append((int(f.split("model_")[-1].split(".pt")[0]), f))
      except ValueError:
        continue
  if not candidates:
    raise FileNotFoundError(f"No {stage}/model_*.pt checkpoints in hf.co/{repo_id}")
  iteration, remote_path = max(candidates, key=lambda x: x[0])
  print(f"[INFO] Latest {stage} checkpoint on HF: {remote_path} (iteration {iteration})")
  os.makedirs(cache_dir, exist_ok=True)
  return hf_hub_download(repo_id, remote_path, repo_type="model", local_dir=cache_dir)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Unitree-Go2-PAS-Oracle")
  ap.add_argument("--checkpoint", help="Local .pt path. Omit to pull from --hf-repo.")
  ap.add_argument("--hf-repo", help="HF model repo to pull the latest checkpoint from.")
  ap.add_argument("--hf-stage", default="stage2", choices=["stage1", "stage2"])
  ap.add_argument("--hf-cache", default="eval_ckpts", help="Where HF downloads land.")
  ap.add_argument("--num-envs", type=int, default=1024)
  ap.add_argument("--steps", type=int, default=1200)
  ap.add_argument("--command-name", default="twist")
  ap.add_argument(
    "--moving-command-threshold",
    type=float,
    default=0.1,
    help="Commanded speed (m/s) above which the robot is expected to actually move.",
  )
  ap.add_argument(
    "--stalled-speed-threshold",
    type=float,
    default=0.05,
    help="Achieved speed (m/s) below which the robot counts as stalled.",
  )
  ap.add_argument(
    "--anneal-prob",
    type=float,
    default=None,
    help=(
      "Override PasActorModel.anneal_prob for this eval. 1.0 = oracle mode (the "
      "actor reads the TRUE privileged terrain latent); 0.0 = estimator-only, i.e. "
      "the proprioception-only policy Stage 2 is distilling toward. Omit to leave "
      "the model's own value untouched."
    ),
  )
  args = ap.parse_args()

  if not args.checkpoint and not args.hf_repo:
    ap.error("pass --checkpoint <path> or --hf-repo <repo_id>")
  checkpoint = args.checkpoint or resolve_hf_checkpoint(
    args.hf_repo, args.hf_stage, args.hf_cache
  )

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"[INFO] device={device} task={args.task} checkpoint={checkpoint}")

  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(args.task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task)
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)

  # `anneal_prob` is runtime state on PasActorModel, seeded from
  # `initial_anneal_prob` (1.0) and decayed only by PasPPO.update() DURING
  # training -- it is not stored in the checkpoint. So without an override,
  # every eval (Stage 1 or Stage 2) silently runs in oracle mode, reading the
  # true privileged terrain latent. That makes a Stage-2 checkpoint look good
  # while never testing the estimator the whole stage exists to train. Report
  # the mode explicitly, and allow forcing it.
  actor = getattr(getattr(runner, "alg", None), "actor", None)
  if actor is not None and hasattr(actor, "anneal_prob"):
    if args.anneal_prob is not None:
      actor.anneal_prob = args.anneal_prob
    mode = (
      "oracle (true privileged terrain latent)"
      if actor.anneal_prob >= 1.0
      else "estimator-only (proprioception, no privileged latent)"
      if actor.anneal_prob <= 0.0
      else "mixed real/predicted latent"
    )
    print(f"[INFO] PAS latent mode: anneal_prob={actor.anneal_prob} -> {mode}")
    if args.anneal_prob is None and actor.anneal_prob >= 1.0:
      print("[WARN] Evaluating in ORACLE mode. For a Stage-2 checkpoint this does NOT")
      print("       test the distilled estimator -- pass --anneal-prob 0.0 for that.")

  policy = runner.get_inference_policy(device=device)

  robot = env.unwrapped.scene["robot"]

  obs, _ = env.reset()
  n = args.num_envs
  ep_len = torch.zeros(n, device=device)
  reward_accum = torch.zeros(n, device=device)

  total_episodes = 0
  total_timeouts = 0
  total_fails = 0
  sum_ep_len_at_end = 0.0
  sum_return_at_end = 0.0

  # Locomotion accumulators. Every per-step quantity is summed over envs and
  # steps, then divided by `locomotion_samples` at the end.
  locomotion_samples = 0
  sum_cmd_speed = 0.0
  sum_actual_speed = 0.0
  sum_lin_vel_error = 0.0
  sum_ang_vel_error = 0.0
  sum_distance = 0.0
  moving_cmd_steps = 0
  stalled_steps = 0

  prev_pos = robot.data.root_link_pos_w[:, :2].clone()

  for step in range(args.steps):
    with torch.inference_mode():
      actions = policy(obs)
    obs, rew, dones, extras = env.step(actions)
    ep_len += 1
    reward_accum += rew

    done_mask = dones.bool()

    with torch.inference_mode():
      command = env.unwrapped.command_manager.get_command(args.command_name)
      lin_vel_b = robot.data.root_link_lin_vel_b
      ang_vel_b = robot.data.root_link_ang_vel_b
      pos_xy = robot.data.root_link_pos_w[:, :2]

      # An env that just reset teleported to a new spawn, so its position
      # delta is meaningless -- exclude those envs from this step's stats.
      valid = ~done_mask
      n_valid = int(valid.sum().item())
      if n_valid:
        cmd_speed = torch.linalg.norm(command[:, :2], dim=1)
        actual_speed = torch.linalg.norm(lin_vel_b[:, :2], dim=1)
        lin_err = torch.linalg.norm(command[:, :2] - lin_vel_b[:, :2], dim=1)
        ang_err = torch.abs(command[:, 2] - ang_vel_b[:, 2])
        distance = torch.linalg.norm(pos_xy - prev_pos, dim=1)

        locomotion_samples += n_valid
        sum_cmd_speed += float(cmd_speed[valid].sum())
        sum_actual_speed += float(actual_speed[valid].sum())
        sum_lin_vel_error += float(lin_err[valid].sum())
        sum_ang_vel_error += float(ang_err[valid].sum())
        sum_distance += float(distance[valid].sum())

        commanded_to_move = valid & (cmd_speed > args.moving_command_threshold)
        moving_cmd_steps += int(commanded_to_move.sum().item())
        stalled_steps += int(
          (commanded_to_move & (actual_speed < args.stalled_speed_threshold))
          .sum()
          .item()
        )

      prev_pos = pos_xy.clone()

    if done_mask.any():
      timeouts = extras.get("time_outs")
      if timeouts is None:
        timeouts = torch.zeros_like(done_mask)
      n_done = done_mask.sum().item()
      n_timeout = (timeouts.bool() & done_mask).sum().item()
      n_fail = n_done - n_timeout

      total_episodes += n_done
      total_timeouts += n_timeout
      total_fails += n_fail
      sum_ep_len_at_end += ep_len[done_mask].sum().item()
      sum_return_at_end += reward_accum[done_mask].sum().item()

      ep_len[done_mask] = 0
      reward_accum[done_mask] = 0

    if (step + 1) % 100 == 0:
      print(f"[step {step + 1}/{args.steps}] episodes so far: {total_episodes}")

  print("\n===== EVAL RESULT =====")
  print(f"Steps simulated: {args.steps}  |  parallel envs: {n}")
  print(f"Checkpoint: {checkpoint}")

  print("\n-- survival --")
  print(f"Episodes completed: {total_episodes}")
  if total_episodes:
    print(f"  time_out (survived full episode): {total_timeouts} "
          f"({100 * total_timeouts / total_episodes:.1f}%)")
    print(f"  failed early (fell / illegal contact / etc): {total_fails} "
          f"({100 * total_fails / total_episodes:.1f}%)")
    print(f"  mean episode length at end: {sum_ep_len_at_end / total_episodes:.1f}")
    print(f"  mean episode return: {sum_return_at_end / total_episodes:.3f}")
  else:
    print("  No episodes completed in this window -- increase --steps.")

  print("\n-- locomotion --")
  if not locomotion_samples:
    print("  No valid samples collected.")
  else:
    mean_cmd = sum_cmd_speed / locomotion_samples
    mean_actual = sum_actual_speed / locomotion_samples
    dt = env.unwrapped.step_dt
    achieved = f"  mean achieved speed:   {mean_actual:.3f} m/s"
    if mean_cmd > 1e-6:
      achieved += f"  ({100 * mean_actual / mean_cmd:.0f}% of commanded)"
    print(f"  mean commanded speed:  {mean_cmd:.3f} m/s")
    print(achieved)
    print(f"  mean linear vel error: {sum_lin_vel_error / locomotion_samples:.3f} m/s")
    print(f"  mean angular vel error:{sum_ang_vel_error / locomotion_samples:.3f} rad/s")
    print(f"  mean distance travelled per env per second: "
          f"{sum_distance / locomotion_samples / dt:.3f} m/s")
    print(f"  total distance travelled (all envs): {sum_distance:.0f} m")
    if moving_cmd_steps:
      print(f"  stalled while commanded to move "
            f"(cmd > {args.moving_command_threshold} m/s, achieved < "
            f"{args.stalled_speed_threshold} m/s): "
            f"{stalled_steps}/{moving_cmd_steps} "
            f"({100 * stalled_steps / moving_cmd_steps:.1f}%)")
      if stalled_steps / moving_cmd_steps > 0.5:
        print("  [WARN] Stalled for most of the steps where movement was commanded --")
        print("         this policy is likely bracing in place, not locomoting,")
        print("         regardless of how good the survival numbers above look.")
    else:
      print(f"  no steps with commanded speed > {args.moving_command_threshold} m/s")


if __name__ == "__main__":
  main()
