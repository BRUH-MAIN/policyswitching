"""Headless numeric eval of a locomotion checkpoint under pinned conditions.

Built for comparing DIFFERENT policies against each other (the specialist
cross-terrain matrix, generalist vs. switching), which is why it does not just
reuse the training env config verbatim:

  1. The terrain curriculum is off. With it on, `terrain_levels_vel` promotes
     an env to harder terrain whenever it walks far enough and demotes it
     otherwise, during the eval rollout -- a better policy gets pushed onto
     harder ground until it too starts failing, so survival measures distance
     to the curriculum's equilibrium rather than capability. Terrain class and
     difficulty are set explicitly with --terrain / --difficulty instead (see
     `apply_eval_conditions` in env_cfgs.py).
  2. The command range is pinned (--lin-vel-x etc., default: the final stage
     of training's `command_vel` curriculum) and that step-keyed curriculum is
     removed. Checkpoints restore `common_step_counter` on load, so the old
     behaviour did land on the final stage -- but only implicitly; this makes
     the commanded range an explicit, printed eval condition.
  3. Velocity error is reported per step. The training log's
     `Metrics/twist/error_vel_xy` accumulates over each episode
     (mjlab velocity_command._update_metrics), so it grows with episode length
     and is not comparable across policies that survive for different lengths.

Pass --keep-curricula for the training config verbatim, e.g. to sanity-check
a number against the checkpoint's own training log.

Reports three groups of metrics:

  1. Survival -- episode completions, timeout vs. early-failure split, mean
     episode length/return. "Does it stay upright?"
  2. Locomotion -- commanded vs. achieved base velocity, per-step tracking
     error, distance travelled, stalled-while-commanded %. "Does it go
     anywhere?" Group 2 exists because group 1 cannot distinguish walking from
     bracing: stage1_model_31800.pt scored 80.5% survival / +39.5 return while
     translating a measured zero metres over an 8-second rollout. Always read
     the two groups together.
  3. Smoothness -- per-step action rate and actuator-force rate over the whole
     rollout. The whole-episode baseline that transition-boundary jerk
     (objective.md) gets normalised against.

--json-out writes every number plus the eval conditions, for aggregation by
a100/summarize_eval_matrix.py.
"""

import argparse
import json
import math
import os
from dataclasses import asdict

import torch

os.environ.setdefault("MUJOCO_GL", "egl")

import src.tasks  # noqa: F401  (registers Unitree-Go2-* tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.config.go2.env_cfgs import EVAL_TERRAINS, apply_eval_conditions

# Final stage of the `command_vel` curriculum in velocity_env_cfg.py: the range
# every policy in this repo was trained to convergence on.
TRAIN_FINAL_LIN_VEL_X = (-1.0, 2.0)
TRAIN_FINAL_LIN_VEL_Y = (-1.0, 1.0)
TRAIN_FINAL_ANG_VEL_Z = (-1.0, 1.0)


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
  ap.add_argument("--hf-stage", default="stage2", help="HF subdir, e.g. stage2.")
  ap.add_argument("--hf-cache", default="eval_ckpts", help="Where HF downloads land.")
  ap.add_argument("--num-envs", type=int, default=1024)
  ap.add_argument("--steps", type=int, default=1200)
  ap.add_argument("--seed", type=int, default=0, help="Env, terrain-generator and torch seed.")
  ap.add_argument("--command-name", default="twist")
  ap.add_argument(
    "--terrain",
    default="native",
    choices=EVAL_TERRAINS,
    help="'native' = the task's own sub-terrain mix; a class name = only that class; "
    "'mixed' = all four classes at equal weight.",
  )
  ap.add_argument(
    "--difficulty",
    type=float,
    default=None,
    help="Pin terrain difficulty to this value in [0, 1]. Omit to spread envs "
    "uniformly over all difficulty rows.",
  )
  ap.add_argument("--lin-vel-x", type=float, nargs=2, default=TRAIN_FINAL_LIN_VEL_X, metavar=("MIN", "MAX"))
  ap.add_argument("--lin-vel-y", type=float, nargs=2, default=TRAIN_FINAL_LIN_VEL_Y, metavar=("MIN", "MAX"))
  ap.add_argument("--ang-vel-z", type=float, nargs=2, default=TRAIN_FINAL_ANG_VEL_Z, metavar=("MIN", "MAX"))
  ap.add_argument(
    "--keep-curricula",
    action="store_true",
    help="Use the training env config verbatim (terrain + command curricula live). "
    "Only for comparing against a checkpoint's own training log -- NOT across policies.",
  )
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
  ap.add_argument(
    "--ablate-height-scan",
    action="store_true",
    help=(
      "Replace the actor's height_scan input with the policy's own normalizer mean "
      "(i.e. 'average terrain', normalized input 0) every step. If a policy's numbers "
      "barely move, it isn't using exteroception. Stock-PPO policies only (not PAS)."
    ),
  )
  ap.add_argument("--label", default=None, help="Name for this policy in --json-out.")
  ap.add_argument("--json-out", default=None, help="Write all metrics + eval conditions here.")
  args = ap.parse_args()

  if not args.checkpoint and not args.hf_repo:
    ap.error("pass --checkpoint <path> or --hf-repo <repo_id>")
  if args.keep_curricula and (args.terrain != "native" or args.difficulty is not None):
    ap.error("--keep-curricula can't be combined with --terrain/--difficulty")
  checkpoint = args.checkpoint or resolve_hf_checkpoint(
    args.hf_repo, args.hf_stage, args.hf_cache
  )

  configure_torch_backends()
  torch.manual_seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"[INFO] device={device} task={args.task} checkpoint={checkpoint}")

  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  conditions: dict = {"keep_curricula": args.keep_curricula, "seed": args.seed}
  if args.keep_curricula:
    print("[WARN] --keep-curricula: terrain difficulty adapts to the policy mid-rollout;")
    print("       these numbers are NOT comparable across policies.")
  else:
    apply_eval_conditions(
      env_cfg, terrain=args.terrain, difficulty=args.difficulty, seed=args.seed
    )
    env_cfg.curriculum.pop("command_vel", None)
    ranges = env_cfg.commands[args.command_name].ranges
    ranges.lin_vel_x = tuple(args.lin_vel_x)
    ranges.lin_vel_y = tuple(args.lin_vel_y)
    ranges.ang_vel_z = tuple(args.ang_vel_z)
    conditions.update(
      terrain=args.terrain,
      difficulty=args.difficulty,
      lin_vel_x=list(args.lin_vel_x),
      lin_vel_y=list(args.lin_vel_y),
      ang_vel_z=list(args.ang_vel_z),
    )
    print(
      f"[INFO] Eval conditions: terrain={args.terrain} "
      f"difficulty={'uniform over rows' if args.difficulty is None else args.difficulty} "
      f"lin_vel_x={tuple(args.lin_vel_x)} lin_vel_y={tuple(args.lin_vel_y)} "
      f"ang_vel_z={tuple(args.ang_vel_z)} seed={args.seed} (curricula off)"
    )
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
    conditions["anneal_prob"] = float(actor.anneal_prob)

  policy = runner.get_inference_policy(device=device)

  unwrapped = env.unwrapped

  # Height-scan ablation: find the height_scan slice of the actor group and the
  # policy's normalizer mean for it. Filling with the mean feeds a normalized
  # input of exactly 0 -- no terrain information, but not out of distribution.
  scan_slice = None
  scan_fill = None
  if args.ablate_height_scan:
    if hasattr(actor, "anneal_prob"):
      raise SystemExit("[ERROR] --ablate-height-scan is for stock-PPO policies; PAS "
                       "encodes height_scan separately (use --anneal-prob 0.0 instead).")
    om = unwrapped.observation_manager
    start = 0
    for name, shape in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"]):
      size = math.prod(shape)
      if name == "height_scan":
        scan_slice = slice(start, start + size)
        break
      start += size
    normalizer = getattr(actor, "obs_normalizer", None)
    if scan_slice is None or not hasattr(normalizer, "_mean"):
      raise SystemExit("[ERROR] --ablate-height-scan: no height_scan term / obs normalizer on this actor.")
    if normalizer._mean.shape[-1] != math.prod(om.group_obs_dim["actor"]):
      raise SystemExit("[ERROR] --ablate-height-scan: actor normalizer doesn't match the actor obs group.")
    scan_fill = normalizer._mean[0, scan_slice].detach().clone()
    conditions["ablate_height_scan"] = True
    print(f"[INFO] Height-scan ablation ON: actor obs[{scan_slice.start}:{scan_slice.stop}] "
          "replaced with the policy's normalizer mean every step.")
  robot = unwrapped.scene["robot"]
  action_manager = unwrapped.action_manager
  dt = unwrapped.step_dt

  obs, _ = env.reset()
  n = args.num_envs
  ep_len = torch.zeros(n, device=device)
  reward_accum = torch.zeros(n, device=device)

  total_episodes = 0
  total_timeouts = 0
  total_fails = 0
  # Per-termination-term counts over completed episodes (an env can trip several
  # terms on one step, so these can sum to more than total_episodes).
  term_manager = unwrapped.termination_manager
  term_counts = {name: 0 for name in term_manager.active_terms}
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

  # Smoothness accumulators. Rates need two consecutive in-episode steps, so an
  # env is excluded on the step it resets AND the step after (a reset zeroes
  # prev_action and teleports the robot, which would read as a huge spike).
  smooth_samples = 0
  smooth_steps = 0
  sum_action_rate = 0.0
  sum_force_rate = 0.0
  sum_force_rate_p99 = 0.0

  prev_pos = robot.data.root_link_pos_w[:, :2].clone()
  prev_force = robot.data.actuator_force.clone()
  prev_done = torch.ones(n, dtype=torch.bool, device=device)

  for step in range(args.steps):
    with torch.inference_mode():
      if scan_slice is not None:
        actor_obs = obs["actor"].clone()
        actor_obs[:, scan_slice] = scan_fill
        obs = obs.clone()
        obs["actor"] = actor_obs
      actions = policy(obs)
    obs, rew, dones, extras = env.step(actions)
    ep_len += 1
    reward_accum += rew

    done_mask = dones.bool()

    with torch.inference_mode():
      command = unwrapped.command_manager.get_command(args.command_name)
      lin_vel_b = robot.data.root_link_lin_vel_b
      ang_vel_b = robot.data.root_link_ang_vel_b
      pos_xy = robot.data.root_link_pos_w[:, :2]
      force = robot.data.actuator_force

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

      smooth_valid = valid & ~prev_done
      n_smooth = int(smooth_valid.sum().item())
      if n_smooth:
        action_rate = torch.sum(
          torch.square(action_manager.action - action_manager.prev_action), dim=1
        )
        force_rate = torch.linalg.norm(force - prev_force, dim=1) / dt
        smooth_samples += n_smooth
        smooth_steps += 1
        sum_action_rate += float(action_rate[smooth_valid].sum())
        sum_force_rate += float(force_rate[smooth_valid].sum())
        sum_force_rate_p99 += float(torch.quantile(force_rate[smooth_valid], 0.99))

      prev_pos = pos_xy.clone()
      prev_force = force.clone()
      prev_done = done_mask.clone()

    if done_mask.any():
      timeouts = extras.get("time_outs")
      if timeouts is None:
        timeouts = torch.zeros_like(done_mask)
      n_done = done_mask.sum().item()
      n_timeout = (timeouts.bool() & done_mask).sum().item()
      n_fail = n_done - n_timeout

      for name in term_counts:
        term_counts[name] += int((term_manager.get_term(name).bool() & done_mask).sum())
      total_episodes += n_done
      total_timeouts += n_timeout
      total_fails += n_fail
      sum_ep_len_at_end += ep_len[done_mask].sum().item()
      sum_return_at_end += reward_accum[done_mask].sum().item()

      ep_len[done_mask] = 0
      reward_accum[done_mask] = 0

    if (step + 1) % 100 == 0:
      print(f"[step {step + 1}/{args.steps}] episodes so far: {total_episodes}")

  result: dict = {
    "label": args.label,
    "task": args.task,
    "checkpoint": checkpoint,
    "num_envs": n,
    "steps": args.steps,
    "conditions": conditions,
  }

  print("\n===== EVAL RESULT =====")
  print(f"Steps simulated: {args.steps}  |  parallel envs: {n}")
  print(f"Checkpoint: {checkpoint}")

  print("\n-- survival --")
  print(f"Episodes completed: {total_episodes}")
  survival: dict = {"episodes": total_episodes}
  if total_episodes:
    survival.update(
      survival_pct=100 * total_timeouts / total_episodes,
      fall_pct=100 * total_fails / total_episodes,
      mean_ep_len=sum_ep_len_at_end / total_episodes,
      mean_return=sum_return_at_end / total_episodes,
    )
    print(f"  time_out (survived full episode): {total_timeouts} "
          f"({survival['survival_pct']:.1f}%)")
    print(f"  failed early (fell / illegal contact / etc): {total_fails} "
          f"({survival['fall_pct']:.1f}%)")
    survival["terminations"] = dict(term_counts)
    print("  terminations by cause (episodes; terms can overlap): "
          + ", ".join(f"{k}={v}" for k, v in term_counts.items()))
    print(f"  mean episode length at end: {survival['mean_ep_len']:.1f}")
    print(f"  mean episode return: {survival['mean_return']:.3f}")
  else:
    print("  No episodes completed in this window -- increase --steps.")
  result["survival"] = survival

  print("\n-- locomotion --")
  locomotion: dict = {}
  if not locomotion_samples:
    print("  No valid samples collected.")
  else:
    mean_cmd = sum_cmd_speed / locomotion_samples
    mean_actual = sum_actual_speed / locomotion_samples
    locomotion.update(
      mean_cmd_speed=mean_cmd,
      mean_actual_speed=mean_actual,
      achieved_pct_of_cmd=100 * mean_actual / mean_cmd if mean_cmd > 1e-6 else None,
      lin_vel_error_per_step=sum_lin_vel_error / locomotion_samples,
      ang_vel_error_per_step=sum_ang_vel_error / locomotion_samples,
      distance_rate=sum_distance / locomotion_samples / dt,
      stalled_pct=100 * stalled_steps / moving_cmd_steps if moving_cmd_steps else None,
      total_distance_m=sum_distance,
      falls_per_100m=100 * total_fails / sum_distance if sum_distance > 1e-6 else None,
    )
    achieved = f"  mean achieved speed:   {mean_actual:.3f} m/s"
    if locomotion["achieved_pct_of_cmd"] is not None:
      achieved += f"  ({locomotion['achieved_pct_of_cmd']:.0f}% of commanded)"
    print(f"  mean commanded speed:  {mean_cmd:.3f} m/s")
    print(achieved)
    print(f"  linear vel error, per step:  {locomotion['lin_vel_error_per_step']:.3f} m/s"
          "  <- the cross-policy tracking metric (not error_vel_xy)")
    print(f"  angular vel error, per step: {locomotion['ang_vel_error_per_step']:.3f} rad/s")
    print(f"  mean distance travelled per env per second: {locomotion['distance_rate']:.3f} m/s")
    print(f"  total distance travelled (all envs): {sum_distance:.0f} m")
    if locomotion["falls_per_100m"] is not None:
      print(f"  falls per 100 m travelled: {locomotion['falls_per_100m']:.2f}")
    if moving_cmd_steps:
      print(f"  stalled while commanded to move "
            f"(cmd > {args.moving_command_threshold} m/s, achieved < "
            f"{args.stalled_speed_threshold} m/s): "
            f"{stalled_steps}/{moving_cmd_steps} "
            f"({locomotion['stalled_pct']:.1f}%)")
      if stalled_steps / moving_cmd_steps > 0.5:
        print("  [WARN] Stalled for most of the steps where movement was commanded --")
        print("         this policy is likely bracing in place, not locomoting,")
        print("         regardless of how good the survival numbers above look.")
    else:
      print(f"  no steps with commanded speed > {args.moving_command_threshold} m/s")
  result["locomotion"] = locomotion

  print("\n-- smoothness (whole rollout) --")
  smoothness: dict = {}
  if not smooth_samples:
    print("  No valid samples collected.")
  else:
    smoothness.update(
      action_rate_l2=sum_action_rate / smooth_samples,
      actuator_force_rate=sum_force_rate / smooth_samples,
      actuator_force_rate_p99=sum_force_rate_p99 / smooth_steps,
    )
    print(f"  action rate (sum sq. change per step): {smoothness['action_rate_l2']:.4f}")
    print(f"  actuator force rate:     {smoothness['actuator_force_rate']:.1f} N*m/s (mean)")
    print(f"  actuator force rate p99: {smoothness['actuator_force_rate_p99']:.1f} N*m/s "
          "(per-step 99th pct across envs, averaged over steps)")
  result["smoothness"] = smoothness

  if args.json_out:
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
      json.dump(result, f, indent=2)
    print(f"\n[INFO] Wrote {args.json_out}")


if __name__ == "__main__":
  main()
