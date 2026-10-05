"""Measure what the terrain curriculum's promote/demote rule does to a policy, row by row.

Question this answers: `terrain_levels` has sat at 1-2 of 10 in every training run, even
for policies that cope with much harder rows in evaluation. Is that the policy's limit, or
the rule in `mjlab.tasks.velocity.mdp.terrain_levels_vel`?

    move_up   = distance_from_origin > patch_size / 2          (4 m on an 8 m patch)
    move_down = distance_from_origin < |command_xy| * episode_length_s * 0.5, and not move_up

This script runs a trained policy with every env held on a fixed row (rows spread uniformly,
no promotion or demotion applied) and records, at every episode end, what that rule WOULD
have done, plus how the episode ended. From that it builds, per terrain type and row:

  * the fraction promoted / demoted / left alone under the current rule,
  * the same under two alternative rules (see ALTERNATIVES),
  * the stationary row distribution the rule would drive a training run to (a Markov chain
    over rows; a promotion past the top row re-draws the row uniformly, as in mjlab).

The chain is a check on the measurement: for a policy that was trained under the current
rule, its predicted mean level should land near the mean `terrain_levels` of that run's log.

Uses the task's own config (curricula live) apart from fixing the rows, so the command
range is the task's (stage 0 only for the V2/V3 tasks, since common_step_counter starts at 0
and their command curriculum has no second stage).
"""

import argparse
import json
import os

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

import src.tasks  # noqa: F401  (registers Unitree-Go2-* tasks)
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

# Alternative promote/demote rules, evaluated offline on the recorded episodes.
#  survive: promote if the episode ran to the time limit AND walked at least MIN_PROGRESS_M
#           from the spawn (the stairs start 1.5 m from the patch centre); demote if it
#           ended early (illegal contact / fell). Stalling on a step neither promotes nor
#           demotes, so a policy that cannot yet climb row r keeps training on row r.
#  survive_or_stall_demote: as survive, but also demote an episode that never got onto the
#           stairs at all (< MIN_ENTER_M) while being commanded to move.
MIN_PROGRESS_M = 2.5
MIN_ENTER_M = 1.0


def rule_current(r):
  up = r["dist"] > r["half_patch"]
  down = (r["dist"] < r["cmd"] * r["episode_s"] * 0.5) & ~up
  return up, down


def rule_survive(r):
  up = r["timeout"] & (r["dist"] > MIN_PROGRESS_M)
  down = r["terminated"]
  return up, down & ~up


def rule_survive_or_stall_demote(r):
  up, down = rule_survive(r)
  stalled = r["timeout"] & (r["dist"] < MIN_ENTER_M) & (r["cmd"] > 0.3)
  return up, (down | stalled) & ~up


ALTERNATIVES = {
  "current": rule_current,
  "survive": rule_survive,
  "survive_or_stall_demote": rule_survive_or_stall_demote,
}


def stationary_mean_level(p_up, p_down, num_rows):
  """Mean row under a birth-death chain with a uniform redraw past the top row."""
  P = np.zeros((num_rows, num_rows))
  for r in range(num_rows):
    up, down = float(p_up[r]), float(p_down[r])
    stay = 1.0 - up - down
    P[r, r] += stay
    P[r, max(r - 1, 0)] += down
    if r + 1 < num_rows:
      P[r, r + 1] += up
    else:
      P[r, :] += up / num_rows  # promoted past the top: redrawn uniformly
  pi = np.full(num_rows, 1.0 / num_rows)
  for _ in range(20000):
    pi = pi @ P
  return float(pi @ np.arange(num_rows)), pi.tolist()


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", default="Unitree-Go2-Spec-StairsV2")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, default=2048)
  ap.add_argument("--steps", type=int, default=4000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--episode-s", type=float, default=None,
                  help="Override episode length (smoke tests only; the rule's thresholds scale with it).")
  ap.add_argument("--json-out", default=None)
  args = ap.parse_args()

  configure_torch_backends()
  torch.manual_seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  if args.episode_s is not None:
    env_cfg.episode_length_s = args.episode_s
  assert "terrain_levels" in env_cfg.curriculum, "task has no terrain_levels curriculum"
  env_cfg.scene.terrain.max_init_terrain_level = None  # rows uniform over 0..num_rows-1
  gen = env_cfg.scene.terrain.terrain_generator
  type_names = list(gen.sub_terrains)
  print(f"[INFO] task={args.task} rows={gen.num_rows} cols={gen.num_cols} types={type_names} "
        f"size={gen.size} difficulty_range={gen.difficulty_range}")
  for name, sub in gen.sub_terrains.items():
    print(f"[INFO]   {name}: step_height_range={getattr(sub, 'step_height_range', None)} "
          f"step_width={getattr(sub, 'step_width', None)} proportion={sub.proportion}")

  agent_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  unwrapped = env
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(args.task)(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  terrain = unwrapped.scene.terrain
  robot = unwrapped.scene["robot"]
  half_patch = gen.size[0] / 2.0
  episode_s = float(unwrapped.max_episode_length_s)
  num_rows, num_cols = gen.num_rows, gen.num_cols

  # Column -> sub-terrain index, same cumulative-proportion rule as the generator.
  props = np.array([s.proportion for s in gen.sub_terrains.values()], dtype=float)
  props /= props.sum()
  col_type = np.array(
    [int(np.min(np.where(c / num_cols + 0.001 < np.cumsum(props))[0])) for c in range(num_cols)]
  )

  records: list[dict] = []

  def hold_and_record(env_ids, move_up, move_down):
    """Replaces terrain.update_env_origins: record what the rule decided, change nothing."""
    if unwrapped.common_step_counter == 0:  # the initial env.reset(): zero-length episodes, not data
      return
    command = unwrapped.command_manager.get_command("twist")
    distance = torch.norm(
      robot.data.root_link_pos_w[env_ids, :2] - unwrapped.scene.env_origins[env_ids, :2], dim=1
    )
    tm = unwrapped.termination_manager
    records.append(
      dict(
        row=terrain.terrain_levels[env_ids].cpu().numpy().copy(),
        col=terrain.terrain_types[env_ids].cpu().numpy().copy(),
        dist=distance.cpu().numpy(),
        cmd=torch.norm(command[env_ids, :2], dim=1).cpu().numpy(),
        timeout=tm.time_outs[env_ids].cpu().numpy().astype(bool),
        terminated=tm.terminated[env_ids].cpu().numpy().astype(bool),
        rule_up=move_up.cpu().numpy().astype(bool),
        rule_down=move_down.cpu().numpy().astype(bool),
      )
    )

  terrain.update_env_origins = hold_and_record

  obs, _ = env.reset()
  for step in range(args.steps):
    with torch.inference_mode():
      actions = policy(obs)
    obs, _, _, _ = env.step(actions)
    if (step + 1) % 500 == 0:
      print(f"[step {step + 1}/{args.steps}] episodes recorded: {sum(len(r['row']) for r in records)}")

  if not records:
    raise SystemExit("[ERROR] no episode ended during the run -- increase --steps")
  cat = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
  # Keep only real episode ends. The curriculum also fires on the initial reset of every
  # env (zero-length episode: neither timed out nor terminated), which the step-counter
  # guard above does not always catch; left in, those records are ~1 in (steps / episode
  # length + 1) of the data and read as demotions at every row.
  real = cat["timeout"] | cat["terminated"]
  print(f"[INFO] dropped {int((~real).sum())} of {len(real)} records that were not episode ends")
  cat = {k: v[real] for k, v in cat.items()}
  cat["half_patch"] = half_patch
  cat["episode_s"] = episode_s
  type_idx = col_type[cat["col"]]
  # The curriculum's own decision must equal rule_current on the same data; if not, this
  # script is not measuring the rule the training run used.
  up_c, down_c = rule_current(cat)
  agree = float(np.mean((up_c == cat["rule_up"]) & (down_c == cat["rule_down"])))
  print(f"[CHECK] rule_current reproduces the curriculum's own decisions on {100 * agree:.2f}% of episodes")

  out = {
    "task": args.task, "checkpoint": args.checkpoint, "num_envs": args.num_envs, "steps": args.steps,
    "episodes": int(len(cat["row"])), "rows": num_rows, "half_patch_m": half_patch, "episode_s": episode_s,
    "rule_check_agreement": agree, "types": {},
  }
  for ti, tname in enumerate(type_names):
    sel = type_idx == ti
    if not sel.any():
      continue
    tout = {"rows": {}, "alternatives": {}}
    for r in range(num_rows):
      m = sel & (cat["row"] == r)
      if m.sum() == 0:
        continue
      sub = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in cat.items()}
      row_stats = {
        "episodes": int(m.sum()),
        "timeout_pct": float(100 * sub["timeout"].mean()),
        "terminated_pct": float(100 * sub["terminated"].mean()),
        "dist_mean_m": float(sub["dist"].mean()),
        "dist_median_m": float(np.median(sub["dist"])),
        "dist_over_half_patch_pct": float(100 * (sub["dist"] > half_patch).mean()),
        "cmd_mean": float(sub["cmd"].mean()),
        "needed_to_avoid_demotion_m_mean": float((sub["cmd"] * episode_s * 0.5).mean()),
      }
      for rname, fn in ALTERNATIVES.items():
        up, down = fn(sub)
        row_stats[f"{rname}_up_pct"] = float(100 * up.mean())
        row_stats[f"{rname}_down_pct"] = float(100 * down.mean())
      tout["rows"][r] = row_stats
    for rname, fn in ALTERNATIVES.items():
      p_up = np.zeros(num_rows)
      p_down = np.zeros(num_rows)
      for r in range(num_rows):
        m = sel & (cat["row"] == r)
        if m.sum() == 0:
          continue
        sub = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in cat.items()}
        up, down = fn(sub)
        p_up[r], p_down[r] = up.mean(), down.mean()
      mean_level, pi = stationary_mean_level(p_up, p_down, num_rows)
      tout["alternatives"][rname] = {"stationary_mean_level": mean_level, "stationary_rows": pi}
    out["types"][tname] = tout

  print("\n===== per terrain type, per row (rule = the curriculum's current promote/demote rule) =====")
  for tname, tout in out["types"].items():
    print(f"\n[{tname}]  (spawn at the patch centre; 'dist' = metres from it at episode end)")
    print("row | n   | timeout% | dist med | >4m% | cmd | needs(m) | up%   down%  || survive: up%  down%")
    for r, s in tout["rows"].items():
      print(f"{r:3d} | {s['episodes']:3d} | {s['timeout_pct']:7.1f}  | {s['dist_median_m']:7.2f}  | "
            f"{s['dist_over_half_patch_pct']:4.0f} | {s['cmd_mean']:.2f} | {s['needed_to_avoid_demotion_m_mean']:7.2f}  | "
            f"{s['current_up_pct']:5.1f} {s['current_down_pct']:6.1f}  ||  {s['survive_up_pct']:5.1f} {s['survive_down_pct']:6.1f}")
    for rname, a in tout["alternatives"].items():
      print(f"  stationary mean level under '{rname}': {a['stationary_mean_level']:.2f}")

  if args.json_out:
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "w") as f:
      json.dump(out, f, indent=2)
    print(f"[INFO] Wrote {args.json_out}")


if __name__ == "__main__":
  main()
