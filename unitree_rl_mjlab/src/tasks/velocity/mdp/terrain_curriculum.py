"""Terrain-curriculum terms for the real-stair-height (V3) task.

Status: `terrain_row_fraction` / `terrain_row_mean` are used by StairsV3 and V4a/V4b.
`terrain_levels_progress` drives StairsV4a/V4b (replayed offline first, never yet trained).
`terrain_levels_survival` is NOT used by any task: it promotes on survival alone, which a
robot that stands still satisfies, so it would reproduce StairsV3's failure one row at a time.

`terrain_levels_vel` (curriculums.py) promotes a robot that ends an episode more than half
a patch (4 m) from its spawn and demotes one that ends below `|command| * T / 2` -- for
every command above 0.4 m/s, anything that did not leave the patch. On an 8 m pyramid
patch that holds a policy that never falls on rows 0-3 near row 2 of 10
(`scripts/diagnose_terrain_curriculum.py`, coordination/results/2026-10-05-terrain-curriculum-diagnosis.md:
stationary mean row ~2.4 for the stairs-v2 policy).

`terrain_levels_survival` keys the move on how the episode ended instead:

  * promote  -- ran to the time limit AND ended at least `min_progress_m` from the spawn
                (the stairs start 1.5 m from the patch centre, so 2.5 m means it got onto them);
  * demote   -- ended early (illegal contact / fell over);
  * otherwise stay. A robot that stalls on a step it cannot climb neither falls nor is
    demoted, so it keeps training on exactly the row it needs to learn.

`terrain_row_fraction` is a monitor only: the fraction of envs currently on rows lo..hi
(inclusive), logged as `Curriculum/<term name>`. The mean level alone hides whether a run
is spending its time on tall steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def terrain_levels_survival(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  min_progress_m: float = 2.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None

  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )
  tm = env.termination_manager
  move_up = tm.time_outs[env_ids] & (distance > min_progress_m)
  move_down = tm.terminated[env_ids] & ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)
  return torch.mean(terrain.terrain_levels.float())


# Staircase geometry (pyramid stairs, 8 m patch, border_width 1.0, 5 steps; the same at both
# treads): steps span Chebyshev distance 1.5-3.0 m from the spawn (centre of the patch), then a
# flat 1 m border out to the 4 m patch edge. Chebyshev, not Euclidean: the stairs are square
# rings, so a robot at (2.1, 2.1) has only crossed 2 steps although it is 3 m away.
PROGRESS_ACROSS_M = 3.2  # clear of the last step: it has climbed (or descended) all 5
PROGRESS_STALL_M = 1.9  # still on the platform or the first step: never got going
PROGRESS_STALL_CMD = 0.3  # a robot told to stand (or barely move) is not "stalled"


def terrain_levels_progress(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "twist",
  across_m: float = PROGRESS_ACROSS_M,
  stall_m: float = PROGRESS_STALL_M,
  stall_cmd: float = PROGRESS_STALL_CMD,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Move a robot up/down a row by whether it actually got across, at episode end.

  cheb = max(|dx|, |dy|) from the spawn, `cmd` = |commanded xy velocity| at the last step.

    promote : ran to the time limit (not terminated) AND cheb >= across_m
    demote  : terminated (illegal contact / fell)
              OR (ran to the time limit AND cmd > stall_cmd AND cheb < stall_m)   [a stall]
    stay    : everything else (e.g. climbed part of the flight, or was told to stand)

  A robot that stands still all episode is therefore demoted, never promoted and never left
  where it is, which is what StairsV3's uniform rows could not do.
  """
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None

  offset = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
  cheb = offset.abs().amax(dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd = torch.norm(command[env_ids, :2], dim=1)
  tm = env.termination_manager
  timed_out = tm.time_outs[env_ids] & ~tm.terminated[env_ids]

  move_up = timed_out & (cheb >= across_m)
  stalled = timed_out & (cmd > stall_cmd) & (cheb < stall_m)
  move_down = (tm.terminated[env_ids] | stalled) & ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)
  return torch.mean(terrain.terrain_levels.float())


def terrain_levels_progress_mixed(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  explore_frac: float = 0.5,
  **progress_kwargs,
) -> torch.Tensor:
  """`terrain_levels_progress`, then re-draw the row of a random `explore_frac` of the resetting
  envs uniformly over all rows (their column, i.e. sub-terrain, is kept).

  Why: stairs v5a under the pure progress rule settled at a mean row of ~3.7 and by iteration
  4,400 only ~6% of robots were on rows 6-7 and none on rows 8-9, so the 15-20 cm risers it must
  learn got almost no data. The adaptive half keeps robots at the frontier; the uniform half
  guarantees every row (5% each at the default 0.5) whatever the rule decides.
  """
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_levels_progress(env, env_ids, **progress_kwargs)
  pick = env_ids[torch.rand(len(env_ids), device=env.device) < explore_frac]
  if len(pick) > 0:
    terrain.terrain_levels[pick] = torch.randint(
      0, terrain.max_terrain_level, (len(pick),), device=env.device
    )
    terrain.env_origins[pick] = terrain.terrain_origins[
      terrain.terrain_levels[pick], terrain.terrain_types[pick]
    ]
  return torch.mean(terrain.terrain_levels.float())


def terrain_row_mean_by_direction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  direction: str,
) -> torch.Tensor:
  """Mean row of the envs on pyramid ("down": spawn on the top platform, walks down) or
  inverted-pyramid ("up": spawn in the pit, walks up) sub-terrains. Envs keep their column, so
  `terrain_row_mean` mixes two populations; a plateau can sit in only one of them."""
  del env_ids
  assert direction in ("down", "up")
  terrain = env.scene.terrain
  assert terrain is not None
  gen = terrain.cfg.terrain_generator
  names = list(gen.sub_terrains)
  props = torch.tensor([s.proportion for s in gen.sub_terrains.values()], dtype=torch.float)
  cum = torch.cumsum(props / props.sum(), dim=0)
  col_type = [
    int(torch.nonzero(c / gen.num_cols + 0.001 < cum)[0]) for c in range(gen.num_cols)
  ]
  want_up = direction == "up"
  col_ok = torch.tensor(
    [("inv" in names[t]) == want_up for t in col_type], device=env.device
  )
  mask = col_ok[terrain.terrain_types]
  if not mask.any():
    return torch.zeros((), device=env.device)
  return torch.mean(terrain.terrain_levels[mask].float())


def terrain_row_mean(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> torch.Tensor:
  del env_ids  # Unused: reported over all envs.
  terrain = env.scene.terrain
  assert terrain is not None
  return torch.mean(terrain.terrain_levels.float())


def terrain_row_fraction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  lo: int,
  hi: int,
) -> torch.Tensor:
  del env_ids  # Unused: reported over all envs.
  terrain = env.scene.terrain
  assert terrain is not None
  levels = terrain.terrain_levels
  return torch.mean(((levels >= lo) & (levels <= hi)).float())


# Per-step metrics, logged by mjlab as Episode_Metrics/<name> (the episode mean of each).
# Read as: achieved/commanded speed ~ actual_speed / cmd_speed; stalled fraction among
# steps told to move ~ stalled / cmd_moving. Thresholds are eval_checkpoint.py's defaults.
_MOVING_CMD = 0.1
_STALLED_SPEED = 0.05


def cmd_speed(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2], dim=1)


def actual_speed(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)


def cmd_moving(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
  return (cmd_speed(env, command_name) > _MOVING_CMD).float()


def stalled(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  moving = cmd_speed(env, command_name) > _MOVING_CMD
  slow = actual_speed(env, asset_cfg) < _STALLED_SPEED
  return (moving & slow).float()


def limb_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """1.0 on a step where any matched geom feels more than `force_threshold` N (a penalty
  term for contacts that are allowed but discouraged; see StairsV4b)."""
  data = env.scene[sensor_name].data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1).float()
  assert data.found is not None
  return torch.any(data.found, dim=-1).float()
