"""Terrain-curriculum terms for the real-stair-height (V3) task.

Status: `terrain_row_fraction` / `terrain_row_mean` are used by Unitree-Go2-Spec-StairsV3.
`terrain_levels_survival` is NOT used by any registered task: it has been evaluated offline
only (scripts/diagnose_terrain_curriculum.py replays its decisions on recorded episodes) and
has never driven a training run. It is the prepared alternative if uniform rows fail.

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
