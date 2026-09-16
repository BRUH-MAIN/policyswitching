"""The three frozen terrain specialists, hot-swappable inside one env.

All specialists share one observation space and network shape (objective.md
scope decisions), so one env serves all of them; each keeps its own observation
normalizer, which lives inside its actor module. Switching is a hard switch at
the next control step: the next action simply comes from a different network.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

from src.vlm_nav.twin_env import BASE_TASK

POLICY_NAMES: tuple[str, ...] = ("flat", "rough", "stairs")


def default_checkpoints(ckpt_root: str | Path) -> dict[str, Path]:
  root = Path(ckpt_root)
  return {name: root / f"go2_spec_{name}" / "model_9999.pt" for name in POLICY_NAMES}


class PolicyBank:
  def __init__(self, env, checkpoints: dict[str, str | Path], device: str) -> None:
    missing = [str(p) for p in checkpoints.values() if not Path(p).exists()]
    if missing:
      raise FileNotFoundError(f"specialist checkpoint(s) not found: {missing}")
    agent_cfg = load_rl_cfg(BASE_TASK)
    runner = load_runner_cls(BASE_TASK)(env, asdict(agent_cfg), device=device)
    self.policies: dict[str, torch.nn.Module] = {}
    for name, path in checkpoints.items():
      # load() also restores the checkpoint's common_step_counter onto the env;
      # harmless here (no step-keyed curricula remain in the twin config).
      runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=device)
      self.policies[name] = copy.deepcopy(runner.get_inference_policy(device=device))
    self.active = next(iter(self.policies))
    self.switch_log: list[tuple[int, str, str]] = []

  def select(self, name: str, step: int) -> bool:
    """Make `name` the active policy. Returns True if this changed the policy."""
    if name not in self.policies:
      raise KeyError(f"unknown policy {name!r}; have {list(self.policies)}")
    if name == self.active:
      return False
    self.switch_log.append((step, self.active, name))
    self.active = name
    return True

  @torch.inference_mode()
  def act(self, obs) -> torch.Tensor:
    return self.policies[self.active](obs)
