"""Reactive terrain classifier on the robot's own height scan (objective.md, arm 2a).

Reads exactly what the specialists read -- the 187-dim `height_scan` slice of the
actor observation, with whatever noise that observation carries -- and predicts
which specialist the footprint rule would have active right now (flat / rough /
stairs). The footprint rule switches 0.3 m before a segment, and the scan covers
0.8 m ahead of the base, so the target is observable in principle; whether it is
recoverable through the observation noise, and how late the switch ends up, is
what the arm measures.

A raw per-step argmax flickers, and every flicker is a policy switch, so the
runtime decision goes through `SwitchFilter`: an exponential average of the class
probabilities, then hold-to-switch hysteresis. Both delay the switch; that delay
is part of what a reactive classifier costs and is reported as its effective lead.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

CLASSES: tuple[str, ...] = ("flat", "rough", "stairs")


def scan_slice(env) -> slice:
  """The `height_scan` term's slice of the actor observation group."""
  om = env.unwrapped.observation_manager
  start = 0
  for name, shape in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"]):
    size = math.prod(shape)
    if name == "height_scan":
      return slice(start, start + size)
    start += size
  raise KeyError("no height_scan term in the actor observation group")


class ScanClassifier(nn.Module):
  def __init__(self, n_in: int = 187, hidden: tuple[int, ...] = (128, 64), n_out: int = len(CLASSES)) -> None:
    super().__init__()
    self.register_buffer("mean", torch.zeros(n_in))
    self.register_buffer("std", torch.ones(n_in))
    layers: list[nn.Module] = []
    prev = n_in
    for h in hidden:
      layers += [nn.Linear(prev, h), nn.ELU()]
      prev = h
    layers.append(nn.Linear(prev, n_out))
    self.net = nn.Sequential(*layers)

  def forward(self, scan: torch.Tensor) -> torch.Tensor:
    return self.net((scan - self.mean) / self.std)


def load_classifier(path: str | Path, device: str) -> ScanClassifier:
  blob = torch.load(str(path), map_location=device, weights_only=True)
  model = ScanClassifier(n_in=blob["n_in"], hidden=tuple(blob["hidden"]))
  model.load_state_dict(blob["state_dict"])
  return model.to(device).eval()


def save_classifier(model: ScanClassifier, path: str | Path, hidden: tuple[int, ...], meta: dict) -> None:
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  torch.save(dict(state_dict=model.state_dict(), n_in=model.mean.shape[0], hidden=list(hidden), meta=meta), str(path))


class SwitchFilter:
  """Per-env smoothing of class probabilities plus hold-to-switch hysteresis.

  `alpha` is the weight of the newest sample in the exponential average (1.0 = no
  smoothing). The active class changes only after the smoothed argmax has
  disagreed with it for `hold` consecutive steps (1 = switch immediately).
  """

  def __init__(self, n: int, alpha: float, hold: int, device: str, initial: int = 0) -> None:
    self.alpha, self.hold = alpha, hold
    self.probs = torch.zeros(n, len(CLASSES), device=device)
    self.probs[:, initial] = 1.0
    self.current = torch.full((n,), initial, dtype=torch.long, device=device)
    self.candidate = self.current.clone()
    self.streak = torch.zeros(n, dtype=torch.long, device=device)

  def step(self, probs: torch.Tensor) -> torch.Tensor:
    self.probs = (1.0 - self.alpha) * self.probs + self.alpha * probs
    top = self.probs.argmax(dim=1)
    differs = top != self.current
    same_candidate = differs & (top == self.candidate)
    self.streak = torch.where(same_candidate, self.streak + 1, differs.long())
    self.candidate = top
    switch = differs & (self.streak >= self.hold)
    self.current = torch.where(switch, top, self.current)
    self.streak = torch.where(switch, torch.zeros_like(self.streak), self.streak)
    return self.current
