"""SARO's closed-loop sub-task executor, plus VLM selection of the locomotion specialist.

SARO (arXiv:2407.16412, Sec. III-B, Fig. 3, Appendix A.2) runs the VLM plan as a
state machine: each sub-task (Action, Ending) is executed by a fixed workflow
(Algorithms 1-4) that turns perception into velocity commands, and the next
sub-task starts only when *both* the workflow's own end condition and the VLM
discriminator agree ("double-check"). SARO has one low-level policy. Here each
sub-task also carries a policy, and the executor decides which of the three
specialists is active at every control step.

How each ending is executed:
  facing intermediation  (Alg. 1) perceive the box, deproject through depth,
      walk to a stand-off point in front of the near edge, aligned with the
      box; done when centred + close + discriminator says the intermediation
      is there.
  across intermediation  (Alg. 2/4) walk straight through along the heading;
      done when odometry puts the hind feet past the far edge AND the
      discriminator says the intermediation is no longer visible.
  to the goal            (Alg. 3) walk to the goal coordinates; done when
      within the radius AND the discriminator says the task is finished.
      Entering this sub-task after a crossing triggers one re-plan (this
      project's extension, so a course with several intermediations -- `multi`
      -- is handled by the same loop; SARO's tasks have exactly one).

When the planned crossing policy is engaged is geometric, not a VLM call: the
specialist switches in when the estimated near edge is within the footprint's
front reach, and the flat policy returns only with the whole footprint past the
far edge (a point rule that ignored the hind legs stalled on the last step in
Phase 1). The VLM decides *which* specialist -- the plan's per-subtask policy,
overridden if the closed-loop policy selector gives a different answer
`selector_agree` times in a row.

`policy_source` makes the comparison arms: "vlm" (the modification under
test), "oracle" (ground-truth terrain under the footprint), or "fixed:<name>"
(one specialist throughout, i.e. SARO's single low-level policy).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from src.vlm_nav import prompts as P
from src.vlm_nav.camera import CameraSpec
from src.vlm_nav.controllers import GotoGains, goto_command
from src.vlm_nav.course import FOOTPRINT_FRONT, FOOTPRINT_REAR, CourseSpec
from src.vlm_nav.perception import (
  EdgeTracker,
  IntermediationEstimate,
  depth_edge_estimate,
  estimate_from_box,
  ground_blind_distance,
)
from src.vlm_nav.vlm_backend import VLM

NAME_TO_POLICY = {"stairs": "stairs", "rough ground": "rough"}


@dataclass
class Subtask:
  action: str
  ending: str
  policy: str


@dataclass
class ExecutorConfig:
  policy_source: str = "vlm"
  nav_source: str = "vlm"
  """"vlm": SARO sub-task execution; "gt": walk straight to the goal (no VLM for navigation)."""
  box_convention: str = "xyxy_px"
  speed: float = 0.5
  goal_radius: float = 0.3
  standoff: float = 0.6
  facing_px_tol: float = 100.0
  engage_margin: float = 0.10
  release_margin: float = 0.10
  selector_agree: int = 2
  subtask_timeout_s: float = 30.0
  max_replans: int = 4
  max_perception_failures: int = 3
  finish_confirm_max: int = 3
  perception_prompt: str = "detect"
  """"saro": SARO's [x0,y0,x1,y1] prompt (parsed with box_convention); "detect": box_2d JSON."""
  depth_fallback: bool = True
  """When the box is degenerate/unusable, locate the intermediation from depth geometry instead."""
  selector_every: int = 2
  perception_every: int = 2
  """In VLM ticks (0.5 s each by default): the two most frequent questions are asked every other tick."""
  """At the goal, stop asking after this many "no" answers (the trial's success is scored on ground truth anyway)."""


@dataclass
class Event:
  step: int
  kind: str
  detail: dict = field(default_factory=dict)


class SaroAgent:
  def __init__(self, env_id: int, course: CourseSpec, camera: CameraSpec, vlm: VLM | None, cfg: ExecutorConfig, dt: float):
    self.env_id = env_id
    self.course = course
    self.camera = camera
    self.vlm = vlm
    self.cfg = cfg
    self.dt = dt
    self.task = course.instruction
    self.goal_w = course.course_to_world(course.goal_course)[:2]
    self.plan: list[Subtask] | None = None
    self.intermediation: str | None = None
    self.idx = 0
    self.subtask_start = 0
    self.blind = ground_blind_distance(camera)
    self.edges = EdgeTracker(self.blind)
    self.ticks = 0
    self.perception_failures = 0
    self.crossing_policy: str | None = None
    self.selector_history: list[str] = []
    self.engaged = False
    self.replans = 0
    self.finish_noes = 0
    self.finished = False
    self.policy = "flat"
    self.events: list[Event] = []
    self.vlm_calls = 0
    self.vlm_time_s = 0.0
    if cfg.nav_source == "gt":
      self.plan = [Subtask("move", "to the goal", "flat")]

  # ---------------------------------------------------------------- VLM side

  def _ask(self, rgb: np.ndarray, prompt: str, schema: dict | None, max_tokens: int, tag: str):
    assert self.vlm is not None
    r = self.vlm.ask(rgb, prompt, schema, max_tokens=max_tokens, tag=f"env{self.env_id}:{tag}")
    self.vlm_calls += 1
    self.vlm_time_s += r.latency_s
    return r

  def _log(self, step: int, kind: str, **detail) -> None:
    self.events.append(Event(step, kind, detail))

  @property
  def current(self) -> Subtask | None:
    if self.plan is None or self.idx >= len(self.plan):
      return None
    return self.plan[self.idx]

  def _make_plan(self, step: int, rgb: np.ndarray) -> list[Subtask] | None:
    r = self._ask(rgb, P.planning(self.task), P.PLANNING_SCHEMA, 256, "planning")
    p = r.parsed
    if not isinstance(p, dict) or not p.get("subtasks"):
      self._log(step, "plan_failed", text=r.text, error=r.error)
      return None
    inter = p.get("intermediation", "none")
    subs = [Subtask(s["action"], s["ending"], s["policy"]) for s in p["subtasks"]]
    if inter == "none":
      subs = [s for s in subs if s.ending == "to the goal"] or [Subtask("move", "to the goal", "flat")]
    elif not any(s.ending == "to the goal" for s in subs):
      subs.append(Subtask("move", "to the goal", "flat"))
    self._log(step, "plan", intermediation=inter, subtasks=[asdict(s) for s in subs])
    self.intermediation = None if inter == "none" else inter
    return subs

  def _perceive(self, step: int, rgb, depth, pos, quat) -> None:
    if self.intermediation is None:
      return
    h, w = depth.shape
    if self.cfg.perception_prompt == "saro":
      r = self._ask(rgb, P.perception(self.intermediation), None, 48, "perception")
      raw = P.parse_box(r.text)
      px = None if raw is None else P.box_to_pixels(raw, self.cfg.box_convention, w, h)
    else:
      r = self._ask(rgb, P.perception_detect(self.intermediation), None, 96, "perception")
      px = P.parse_detect_box(r.text, w, h)
    est = IntermediationEstimate(valid=False)
    source = "none"
    if not P.box_is_degenerate(px, w, h):
      est = estimate_from_box(px, depth, self.camera, pos, quat)
      source = "vlm_box"
    if not est.valid and self.cfg.depth_fallback:
      est = depth_edge_estimate(depth, self.camera, pos, quat)
      source = "depth_fallback" if est.valid else "none"
    if est.valid:
      self.edges.update(est)
      self.perception_failures = 0
    else:
      self.perception_failures += 1
    self._log(step, "perception", text=r.text, source=source, valid=est.valid, near=est.near_along,
              far=est.far_along, lateral=est.lateral, n=est.n_points)

  def _selector(self, step: int, rgb) -> None:
    r = self._ask(rgb, P.policy_selector(), P.POLICY_SCHEMA, 32, "selector")
    choice = (r.parsed or {}).get("policy")
    if choice in ("flat", "rough", "stairs"):
      self.selector_history.append(choice)
    self._log(step, "selector", choice=choice)

  def _present(self, step: int, rgb) -> bool | None:
    if self.intermediation is None:
      return None
    # Free text, SARO's format: under the yes/no JSON schema Gemma-4-E4B spends
    # max_tokens and returns empty content (the policy schema works).
    r = self._ask(rgb, P.discriminator_present(self.intermediation), None, 8, "present")
    ans = P.parse_yes_no(r.text)
    self._log(step, "present", answer=ans, text=r.text)
    return ans

  def _finished(self, step: int, rgb) -> bool | None:
    r = self._ask(rgb, P.discriminator_finished(self.task), None, 8, "finished")
    ans = P.parse_yes_no(r.text)
    self._log(step, "finished", answer=ans, text=r.text)
    return ans

  def _advance(self, step: int, reason: str) -> None:
    self._log(step, "subtask_done", subtask=asdict(self.current) if self.current else None, reason=reason)
    self.idx += 1
    self.subtask_start = step
    cur = self.current
    if cur is not None and cur.ending == "across intermediation":
      self.engaged = False

  def think(self, step: int, rgb: np.ndarray, depth: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> None:
    """One VLM tick (sim paused). Called every `vlm_period` control steps."""
    if self.finished or self.cfg.nav_source == "gt":
      return
    if self.plan is None:
      self.plan = self._make_plan(step, rgb)
      self.subtask_start = step
      if self.plan is None:
        return
    cur = self.current
    if cur is None:
      return
    base_xy = pos[:2]
    timed_out = (step - self.subtask_start) * self.dt > self.cfg.subtask_timeout_s
    tick = self.ticks
    self.ticks += 1

    if self.cfg.policy_source == "vlm" and tick % self.cfg.selector_every == 0:
      self._selector(step, rgb)

    if cur.ending == "facing intermediation":
      if tick % self.cfg.perception_every == 0 or not self.edges.seen:
        self._perceive(step, rgb, depth, pos, quat)
      if self.edges.seen:
        near = self.edges.near_remaining(base_xy)
        # near is None when the edge was never seen beyond the camera's blind zone: it is already close.
        close = near is None or near <= self.cfg.standoff + 0.15
        centred = self.edges.center_u is None or abs(self.edges.center_u - self.camera.width / 2) < self.cfg.facing_px_tol
        if close and centred and self._present(step, rgb) is not False:
          self._advance(step, "facing reached")
      elif self.perception_failures >= self.cfg.max_perception_failures:
        self._advance(step, "perception failed")
      if timed_out and self.current is cur:
        self._advance(step, "timeout")

    elif cur.ending == "across intermediation":
      far = self.edges.far_remaining(base_xy)
      if (far is None or far > 0.5) and tick % self.cfg.perception_every == 0:
        self._perceive(step, rgb, depth, pos, quat)  # far edge still ahead: keep refining it
        far = self.edges.far_remaining(base_xy)
      past = far is not None and far < -(FOOTPRINT_REAR + self.cfg.release_margin)
      if past and self._present(step, rgb) is False:
        self._advance(step, "crossed")
      elif timed_out:
        self._advance(step, "timeout")

    elif cur.ending == "to the goal":
      if self.intermediation is not None and self.replans < self.cfg.max_replans and self._crossed_any():
        # Re-plan once after each crossing: a further intermediation (multi course) gets its own sub-tasks.
        self.replans += 1
        new = self._make_plan(step, rgb)
        if new is not None and self.intermediation is not None:
          self.plan = self.plan[: self.idx] + new
          self.edges = EdgeTracker(self.blind)
          self.subtask_start = step
          return
        self.intermediation = None
      dist = float(np.linalg.norm(self.goal_w - base_xy))
      if dist < self.cfg.goal_radius:
        ok = self._finished(step, rgb)
        self.finish_noes += ok is not True
        if ok or self.finish_noes >= self.cfg.finish_confirm_max:
          self.finished = True
          self._advance(step, "finished" if ok else "at goal, discriminator unconvinced")

  def _crossed_any(self) -> bool:
    return any(e.kind == "subtask_done" and e.detail.get("reason") == "crossed" for e in self.events[-6:])

  # ------------------------------------------------------------ control side

  def _selector_consensus(self) -> str | None:
    k = self.cfg.selector_agree
    h = self.selector_history[-k:]
    return h[0] if len(h) == k and len(set(h)) == 1 else None

  def _policy_now(self, x_course: float, base_xy: np.ndarray) -> str:
    src = self.cfg.policy_source
    if src == "oracle":
      return self.course.required_terrain(x_course)
    if src.startswith("fixed:"):
      return src.split(":", 1)[1]
    cur = self.current
    consensus = self._selector_consensus()
    if cur is None:
      return "flat"
    if cur.ending == "across intermediation":
      planned = cur.policy if cur.policy != "flat" else NAME_TO_POLICY.get(self.intermediation or "", "flat")
      choice = consensus if consensus in ("rough", "stairs") else planned
      if not self.engaged:
        # Engage when the front feet reach the near edge; with no usable estimate, engage now
        # (a specialist walks flat ground safely, the flat policy on steps does not).
        near = self.edges.near_remaining(base_xy)
        self.engaged = near is None or near <= FOOTPRINT_FRONT + self.cfg.engage_margin
      return choice if self.engaged else self.policy
    if cur.ending == "facing intermediation":
      return cur.policy if cur.policy == "flat" else self.policy
    # "to the goal": the plan's policy, unless the selector consistently sees terrain the plan missed.
    return consensus if consensus in ("rough", "stairs") else cur.policy

  def control(self, step: int, pos_w: torch.Tensor, quat_w: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Per-step velocity command (base frame) and specialist name."""
    base_xy = pos_w[:2].detach().cpu().numpy()
    x_course = self.course.world_to_course_x(float(base_xy[0]))
    gains = GotoGains(v_max=self.cfg.speed)
    cur = self.current
    target = self.goal_w
    if self.finished or cur is None:
      cmd = torch.zeros(3, device=pos_w.device)
    else:
      edges = self.edges
      if cur.ending == "facing intermediation" and edges.seen and edges.heading is not None:
        near = edges.near_remaining(base_xy)
        along = 0.0 if near is None else max(0.0, near - self.cfg.standoff)
        left = np.array([-edges.heading[1], edges.heading[0]])
        target = base_xy + edges.heading * along + left * edges.lateral_offset(base_xy)
      elif cur.ending == "across intermediation" and edges.seen and edges.heading is not None:
        far = edges.far_remaining(base_xy)
        target = base_xy + edges.heading * (far + FOOTPRINT_REAR + self.cfg.release_margin + 0.6)
      cmd, _ = goto_command(pos_w[None], quat_w[None], torch.as_tensor(target, device=pos_w.device, dtype=pos_w.dtype)[None], gains)
      cmd = cmd[0]
    new_policy = self._policy_now(x_course, base_xy)
    if new_policy != self.policy:
      self._log(step, "switch", frm=self.policy, to=new_policy, x_course=round(x_course, 3))
      self.policy = new_policy
    return cmd, self.policy

  def summary(self) -> dict:
    return dict(
      env_id=self.env_id, plan=[asdict(s) for s in self.plan] if self.plan else None, finished=self.finished,
      subtask_idx=self.idx, replans=self.replans, vlm_calls=self.vlm_calls, vlm_time_s=round(self.vlm_time_s, 2),
      events=[asdict(e) for e in self.events],
    )


def yaw_quat_to_heading(quat) -> float:
  w, x, y, z = quat
  return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
