"""A VLM stand-in that answers every pipeline question from ground truth.

Running the SARO executor against this isolates the executor (sub-task logic,
box-to-depth geometry, switch timing, controller) from the real model's
perception errors: if a trial fails with perfect answers, the fault is in the
pipeline, not in Gemma. It speaks the same interface and reply formats as
OpenAICompatVLM, so the executor code path is identical.

Questions are recognised from the executor's tag ("env<i>:<kind>"); the
robot's current state comes from a callback, because an image alone doesn't
carry pose.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np

from src.vlm_nav.camera import CameraSpec
from src.vlm_nav.course import CourseSpec
from src.vlm_nav.frames import _next_intermediation, visible_region_bbox
from src.vlm_nav.vlm_backend import VLMReply

NAME = {"stairs": "stairs", "rough": "rough ground"}
StateFn = Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]]  # env_id -> (depth, base_pos_w, base_quat_w)


class OracleVLM:
  def __init__(self, course: CourseSpec, camera: CameraSpec, state: StateFn, goal_radius: float = 0.3):
    self.course = course
    self.camera = camera
    self.state = state
    self.goal_radius = goal_radius

  def ask(self, image: np.ndarray, prompt: str, schema: dict | None = None, max_tokens: int = 256, tag: str = "") -> VLMReply:
    env_tag, _, kind = tag.partition(":")
    env_id = int(env_tag.removeprefix("env"))
    depth, pos, quat = self.state(env_id)
    x = self.course.world_to_course_x(float(pos[0]))
    region = _next_intermediation(self.course, x)
    visible_box = None
    if region is not None:
      visible_box, _ = visible_region_bbox(self.course, region, self.camera, depth, pos, quat)

    if kind == "planning":
      if region is None or visible_box is None:
        ans = {"intermediation": "none", "subtasks": [{"action": "move", "ending": "to the goal", "policy": "flat"}]}
      else:
        action = "climb" if region.terrain == "stairs" else "move"
        ans = {"intermediation": NAME[region.terrain], "subtasks": [
          {"action": "move", "ending": "facing intermediation", "policy": "flat"},
          {"action": action, "ending": "across intermediation", "policy": region.terrain},
          {"action": "move", "ending": "to the goal", "policy": "flat"},
        ]}
      return VLMReply(text=json.dumps(ans), parsed=ans)
    if kind == "perception":
      text = "[]" if visible_box is None else json.dumps(visible_box)
      return VLMReply(text=text)
    if kind == "present":
      return VLMReply(text="yes" if visible_box is not None else "no")
    if kind == "selector":
      ans = {"policy": self.course.required_terrain(x + 0.5)}
      return VLMReply(text=json.dumps(ans), parsed=ans)
    if kind == "finished":
      goal = self.course.course_to_world(self.course.goal_course)[:2]
      done = float(np.linalg.norm(goal - pos[:2])) < self.goal_radius + 0.1
      return VLMReply(text="yes" if done else "no")
    return VLMReply(text="", error=f"oracle VLM: unknown question kind {kind!r}")
