"""Prompts: SARO's three (arXiv:2407.16412, Appendix A.1) plus this project's policy selector.

SARO asks the VLM three kinds of question:
  1. Planning -- name the intermediation, decompose the task into (Action, Ending) subtasks.
  2. Perception -- a bounding box for the intermediation.
  3. Discriminator -- yes/no checks that close the loop ("is there any <I>?",
     "is the task finished?").
The wording below is SARO's, with two substitutions: the intermediation list is
the terrain this project has specialists for (stairs, rough ground) instead of
stair/ramp/gap/door, and answers are requested as JSON under a schema the
server enforces, so a malformed reply can't silently become a wrong action.

The modification under test: each subtask also names the locomotion *policy*
(one of the three frozen specialists), and a fourth question -- the policy
selector -- asks which specialist suits the ground immediately ahead, so the
choice is re-checked in closed loop rather than fixed at planning time.
"""

from __future__ import annotations

import re

from src.vlm_nav.policy_bank import POLICY_NAMES

INTERMEDIATIONS = ("stairs", "rough ground")
ACTIONS = ("move", "climb")
ENDINGS = ("facing intermediation", "across intermediation", "to the goal")

POLICY_DESCRIPTIONS = {
  "flat": "for smooth, level floor",
  "rough": "for uneven, bumpy ground",
  "stairs": "for steps going up or down",
}
"""Label-based cards: each specialist described by the terrain it was trained on.

Phase-1 exploratory runs (single seed) found the specialist named for a terrain
is not always the best one on it -- the rough specialist crossed 0.05 m up-stairs
more reliably than the stairs specialist. "The best policy for the situation" can
then only be chosen from measured competence, so every prompt takes an optional
`cards` dict to swap in measured descriptions once confirmed; the default stays
label-based so the two can be compared as arms."""


def _policy_list(cards: dict[str, str] | None) -> str:
  return "; ".join(f"'{p}' {d}" for p, d in (cards or POLICY_DESCRIPTIONS).items())


def planning(task: str, cards: dict[str, str] | None = None) -> str:
  return (
    "Ignore anything on the wall. You are a robot dog. The intermediation may be stairs or rough ground. "
    f"The task is {task}. First answer the question: 1. What is the only intermediation you need to cross "
    "or climb to finish the task? If there is none between you and the goal, answer 'none'. "
    "Based on previous questions, decompose this task into a sequence of subtasks. "
    "The subtask is (Action, Ending, Policy). Action is one of ['move', 'climb']. "
    "The ending is one of ['facing intermediation', 'across intermediation', 'to the goal']. "
    "Replace the intermediation with the answer to question 1. "
    f"Policy is the walking controller to use during that subtask, one of: {_policy_list(cards)}. "
    'Answer as JSON: {"intermediation": ..., "subtasks": [{"action": ..., "ending": ..., "policy": ...}]}.'
  )


PLANNING_SCHEMA = {
  "type": "object",
  "properties": {
    "intermediation": {"type": "string", "enum": [*INTERMEDIATIONS, "none"]},
    "subtasks": {
      "type": "array",
      "minItems": 1,
      "maxItems": 4,
      "items": {
        "type": "object",
        "properties": {
          "action": {"type": "string", "enum": list(ACTIONS)},
          "ending": {"type": "string", "enum": list(ENDINGS)},
          "policy": {"type": "string", "enum": list(POLICY_NAMES)},
        },
        "required": ["action", "ending", "policy"],
      },
    },
  },
  "required": ["intermediation", "subtasks"],
}


def perception(intermediation: str) -> str:
  return f"Where is the {intermediation}? Answer in [x0,y0,x1,y1] format, don't say anything else."


def perception_detect(intermediation: str) -> str:
  """Detection-style box request. Not SARO's wording: Gemma-4-E4B answers SARO's
  "[x0,y0,x1,y1]" prompt with [0,0,0,0] or the full frame almost every time
  (Phase-2 probe), and does noticeably better with this Gemini-style format."""
  return (f"Detect the {intermediation} in the image. Return a JSON list like "
          f'[{{"box_2d": [ymin, xmin, ymax, xmax], "label": "{intermediation}"}}] with coordinates normalized to 0-1000.')


def parse_detect_box(text: str, width: int, height: int) -> list[float] | None:
  from src.vlm_nav.vlm_backend import parse_json  # noqa: PLC0415

  p = parse_json(text)
  if isinstance(p, dict):
    p = [p]
  if not (isinstance(p, list) and p and isinstance(p[0], dict) and len(p[0].get("box_2d", [])) == 4):
    return None
  return box_to_pixels([float(v) for v in p[0]["box_2d"]], "yxyx_1000", width, height)


def box_is_degenerate(box_px: list[float] | None, width: int, height: int) -> bool:
  """Empty, zero-area, or (near) the whole frame -- the answers that carry no location."""
  if box_px is None:
    return True
  bw, bh = box_px[2] - box_px[0], box_px[3] - box_px[1]
  return bw < 4 or bh < 4 or (bw > 0.97 * width and bh > 0.97 * height)


def discriminator_present(intermediation: str) -> str:
  return f"Is there any {intermediation}? Just answer yes or no."


def discriminator_finished(task: str) -> str:
  return f"Is the task {task} finished at current state? Just answer yes or no."


def policy_selector(cards: dict[str, str] | None = None) -> str:
  return (
    "You are a robot dog choosing which walking controller to use. Look at the ground directly in front of "
    f"you, within about one metre. The controllers are: {_policy_list(cards)}. "
    "Which controller should you use now? Answer with one word: flat, rough, or stairs."
  )


YES_NO_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}}, "required": ["answer"]}
POLICY_SCHEMA = {"type": "object", "properties": {"policy": {"type": "string", "enum": list(POLICY_NAMES)}}, "required": ["policy"]}


def parse_yes_no(text: str) -> bool | None:
  t = text.strip().lower()
  if re.match(r"^\W*yes\b", t):
    return True
  if re.match(r"^\W*no\b", t):
    return False
  return None


def parse_policy(text: str) -> str | None:
  hits = [p for p in POLICY_NAMES if re.search(rf"\b{p}\b", text.lower())]
  return hits[0] if len(hits) == 1 else None


def parse_box(text: str) -> list[float] | None:
  """First four numbers in the reply, in the order the model wrote them."""
  nums = re.findall(r"-?\d+(?:\.\d+)?", text)
  return [float(n) for n in nums[:4]] if len(nums) >= 4 else None


BOX_CONVENTIONS = ("xyxy_px", "xyxy_1000", "yxyx_1000", "xyxy_unit")


def box_to_pixels(box: list[float], convention: str, width: int, height: int) -> list[float]:
  """Convert a model's 4-number box to [x0, y0, x1, y1] pixels.

  VLMs disagree on box conventions (absolute pixels, 0-1000 normalized, 0-1
  normalized, and Gemma-family models commonly emit [y0, x0, y1, x1] on a 0-1000
  grid). Which one this model uses is measured in the Phase-2 perception gate,
  not assumed.
  """
  a, b, c, d = box
  if convention == "xyxy_px":
    x0, y0, x1, y1 = a, b, c, d
  elif convention == "xyxy_1000":
    x0, y0, x1, y1 = a * width / 1000, b * height / 1000, c * width / 1000, d * height / 1000
  elif convention == "yxyx_1000":
    x0, y0, x1, y1 = b * width / 1000, a * height / 1000, d * width / 1000, c * height / 1000
  elif convention == "xyxy_unit":
    x0, y0, x1, y1 = a * width, b * height, c * width, d * height
  else:
    raise ValueError(f"unknown box convention {convention!r}")
  return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def iou(a: list[float], b: list[float]) -> float:
  ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
  iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
  inter = ix * iy
  union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
  return inter / union if union > 0 else 0.0


def follow_target(task: str, classes: tuple[str, ...]) -> str:
  """Ask the planner which detector class accomplishes the task.

  This is the VLM's whole job in the two-rate design: turn a task stated in
  language into a class name the fast detector can track every control step.
  """
  opts = ", ".join(f"'{c}'" for c in classes)
  return (
    "You are a robot dog looking through your forward camera. "
    f"Your task is: {task}. "
    "Answer which single kind of object in view you must keep track of to do this. "
    f"Choose exactly one of: {opts}. "
    'Answer as JSON: {"target": ...}'
  )


def follow_target_schema(classes: tuple[str, ...]) -> dict:
  return {
    "type": "object",
    "properties": {"target": {"type": "string", "enum": list(classes)}},
    "required": ["target"],
  }


def parse_target(text: str, classes: tuple[str, ...]) -> str | None:
  from src.vlm_nav.vlm_backend import parse_json  # noqa: PLC0415

  p = parse_json(text)
  if isinstance(p, dict):
    v = str(p.get("target", "")).strip().lower()
    if v in {c.lower() for c in classes}:
      return v
  return None
