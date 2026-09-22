"""Draw the decision layer onto recorded ego frames.

A raw ego video shows the outcome but not the architecture: you cannot see which
component decided what, or when. This overlay makes the two rates visible --
the detector's box updating every frame, and the planner's banner firing only
when a VLM answer actually lands (and whether it blocked the control loop).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = (
  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)
_BOX_RGB = (60, 230, 90)
_PLAN_RGB = (255, 170, 40)
_ASYNC_RGB = (90, 180, 255)


def _font(size: int):
  for p in _FONT_PATHS:
    try:
      return ImageFont.truetype(p, size)
    except OSError:
      continue
  return ImageFont.load_default()


@dataclass(frozen=True)
class VlmEvent:
  """A planner answer that landed at `t` seconds."""

  t: float
  kind: str
  answer: str | None
  latency_s: float
  blocking: bool


def _panel(draw: ImageDraw.ImageDraw, xy, w, h, alpha=150) -> None:
  x, y = xy
  draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0, alpha))


def annotate(
  rgb: np.ndarray,
  *,
  t: float,
  target: str,
  box_px: tuple[float, float, float, float] | None,
  conf: float | None,
  det_ms: float | None,
  range_m: float | None,
  gap_m: float,
  events: list[VlmEvent],
  banner_hold_s: float = 1.6,
) -> np.ndarray:
  """Return `rgb` with the detector box and any live planner banner drawn on it."""
  im = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
  d = ImageDraw.Draw(im, "RGBA")
  f_small, f_big = _font(16), _font(22)

  if box_px is not None:
    x0, y0, x1, y1 = box_px
    d.rectangle([x0, y0, x1, y1], outline=_BOX_RGB, width=3)
    tag = f"{target} {conf:.2f}" if conf is not None else target
    tw = d.textlength(tag, font=f_small)
    ty = max(0.0, y0 - 20)
    d.rectangle([x0, ty, x0 + tw + 8, ty + 19], fill=(*_BOX_RGB, 210))
    d.text((x0 + 4, ty + 2), tag, fill=(0, 0, 0), font=f_small)

  # HUD: who is doing what, at what rate.
  _panel(d, (8, 8), 268, 74)
  d.text((16, 12), f"t {t:5.1f}s", fill=(235, 235, 235), font=f_small)
  yolo_txt = f"YOLO  {det_ms:.1f} ms" if det_ms is not None else "YOLO  --"
  d.text((16, 31), yolo_txt, fill=_BOX_RGB, font=f_small)
  det_state = "tracking" if box_px is not None else "no detection"
  d.text((150, 31), det_state, fill=(235, 235, 235) if box_px is not None else (255, 120, 120), font=f_small)
  rng_txt = f"range {range_m:.2f} m   gap {gap_m:.1f} m" if range_m is not None else f"gap {gap_m:.1f} m"
  d.text((16, 50), rng_txt, fill=(235, 235, 235), font=f_small)

  # Planner banner: only visible for a moment after an answer lands.
  live = [e for e in events if 0.0 <= t - e.t <= banner_hold_s]
  if live:
    e = live[-1]
    colour = _PLAN_RGB if e.blocking else _ASYNC_RGB
    label = "VLM PLAN" if e.kind == "plan" else "VLM re-confirm"
    mode = "blocking" if e.blocking else "async (loop keeps running)"
    txt = f"{label} -> {e.answer}"
    sub = f"{e.latency_s:.2f} s  |  {mode}"
    w = int(max(d.textlength(txt, font=f_big), d.textlength(sub, font=f_small))) + 24
    # Bottom-right: detector boxes cluster near the top-centre, and a banner up
    # there overlaps their confidence label exactly when the target is in view.
    x = im.width - w - 10
    y = im.height - 66
    _panel(d, (x, y), w, 56, alpha=170)
    d.rectangle([x, y, x + 5, y + 56], fill=colour)
    d.text((x + 14, y + 4), txt, fill=colour, font=f_big)
    d.text((x + 14, y + 30), sub, fill=(225, 225, 225), font=f_small)

  return np.asarray(im)


_POLICY_RGB = {"flat": (200, 200, 210), "rough": (120, 220, 140), "stairs": (255, 150, 90)}


def annotate_saro(
  rgb: np.ndarray,
  *,
  t: float,
  intermediation: str | None,
  subtask: tuple[str, str] | None,
  policy: str | None,
  subtask_idx: int,
  n_subtasks: int,
  events: list[tuple[float, str]],
  banner_hold_s: float = 2.0,
) -> np.ndarray:
  """Overlay for SARO's goal-tracking runner (`scripts/vlm_nav_run.py`).

  Shows the two things the numbers alone hide: what the planner decided the
  intermediation was, and which specialist is actually driving right now. On a
  staircase course where the planner answered "none", that pairing is the whole
  story -- the robot walks at a staircase on the flat policy and stops.
  """
  im = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
  d = ImageDraw.Draw(im, "RGBA")
  f_small, f_big = _font(16), _font(20)

  _panel(d, (8, 8), 330, 94)
  d.text((16, 12), f"t {t:5.1f}s", fill=(235, 235, 235), font=f_small)

  inter = intermediation if intermediation is not None else "(not planned yet)"
  bad = inter == "none"
  _lbl = "planner says:"
  d.text((16, 31), _lbl, fill=(180, 180, 190), font=f_small)
  d.text((16 + d.textlength(_lbl, font=f_small) + 8, 31), inter,
         fill=(255, 110, 110) if bad else (120, 220, 140), font=f_small)

  if subtask is not None:
    _st = f"subtask {subtask_idx + 1}/{n_subtasks}:"
    d.text((16, 50), _st, fill=(180, 180, 190), font=f_small)
    d.text((16 + d.textlength(_st, font=f_small) + 8, 50), f"{subtask[0]} -> {subtask[1]}",
           fill=(235, 235, 235), font=f_small)

  if policy is not None:
    d.text((16, 71), "policy:", fill=(180, 180, 190), font=f_small)
    d.text((80, 69), policy, fill=_POLICY_RGB.get(policy, (235, 235, 235)), font=f_big)

  live = [e for e in events if 0.0 <= t - e[0] <= banner_hold_s]
  if live:
    txt = live[-1][1]
    w = int(d.textlength(txt, font=f_big)) + 28
    x, y = im.width - w - 10, im.height - 44
    _panel(d, (x, y), w, 34, alpha=175)
    d.rectangle([x, y, x + 5, y + 34], fill=_ASYNC_RGB)
    d.text((x + 14, y + 6), txt, fill=_ASYNC_RGB, font=f_big)
  return np.asarray(im)
