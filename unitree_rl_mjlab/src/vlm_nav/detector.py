"""Fast per-frame object localization, so the VLM never sits in the control loop.

Measured division of labour (see findings.md, "Person-following"):

- The **VLM** is a planner. It reasons about the task in language, names the target
  class, picks the specialist and judges whether a sub-task is finished. It takes
  ~2 s per call and answered with a usable position on only 11 of 84 queries during
  a closed-loop follow run -- fine at 0.2 Hz, unusable at 50 Hz.
- A **detector** is a tracker. It answers only "where in this frame is <class>" and
  must be fast enough to run every control step (YOLO11s measured at 3.5-3.9 ms on
  this laptop's RTX 5060, ~500x the VLM).
- **Depth geometry** turns a pixel column into metres, already validated to 0-8 cm.

`Detector` is the seam between them. Swapping the detector never touches the
planner, which is what makes the pipeline retargetable: the VLM says *what* to look
for in words, the detector says *where*, every frame.

Note on the simulator: stock COCO YOLO does **not** recognise this project's leader
(it reads the capsule legs as "baseball bat" at 0.79 confidence and finds nothing at
all at the 6 m follow distance). That is an appearance gap specific to flat-shaded
mjlab geoms -- on real hardware a real person is exactly COCO's home ground. For sim,
fine-tune on the auto-labelled set from
`scripts/vlm_nav_make_detector_dataset.py` and point `YoloDetector` at those weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
  """Axis-aligned pixel box (x0, y0, x1, y1) with a confidence and class name."""

  box_px: tuple[float, float, float, float]
  conf: float
  label: str

  @property
  def centre_u(self) -> float:
    return 0.5 * (self.box_px[0] + self.box_px[2])

  @property
  def centre_v(self) -> float:
    return 0.5 * (self.box_px[1] + self.box_px[3])


class Detector(Protocol):
  def detect(self, rgb: np.ndarray, label: str) -> list[Detection]:
    """Every instance of `label` in the frame, most confident first."""
    ...


@dataclass
class YoloDetector:
  """Ultralytics YOLO behind the `Detector` seam.

  `weights` may be a stock COCO checkpoint or one fine-tuned by
  `scripts/vlm_nav_train_detector.py`. `label` is matched case-insensitively
  against the model's own class names, so a fine-tuned single-class model that
  calls its class "person" answers a "person" request unchanged.
  """

  weights: str
  device: int | str = 0
  conf: float = 0.25
  imgsz: int = 640
  _model: object | None = field(default=None, repr=False)
  latencies_ms: list[float] = field(default_factory=list, repr=False)

  def __post_init__(self) -> None:
    from ultralytics import YOLO  # noqa: PLC0415  (heavy, and optional)

    self._model = YOLO(self.weights)
    self._names = {i: str(n).lower() for i, n in self._model.names.items()}

  def warmup(self, rgb: np.ndarray, rounds: int = 3) -> None:
    """First inference includes CUDA graph/kernel setup; exclude it from timings."""
    for _ in range(rounds):
      self._model.predict(rgb, verbose=False, device=self.device, imgsz=self.imgsz)

  def detect(self, rgb: np.ndarray, label: str) -> list[Detection]:
    want = label.strip().lower()
    t0 = time.perf_counter()
    res = self._model.predict(rgb, verbose=False, device=self.device, conf=self.conf, imgsz=self.imgsz)[0]
    self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    out: list[Detection] = []
    for cls, cf, box in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
      name = self._names.get(int(cls), str(int(cls)))
      if name == want:
        out.append(Detection(tuple(float(v) for v in box), float(cf), name))
    out.sort(key=lambda d: -d.conf)
    return out

  @property
  def median_latency_ms(self) -> float:
    return float(np.median(self.latencies_ms)) if self.latencies_ms else float("nan")
