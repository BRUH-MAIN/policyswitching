"""Fine-tune YOLO on the auto-labelled leader set so the fast layer works in sim.

Why this exists: stock COCO YOLO cannot see this project's leader (see
`src/vlm_nav/detector.py`). One class, one fixed appearance and exact labels, so
this converges in a few minutes on the laptop GPU -- it is a domain patch, not a
research artefact, and it is sim-only: on real hardware stock COCO weights are the
right choice.

Usage (from unitree_rl_mjlab/):
  PYTHONPATH=$PWD python scripts/vlm_nav_train_detector.py \
      --data /tmp/leader_ds/data.yaml --weights yolo11n.pt --epochs 40
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", required=True, help="data.yaml from vlm_nav_make_detector_dataset.py")
  ap.add_argument("--weights", default="yolo11n.pt", help="Starting checkpoint (nano keeps inference ~3 ms).")
  ap.add_argument("--epochs", type=int, default=40)
  ap.add_argument("--imgsz", type=int, default=640)
  ap.add_argument("--batch", type=int, default=16)
  ap.add_argument("--device", default="0")
  ap.add_argument("--project", default="logs/vlm_nav/detector")
  ap.add_argument("--name", default="leader")
  args = ap.parse_args()

  from ultralytics import YOLO  # noqa: PLC0415

  model = YOLO(args.weights)
  model.train(
    data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
    device=args.device, project=args.project, name=args.name, exist_ok=True,
    # The scene is already the deployment distribution: no mosaic/flip games that
    # would invent views the ego camera can never produce.
    mosaic=0.0, fliplr=0.0, scale=0.2, degrees=0.0, translate=0.05,
    pretrained=True, verbose=True, plots=False,
  )
  best = Path(args.project) / args.name / "weights" / "best.pt"
  print(f"TRAIN_DONE best={best}")


if __name__ == "__main__":
  main()
