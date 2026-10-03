"""Train the reactive height-scan terrain classifier (src/vlm_nav/scan_classifier.py).

Input: one or more files written by `switch_follow.py --record-scans`. The split is
by trial, never by sample: consecutive scans from one robot are near-duplicates, so
a random split would put the same stretch of terrain on both sides and report an
accuracy that means nothing (same reasoning as `height_scan_classifier.py`).

Usage (from unitree_rl_mjlab/):
  python scripts/scan_classifier_train.py logs/switch_follow/scans_noisy_*.pt \
      --out logs/switch_follow/clf_noisy.pt
"""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn

from src.vlm_nav.scan_classifier import CLASSES, ScanClassifier, save_classifier


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("files", nargs="+")
  ap.add_argument("--out", required=True)
  ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
  ap.add_argument("--epochs", type=int, default=30)
  ap.add_argument("--val-frac", type=float, default=0.15)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  torch.manual_seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  scans, labels, trials, offset = [], [], [], 0
  for f in args.files:
    d = torch.load(f, weights_only=True)
    assert d["classes"] == list(CLASSES), f"{f}: class order {d['classes']} != {CLASSES}"
    scans.append(d["scan"])
    labels.append(d["label"])
    trials.append(d["trial"] + offset)
    offset += int(d["trial"].max()) + 1
  x, y, t = torch.cat(scans).float(), torch.cat(labels).long(), torch.cat(trials)

  ids = torch.unique(t)
  ids = ids[torch.randperm(len(ids))]
  val_ids = ids[: max(1, int(len(ids) * args.val_frac))]
  is_val = torch.isin(t, val_ids)
  xt, yt, xv, yv = x[~is_val].to(device), y[~is_val].to(device), x[is_val].to(device), y[is_val].to(device)

  model = ScanClassifier(n_in=x.shape[1], hidden=tuple(args.hidden)).to(device)
  model.mean.copy_(xt.mean(dim=0))
  model.std.copy_(xt.std(dim=0).clamp_min(1e-6))
  counts = torch.bincount(yt, minlength=len(CLASSES)).float()
  loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / (len(CLASSES) * counts)).to(device))
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)

  for epoch in range(args.epochs):
    model.train()
    perm = torch.randperm(len(xt), device=device)
    for i in range(0, len(xt), 4096):
      idx = perm[i : i + 4096]
      opt.zero_grad()
      loss_fn(model(xt[idx]), yt[idx]).backward()
      opt.step()
  model.eval()
  with torch.inference_mode():
    pred = model(xv).argmax(dim=1)
  confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
  for a, b in zip(yv.cpu().tolist(), pred.cpu().tolist()):
    confusion[a, b] += 1
  recall = {c: float(confusion[i, i] / confusion[i].sum().clamp_min(1)) for i, c in enumerate(CLASSES)}
  meta = dict(
    files=args.files, samples=int(len(x)), trials=int(len(ids)), val_trials=int(len(val_ids)),
    val_accuracy=float((pred == yv).float().mean()), val_recall=recall,
    val_confusion_rows_true_cols_pred=confusion.tolist(), train_class_counts=counts.tolist(),
  )
  save_classifier(model, args.out, tuple(args.hidden), meta)
  print(json.dumps(meta, indent=1))


if __name__ == "__main__":
  main()
