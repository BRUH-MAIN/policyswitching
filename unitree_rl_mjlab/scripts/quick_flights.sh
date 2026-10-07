#!/bin/bash
# Quick look at a checkpoint on straight flights: 64 trials per cell, one seed, tipping-only
# falls, training sensor noise on. For watching a run, not for a result: the result is
# switch_follow_real_stairs.sh (256 trials per cell, both fall definitions).
# Usage: scripts/quick_flights.sh <checkpoint> <tag> [heights...]
set -uo pipefail
CK="$(realpath "$1")"; TAG="$2"; shift 2
HEIGHTS="${*:-0.12 0.15 0.17}"
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$W"
PY="${PY:-$HOME/miniconda3/envs/unitree_rl_mjlab/bin/python}"
export PYTHONPATH="$W" MUJOCO_GL=egl
O=eval_results/switch_follow/real_stairs; mkdir -p "$O" logs/switch_follow
for c in stairs_up stairs_down; do for h in $HEIGHTS; do
  "$PY" scripts/switch_follow.py --course "$c" --level L2 --step-height "$h" --stair-steps 5 --seed 800 \
    --num-envs 64 --obs-noise --terminations saro --resume --arms fixed:stairs \
    --extra-policy "stairs=$CK" --ckpt-root "$W/eval_ckpts" \
    --json-out "$O/${TAG}_n5_${c}_h${h}_saro_s800.json" > "logs/switch_follow/quick_${TAG}.log" 2>&1 || echo "FAILED $c $h"
done; done
"$PY" - "$O" "$TAG" $HEIGHTS <<'PY'
import json, sys
out, tag, heights = sys.argv[1], sys.argv[2], sys.argv[3:]
print(f"{tag}: success / fall / lost %, 64 trials per cell, tipping-only")
for c in ("stairs_up", "stairs_down"):
  cells = []
  for h in heights:
    t = json.load(open(f"{out}/{tag}_n5_{c}_h{h}_saro_s800.json"))["arms"]["fixed:stairs"]["trials"]
    p = lambda o: 100 * sum(x["outcome"] == o for x in t) / len(t)
    cells.append(f"{h} m: {p('success'):5.1f} / {p('fall'):4.1f} / {p('lost'):5.1f}")
  print(f"  {c:11s} | " + " | ".join(cells))
PY
