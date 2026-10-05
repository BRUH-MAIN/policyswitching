#!/bin/bash
# A stairs policy on real riser heights: single straight flights, up and down.
# Usage: STAIRS_CKPT=<ckpt> TAG=<name> [STEPS=5] switch_follow_real_stairs.sh
# 128 trials x seeds 800, 801 per cell; training sensor noise on; leader 0.5 m/s;
# both fall definitions (training: any non-foot contact > 10 N; saro: tipping over only).
set -uo pipefail
: "${STAIRS_CKPT:?set STAIRS_CKPT}"; : "${TAG:?set TAG}"
STEPS="${STEPS:-5}"
STAIRS_CKPT="$(realpath "$STAIRS_CKPT")"
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/unitree_rl_mjlab/bin/python}"
OUT=eval_results/switch_follow/real_stairs
cd "$W"; mkdir -p "$OUT" logs/switch_follow
export PYTHONPATH="$W" MUJOCO_GL=egl
for term in training saro; do for course in stairs_up stairs_down; do for h in 0.09 0.12 0.15 0.17; do for s in 800 801; do
  "$PY" scripts/switch_follow.py --course "$course" --level L2 --step-height "$h" --stair-steps "$STEPS" --seed "$s" \
    --num-envs 128 --obs-noise --terminations "$term" --resume --arms fixed:stairs \
    --extra-policy "stairs=$STAIRS_CKPT" --ckpt-root "$W/eval_ckpts" \
    --json-out "$OUT/${TAG}_n${STEPS}_${course}_h${h}_${term}_s$s.json" > "logs/switch_follow/realstairs_${TAG}.log" 2>&1 \
    || echo "FAILED $course $h $term $s"
done; done; done; done
"$PY" - "$OUT" "$TAG" "$STEPS" <<'PY'
import json, sys
out, tag, steps = sys.argv[1:4]
print(f"{tag}, {steps}-step flights: success % / fall % / lost % (256 trials per cell)")
for term in ("training", "saro"):
  for course in ("stairs_up", "stairs_down"):
    cells = []
    for h in ("0.09", "0.12", "0.15", "0.17"):
      t = []
      for s in (800, 801):
        t += json.load(open(f"{out}/{tag}_n{steps}_{course}_h{h}_{term}_s{s}.json"))["arms"]["fixed:stairs"]["trials"]
      pct = lambda o: 100 * sum(x["outcome"] == o for x in t) / len(t)
      cells.append(f"{h} m: {pct('success'):5.1f} / {pct('fall'):4.1f} / {pct('lost'):4.1f}")
    print(f"{term:8s} {course:11s} | " + " | ".join(cells))
PY
