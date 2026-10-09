#!/bin/bash
# Sim-to-real robustness of a stairs policy: control delay, motor strength, payload.
# Usage: STAIRS_CKPT=<ckpt> TAG=<name> [STEPS=10] scripts/stairs_robustness.sh
# 128 trials per cell (seed 800), training sensor noise on, tipping-only falls, leader 0.5 m/s.
set -uo pipefail
: "${STAIRS_CKPT:?set STAIRS_CKPT}"; : "${TAG:?set TAG}"
STEPS="${STEPS:-10}"
STAIRS_CKPT="$(realpath "$STAIRS_CKPT")"
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/unitree_rl_mjlab/bin/python}"
OUT=eval_results/switch_follow/robustness
cd "$W"; mkdir -p "$OUT" logs/switch_follow
export PYTHONPATH="$W" MUJOCO_GL=egl
CONDS=("nominal:" "delay1:--action-delay 1" "delay2:--action-delay 2"
       "motor80:--motor-strength 0.8" "motor120:--motor-strength 1.2"
       "payload1:--payload 1.0" "payload2:--payload 2.0"
       "combo:--action-delay 1 --motor-strength 0.85 --payload 1.0")
for h in 0.15 0.17; do for course in stairs_up stairs_down; do for c in "${CONDS[@]}"; do
  name="${c%%:*}"; flags="${c#*:}"
  "$PY" scripts/switch_follow.py --course "$course" --level L2 --step-height "$h" --stair-steps "$STEPS" --seed 800 \
    --num-envs 128 --obs-noise --terminations saro --resume --arms fixed:stairs $flags \
    --extra-policy "stairs=$STAIRS_CKPT" --ckpt-root "$W/eval_ckpts" \
    --json-out "$OUT/${TAG}_n${STEPS}_${course}_h${h}_${name}.json" > "logs/switch_follow/robust_${TAG}.log" 2>&1 \
    || echo "FAILED $course $h $name"
done; done; done
"$PY" - "$OUT" "$TAG" "$STEPS" <<'PY'
import json, sys
out, tag, steps = sys.argv[1:4]
names = ["nominal", "delay1", "delay2", "motor80", "motor120", "payload1", "payload2", "combo"]
label = {"nominal": "nominal", "delay1": "20 ms delay", "delay2": "40 ms delay", "motor80": "motors 80%",
         "motor120": "motors 120%", "payload1": "+1 kg", "payload2": "+2 kg", "combo": "20 ms + 85% + 1 kg"}
print(f"{tag}, {steps}-step flights, 128 trials per cell, tipping-only: success / fall / lost %")
print(f"{'condition':22s} | " + " | ".join(f"{c} {h} cm" for h in ("15", "17") for c in ("up", "down")))
for n in names:
  cells = []
  for h in ("0.15", "0.17"):
    for c in ("stairs_up", "stairs_down"):
      t = json.load(open(f"{out}/{tag}_n{steps}_{c}_h{h}_{n}.json"))["arms"]["fixed:stairs"]["trials"]
      p = lambda o: 100 * sum(x["outcome"] == o for x in t) / len(t)
      cells.append(f"{p('success'):5.1f}/{p('fall'):4.1f}/{p('lost'):4.1f}")
  print(f"{label[n]:22s} | " + " | ".join(cells))
PY
