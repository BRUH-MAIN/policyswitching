#!/bin/bash
# How does a stairs policy tolerate a robot-like height scan? (delay, missing cells, bias)
# Usage: STAIRS_CKPT=<ckpt> TAG=<name> switch_follow_scan_faults.sh FIRST_SEED LAST_SEED
# Runs the policy alone on randomised layouts FIRST..LAST under each fault condition.
set -uo pipefail
: "${STAIRS_CKPT:?set STAIRS_CKPT}"; : "${TAG:?set TAG}"
STAIRS_CKPT="$(realpath "$STAIRS_CKPT")"
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/unitree_rl_mjlab/bin/python}"
OUT=eval_results/switch_follow/scan_faults
cd "$W"; mkdir -p "$OUT" logs/switch_follow
export PYTHONPATH="$W" MUJOCO_GL=egl
CONDS=("clean:" "delay2:--scan-delay 2" "delay5:--scan-delay 5" "delay10:--scan-delay 10"
       "drop30:--scan-dropout 0.3" "drop60:--scan-dropout 0.6" "bias3:--scan-bias 0.03" "bias6:--scan-bias 0.06"
       "combo:--scan-delay 5 --scan-dropout 0.3 --scan-bias 0.03")
for s in $(seq "$1" "$2"); do
  for c in "${CONDS[@]}"; do
    name="${c%%:*}"; flags="${c#*:}"
    "$PY" scripts/switch_follow.py --random-layout "$s" --seed "$s" --num-envs 128 --obs-noise --resume \
      --arms fixed:stairs --extra-policy "stairs=$STAIRS_CKPT" $flags ${EXTRA_FLAGS:-} \
      --ckpt-root "$W/eval_ckpts" --json-out "$OUT/${TAG}_${name}_s$s.json" \
      > "logs/switch_follow/scanfault_${TAG}.log" 2>&1 || echo "FAILED $name $s"
  done
done
echo "SCAN_FAULTS_DONE $1-$2"
