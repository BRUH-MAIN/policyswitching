#!/bin/bash
# Re-run the lead sweep with a different checkpoint in the stairs slot.
#
# Same course, task, seeds and arms as Addendum 2B of
# coordination/results/switch-follow-preregistration.md: L1 on seeds 500-502 with
# observation noise off and on, L2 on seeds 600-601 with it off.
#
# Usage:
#   TAG=stairsv2 STAIRS_CKPT=eval_ckpts/go2_spec_stairs_v2/model_9999.pt \
#       unitree_rl_mjlab/scripts/switch_follow_stairs_ckpt.sh
set -euo pipefail
: "${TAG:?set TAG, the output file prefix (e.g. stairsv2)}"
: "${STAIRS_CKPT:?set STAIRS_CKPT, the checkpoint to put in the stairs slot}"

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/unitree_rl_mjlab/bin/python}"
CK="${CKPT_ROOT:-$W/eval_ckpts}"
OUT="eval_results/switch_follow"
ARMS="fixed:flat fixed:rough fixed:stairs hard:-0.3:label hard:0.0:label hard:0.3:label hard:0.8:label hard:1.5:label"

cd "$W"
STAIRS_CKPT="$(realpath "$STAIRS_CKPT")"
export PYTHONPATH="$W" MUJOCO_GL=egl
mkdir -p logs/switch_follow

run() {  # seed level noise-flag condition
  "$PY" scripts/switch_follow.py --seed "$1" --level "$2" --num-envs 256 --arms $ARMS $3 --resume \
    --extra-policy "stairs=$STAIRS_CKPT" --ckpt-root "$CK" \
    --json-out "$OUT/${TAG}_$2_$4_s$1.json" > "logs/switch_follow/${TAG}_$2_$4_s$1.log" 2>&1
}
for s in 500 501 502; do run "$s" L1 "" clean; done
for s in 500 501 502; do run "$s" L1 "--obs-noise" noisy; done
for s in 600 601; do run "$s" L2 "" clean; done

PAIRS=("hard:1.5:label>hard:0.3:label" "hard:0.3:label>fixed:stairs" "hard:0.0:label>hard:0.3:label" "hard:-0.3:label>hard:0.3:label")
for spec in "L1 clean 500 501 502" "L1 noisy 500 501 502" "L2 clean 600 601"; do
  set -- $spec
  level=$1; cond=$2; shift 2
  echo "=========== $TAG, $level, observation noise: $cond ==========="
  "$PY" scripts/switch_follow_analyze.py confirm $(for s in "$@"; do echo "$OUT/${TAG}_${level}_${cond}_s$s.json"; done) --pairs "${PAIRS[@]}"
done
