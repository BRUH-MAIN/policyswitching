#!/bin/bash
# Randomised-layout evaluation (Addendum 7 of switch-follow-preregistration.md).
# Usage: switch_follow_random_layouts.sh FIRST_SEED LAST_SEED     e.g. 900 919
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/unitree_rl_mjlab/bin/python}"
CK="${CKPT_ROOT:-$W/eval_ckpts}"
OUT=eval_results/switch_follow/random
cd "$W"; mkdir -p "$OUT" logs/switch_follow
export PYTHONPATH="$W" MUJOCO_GL=egl
for s in $(seq "$1" "$2"); do
  # Run A: stairs slot = stairs v2.
  "$PY" scripts/switch_follow.py --random-layout "$s" --seed "$s" --num-envs 128 --obs-noise --resume \
    --arms fixed:stairs fixed:gen1 fixed:gen2 hard:-0.3:label hard:0.3:label hard:1.5:label blind:stairs \
    --extra-policy "stairs=$CK/go2_spec_stairs_v2/model_9999.pt" \
    --extra-policy "gen1=$CK/go2_generalist/model_9999.pt" --extra-policy "gen2=$CK/go2_generalist_v2/model_9999.pt" \
    --ckpt-root "$CK" --json-out "$OUT/A_s$s.json" > "logs/switch_follow/random_A_s$s.log" 2>&1 || echo "FAILED A $s"
  # Run B: stairs slot = the as-trained stairs specialist (the bank's default).
  "$PY" scripts/switch_follow.py --random-layout "$s" --seed "$s" --num-envs 128 --obs-noise --resume \
    --arms fixed:stairs hard:0.3:label hard:1.5:label \
    --ckpt-root "$CK" --json-out "$OUT/B_s$s.json" > "logs/switch_follow/random_B_s$s.log" 2>&1 || echo "FAILED B $s"
done
echo "RANDOM_DONE $1-$2"
