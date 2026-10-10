#!/bin/bash
# Presentation evaluation (coordination/results/switch-follow-preregistration.md, Addendum 8):
# three switching controllers on the bank flat + rough + stairs v8b against GeneralistV3,
# on randomised layouts with real risers. One array task = one layout.
#
#   MODE=smoke  SEED_BASE=1000 sbatch --array=0    a100/switch_follow_present_slurm.sh   # timing check
#   MODE=record SEED_BASE=2000 sbatch --array=0-11 a100/switch_follow_present_slurm.sh   # classifier scans
#   MODE=main   SEED_BASE=1000 CLF=<clf.pt> sbatch --array=0-23 a100/switch_follow_present_slurm.sh
#   MODE=genv3  SEED_BASE=1000 GENV3=<ckpt> sbatch --array=0-23 a100/switch_follow_present_slurm.sh
#
# Riser height per layout: [0.09, 0.12, 0.15, 0.17][seed % 4] m, 10-step flights.
# Conditions per layout: saro (primary), training (strict), saro at 20 ms actuator delay.
# Outputs: unitree_rl_mjlab/eval_results/switch_follow/present/<cond>/L<seed>_<main|genv3>.json
#          (genv3 has its own file: --resume refuses a different --extra-policy list).

#SBATCH --job-name=sf-present
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/dist_home/d_palmani/c-08/policyswitching/sf-present-%A_%a.out
#SBATCH --error=/dist_home/d_palmani/c-08/policyswitching/sf-present-%A_%a.err

set -uo pipefail
: "${MODE:?set MODE to smoke, record, main or genv3}"
TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"
SEED=$(( ${SEED_BASE:-0} + TASK ))  # the cluster caps array indices, so layouts are SEED_BASE + index

REPO=/dist_home/d_palmani/c-08/policyswitching
W="$REPO/unitree_rl_mjlab"
PY=/dist_home/d_palmani/.venvs/policyswitching-pas/bin/python3
RL="$W/logs/rsl_rl"
FLAT="$RL/go2_spec_flat/2026-09-06_17-17-15/model_9999.pt"
ROUGH="$RL/go2_spec_rough/2026-09-06_12-07-49/model_9999.pt"
STAIRS="$RL/go2_spec_stairs_v8b/2026-10-09_12-49-32/model_1999.pt"
GEN2="$RL/go2_generalist_v2/2026-10-04_15-51-21/model_9999.pt"
BANK=(--extra-policy "flat=$FLAT" --extra-policy "rough=$ROUGH" --extra-policy "stairs=$STAIRS")

HEIGHTS=(0.09 0.12 0.15 0.17)
H="${HEIGHTS[$((SEED % 4))]}"
OUT="$W/eval_results/switch_follow/present"
LOGD="$W/logs/switch_follow/present"
mkdir -p "$OUT"/{saro,training,saro_d20} "$LOGD"
cd "$W"
export PYTHONPATH="$W" MUJOCO_GL=egl PYTHONUNBUFFERED=1
echo "[INFO] mode=$MODE layout=$SEED riser=$H host=$(hostname)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

run() {  # run <cond> <json> <arms...> -- <extra args...>
  local cond="$1" json="$2"; shift 2
  local arms=() extra=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do arms+=("$1"); shift; done
  [ "${1:-}" = "--" ] && shift
  extra=("$@")
  local term=saro delay=()
  [ "$cond" = "training" ] && term=training
  [ "$cond" = "saro_d20" ] && delay=(--action-delay 1)
  "$PY" scripts/switch_follow.py --random-layout "$SEED" --seed "$SEED" --num-envs 256 --obs-noise --resume \
    --step-height "$H" --stair-steps 10 --terminations "$term" "${delay[@]}" \
    --arms "${arms[@]}" --ckpt-root "$RL" "${BANK[@]}" "${extra[@]}" \
    --json-out "$json" || echo "FAILED $MODE $cond layout $SEED"
}

case "$MODE" in
  smoke)
    run saro "$OUT/smoke_L$SEED.json" hard:0.8:label fixed:stairs
    ;;
  record)
    # Scans for the classifier, from layouts never used for testing (2000-2011).
    mkdir -p "$W/logs/switch_follow/clf_v8b_scans"
    run saro "$OUT/record_L$SEED.json" hard:0.8:label -- \
      --record-scans "$W/logs/switch_follow/clf_v8b_scans/scans_L$SEED.pt"
    ;;
  main)
    : "${CLF:?set CLF to the retrained classifier weights}"
    for cond in saro training saro_d20; do
      run "$cond" "$OUT/$cond/L${SEED}_main.json" \
        hard:0.8:label "clf:$CLF:0.1:5" soft:0.8:0.0:label fixed:stairs fixed:gen2 -- \
        --extra-policy "gen2=$GEN2"
    done
    ;;
  genv3)
    : "${GENV3:?set GENV3 to the GeneralistV3 checkpoint}"
    for cond in saro training saro_d20; do
      run "$cond" "$OUT/$cond/L${SEED}_genv3.json" fixed:genv3 -- --extra-policy "genv3=$GENV3"
    done
    ;;
  *) echo "[ERROR] unknown MODE=$MODE" >&2; exit 1 ;;
esac
echo "[DONE] mode=$MODE layout=$SEED"
