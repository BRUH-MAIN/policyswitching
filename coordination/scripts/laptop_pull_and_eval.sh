#!/usr/bin/env bash
# Run on the LAPTOP. Reads coordination/status/cluster.json for the given
# run_id, fetches the checkpoint (from HF directly for PAS, or via rsync over
# SSH from the cluster's local disk for specialists -- see findings.md bug #4
# for why specialists have no HF copy), runs a local numeric eval with
# unitree_rl_mjlab/scripts/eval_checkpoint.py, records the result, and commits.
#
# Usage:
#   coordination/scripts/laptop_pull_and_eval.sh <run_id> [-- extra eval_checkpoint.py args]
#
# run_id is one of: go2_pas_stage1, go2_pas_stage2,
#                    go2_spec_flat, go2_spec_stairs, go2_spec_rough, go2_spec_gaps
#
# Examples:
#   coordination/scripts/laptop_pull_and_eval.sh go2_pas_stage2 -- --anneal-prob 0.0
#   coordination/scripts/laptop_pull_and_eval.sh go2_spec_gaps
#
# CLUSTER_HOST / CLUSTER_REPO_PATH below match this project's actual SSH
# alias and canonical checkout (see docs/CLAUDE.cluster.md) -- override via
# env var if either ever changes.

set -euo pipefail

RUN_ID="${1:?usage: $0 <run_id> [-- extra eval_checkpoint.py args]}"
shift
EXTRA_ARGS=()
if [ "${1:-}" = "--" ]; then
  shift
  EXTRA_ARGS=("$@")
fi

CLUSTER_HOST="${CLUSTER_HOST:-172.17.16.11}"     # d_palmani@172.17.16.11 per ~/.ssh/config
CLUSTER_USER="${CLUSTER_USER:-d_palmani}"
CLUSTER_REPO_PATH="${CLUSTER_REPO_PATH:-/dist_home/d_palmani/c-08/policyswitching}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MJLAB_DIR="$REPO_ROOT/unitree_rl_mjlab"
CONDA_ENV="${CONDA_ENV:-unitree_rl_mjlab}"
NUM_ENVS="${NUM_ENVS:-256}"   # empirically fits this laptop's 8GB VRAM as of 2026-09-11, see coordination/status/laptop.json
STEPS="${STEPS:-1200}"        # match the cluster's eval horizon so numbers are comparable

echo "==> Pulling latest repo state"
cd "$REPO_ROOT"
git pull --rebase

echo "==> Reading coordination/status/cluster.json for ${RUN_ID}"
RUN_INFO=$(python3 - "$RUN_ID" <<'PY'
import json, sys
run_id = sys.argv[1]
data = json.load(open("coordination/status/cluster.json"))
run = data["runs"].get(run_id)
if run is None:
    print("MISSING", "", "", "", "", "")
    sys.exit(0)
print(
    run.get("task", "MISSING"),
    run.get("source", "MISSING"),
    run.get("hf_repo") or "-",
    run.get("hf_stage") or "-",
    run.get("path_on_cluster") or "-",
    run.get("iteration") or "-",
)
PY
)
read -r TASK SOURCE HF_REPO HF_STAGE PATH_ON_CLUSTER ITERATION <<< "$RUN_INFO"

if [ "$TASK" = "MISSING" ]; then
  echo "[ERROR] '${RUN_ID}' not found in coordination/status/cluster.json -- check the run_id, or ask the cluster session to run cluster_update_status.sh for it first." >&2
  exit 1
fi

CKPT_ARGS=(--task "$TASK")

if [ "$SOURCE" = "hf" ]; then
  echo "==> ${RUN_ID}: pulling directly from HF (${HF_REPO}, ${HF_STAGE}) -- no SSH needed"
  CKPT_ARGS+=(--hf-repo "$HF_REPO" --hf-stage "$HF_STAGE")
elif [ "$SOURCE" = "local" ]; then
  if [ "$PATH_ON_CLUSTER" = "-" ]; then
    echo "[ERROR] '${RUN_ID}' has source=local but no path_on_cluster yet -- checkpoint isn't ready (check coordination/status/cluster.json / coordination/inbox/to-laptop.md)." >&2
    exit 1
  fi
  LOCAL_PATH="$REPO_ROOT/$PATH_ON_CLUSTER"
  echo "==> ${RUN_ID}: rsyncing from ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_REPO_PATH}/${PATH_ON_CLUSTER}"
  mkdir -p "$(dirname "$LOCAL_PATH")"
  rsync -avz --progress \
    "${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_REPO_PATH}/${PATH_ON_CLUSTER}" \
    "$LOCAL_PATH"
  CKPT_ARGS+=(--checkpoint "$LOCAL_PATH")
else
  echo "[ERROR] unrecognized source '${SOURCE}' for ${RUN_ID}" >&2
  exit 1
fi

echo "==> Running local evaluation (conda env: ${CONDA_ENV}, num-envs=${NUM_ENVS}, steps=${STEPS})"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# PYTHONPATH must point at THIS repo's unitree_rl_mjlab/ explicitly -- the
# unitree_rl_mjlab conda env's own editable install can otherwise resolve
# `import src` to the unrelated /home/rohan/unitree_rl_mjlab project. See
# CLAUDE.md.
export PYTHONPATH="$MJLAB_DIR:${PYTHONPATH:-}"
export MUJOCO_GL=egl

RESULT_LOG="coordination/results/${RUN_ID}-step${ITERATION}-eval.log"
cd "$MJLAB_DIR"
python3 scripts/eval_checkpoint.py \
  "${CKPT_ARGS[@]}" \
  --num-envs "$NUM_ENVS" --steps "$STEPS" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  2>&1 | tee "$REPO_ROOT/$RESULT_LOG"
cd "$REPO_ROOT"

echo "==> Updating coordination/status/laptop.json"
python3 - "$RUN_ID" "$ITERATION" "$RESULT_LOG" <<'PY'
import json, sys, datetime
run_id, iteration, result_log = sys.argv[1:4]
path = "coordination/status/laptop.json"
data = json.load(open(path))
data["last_updated_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
data.setdefault("evaluated", {})[run_id] = {
    "iteration": int(iteration) if iteration != "-" else None,
    "verdict": "pending_review",
    "via": f"laptop_pull_and_eval.sh -> {result_log}",
}
json.dump(data, open(path, "w"), indent=2)
PY

echo ""
echo "==> Eval done. Raw output at ${RESULT_LOG}"
echo "==> Next: read the log critically (see docs/CLAUDE.laptop.md -- watch for the"
echo "    survival-vs-bracing and anneal_prob gotchas in findings.md before trusting it),"
echo "    write coordination/results/${RUN_ID}-step${ITERATION}-analysis.md, update"
echo "    coordination/inbox/to-cluster.md if a next step is warranted, then:"
echo "      git add coordination/ && git commit -m '...' && git push"
