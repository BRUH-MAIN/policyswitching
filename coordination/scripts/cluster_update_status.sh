#!/usr/bin/env bash
# Run on the CLUSTER (login node / wherever git+network access exists -- the
# training job itself may not have outbound internet, see the network note in
# coordination/../docs/CLAUDE.cluster.md) after a checkpoint is saved, to
# update the shared heartbeat and notify the laptop side.
#
# This project has TWO checkpoint sources (see findings.md bug #4), so this
# script has two modes instead of one rsync path:
#
#   PAS (go2_pas_stage1 / go2_pas_stage2) -- already synced to HF by
#   a100/hf_sync.py during training, so there is no local file to hash here:
#     coordination/scripts/cluster_update_status.sh go2_pas_stage2 79998 \
#       hf RohanRamesh/go2-pas-saro stage2 [slurm_job_id]
#
#   Specialists (go2_spec_flat/stairs/rough/gaps) -- checkpoints live ONLY on
#   cluster local disk (the stock VelocityOnPolicyRunner has no HF upload
#   logic), so this hashes the file directly:
#     coordination/scripts/cluster_update_status.sh go2_spec_gaps 9999 \
#       local unitree_rl_mjlab/logs/rsl_rl/go2_spec_gaps/<run_dir>/model_9999.pt [slurm_job_id]
#
# Run from the repo root (paths above are relative to it).

set -euo pipefail

RUN_ID="${1:?usage: $0 <run_id> <iteration> hf <hf_repo> <hf_stage> [job_id]  OR  $0 <run_id> <iteration> local <path> [job_id]}"
ITERATION="${2:?see usage}"
MODE="${3:?mode must be 'hf' or 'local'}"

case "$MODE" in
  hf)
    HF_REPO="${4:?usage: $0 <run_id> <iteration> hf <hf_repo> <hf_stage> [job_id]}"
    HF_STAGE="${5:?usage: $0 <run_id> <iteration> hf <hf_repo> <hf_stage> [job_id]}"
    JOB_ID="${6:-}"
    LOCAL_PATH=""
    SHA256=""
    ;;
  local)
    LOCAL_PATH="${4:?usage: $0 <run_id> <iteration> local <path> [job_id]}"
    JOB_ID="${5:-}"
    HF_REPO=""
    HF_STAGE=""
    if [ ! -f "$LOCAL_PATH" ]; then
      echo "[ERROR] $LOCAL_PATH does not exist (run from repo root?)" >&2
      exit 1
    fi
    SHA256=$(sha256sum "$LOCAL_PATH" | cut -d' ' -f1)
    ;;
  *)
    echo "[ERROR] mode must be 'hf' or 'local', got '$MODE'" >&2
    exit 1
    ;;
esac

git pull --rebase

python3 - "$RUN_ID" "$ITERATION" "$MODE" "$HF_REPO" "$HF_STAGE" "$LOCAL_PATH" "$SHA256" "$JOB_ID" <<'PY'
import json, sys, datetime

run_id, iteration, mode, hf_repo, hf_stage, local_path, sha256, job_id = sys.argv[1:9]
path = "coordination/status/cluster.json"
data = json.load(open(path))
data["last_updated_utc"] = datetime.datetime.utcnow().isoformat() + "Z"

run = data["runs"].setdefault(run_id, {})
run["iteration"] = int(iteration)
run["status"] = "complete"  # cluster session: edit to "in_progress"/"timeout" by hand if this call is a checkpoint-in-progress update, not a finish.
if mode == "hf":
    run["source"] = "hf"
    run["hf_repo"] = hf_repo
    run["hf_stage"] = hf_stage
    run["path_on_cluster"] = None
    run["sha256"] = None
else:
    run["source"] = "local"
    run["path_on_cluster"] = local_path
    run["sha256"] = sha256
    run["hf_repo"] = None
    run["hf_stage"] = None
if job_id:
    run["slurm_job_id"] = job_id

json.dump(data, open(path, "w"), indent=2)
print(f"Updated coordination/status/cluster.json: {run_id} -> iteration {iteration} ({mode})")
PY

# Insert the "please evaluate" note right below "## Open" (not appended at
# the file's end, which would land it after "## Done").
python3 - "$RUN_ID" "$ITERATION" "$MODE" "$HF_REPO" "$HF_STAGE" "$LOCAL_PATH" "$JOB_ID" <<'PY'
import sys, datetime

run_id, iteration, mode, hf_repo, hf_stage, local_path, job_id = sys.argv[1:8]
path = "coordination/inbox/to-laptop.md"
lines = open(path).read().splitlines(keepends=True)

entry = [f"\n## {datetime.date.today().isoformat()} -- {run_id} step {iteration} ready\n"]
if mode == "hf":
    entry.append(f"Source: hf ({hf_repo}, {hf_stage}/model_{iteration}.pt).\n")
else:
    entry.append(f"Source: local (cluster disk only, not on HF -- see findings.md bug #4).\n")
    entry.append(f"Path: {local_path} (sha256 in coordination/status/cluster.json).\n")
if job_id:
    entry.append(f"SLURM job {job_id}.\n")

out = []
inserted = False
for line in lines:
    out.append(line)
    if not inserted and line.strip() == "## Open":
        out.extend(entry)
        inserted = True
if not inserted:
    out.extend(entry)

open(path, "w").writelines(out)
print("Inserted please-evaluate note into coordination/inbox/to-laptop.md")
PY

git add coordination/status/cluster.json coordination/inbox/to-laptop.md
git commit -m "status: ${RUN_ID} step ${ITERATION} checkpoint ready for eval"
git push

echo ""
echo "==> If /list-agents shows the laptop session reachable, message it by name now"
echo "    (e.g. 'tell @laptop-eval that ${RUN_ID} step ${ITERATION} is ready') instead"
echo "    of leaving it to poll -- see docs/CLAUDE.cluster.md. This commit is the record"
echo "    either way."
