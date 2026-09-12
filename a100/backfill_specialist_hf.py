"""One-off backfill: upload existing specialist checkpoints to a private HF repo.

The three finished specialists (go2_spec_flat/rough/stairs) were trained before
HfSyncVelocityOnPolicyRunner existed, so their checkpoints only ever landed on
local disk (findings.md bug #4) -- and now that the laptop has no SSH route to
this cluster's local disk either, HF is the only path they reach that machine
by (user decision, 2026-09-12: a NEW PRIVATE repo, not the public go2-pas-saro).

Unlike push_checkpoint_to_hf() (rl/hf_upload.py, used during live training,
which swallows upload failures so a flaky network never crashes a multi-day PPO
run), this fails loudly -- a one-time backfill should be verified, not silently
skipped. Needs no GPU: run this on the login node any time, drained nodes or not.

Usage:
  export HF_TOKEN=...
  python3 a100/backfill_specialist_hf.py --repo-id RohanRamesh/go2-specialists
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# (experiment_name, checkpoint path relative to --mjlab-dir). Source: findings.md
# "Terrain specialists" table.
CHECKPOINTS = [
  ("go2_spec_flat", "logs/rsl_rl/go2_spec_flat/2026-09-06_17-17-15/model_9999.pt"),
  ("go2_spec_rough", "logs/rsl_rl/go2_spec_rough/2026-09-06_12-07-49/model_9999.pt"),
  ("go2_spec_stairs", "logs/rsl_rl/go2_spec_stairs/2026-09-05_22-37-43/model_9999.pt"),
]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--repo-id", default="RohanRamesh/go2-specialists",
                       help="Must match SPECIALIST_HF_REPO in a100/eval_matrix.py and "
                       "HF_CHECKPOINT_REPO's default in a100/train_specialist_slurm.sh.")
  parser.add_argument("--mjlab-dir", default=str(Path(__file__).resolve().parent.parent / "unitree_rl_mjlab"))
  parser.add_argument("--public", action="store_true",
                       help="Create the repo public instead of private. Do NOT pass this for "
                       "go2-specialists -- the user explicitly declined publishing these "
                       "(see findings.md bug #12's neighbouring decision record, 2026-09-12).")
  args = parser.parse_args()

  token = os.environ.get("HF_TOKEN")
  if not token:
    sys.exit("[ERROR] HF_TOKEN not set.")

  mjlab_dir = Path(args.mjlab_dir)
  local_paths = []
  for experiment, rel_path in CHECKPOINTS:
    local_path = mjlab_dir / rel_path
    if not local_path.is_file():
      sys.exit(f"[ERROR] {local_path} not found -- aborting before any upload "
                "(checked all three first so a missing file doesn't leave a partial backfill).")
    local_paths.append((experiment, local_path))

  # Pass the token per-call rather than huggingface_hub.login(token=...): login()
  # writes it to a local credential file (~/.cache/huggingface/token) that would
  # persist on this shared cluster filesystem after the process exits -- exactly
  # the plaintext-token-on-shared-disk exposure the user's deploy-key decision
  # steered git auth away from (`gh` on the cluster). No reason to accept it here.
  from huggingface_hub import HfApi, create_repo
  private = not args.public
  create_repo(args.repo_id, repo_type="model", exist_ok=True, private=private, token=token)
  print(f"[INFO] Repo ready: hf.co/{args.repo_id} (private={private})")

  api = HfApi(token=token)
  for experiment, local_path in local_paths:
    path_in_repo = f"{experiment}/{local_path.name}"
    print(f"[INFO] Uploading {local_path} -> hf.co/{args.repo_id}/{path_in_repo}")
    api.upload_file(
      path_or_fileobj=str(local_path),
      path_in_repo=path_in_repo,
      repo_id=args.repo_id,
      repo_type="model",
    )

  print("[INFO] Verifying...")
  files = set(api.list_repo_files(args.repo_id, repo_type="model"))
  all_ok = True
  for experiment, local_path in local_paths:
    expected = f"{experiment}/{local_path.name}"
    ok = expected in files
    all_ok &= ok
    print(f"  [{'OK' if ok else 'MISSING'}] {expected}")
  if not all_ok:
    sys.exit("[ERROR] one or more uploads did not verify -- see MISSING lines above.")

  print("\n[INFO] Done. Next: for each specialist, register it in cluster.json so the laptop")
  print("       pulls from HF instead of the (now-dead) rsync path, e.g.:")
  for experiment, local_path in local_paths:
    run_id = experiment  # e.g. go2_spec_flat
    print(f"  coordination/scripts/cluster_update_status.sh {run_id} 9999 hf {args.repo_id} {experiment}")


if __name__ == "__main__":
  main()
