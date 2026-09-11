"""Best-effort checkpoint upload to a Hugging Face model repo.

A leaf module (imports nothing from this package) so both runner.py and
mdp/pas.py can use it without an import cycle -- pas.py already imports
runner.py.
"""

import os


def push_checkpoint_to_hf(repo_id: str, checkpoint_path: str) -> None:
  """Best-effort upload of one checkpoint to a Hugging Face model repo.

  No-ops (with a warning) if `huggingface_hub` isn't installed or the upload
  fails for any reason -- this must never be what breaks a training run.
  Requires `huggingface_hub.login()` to have already been called (e.g. via
  an `HF_TOKEN` env var / `huggingface-cli login` in the calling notebook).
  """
  try:
    from huggingface_hub import HfApi
  except ImportError:
    print("[WARN] huggingface_hub not installed; skipping checkpoint upload.")
    return
  stage = os.environ.get("HF_CHECKPOINT_STAGE", "stage1")
  filename = os.path.basename(checkpoint_path)
  try:
    HfApi().upload_file(
      path_or_fileobj=checkpoint_path,
      path_in_repo=f"{stage}/{filename}",
      repo_id=repo_id,
      repo_type="model",
    )
    print(f"[INFO] Uploaded {filename} to hf.co/{repo_id}/{stage}/")
  except Exception as exc:  # noqa: BLE001
    print(f"[WARN] HF checkpoint upload failed, continuing training. ({exc})")
