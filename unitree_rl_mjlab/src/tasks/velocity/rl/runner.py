import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

from .hf_upload import push_checkpoint_to_hf


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class HfSyncVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """VelocityOnPolicyRunner that also pushes every checkpoint to Hugging Face
  when HF_CHECKPOINT_REPO is set (path prefix: HF_CHECKPOINT_STAGE).

  The stock runner never uploads, which is why the terrain specialists had no
  off-cluster copy at all (findings.md, bug #4). Unset, this behaves exactly
  like VelocityOnPolicyRunner.
  """

  def save(self, path: str, infos=None):
    super().save(path, infos)
    repo_id = os.environ.get("HF_CHECKPOINT_REPO")
    if repo_id:
      push_checkpoint_to_hf(repo_id, path)
