"""PAS (Probability Annealing Selection) building blocks.

Replicates the low-level locomotion training technique from SARO
(arXiv:2407.16412): an "oracle" actor that is privileged during training
(sees a terrain-height-scan latent and a small privileged state alongside
proprioception) is annealed, over a second training stage, toward relying on
an estimator's prediction of that same latent from proprioception history
alone -- ending in a policy deployable with proprioception only.

Two stages, one model/algorithm pair:
  - "oracle" (Stage 1): `PasPpoAlgorithmCfg(enable_annealing=False)`. The
    actor's latent is always the real, privileged one. The estimator is
    trained purely by an auxiliary MSE-reconstruction loss
    (`PasPPO._estimator_aux_step`), fully decoupled from the PPO graph.
  - "anneal" (Stage 2): `PasPpoAlgorithmCfg(enable_annealing=True)`, resumed
    from the Stage-1 checkpoint. `PasPPO.update()` decays `anneal_prob` every
    iteration; the actor mixes real/predicted latents per-env via a Bernoulli
    draw. Both branches stay in the autograd graph, so PPO's own backward
    pass trains the estimator jointly with the rest of the actor whenever its
    branch was sampled -- matching the paper's Stage 2 description ("the loss
    of RL is utilized to simultaneously optimize both the estimator and the
    low-level MLP").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl.config import RslRlModelCfg, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, HiddenState, RNN
from rsl_rl.utils import unpad_trajectories

from src.tasks.velocity.rl.runner import VelocityOnPolicyRunner

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_friction(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Read back the DR-sampled tangential friction coefficient (privileged).

  `dr.geom_friction` (mode="startup", shared_random=True) writes this value
  directly into the batched MuJoCo model at `geom_friction[:, geom_id, 0]`; it
  is otherwise only used for domain randomization and never observed.
  """
  friction = env.sim.model.geom_friction[:, asset_cfg.geom_ids, 0]  # [B, N]
  return friction.mean(dim=1, keepdim=True)  # [B, 1] -- all N share one value.


@dataclass
class PasModelCfg(RslRlModelCfg):
  """Actor config for the PAS oracle/anneal architecture.

  Dimensions default to the paper's Table VIII/IX values for the Go2.
  """

  proprio_dim: int = 47
  height_scan_dim: int = 187
  privileged_dim: int = 4
  terrain_latent_dim: int = 32
  estimator_hidden: int = 256
  estimator_num_layers: int = 2
  initial_anneal_prob: float = 1.0


@dataclass
class PasPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO config extended with PAS's estimator auxiliary loss + annealing."""

  enable_annealing: bool = False
  """False for Stage 1 (oracle): anneal_prob stays pinned at 1.0."""
  anneal_base: float = 0.9998
  """P_t = anneal_base ** iteration. Paper's best-performing ablation (Table III)."""
  estimator_lr: float = 1.0e-3
  estimator_loss_coef: float = 1.0


class PasActorModel(MLPModel):
  """Actor for SARO's PAS: terrain-encoder + estimator + annealed latent mixing.

  Expects two obs groups for its obs_set: "actor" (proprioception followed by
  a raw, noisy height-scan -- the *last* `height_scan_dim` columns, matching
  this repo's convention of declaring height_scan last) and
  "privileged_state" (base linear velocity + foot friction, in that order).
  """

  is_recurrent: bool = True

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    cnn_cfg: dict | None = None,
    distribution_cfg: dict | None = None,
    proprio_dim: int = 47,
    height_scan_dim: int = 187,
    privileged_dim: int = 4,
    terrain_latent_dim: int = 32,
    estimator_hidden: int = 256,
    estimator_num_layers: int = 2,
    initial_anneal_prob: float = 1.0,
  ) -> None:
    del cnn_cfg  # Unused by this model.
    self._proprio_dim = proprio_dim
    self._height_scan_dim = height_scan_dim
    self._privileged_dim = privileged_dim
    self._latent_state_dim = terrain_latent_dim + privileged_dim
    # Mutable; 1.0 = always the real, privileged latent (oracle / Stage 1).
    # Updated once per PPO iteration by PasPPO.update() in Stage 2.
    self.anneal_prob = initial_anneal_prob

    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    # Terrain encoder E_t (Table VIII: hidden [128, 64]). Trained by the RL
    # loss whenever its output is selected -- it sits on the same backprop
    # path as the low-level actor MLP, exactly as in the paper's Fig. 4.
    self.terrain_encoder = MLP(height_scan_dim, terrain_latent_dim, (128, 64), activation)

    # Estimator: LSTM (Table VIII: hidden [256, 256], i.e. 2 layers of 256)
    # over proprioception history, then an MLP head (hidden [256, 128]) into
    # the same latent space as the oracle's l_t.
    self.estimator_rnn = RNN(proprio_dim, estimator_hidden, estimator_num_layers, "lstm")
    self.estimator_head = MLP(estimator_hidden, self._latent_state_dim, (256, 128), activation)

  def _get_latent_dim(self) -> int:
    return self._proprio_dim + self._latent_state_dim

  def _raw_obs_dim_split(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalized (proprio, height_scan, privileged_state), split from the
    base class's naive concat-and-normalize over ("actor", "privileged_state").

    `self.obs_normalizer` is sized for that raw concatenated width (not our
    post-encoding latent), so normalization must happen here, before any
    encoding -- mirroring how `RNNModel.get_latent()` normalizes raw inputs
    before running its RNN.
    """
    raw = super().get_latent(obs)  # [B, actor_dim + privileged_dim], normalized.
    proprio = raw[..., : self._proprio_dim]
    height_scan = raw[..., self._proprio_dim : self._proprio_dim + self._height_scan_dim]
    privileged_state = raw[..., self._proprio_dim + self._height_scan_dim :]
    return proprio, height_scan, privileged_state

  def real_latent(self, obs: TensorDict, masks: torch.Tensor | None = None) -> torch.Tensor:
    """l_t = concat(terrain_encoder(height_scan), privileged_state).

    During the batched PPO update, sequences are time-padded; `masks`
    unpads the result to match `predicted_latent`'s output, which is
    unpadded internally by the RNN (see `RNN.forward`'s batch-mode branch).
    """
    _, height_scan, privileged_state = self._raw_obs_dim_split(obs)
    terrain_latent = self.terrain_encoder(height_scan)
    real_l_t = torch.cat([terrain_latent, privileged_state], dim=-1)
    if masks is not None:
      real_l_t = unpad_trajectories(real_l_t, masks)
    return real_l_t

  def predicted_latent(
    self,
    proprio: torch.Tensor,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    """The estimator's predicted l_t from (normalized) proprioception history."""
    rnn_out = self.estimator_rnn(proprio, masks, hidden_state).squeeze(0)
    return self.estimator_head(rnn_out)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    proprio, height_scan, privileged_state = self._raw_obs_dim_split(obs)
    terrain_latent = self.terrain_encoder(height_scan)
    real_l_t = torch.cat([terrain_latent, privileged_state], dim=-1)
    # predicted_l_t is unpadded internally by the RNN when masks is not None;
    # real_l_t and proprio must be unpadded the same way to stay aligned.
    predicted_l_t = self.predicted_latent(proprio, masks, hidden_state)
    if masks is not None:
      proprio = unpad_trajectories(proprio, masks)
      real_l_t = unpad_trajectories(real_l_t, masks)

    # At anneal_prob == 1.0 (oracle / Stage 1) this deterministically selects
    # real_l_t for every env, so predicted_l_t receives no gradient here --
    # the estimator is trained only via PasPPO's auxiliary MSE step.
    # real_l_t is [B, D] during rollout but [T, B_chunk, D] during the
    # batched recurrent update, so build the mask from all but the last dim.
    use_real = torch.bernoulli(
      torch.full((*real_l_t.shape[:-1], 1), self.anneal_prob, device=real_l_t.device)
    ).bool()
    mixed_l_t = torch.where(use_real, real_l_t, predicted_l_t)

    return torch.cat([proprio, mixed_l_t], dim=-1)

  def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
    self.estimator_rnn.reset(dones, hidden_state)

  def get_hidden_state(self) -> HiddenState:
    return self.estimator_rnn.hidden_state  # type: ignore[return-value]

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    self.estimator_rnn.detach_hidden_state(dones)


class PasPPO(PPO):
  """PPO plus PAS's estimator auxiliary reconstruction loss and P_t schedule."""

  def __init__(
    self,
    actor,
    critic,
    storage,
    *,
    enable_annealing: bool = False,
    anneal_base: float = 0.9998,
    estimator_lr: float = 1.0e-3,
    estimator_loss_coef: float = 1.0,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    if not isinstance(actor, PasActorModel):
      raise TypeError("PasPPO requires a PasActorModel actor.")
    self.enable_annealing = enable_annealing
    self.anneal_base = anneal_base
    self.estimator_loss_coef = estimator_loss_coef
    self._iteration = 0
    self.estimator_optimizer = torch.optim.Adam(
      list(actor.estimator_rnn.parameters()) + list(actor.estimator_head.parameters()),
      lr=estimator_lr,
    )

  def _estimator_aux_step(self) -> float:
    """One epoch of MSE-reconstruction training for the estimator.

    Runs against the still-populated rollout storage, *before*
    `super().update()` clears it. This is a fully separate forward pass,
    loss, and optimizer from PPO's own update -- sequential, not
    interleaved, so there is no shared-autograd-graph conflict between the
    two backward passes.
    """
    actor: PasActorModel = self.actor  # type: ignore[assignment]
    generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, 1)
    total_loss = 0.0
    num_batches = 0
    for batch in generator:
      proprio, _, _ = actor._raw_obs_dim_split(batch.observations)
      with torch.no_grad():
        target = actor.real_latent(batch.observations, masks=batch.masks)
      pred = actor.predicted_latent(proprio, masks=batch.masks, hidden_state=batch.hidden_states[0])
      loss = self.estimator_loss_coef * F.mse_loss(pred, target)
      self.estimator_optimizer.zero_grad()
      loss.backward()
      self.estimator_optimizer.step()
      total_loss += loss.item()
      num_batches += 1
    return total_loss / max(num_batches, 1)

  def update(self) -> dict[str, float]:
    estimator_mse = self._estimator_aux_step()
    loss_dict = super().update()
    loss_dict["estimator_mse"] = estimator_mse
    if self.enable_annealing:
      self._iteration += 1
      anneal_prob = self.anneal_base**self._iteration
      self.actor.anneal_prob = anneal_prob  # type: ignore[attr-defined]
      loss_dict["anneal_prob"] = anneal_prob
    return loss_dict


class PasOnPolicyRunner(VelocityOnPolicyRunner):
  """VelocityOnPolicyRunner, but tolerant of ONNX export failing on PasActorModel.

  `MLPModel.as_onnx()`'s generic wrapper assumes `get_latent()` is a plain
  concat-and-normalize (true for a stock actor, false for `PasActorModel`,
  which encodes/mixes a privileged latent). Until a PAS-specific
  `as_onnx()`/`as_jit()` override exists (deployment export is a follow-up,
  not needed for training), skip ONNX export rather than crash training at
  every `save_interval` checkpoint -- the `.pt` checkpoint (used to resume
  between Stage 1 and Stage 2) is saved first and is unaffected.
  """

  def save(self, path: str, infos=None) -> None:
    try:
      super().save(path, infos)
    except Exception as exc:  # noqa: BLE001
      print(f"[WARN] PasOnPolicyRunner: ONNX export failed, .pt checkpoint still saved. ({exc})")
