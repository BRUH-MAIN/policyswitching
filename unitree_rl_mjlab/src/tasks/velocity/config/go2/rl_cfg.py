"""RL configuration for Unitree Go2 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from src.tasks.velocity.mdp.pas import PasModelCfg, PasPpoAlgorithmCfg


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_go2_pas_ppo_runner_cfg(enable_annealing: bool) -> RslRlOnPolicyRunnerCfg:
  """RL runner config for SARO's PAS training (arXiv:2407.16412).

  ``enable_annealing=False`` is Stage 1 (oracle): the actor's latent stays
  pinned at the real, privileged value while the estimator trains via an
  auxiliary reconstruction loss only. ``enable_annealing=True`` is Stage 2:
  resume from the Stage-1 checkpoint (``--agent.resume --agent.load-run ...``)
  and the actor/estimator are annealed and jointly fine-tuned by PPO itself.
  """
  return RslRlOnPolicyRunnerCfg(
    obs_groups={
      "actor": ("actor", "privileged_state"),
      "critic": ("critic", "privileged_state"),
    },
    actor=PasModelCfg(
      class_name="src.tasks.velocity.mdp.pas:PasActorModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      initial_anneal_prob=1.0,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=PasPpoAlgorithmCfg(
      class_name="src.tasks.velocity.mdp.pas:PasPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      enable_annealing=enable_annealing,
      anneal_base=0.9998,
      estimator_lr=1.0e-3,
      estimator_loss_coef=1.0,
    ),
    experiment_name="go2_pas",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=40000,
  )
