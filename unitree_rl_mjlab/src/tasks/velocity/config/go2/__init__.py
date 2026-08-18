from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.mdp.pas import PasOnPolicyRunner
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_pas_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .rl_cfg import unitree_go2_pas_ppo_runner_cfg, unitree_go2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# SARO's PAS low-level policy (arXiv:2407.16412). Train Stage 1 to
# convergence, then train Stage 2 with
# `--agent.resume --agent.load-run <stage-1 run dir>` to anneal from the
# oracle latent to the estimator's prediction.
register_mjlab_task(
  task_id="Unitree-Go2-PAS-Oracle",
  env_cfg=unitree_go2_pas_env_cfg(),
  play_env_cfg=unitree_go2_pas_env_cfg(play=True),
  rl_cfg=unitree_go2_pas_ppo_runner_cfg(enable_annealing=False),
  runner_cls=PasOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-PAS-Anneal",
  env_cfg=unitree_go2_pas_env_cfg(),
  play_env_cfg=unitree_go2_pas_env_cfg(play=True),
  rl_cfg=unitree_go2_pas_ppo_runner_cfg(enable_annealing=True),
  runner_cls=PasOnPolicyRunner,
)
