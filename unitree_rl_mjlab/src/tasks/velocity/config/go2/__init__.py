from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.mdp.pas import PasOnPolicyRunner
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_pas_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_spec_flat_env_cfg,
  unitree_go2_spec_gaps_env_cfg,
  unitree_go2_spec_rough_env_cfg,
  unitree_go2_spec_stairs_env_cfg,
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

# Terrain specialists. One policy per terrain class, to be frozen and blended by
# the switching module. Stock PPO (no terrain encoder, no privileged state): a
# specialist only ever sees one terrain, so it has nothing to disambiguate.
# All four share an identical observation space and network shape so the gating
# network can blend them -- see `_unitree_go2_specialist_env_cfg`.
for _spec_name, _spec_cfg_fn in (
  ("Flat", unitree_go2_spec_flat_env_cfg),
  ("Rough", unitree_go2_spec_rough_env_cfg),
  ("Stairs", unitree_go2_spec_stairs_env_cfg),
  ("Gaps", unitree_go2_spec_gaps_env_cfg),
):
  register_mjlab_task(
    task_id=f"Unitree-Go2-Spec-{_spec_name}",
    env_cfg=_spec_cfg_fn(),
    play_env_cfg=_spec_cfg_fn(play=True),
    rl_cfg=unitree_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
