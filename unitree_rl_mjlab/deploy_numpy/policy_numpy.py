# Created by Claude (policyswitching repo, deploy_numpy/policy_numpy.py).
# Purpose: evaluate an exported locomotion policy (MLP + normaliser) with numpy only.
"""Run a trained Go2 locomotion policy with numpy only (no torch, no onnxruntime).

For the robot's Jetson, which has Python 3.8 and numpy 1.17 and neither torch
nor onnxruntime. Deliberately written for that interpreter: no type-union
syntax, no f-string "=" specifier, nothing newer than Python 3.8.

The policy is an observation normaliser followed by an MLP with ELU
activations. Weights come from an .npz written on the laptop by
scripts/export_policy_numpy.py, which also checks this file's output against
the PyTorch policy on real simulator observations.

Observation (float32), in this order:
  base angular velocity, body frame (3)
  projected gravity, body frame (3)
  velocity command vx, vy, wz (3)
  gait phase sin, cos; period 0.6 s; both zero when |command xy| < 0.1 (2)
  joint position minus default (12)
  joint velocity (12)
  previous action (12)
  height scan (187), only for policies exported with a scan: height of the base
    above the ground on a 17 x 11 grid, 0.1 m spacing, 1.6 m x 1.0 m, centred
    on the base and aligned with its heading, divided by 5

Action -> joint position target = default_joint_pos + action_scale * action,
with the action clipped to +-clip_actions first. Joint order is the
simulator's (FL, FR, RL, RR, each hip, thigh, calf); the robot's motor order
is FR, FL, RR, RL, so map with JOINT_IDS_MAP before sending.

This file only computes actions. It sends nothing to the robot.
"""

import numpy as np

# Simulator joint index for each robot motor index (from the Go2 deploy config).
JOINT_IDS_MAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
DEFAULT_JOINT_POS = np.array([-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8], dtype=np.float32)
ACTION_SCALE = 0.25
STIFFNESS = [20, 20, 40] * 4
DAMPING = [1, 1, 2] * 4
CONTROL_DT = 0.02
GAIT_PERIOD = 0.6


def elu(x):
    return np.where(x > 0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


class NumpyPolicy(object):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.mean = data["obs_mean"].astype(np.float32)
        self.std = data["obs_std"].astype(np.float32)
        self.eps = float(data["obs_eps"])
        self.clip_actions = float(data["clip_actions"])
        n_layers = int(data["n_layers"])
        self.weights = [data["w%d" % i].astype(np.float32) for i in range(n_layers)]
        self.biases = [data["b%d" % i].astype(np.float32) for i in range(n_layers)]
        self.obs_dim = int(self.mean.shape[-1])
        self.action_dim = int(self.biases[-1].shape[0])

    def act(self, obs):
        """obs: (obs_dim,) or (N, obs_dim). Returns clipped actions, same leading shape."""
        x = np.asarray(obs, dtype=np.float32)
        if x.shape[-1] != self.obs_dim:
            raise ValueError("observation has %d entries, policy expects %d" % (x.shape[-1], self.obs_dim))
        x = (x - self.mean) / (self.std + self.eps)
        last = len(self.weights) - 1
        for i in range(len(self.weights)):
            x = x.dot(self.weights[i].T) + self.biases[i]
            if i < last:
                x = elu(x)
        return np.clip(x, -self.clip_actions, self.clip_actions)

    def joint_targets(self, action):
        """Joint position targets in SIMULATOR joint order for one action vector."""
        return DEFAULT_JOINT_POS + ACTION_SCALE * np.asarray(action, dtype=np.float32)


def gait_phase(t, command_xy_norm):
    """sin, cos of the gait clock at time t (s since the policy started); zeros when nearly standing."""
    if command_xy_norm < 0.1:
        return np.zeros(2, dtype=np.float32)
    phase = (t % GAIT_PERIOD) / GAIT_PERIOD
    return np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32)
