"""Vectorized HybridShoot environment.

Thin Python surface over the C++ ``VecHybridJamShoot`` class, which holds
``num_envs`` independent simulations and steps them in parallel with OpenMP.
All of the multiprocessing stays inside C++ -- there are no Python
subprocesses, pipes, or pickling, unlike Gym's ``AsyncVectorEnv``.

The API mirrors ``gymnasium.vector.VectorEnv`` / envpool:

    env = HybridShootVecEnv(num_envs=64)
    obs, infos = env.reset()
    # actions: (discrete[num_envs], continuous[num_envs, cont_dim])
    obs, rewards, terminations, truncations, infos = env.step(actions)

Auto-reset is handled inside C++ in the envpool / SB3 style: when an env
finishes, ``obs`` already contains the first observation of the next episode
(not the terminal one).
"""

import numpy as np
from gymnasium import spaces

from . import _hybrid_shoot


class HybridShootVecEnv:
    def __init__(
        self,
        num_envs=8,
        independent_mode=False,
        n_enemies=3,
        map_size=1.0,
        hit_radius=0.05,
        joint_xy_action=False,
        xy_hilbert_width=16,
        num_threads=0,
    ):
        self.num_envs = num_envs
        self.joint_xy_action = joint_xy_action

        self._vec = _hybrid_shoot.VecHybridJamShoot(
            num_envs,
            independent_mode,
            n_enemies,
            map_size,
            hit_radius,
            joint_xy_action,
            xy_hilbert_width,
            num_threads,
        )

        self.n_enemies = self._vec.get_num_enemies()
        self.obs_dim = self._vec.get_obs_dim()
        self.cont_dim = self._vec.get_cont_dim()
        self.map_size = map_size

        # Per-env (single) spaces.
        high_obs = np.array([map_size, map_size, 1.0] * self.n_enemies, dtype=np.float64)
        self.single_observation_space = spaces.Box(
            low=np.zeros(self.obs_dim, dtype=np.float64), high=high_obs, dtype=np.float64
        )
        shoot_high = 1.0 if joint_xy_action else map_size
        self.single_action_space = spaces.Tuple(
            (
                spaces.Discrete(self.n_enemies),
                spaces.Box(low=0.0, high=shoot_high, shape=(self.cont_dim,), dtype=np.float64),
            )
        )
        # Batched spaces.
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.tile(high_obs, (num_envs, 1)),
            shape=(num_envs, self.obs_dim),
            dtype=np.float64,
        )
        self.action_space = spaces.Tuple(
            (
                spaces.MultiDiscrete([self.n_enemies] * num_envs),
                spaces.Box(low=0.0, high=shoot_high, shape=(num_envs, self.cont_dim), dtype=np.float64),
            )
        )

    def reset(self, seed=None, options=None):
        obs = self._vec.reset()
        infos = {}
        return obs, infos

    def step(self, actions):
        """Step all envs.

        ``actions`` is a ``(discrete, continuous)`` tuple:
          * discrete:   array-like of shape (num_envs,)
          * continuous: array-like of shape (num_envs, cont_dim)

        Returns ``(obs, rewards, terminations, truncations, infos)``. The C++
        core reports a single ``done`` flag (terminal or step-limit); it is
        surfaced as ``terminations`` with ``truncations`` left all-False.
        """
        discrete, continuous = actions
        discrete = np.ascontiguousarray(discrete, dtype=np.int32)
        continuous = np.ascontiguousarray(continuous, dtype=np.float64)
        if continuous.ndim == 1:
            continuous = continuous.reshape(self.num_envs, self.cont_dim)

        obs, rewards, dones = self._vec.step(discrete, continuous)
        truncations = np.zeros(self.num_envs, dtype=bool)
        return obs, rewards, dones, truncations, {}

    def close(self):
        pass
