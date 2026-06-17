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
        max_len=5,
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
            max_len,
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

        Returns ``(obs, rewards, terminations, truncations, infos)`` following
        the Gymnasium VectorEnv convention:

          * ``obs`` is the next observation; for an env that just finished it is
            the first observation of the freshly auto-reset episode.
          * ``terminations`` marks true terminal states (bootstrap target 0).
          * ``truncations`` marks step-limit endings (still bootstrap from the
            final observation).
          * For every env that ended, ``infos["final_observation"][i]`` holds the
            real post-step observation that actually occurred (the ``s'`` to
            bootstrap from), with ``infos["_final_observation"]`` the boolean
            mask of which entries are populated. This mirrors
            ``gymnasium.vector.SyncVectorEnv`` so RL libraries can consume it.
        """
        discrete, continuous = actions
        discrete = np.ascontiguousarray(discrete, dtype=np.int32)
        continuous = np.ascontiguousarray(continuous, dtype=np.float64)
        if continuous.ndim == 1:
            continuous = continuous.reshape(self.num_envs, self.cont_dim)

        obs, rewards, terminations, truncations, final_obs = self._vec.step(
            discrete, continuous
        )

        infos = {}
        done = terminations | truncations
        if done.any():
            final_observation = np.full(self.num_envs, None, dtype=object)
            idx = np.flatnonzero(done)
            for i in idx:
                final_observation[i] = final_obs[i]
            infos["final_observation"] = final_observation
            infos["_final_observation"] = done

        return obs, rewards, terminations, truncations, infos

    def close(self):
        pass
