import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .gym_wrapper import HybridShootEnv


class HybridShootPettingZooEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "hybrid_shoot_v1"}

    def __init__(
        self,
        independent_mode=False,
        n_enemies=3,
        map_size=1.0,
        hit_radius=0.05,
        render_mode=None,
    ):
        self.env = HybridShootEnv(
            independent_mode, n_enemies, map_size, hit_radius, render_mode
        )
        self.possible_agents = ["jammer", "shooter_x", "shooter_y"]
        self.agents = self.possible_agents[:]

        self.observation_spaces = {
            agent: self.env.observation_space for agent in self.possible_agents
        }

        self.action_spaces = {
            "jammer": spaces.Discrete(n_enemies),
            "shooter_x": spaces.Box(
                low=0.0, high=map_size, shape=(1,), dtype=np.float64
            ),
            "shooter_y": spaces.Box(
                low=0.0, high=map_size, shape=(1,), dtype=np.float64
            ),
        }
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        obs, info = self.env.reset(seed=seed, options=options)
        observations = {agent: obs for agent in self.agents}
        infos = {agent: info for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # Default actions if missing
        jam_act = actions.get("jammer", 0)
        x_act = actions.get("shooter_x", np.array([0.0]))
        y_act = actions.get("shooter_y", np.array([0.0]))

        # Handle scalar vs array inputs for Box spaces
        if hasattr(x_act, "item"):
            x_act = x_act.item()
        if hasattr(y_act, "item"):
            y_act = y_act.item()

        # Construct action for underlying gym env
        # Gym env expects: (discrete_act, [x, y])
        gym_action = (
            int(jam_act),
            np.array([float(x_act), float(y_act)], dtype=np.float64),
        )

        obs, reward, terminated, truncated, info = self.env.step(gym_action)

        observations = {agent: obs for agent in self.agents}
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: info for agent in self.agents}

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test

    env = HybridShootPettingZooEnv()
    parallel_api_test(env, num_cycles=1000)
    print("PettingZoo Parallel API test passed!")
