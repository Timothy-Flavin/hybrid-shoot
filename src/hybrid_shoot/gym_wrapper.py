import gymnasium as gym
import numpy as np
from gymnasium import spaces
from . import _hybrid_shoot

class HybridShootEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, 
                 independent_mode=False, 
                 n_enemies=3, 
                 map_size=1.0, 
                 hit_radius=0.05):
        
        super().__init__()
        
        # Pass configuration to C++ to setup the difficulty/precision
        self.cpp_env = _hybrid_shoot.HybridJamShoot(
            independent_mode, 
            n_enemies, 
            map_size, 
            hit_radius
        )
        
        self.n_enemies = self.cpp_env.get_num_enemies()
        self.map_size = map_size
        
        # Tuple(Target_ID, [Shot_X, Shot_Y])
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.n_enemies),
            spaces.Box(low=0.0, high=map_size, shape=(2,), dtype=np.float64)
        ))
        
        # Obs: [x, y, alive] * n_enemies
        low_obs = np.zeros(self.n_enemies * 3, dtype=np.float64)
        # Upper bound for x, y is map_size; for alive is 1.0
        high_obs = np.array([map_size, map_size, 1.0] * self.n_enemies, dtype=np.float64)
        
        self.observation_space = spaces.Box(
            low=low_obs, 
            high=high_obs, 
            dtype=np.float64
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs = self.cpp_env.reset()
        return np.array(raw_obs, dtype=np.float64), {}

    def step(self, action):
        discrete_act, continuous_act = action
        
        # Ensure conversion to list of doubles for C++
        cont_list = continuous_act.tolist() if isinstance(continuous_act, np.ndarray) else list(continuous_act)
        
        result = self.cpp_env.step(int(discrete_act), cont_list)
        
        return (np.array(result.observation, dtype=np.float64), 
                float(result.reward), 
                result.done, 
                False, 
                {"msg": result.info})