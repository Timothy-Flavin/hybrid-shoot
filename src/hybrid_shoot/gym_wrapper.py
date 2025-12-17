import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from . import _hybrid_shoot
import math


class HybridShootEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        independent_mode=False,
        n_enemies=3,
        map_size=1.0,
        hit_radius=0.05,
        render_mode=None,
        joint_xy_action=False,
    ):

        super().__init__()

        self.joint_xy_action = joint_xy_action
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None
        self.hit_radius = hit_radius

        # State tracking for rendering
        self.prev_state = None
        self.state = None
        self.last_action = None

        # Pass configuration to C++ to setup the difficulty/precision
        self.cpp_env = _hybrid_shoot.HybridJamShoot(
            independent_mode, n_enemies, map_size, hit_radius
        )

        self.n_enemies = self.cpp_env.get_num_enemies()
        self.map_size = map_size

        # Tuple(Target_ID, [Shot_X, Shot_Y])
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(self.n_enemies),
                spaces.Box(low=0.0, high=map_size, shape=(2,), dtype=np.float64),
            )
        )

        # Obs: [x, y, alive] * n_enemies
        low_obs = np.zeros(self.n_enemies * 3, dtype=np.float64)
        # Upper bound for x, y is map_size; for alive is 1.0
        high_obs = np.array(
            [map_size, map_size, 1.0] * self.n_enemies, dtype=np.float64
        )

        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float64
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs = self.cpp_env.reset()
        self.state = np.array(raw_obs, dtype=np.float64)
        self.prev_state = self.state.copy()
        self.last_action = None

        if self.render_mode == "human":
            self.render()

        return self.state, {}

    def _scalar_to_xy_hilbert(self, d, n):
        """
        Map a 1D scalar 'd' to (x, y) coordinates using a Hilbert Curve.

        Args:
            d (int): The 1D scalar input (0 to n*n - 1)
            n (int): The width of the square grid (must be a power of 2)
        """
        d = round(d)
        rx, ry, t = 0, 0, d
        x, y = 0, 0
        s = 1

        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)

            # Rotate/flip a quadrant quadrant
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x  # Swap x and y

            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    def step(self, action):
        discrete_act, continuous_act = action

        if self.joint_xy_action:
            # If joint action, continuous_act is expected to be a single number
            continuous_act = self._scalar_to_xy_hilbert(
                continuous_act * 31, 32
            )  # Assuming 32x32 grid for mapping
        # Store state before step for rendering
        self.prev_state = self.state.copy()
        self.last_action = action

        # Ensure conversion to list of doubles for C++
        cont_list = (
            continuous_act.tolist()
            if isinstance(continuous_act, np.ndarray)
            else list(continuous_act)
        )

        result = self.cpp_env.step(int(discrete_act), cont_list)

        self.state = np.array(result.observation, dtype=np.float64)

        if self.render_mode == "human":
            self.render()

        return (
            self.state,
            float(result.reward),
            result.done,
            False,
            {"msg": result.info},
        )

    def render(self):
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Helper to scale coordinates
        def to_screen(x, y):
            return (
                int(x / self.map_size * self.window_size),
                int(y / self.map_size * self.window_size),
            )

        # Helper to draw enemies
        def draw_enemies(surface, state):
            for i in range(self.n_enemies):
                idx = i * 3
                x, y, alive = state[idx], state[idx + 1], state[idx + 2]
                if alive > 0.5:
                    pos = to_screen(x, y)
                    # Draw filled circle (Red for enemy)
                    pygame.draw.circle(surface, (200, 0, 0), pos, 15)

        # --- Frame 1: Action Visualization (if we have a last action) ---
        if self.last_action is not None:
            # Draw previous state
            draw_enemies(canvas, self.prev_state)

            jam_idx, shoot_coords = self.last_action

            # Draw Jamming (Blue unfilled circle)
            if 0 <= jam_idx < self.n_enemies:
                # Get target position from prev_state
                idx = int(jam_idx) * 3
                tx, ty, talive = (
                    self.prev_state[idx],
                    self.prev_state[idx + 1],
                    self.prev_state[idx + 2],
                )
                # Even if dead, we might have tried to jam it?
                # Assuming we draw it if we targeted it, or maybe only if it was there.
                # Let's draw the marker at the target's last known position.
                t_pos = to_screen(tx, ty)
                pygame.draw.circle(
                    canvas, (0, 0, 255), t_pos, 20, 2
                )  # Width 2 = unfilled

            # Draw Firing (Orange unfilled circle)
            sx, sy = shoot_coords
            s_pos = to_screen(sx, sy)
            # Radius based on hit_radius scaled to screen
            r_screen = int(self.hit_radius / self.map_size * self.window_size)
            if r_screen < 2:
                r_screen = 2
            pygame.draw.circle(canvas, (255, 165, 0), s_pos, r_screen, 2)

            if self.render_mode == "human":
                self.window.blit(canvas, (0, 0))
                pygame.event.pump()
                pygame.display.flip()
                self.clock.tick(self.metadata["render_fps"])
                pygame.time.wait(200)  # Pause to show action

        # --- Frame 2: Result Visualization ---
        canvas.fill((255, 255, 255))
        draw_enemies(canvas, self.state)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
