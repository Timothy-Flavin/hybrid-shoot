# Hybrid Shoot Environment

This environment is designed as a sanity check for reinforcement learning with hybrid action spaces (discrete + continuous). It supports both **Gymnasium** (single agent with hybrid actions) and **PettingZoo** (multi-agent decomposition).

There are `num_enemies` enemies in a 2D space. The goal is to Jam and Shoot them.

## Installation

```bash
pip install .
```

## Gymnasium Usage

The Gymnasium environment presents a single agent with a Tuple action space.

```python
from hybrid_shoot import HybridShootEnv

env = HybridShootEnv()
obs, info = env.reset()
# Action: (Jam_Target_Index, [Shoot_X, Shoot_Y])
action = (0, [0.5, 0.5]) 
obs, reward, terminated, truncated, info = env.step(action)
```

### Action Space (Gymnasium)
A `spaces.Tuple` containing:
1.  **Jam**: `Discrete(num_enemies)` - Selects which enemy to jam.
2.  **Shoot**: `Box(low=0, high=map_size, shape=(2,))` - `[x, y]` coordinates to shoot at.

## PettingZoo Usage

The PettingZoo environment decomposes the task into two cooperating agents.

```python
from hybrid_shoot import HybridShootPettingZooEnv

env = HybridShootPettingZooEnv()
observations, infos = env.reset()
```

### Agents & Action Spaces (PettingZoo)
1.  `jammer`: `Discrete(num_enemies)` - Selects which enemy to jam. (jammed enemy does not cause negative reward this turn)
2.  `shooter`: `Box(low=0, high=map_size, shape=(2,))` - Selects the `[x, y]` coordinates to shoot.

## Vectorized Usage (Parallel)

For high-throughput training, `HybridShootVecEnv` runs `num_envs` independent
copies of the environment and steps them **in parallel inside C++ using OpenMP**.
All batching stays in C++ — there are no Python subprocesses, pipes, or pickling
(unlike Gymnasium's `AsyncVectorEnv`), and observations are returned as batched
NumPy arrays. The API mirrors `gymnasium.vector.VectorEnv` / `envpool`.

```python
import numpy as np
from hybrid_shoot import HybridShootVecEnv

num_envs = 1024
env = HybridShootVecEnv(num_envs=num_envs)
obs, infos = env.reset()                      # obs: (num_envs, obs_dim)

# Actions are a (discrete, continuous) tuple, batched over envs:
#   discrete:   (num_envs,)            int   - jam target per env
#   continuous: (num_envs, cont_dim)   float - shoot [x, y] per env
discrete = np.zeros(num_envs, dtype=np.int32)
continuous = np.random.rand(num_envs, 2)

obs, rewards, terminations, truncations, infos = env.step((discrete, continuous))
```

`num_threads` controls the OpenMP thread count (`0` = use all available cores).

### Auto-reset and bootstrapping

Each env auto-resets the moment it ends (envpool / SB3 style). On the step where
an env finishes, the returned `obs` for that env is already the **first
observation of the next episode**, while the genuine final observation (the `s'`
needed for value bootstrapping) is preserved in `infos`:

-   `infos["final_observation"]`: object array; for every env that just ended it
    holds that env's true post-step observation (`None` otherwise).
-   `infos["_final_observation"]`: boolean mask of which entries are populated.

`terminations` and `truncations` are reported **separately**:

-   `terminations[i] == True`  → a true terminal state (all enemies cleared);
    bootstrap target is `0`.
-   `truncations[i] == True`   → the step limit was reached with enemies alive;
    still bootstrap from `infos["final_observation"][i]`.

This matches `gymnasium.vector.SyncVectorEnv`, so RL libraries (Stable-Baselines3,
CleanRL, etc.) can consume it directly.

### Hilbert joint action (optional)

Passing `joint_xy_action=True` (available on every env class) replaces the 2D
`[x, y]` shoot action with a **single scalar in `[0, 1]`**, mapped onto the map
via a Hilbert curve of resolution `xy_hilbert_width` (default `16`). This is handy
for algorithms that prefer a single continuous output; the mapping covers the full
`[0, map_size]²` map.

## Game Mechanics

**Jamming**: Stops the targeted enemy from dealing damage this turn.
**Shooting**: Fires at location `(x, y)`.

-   **Standard Mode** (`independent_mode=False`): An enemy must be **jammed** to be vulnerable to being shot. Shooting an unjammed enemy does nothing.
-   **Independent Mode** (`independent_mode=True`): Jamming prevents damage, and Shooting kills enemies regardless of whether they are jammed.
 
