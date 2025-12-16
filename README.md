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
obs, reward, done, truncated, info = env.step(action)
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
1.  `jammer`: `Discrete(num_enemies)` - Selects which enemy to jam.
2.  `shooter`: `Box(low=0, high=map_size, shape=(2,))` - Selects the `[x, y]` coordinates.

## Game Mechanics

**Jamming**: Stops the targeted enemy from dealing damage this turn.
**Shooting**: Fires at location `(x, y)`.

-   **Standard Mode** (`independent_mode=False`): An enemy must be **jammed** to be vulnerable to shooting. Shooting an unjammed enemy does nothing (or incurs a penalty).
-   **Independent Mode** (`independent_mode=True`): Jamming prevents damage, and Shooting kills enemies regardless of whether they are jammed.
 
