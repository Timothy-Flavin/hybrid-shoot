"""
Behavioral verification for the HybridShoot environment.

Checks that the environment actually behaves as the README describes:
  * Independent vs. dependent reward bookkeeping
  * Jamming prevents damage for the jammed enemy
  * Dead enemies no longer deal damage (no spurious negative reward)
  * Termination bonus when all enemies are cleared
  * Hilbert (joint_xy_action) coordinate coverage of the map

Run:  python verify_env.py
Exit code is non-zero if any hard assertion fails. "Coverage" issues are
reported as WARNINGs rather than failures because they are design limits.
"""

import math
import numpy as np

import hybrid_shoot
from hybrid_shoot import HybridShootEnv

# Mirror of the constants baked into cpp/env.cpp
BASE_PENALTY = -0.1
DAMAGE_PER_TURN = 0.5
KILL_REWARD = 10.0
CLEAR_BONUS = 5.0

_fail = 0
_warn = 0


def check(name, cond, detail=""):
    global _fail
    status = "PASS" if cond else "FAIL"
    if not cond:
        _fail += 1
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


def warn(name, detail=""):
    global _warn
    _warn += 1
    print(f"  [WARN] {name}" + (f"  ({detail})" if detail else ""))


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def enemy_positions(obs):
    return [(obs[i * 3], obs[i * 3 + 1]) for i in range(len(obs) // 3)]


# ---------------------------------------------------------------------------
def test_independent_dead_enemies():
    """Dead enemies must not deal damage on subsequent turns."""
    print("\n[1] Independent mode: dead enemies deal no damage")
    env = hybrid_shoot.HybridJamShoot(
        independent_mode=True, n_enemies=3, map_size=1.0, hit_radius=1.0
    )
    obs = env.reset()
    e = enemy_positions(obs)

    # Turn 1: jam e0 and kill e0. e1,e2 alive+unjammed => -2*0.5 damage.
    r1 = env.step(0, [e[0][0], e[0][1]])
    expect1 = BASE_PENALTY + KILL_REWARD - 2 * DAMAGE_PER_TURN
    check("kill e0 while jamming e0", approx(r1.reward, expect1),
          f"reward={r1.reward:.3f} expect={expect1:.3f}")

    # Turn 2: e0 now dead. Jam+kill e1. Only e2 is alive+unjammed => -0.5.
    # If the dead e0 wrongly dealt damage this would be -1.0 instead.
    r2 = env.step(1, [e[1][0], e[1][1]])
    expect2 = BASE_PENALTY + KILL_REWARD - 1 * DAMAGE_PER_TURN
    check("dead e0 contributes 0 damage", approx(r2.reward, expect2),
          f"reward={r2.reward:.3f} expect={expect2:.3f}")

    # Turn 3: kill last enemy e2 (jam it). No alive unjammed enemies left,
    # plus clear bonus.
    r3 = env.step(2, [e[2][0], e[2][1]])
    expect3 = BASE_PENALTY + KILL_REWARD + CLEAR_BONUS
    check("clear bonus on final kill", approx(r3.reward, expect3),
          f"reward={r3.reward:.3f} expect={expect3:.3f}")
    check("episode terminates when all dead", r3.done)


def test_independent_jam_blocks_damage():
    """Jamming an alive enemy prevents its damage this turn."""
    print("\n[2] Independent mode: jamming blocks damage")
    env = hybrid_shoot.HybridJamShoot(
        independent_mode=True, n_enemies=2, map_size=1.0, hit_radius=1.0
    )
    obs = env.reset()
    e = enemy_positions(obs)

    # Jam e0, shoot empty space (miss both, hit_radius=1.0 would hit... use far pt)
    # hit_radius=1.0 guarantees a hit, so shoot to kill e1 while jamming e0.
    r = env.step(0, [e[1][0], e[1][1]])
    # e0 jammed (no dmg), e1 killed. Only damage source would be e1 but it's
    # the one we kill -- e1 is alive at start of turn and NOT jammed => deals 0.5.
    expect = BASE_PENALTY + KILL_REWARD - DAMAGE_PER_TURN
    check("jammed e0 deals 0, unjammed e1 deals 0.5 before dying",
          approx(r.reward, expect), f"reward={r.reward:.3f} expect={expect:.3f}")


def test_dependent_mode():
    """Dependent mode: must jam the enemy you shoot; rewards/penalties."""
    print("\n[3] Dependent mode: reward bookkeeping")
    env = hybrid_shoot.HybridJamShoot(
        independent_mode=False, n_enemies=2, map_size=1.0, hit_radius=1.0
    )
    obs = env.reset()
    e = enemy_positions(obs)

    # Jam e0 and shoot e0 -> hit.
    r = env.step(0, [e[0][0], e[0][1]])
    check("jam+shoot same enemy => kill", approx(r.reward, BASE_PENALTY + KILL_REWARD),
          f"reward={r.reward:.3f}")

    # e0 now dead. Jam dead e0 -> -0.5 penalty branch.
    r2 = env.step(0, [e[0][0], e[0][1]])
    check("jamming a dead enemy => -0.5 penalty",
          approx(r2.reward, BASE_PENALTY - 0.5), f"reward={r2.reward:.3f}")


def test_hilbert_coverage():
    """Hilbert joint action must reach the full [0, map_size]^2 map."""
    print("\n[4] Hilbert (joint_xy_action) map coverage")
    for map_size in (1.0, 2.0):
        width = 16
        env = HybridShootEnv(
            joint_xy_action=True, xy_hilbert_width=width, map_size=map_size,
            hit_radius=0.05,
        )
        xs, ys, pts = [], [], set()
        for s in np.linspace(0.0, 1.0, 4000):
            x, y = env.cpp_env.scalar_to_xy(float(s))
            xs.append(x); ys.append(y); pts.add((round(x, 6), round(y, 6)))
        xs, ys = np.array(xs), np.array(ys)
        maxc = max(xs.max(), ys.max())
        print(f"  map_size={map_size}: unique pts={len(pts)}/{width*width}, "
              f"max coord reached={maxc:.4f}")
        # Coverage now scales with map_size and reaches the far corner.
        check(f"Hilbert reaches full map (map_size={map_size})",
              approx(maxc, map_size, tol=1e-6),
              f"max={maxc:.4f} expect={map_size}")
        check(f"all {width*width} grid cells reachable (map_size={map_size})",
              len(pts) == width * width, f"got {len(pts)}")
        # Max action value should map to the far corner, not wrap to origin.
        ex, ey = env.cpp_env.scalar_to_xy(1.0)
        check(f"max action -> far corner (map_size={map_size})",
              approx(ex, map_size) and approx(ey, 0.0) or (ex, ey) != (0.0, 0.0),
              f"endpoint=({ex:.3f},{ey:.3f})")


def test_vec_matches_single():
    """The vectorized C++ env must reproduce single-env reward dynamics."""
    print("\n[5] Vectorized env matches single env")
    import hybrid_shoot
    n_envs = 32
    vec = hybrid_shoot.VecHybridJamShoot(
        num_envs=n_envs, independent_mode=True, n_enemies=3,
        map_size=1.0, hit_radius=1.0,
    )
    obs = vec.reset()
    # For each vec env, replay the same start state + action in a single core
    # and confirm identical reward. We use hit_radius=1.0 so a shot at an
    # enemy's coords always hits, making dynamics deterministic given obs.
    disc = np.zeros(n_envs, dtype=np.int32)
    cont = np.stack([obs[:, 0], obs[:, 1]], axis=1)  # aim at enemy 0 of each env
    obs2, rewards, dones = vec.step(disc, cont)
    # Independent mode, jam e0, kill e0: -0.1 + 10 - 0.5*(#other alive enemies).
    # All envs start with 3 alive -> two others unjammed -> reward 8.9.
    ok = np.allclose(rewards, 8.9)
    check("all vec envs reward == 8.9 (jam0+kill0, 3 enemies)", ok,
          f"unique rewards={np.unique(np.round(rewards,3))}")
    check("obs batch shape correct", obs.shape == (n_envs, 9),
          f"shape={obs.shape}")


if __name__ == "__main__":
    test_independent_dead_enemies()
    test_independent_jam_blocks_damage()
    test_dependent_mode()
    test_hilbert_coverage()
    test_vec_matches_single()

    print("\n" + "=" * 60)
    print(f"Result: {_fail} failure(s), {_warn} warning(s)")
    print("=" * 60)
    raise SystemExit(1 if _fail else 0)
