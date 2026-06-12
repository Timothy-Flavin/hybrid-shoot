"""
Throughput / bottleneck profiler for the HybridShoot environment.

Measures steps-per-second at four layers and uses cProfile to attribute time
to individual functions so you can see where the cost lives:

  1. raw C++       -- HybridJamShoot.step() called directly (no Python wrapper)
  2. gym xy        -- HybridShootEnv.step() with a 2D [x, y] continuous action
  3. gym hilbert   -- HybridShootEnv.step() with joint_xy_action (C++ Hilbert)
  4. vec (OpenMP)  -- VecHybridJamShoot stepping num_envs in parallel in C++

Usage:
    python profile_env.py                  # defaults
    python profile_env.py --steps 500000
    python profile_env.py --vec-envs 1024 --vec-steps 5000
    python profile_env.py --no-cprofile    # wall-clock timing only
"""

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

import hybrid_shoot
from hybrid_shoot import HybridShootEnv, VecHybridJamShoot


def fmt_rate(n_steps, seconds):
    sps = n_steps / seconds if seconds > 0 else float("inf")
    return f"{sps:>14,.0f} steps/s  ({seconds*1e6/n_steps:7.3f} us/step)"


# ---------------------------------------------------------------------------
# Single-step workloads. Actions are precomputed so the loop times the env,
# not numpy RNG.
# ---------------------------------------------------------------------------
def make_raw_cpp(n, n_enemies=3):
    env = hybrid_shoot.HybridJamShoot(
        independent_mode=True, n_enemies=n_enemies, map_size=1.0, hit_radius=0.05
    )
    env.reset()
    rng = np.random.default_rng(0)
    jams = rng.integers(0, n_enemies, size=n).tolist()
    xys = rng.random((n, 2))

    def run():
        for i in range(n):
            res = env.step(jams[i], xys[i])
            if res.done:
                env.reset()

    return run


def make_gym_xy(n, n_enemies=3):
    env = HybridShootEnv(
        independent_mode=True, n_enemies=n_enemies, map_size=1.0,
        hit_radius=0.05, joint_xy_action=False,
    )
    env.reset()
    rng = np.random.default_rng(0)
    jams = rng.integers(0, n_enemies, size=n)
    xys = rng.random((n, 2))

    def run():
        for i in range(n):
            _, _, term, trunc, _ = env.step((int(jams[i]), xys[i]))
            if term or trunc:
                env.reset()

    return run


def make_gym_hilbert(n, n_enemies=3):
    env = HybridShootEnv(
        independent_mode=True, n_enemies=n_enemies, map_size=1.0,
        hit_radius=0.05, joint_xy_action=True, xy_hilbert_width=16,
    )
    env.reset()
    rng = np.random.default_rng(0)
    jams = rng.integers(0, n_enemies, size=n)
    scalars = rng.random((n, 1))

    def run():
        for i in range(n):
            _, _, term, trunc, _ = env.step((int(jams[i]), scalars[i]))
            if term or trunc:
                env.reset()

    return run


SINGLE_WORKLOADS = {
    "raw C++ step": make_raw_cpp,
    "gym wrapper (xy)": make_gym_xy,
    "gym wrapper (hilbert)": make_gym_hilbert,
}


def time_workload(run):
    t0 = time.perf_counter()
    run()
    return time.perf_counter() - t0


def cprofile_workload(label, run):
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime").print_stats(10)
    print(f"\n--- cProfile: {label} (top functions by tottime) ---")
    print(s.getvalue())


# ---------------------------------------------------------------------------
def bench_vec(num_envs, n_steps, n_enemies, threads):
    env = VecHybridJamShoot(
        num_envs=num_envs, independent_mode=True, n_enemies=n_enemies,
        map_size=1.0, hit_radius=0.05, num_threads=threads,
    )
    env.reset()
    disc = np.random.default_rng(0).integers(0, n_enemies, size=num_envs).astype(np.int32)
    cont = np.random.default_rng(1).random((num_envs, 2))
    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step(disc, cont)
    dt = time.perf_counter() - t0
    return num_envs * n_steps / dt, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--enemies", type=int, default=3)
    ap.add_argument("--vec-envs", type=int, default=2048)
    ap.add_argument("--vec-steps", type=int, default=3000)
    ap.add_argument("--no-cprofile", action="store_true")
    args = ap.parse_args()

    n = args.steps
    print(f"HybridShoot profiler: {n:,} single-env steps/config, "
          f"n_enemies={args.enemies}\n")

    # --- Single-env wall-clock throughput ---
    print("=" * 72)
    print("SINGLE-ENV WALL-CLOCK THROUGHPUT")
    print("=" * 72)
    results = {}
    for label, factory in SINGLE_WORKLOADS.items():
        secs = time_workload(factory(n, args.enemies))
        results[label] = secs
        print(f"  {label:24s}: {fmt_rate(n, secs)}")

    raw = results["raw C++ step"]
    xy = results["gym wrapper (xy)"]
    hil = results["gym wrapper (hilbert)"]
    print("\n  Overhead attribution (per step):")
    print(f"    C++ core            : {raw*1e6/n:8.3f} us")
    print(f"    + gym/numpy wrapper : {(xy-raw)*1e6/n:8.3f} us")
    print(f"    + Hilbert (now C++) : {(hil-xy)*1e6/n:8.3f} us")

    # --- Vectorized (OpenMP) throughput + thread scaling ---
    print("\n" + "=" * 72)
    print(f"VECTORIZED (OpenMP) THROUGHPUT  "
          f"[{args.vec_envs} envs x {args.vec_steps} steps]")
    print("=" * 72)
    best_vec = 0.0
    for th in (1, 2, 4, 0):
        sps, dt = bench_vec(args.vec_envs, args.vec_steps, args.enemies, th)
        best_vec = max(best_vec, sps)
        tag = "auto" if th == 0 else str(th)
        print(f"  threads={tag:>4s}: {sps:>16,.0f} env-steps/s  ({dt:.2f}s)")

    single_best = n / min(results.values())
    print(f"\n  Best single-env throughput : {single_best:>16,.0f} steps/s")
    print(f"  Best vectorized throughput : {best_vec:>16,.0f} env-steps/s")
    print(f"  Speedup (vec / best single): {best_vec/single_best:>16.1f}x")

    # --- cProfile attribution for the single-env paths ---
    if not args.no_cprofile:
        print("\n" + "=" * 72)
        print("FUNCTION-LEVEL ATTRIBUTION (cProfile, single-env paths)")
        print("=" * 72)
        cn = max(20_000, n // 5)
        for label, factory in SINGLE_WORKLOADS.items():
            cprofile_workload(label, factory(cn, args.enemies))


if __name__ == "__main__":
    main()
