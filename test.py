import hybrid_shoot
import numpy as np

def run_test_scenario(scenario_name, jam_target_idx, kill_target_idx):
    print(f"\n--- Scenario: {scenario_name} ---")
    
    # 2 Enemies, Independent Mode
    # Damage per turn is 0.5 (from C++ config)
    env = hybrid_shoot.HybridJamShoot(
        independent_mode=True, 
        n_enemies=2, 
        map_size=1.0, 
        hit_radius=1.0 # Huge radius guarantees a hit
    )
    
    # Reset and get positions
    obs = env.reset()
    # obs structure: [x0, y0, alive0, x1, y1, alive1]
    
    e0_x, e0_y = obs[0], obs[1]
    e1_x, e1_y = obs[3], obs[4]
    
    targets = [(e0_x, e0_y), (e1_x, e1_y)]
    kill_coords = targets[kill_target_idx]
    
    print(f"Action: Jam Enemy {jam_target_idx}, Shoot Enemy {kill_target_idx}")
    
    # Step the environment
    result = env.step(jam_target_idx, [kill_coords[0], kill_coords[1]])
    
    # Expected Reward Calculation:
    # Base Penalty: -0.1
    # Kill Reward: +10.0
    # Damage Penalty: -0.5 per unjammed alive enemy
    
    print(f"Total Reward Received: {result.reward}")
    
    # Analyze Damage
    # If we killed E0, but didn't jam him, did we take damage?
    damage_taken = False
    
    # Inverse calculate damage from reward
    # reward = -0.1 + 10.0 - (damage)
    damage = (-0.1 + 10.0) - result.reward
    
    if damage > 0.01:
        print(f"RESULT: Took {damage:.1f} damage this turn.")
    else:
        print("RESULT: Took ZERO damage this turn.")

# --- TEST 1: The "Split Focus" Flaw ---
# We Kill Enemy 0, but we Jam Enemy 1.
# Logic check: Enemy 0 is alive at start of turn. He is NOT jammed.
# He should deal 0.5 damage BEFORE he dies.
run_test_scenario("Split Focus (Jam 1, Kill 0)", jam_target_idx=1, kill_target_idx=0)

# --- TEST 2: The "Unified Focus" Strategy ---
# We Kill Enemy 0 AND Jam Enemy 0.
# Logic check: Enemy 0 is alive at start. He IS jammed. 
# He deals 0 damage. Then he dies.
run_test_scenario("Unified Focus (Jam 0, Kill 0)", jam_target_idx=0, kill_target_idx=0)