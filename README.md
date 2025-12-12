# Hybrid Shoot Environment

This C++ gymnasium compliant env is designed as a sanity check
for reinforcement learning with multiple actions with hybrid
reprisentation. There are `num_enemies` enemies and two actions that need
to be taken 1. Jam and 2. Shoot.

## Action Space
1. Jam Discrete cardonality: |num_enemies|
2. Shoot Continuous dims:    2, box2D(-1,1), `[x,y]`

## Action Descriptions
Jamming an enemy agent stops it's attack from going through. The
effect of shooting within 'hit_radius' of an enemy depends on the 
gamemode. If `independent_mode=True` then shooting an enemy removes
them at the end of the round. Otherwise, an enemy needs to be actively
jammed in order to be removed by shoot. 
