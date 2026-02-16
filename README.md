# Learning Machines Robobo (University RL Project)

This repository contains our 4-week Learning Machines project, where we trained a small wheeled Robobo robot in simulation (CoppeliaSim + ROS) and then tested policies on real hardware.

The most relevant work is in:
`examples/full_project_setup/catkin_ws/src/learning_machines`

## What We Focused On

- Reinforcement learning for robot control under partial observability (IR sensors + camera).
- Sim-first training with hardware evaluation (sim-to-real transfer).
- Iterative environment design: better state representations, reward shaping, and reset randomization.

## Week-by-Week Approach

- Week 1: baseline robot control and environment interaction.
- Week 2: early RL baselines (custom DQN/PPO-style training loops).
- Week 3 (foraging, stronger results):
  - Built a Gym environment combining normalized IR readings with visual detections (green object bounding boxes).
  - Trained DQN with shaped rewards for progress, object collection, collision penalties, and episode efficiency.
  - Added domain variation through periodic reset perturbations.
- Week 4 (puck pushing, strongest RL focus):
  - Reformulated task as two-stage learning (reach puck, then push to goal zone).
  - Compared DQN, PPO, and SAC in Stable-Baselines3.
  - Used vision + proximity sensing in the observation and task-specific reward shaping.
  - Introduced randomization schedules (including performance-based randomization for SAC) to improve robustness.

## ML/RL Takeaway

The core lesson was that RL performance depended less on model choice alone and more on environment design: reward decomposition, staged curricula, and controlled randomization were the key levers for getting policies to learn reliably and transfer better from simulation to hardware.
