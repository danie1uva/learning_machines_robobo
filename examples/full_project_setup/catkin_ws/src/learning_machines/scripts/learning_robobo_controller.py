#!/usr/bin/env python3
import sys
from learning_machines import run_all_actions, test_irs, stop_at_obstacle, run_ppo_training
from robobo_interface import SimulationRobobo, HardwareRobobo

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware or simulation.
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
        stop_at_obstacle(rob, 'FrontC')  # Example: hardware-specific action
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        if len(sys.argv) > 2 and sys.argv[2] == "--train-ppo":
            run_ppo_training(rob)  # Train PPO for obstacle avoidance
        else:
            raise ValueError("Specify --train-ppo for simulation training.")
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")
