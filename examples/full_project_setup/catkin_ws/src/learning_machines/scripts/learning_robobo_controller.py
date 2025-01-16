#!/usr/bin/env python3
import sys
from learning_machines import train_model, test_model
from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions, test_irs, stop_at_obstacle

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
        stop_at_obstacle(rob, 'FrontC')  # Example: hardware-specific action
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        if len(sys.argv) > 2 and sys.argv[2] == "--train":
            train_model()  # Train PPO for obstacle avoidance
        elif len(sys.argv) > 2 and sys.argv[2] == "--test":
            test_model()  # Test trained PPO model
        else:
            raise ValueError("Specify --train or --test for simulation.")
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")
