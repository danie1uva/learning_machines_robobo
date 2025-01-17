#!/usr/bin/env python3
import sys
import torch
import os

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions, test_irs, stop_at_obstacle, run_qlearning_classification, run_ppo, rob_move, go_to_space


if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware or simulation.
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)

        # Define the path to the model weights
        weights_path = "/app/model.pth"  # Use the absolute path inside the Docker container

        # Check if the weights file exists
        if os.path.exists(weights_path):
            model = QNetwork()
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            print("Model weights loaded successfully.")
            rob_move(model, rob)
        else:
            raise FileNotFoundError(f"Model weights file not found at {weights_path}")
        
    elif sys.argv[1] == "--debug":
        # Define the path to the model weights
        weights_path = "/app/model.pth"  # Same path as in the hardware section
        print(f"Checking for model weights at: {weights_path}")
        
        if os.path.exists(weights_path):
            model = QNetwork()
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            print("Model weights loaded successfully for debugging.")
        else:
            raise FileNotFoundError(f"Model weights file not found at {weights_path}")
        
    elif sys.argv[1] == "--simulation":
        print(f"Arguments received: {sys.argv}")
        rob = SimulationRobobo(identifier=1)
        run_qlearning_classification(rob)

    elif sys.argv[1] == "--hardcode":
        rob = HardwareRobobo(camera=True)
        go_to_space(rob)
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")
