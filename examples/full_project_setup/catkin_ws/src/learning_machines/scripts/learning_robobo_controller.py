#!/usr/bin/env python3
import sys
import torch 

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions, test_irs, stop_at_obstacle, run_qlearning_classification, run_ppo, rob_move, QNetwork


if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
        model = QNetwork()
        model.load_state_dict(torch.load("model.pth"))
        rob_move(model, rob)
        
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo(identifier = 1) 
        run_qlearning_classification(rob) 
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")
