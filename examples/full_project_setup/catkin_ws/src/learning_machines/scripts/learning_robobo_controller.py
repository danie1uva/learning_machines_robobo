#!/usr/bin/env python3
import sys
import torch 

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import (run_all_actions,
                               train_dqn_with_coppeliasim, 
                               rob_move, 
                               go_to_space, 
                               QNetwork, 
                               run_dqn_with_coppeliasim, 
                               run_all_actions,
                               RobotNavigator,
                               train_dqn_forage,
                               run_dqn_forage
)


if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
        run_all_actions(rob)
        
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo() 
        run_all_actions(rob)

    elif sys.argv[1] == "--simulation_inf":
        weights_path = "/root/catkin_ws/policy.pth" 
        rob = SimulationRobobo() 
        run_dqn_with_coppeliasim(rob, weights_path) 

    elif sys.argv[1] == "--debug":
        model = QNetwork()
        model.load_state_dict(torch.load("model.pth"))

    elif sys.argv[1] == "--hardcode":
        rob = HardwareRobobo(camera=True)
        go_to_space(rob)

    elif sys.argv[1] == "--test_controls":
        rob = HardwareRobobo(camera=True)
        run_all_actions(rob)

    elif sys.argv[1] == "--forage" and sys.argv[2] == "--simulation":
        rob = SimulationRobobo()
        RobotNavigator(rob).forage() 

    elif sys.argv[1] == "--forage" and sys.argv[2] == "--hardware":
        rob = HardwareRobobo(camera=True)
        RobotNavigator(rob, debug=True).forage() 

    elif sys.argv[1] == "--train_forage":
        rob = SimulationRobobo()
        train_dqn_forage(rob)

    elif sys.argv[1] == "--run_forage":
        weights_path = "/root/catkin_ws/policy.pth" 
        rob = SimulationRobobo()
        run_dqn_forage(rob, weights_path)

    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")
