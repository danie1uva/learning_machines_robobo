import torch
import numpy as np
import joblib

from robobo_interface import (
    IRobobo,
)

from .run_DQN import (
    check_collision,
    process_irs,
    determine_action,
)

def rob_move(policy, rob: IRobobo):
    '''
    policy: torch model 
    rob: robot
    '''

    while True:
        irs = rob.read_irs()
        print(irs)
        coll = check_collision(irs)
        if coll:
            break
        logits = policy(torch.tensor(irs).float())
        action_idx = logits.argmax().item()
        action_cmd = determine_action(action_idx)
        rob.move(*action_cmd)
        
def dir_given_sensor(sensor_idx):
    if sensor_idx == 0:
        return [0, 50, 100]
    elif sensor_idx == 1:
        return [25, 50, 100]
    elif sensor_idx == 2:
        return [50, 50, 100]
    elif sensor_idx == 3:
        return [50, 25, 100]
    else:
        return [50, 0, 100]
    
def go_to_space(rob: IRobobo):
    '''
    policy: torch model 
    rob: robot
    '''

    while True:
        irs = rob.read_irs()
        irs = process_irs(irs)
        coll = check_collision(irs)
        irs[1] = irs[1] * .75
        irs[3] = irs[3] * .75
        irs[2] = irs[2] * .5 
        if coll:
            break
        small_dir = irs.index(min(irs))
        action_cmd = dir_given_sensor(small_dir)
        rob.move(*action_cmd)
        
        
        

