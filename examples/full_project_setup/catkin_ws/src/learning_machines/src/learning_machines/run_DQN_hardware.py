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
        irs = process_irs(irs)
        irs = irs[-1::-1]
        print(irs)
        coll = check_collision(irs)
        if coll:
            break
        logits = policy(torch.tensor(irs).float())

        action_idx = logits.argmax().item()
        action_cmd = determine_action(action_idx, rob)
        rob.move(*action_cmd)
        

        

