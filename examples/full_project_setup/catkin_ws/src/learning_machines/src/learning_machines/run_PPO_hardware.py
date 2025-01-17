import torch
import numpy as np
import joblib

from robobo_interface import (
    IRobobo,
)

def check_collision(state):
    # Adjust thresholds for normalized sensor values
    coll_FrontLL = state[0] > 600
    coll_FrontL = state[1] > 600
    coll_FrontC = state[2] > 600
    coll_FrontR = state[3] > 600
    coll_FrontRR = state[4] > 600
    coll_BackL = state[5] > 600
    coll_BackC = state[6] > 600
    coll_BackR = state[7] > 600

    return any([coll_FrontLL, coll_FrontL, coll_FrontC, coll_FrontR, coll_FrontRR, coll_BackL, coll_BackC, coll_BackR])

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
        action = policy(torch.tensor(irs)).detach().numpy()
        print(action)
        rob.move(action[0], action[1], 250)
        

        

