# week_3/foraging.py
from .dqn_foraging import DQN, train_dqn_foraging
import torch
from .foraging_env import ForagingEnv
from robobo_interface import (
    IRobobo,
    SimulationRobobo,
    HardwareRobobo,
)

def forage(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        train_dqn_foraging(rob)  # Start training
    else:
        # Load trained model for hardware
        model = load_trained_model()
        run_trained_model(rob, model)

def load_trained_model():
    # Load pretrained weights
    model = DQN(input_shape=(3, 64, 64), n_actions=5)
    model.load_state_dict(torch.load("foraging_dqn.pth"))
    return model

def run_trained_model(rob, model):
    env = ForagingEnv(rob)
    state = env.reset()
    
    while True:
        with torch.no_grad():
            image_tensor = torch.FloatTensor(state['image']).permute(2, 0, 1).unsqueeze(0)
            irs_tensor = torch.FloatTensor(state['irs']).unsqueeze(0)
            q_values = model(image_tensor, irs_tensor)
            action = q_values.argmax().item()
        
        state, _, done, _ = env.step(action)
        if done:
            state = env.reset()