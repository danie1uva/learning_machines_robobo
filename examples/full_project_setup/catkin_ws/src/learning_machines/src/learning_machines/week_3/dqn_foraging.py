# week_3/dqn_foraging.py
import wandb
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from collections import deque
import random
from robobo_interface import (
    IRobobo,
    SimulationRobobo,
    HardwareRobobo,
)
from .foraging_env import ForagingEnv

class DQN(nn.Module):
    """Deep Q-Network with CNN for image processing"""
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, *input_shape))
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_out.shape[1] + 5, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, image, irs):
        cnn_features = self.cnn(image)
        combined = torch.cat([cnn_features, irs], dim=1)
        return self.fc(combined)

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train_dqn_foraging(rob: IRobobo):
    """Training loop for foraging task"""
    # Initialize environment and models
    env = ForagingEnv(rob)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DQN(input_shape=(3, 64, 64), n_actions=5).to(device)
    target_model = DQN(input_shape=(3, 64, 64), n_actions=5).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(50000)
    
    # Training parameters
    batch_size = 32
    gamma = 0.99
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    update_interval = 1000
    epsilon = eps_start
    
    # Training loop
    episode = 0
    while True:
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    image_tensor = torch.FloatTensor(state['image']).permute(2, 0, 1).unsqueeze(0).to(device)
                    irs_tensor = torch.FloatTensor(state['irs']).unsqueeze(0).to(device)
                    q_values = model(image_tensor, irs_tensor)
                    action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Train model
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                # Training logic here...
            
            # Update target network
            if episode % update_interval == 0:
                target_model.load_state_dict(model.state_dict())
            
            state = next_state
        
        # Log metrics
        wandb.log({
            "episode_reward": episode_reward,
            "packages_collected": env.package_count,
            "epsilon": epsilon
        })
        
        # Decay epsilon
        epsilon = max(eps_end, epsilon * eps_decay)
        episode += 1