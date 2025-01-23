# week_3/dqn_foraging.py
import wandb
import torch
import numpy as np
import time
from torch import nn
from collections import deque
import random
from robobo_interface import IRobobo
from .foraging_env import ForagingEnv

class DQN(nn.Module):
    """Deep Q-Network with 20 output actions"""
    def __init__(self, input_shape, n_actions=20):
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
        
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, *input_shape))
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_out.shape[1] + 5, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, image, irs):
        return self.fc(torch.cat([self.cnn(image), irs], dim=1))

class ReplayBuffer(deque):
    def __init__(self, capacity):
        super().__init__(maxlen=capacity)
    
    def sample(self, batch_size):
        return random.sample(self, batch_size)

def train_dqn_foraging(rob: IRobobo):
    """Training loop with episode management"""
    wandb.init(project="robobo-foraging", config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "gamma": 0.99,
        "epsilon_decay": 0.999,
        "min_epsilon": 0.1,
        "eps_decay_steps": 20000
    })
    
    env = ForagingEnv(rob)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DQN((3, 64, 64)).to(device)
    target = DQN((3, 64, 64)).to(device)
    target.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    buffer = ReplayBuffer(50000)
    
    episode = total_steps = 0
    epsilon = 1.0
    
    while True:
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            total_steps += 1
            epsilon = max(wandb.config.min_epsilon, 1.0 - total_steps/wandb.config.eps_decay_steps)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    img_tensor = torch.FloatTensor(state['image']).permute(2,0,1).unsqueeze(0).to(device)
                    irs_tensor = torch.FloatTensor(state['irs']).unsqueeze(0).to(device)
                    action = model(img_tensor, irs_tensor).argmax().item()
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            buffer.append((state, action, reward, next_state, done))
            
            # Training step
            if len(buffer) >= wandb.config.batch_size:
                batch = buffer.sample(wandb.config.batch_size)
                state_batch = {
                    'image': torch.stack([torch.FloatTensor(s['image']).permute(2,0,1) for s,_,_,_,_ in batch]).to(device),
                    'irs': torch.stack([torch.FloatTensor(s['irs']) for s,_,_,_,_ in batch]).to(device)
                }
                next_state_batch = {
                    'image': torch.stack([torch.FloatTensor(ns['image']).permute(2,0,1) for _,_,_,ns,_ in batch]).to(device),
                    'irs': torch.stack([torch.FloatTensor(ns['irs']) for _,_,_,ns,_ in batch]).to(device)
                }
                
                with torch.no_grad():
                    target_q = (
                        torch.FloatTensor([r for _,_,r,_,_ in batch]).to(device)
                        + (1 - torch.FloatTensor([d for _,_,_,_,d in batch]).to(device))
                        * wandb.config.gamma
                        * target(next_state_batch['image'], next_state_batch['irs']).max(1)[0]
                    )
                
                current_q = model(state_batch['image'], state_batch['irs']).gather(1, 
                    torch.LongTensor([a for _,a,_,_,_ in batch]).unsqueeze(1).to(device))
                
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                wandb.log({"loss": loss.item()})
            
            if total_steps % 1000 == 0:
                target.load_state_dict(model.state_dict())
            
            state = next_state

        # Episode logging
        wandb.log({
            "episode": episode,
            "reward": episode_reward,
            "collected": info["collected"],
            "epsilon": epsilon,
            "duration": info["duration"]
        })
        episode += 1