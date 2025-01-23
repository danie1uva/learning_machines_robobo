# week_3/dqn_foraging.py
import wandb
import torch
import numpy as np
import time
from torch import nn
from torch.utils.data import DataLoader
from collections import deque
import random
from robobo_interface import IRobobo
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
    # Initialize W&B with adjusted parameters
    wandb.init(
        project="robobo-foraging",
        config={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "gamma": 0.99,
            "epsilon_decay": 0.998,  # Slower decay
            "min_epsilon": 0.01,
            "eps_decay_steps": 20000  # Decay over steps instead of episodes
        }
    )
    
    # Initialize environment and models
    env = ForagingEnv(rob)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DQN(input_shape=(3, 64, 64), n_actions=5).to(device)
    target_model = DQN(input_shape=(3, 64, 64), n_actions=5).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    replay_buffer = ReplayBuffer(50000)
    
    # Training parameters
    total_steps = 0
    epsilon = 1.0
    episode = 0
    
    # Training loop
    while True:
        state = env.reset()
        episode_reward = 0
        done = False
        steps_in_episode = 0
        
        while not done:
            steps_in_episode += 1
            total_steps += 1
            
            # Adjusted epsilon decay
            epsilon = max(wandb.config.min_epsilon, 
                        1.0 - (total_steps / wandb.config.eps_decay_steps))
            
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
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Train model (keep existing training code)
            # ... [existing training code] ...
            
            # Update target network periodically
            if total_steps % 1000 == 0:
                target_model.load_state_dict(model.state_dict())
                print(f"Updated target network at step {total_steps}")
            
            state = next_state

        # Episode summary
        duration = time.time() - env.episode_start_time
        print(f"\nEpisode {episode} completed in {duration:.1f}s")
        print(f"Total steps: {total_steps}")
        print(f"Packages collected: {env.total_packages}")
        print(f"Epsilon: {epsilon:.3f}")
        print(f"Average reward per step: {episode_reward/steps_in_episode:.2f}\n")
        
        # Log metrics
        wandb.log({
            "episode": episode,
            "total_steps": total_steps,
            "epsilon": epsilon,
            "episode_reward": episode_reward,
            "packages_collected": env.total_packages,
            "episode_duration": duration,
            "average_reward_per_step": episode_reward/steps_in_episode
        })
        
        episode += 1