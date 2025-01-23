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
    """Deep Q-Network with 7 output actions matching the environment"""
    def __init__(self, input_shape, n_actions=20):  # Corrected to 7 actions
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
            nn.Linear(cnn_out.shape[1] + 5, 512),  # 5 comes from IRS sensors
            nn.ReLU(),
            nn.Linear(512, n_actions)  # Now outputs 7 values
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
    """Training loop with action dimension fix"""
    wandb.init(
        project="robobo-foraging",
        config={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "gamma": 0.99,
            "epsilon_decay": 0.9995,  # Slower decay
            "min_epsilon": 0.1,
            "eps_decay_steps": 40000  # More gradual exploration
        }
    )
    
    env = ForagingEnv(rob)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with correct number of actions
    model = DQN(input_shape=(3, 64, 64)).to(device)
    target_model = DQN(input_shape=(3, 64, 64)).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    replay_buffer = ReplayBuffer(50000)
    
    total_steps = 0
    epsilon = 1.0
    episode = 0
    
    while True:
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            total_steps += 1
            
            # Epsilon decay
            epsilon = max(
                wandb.config.min_epsilon,
                1.0 - (total_steps / wandb.config.eps_decay_steps)
            )
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()  # Returns 0-6
            else:
                with torch.no_grad():
                    image_tensor = torch.FloatTensor(state['image']).permute(2, 0, 1).unsqueeze(0).to(device)
                    irs_tensor = torch.FloatTensor(state['irs']).unsqueeze(0).to(device)
                    q_values = model(image_tensor, irs_tensor)
                    action = q_values.argmax().item()
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store experience
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Train model
            if len(replay_buffer) >= wandb.config.batch_size:
                batch = replay_buffer.sample(wandb.config.batch_size)
                
                # Convert batch to tensors
                state_images = torch.stack([torch.FloatTensor(s['image']).permute(2, 0, 1) for s, _, _, _, _ in batch]).to(device)
                state_irs = torch.stack([torch.FloatTensor(s['irs']) for s, _, _, _, _ in batch]).to(device)
                actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(device)
                rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(device)
                next_state_images = torch.stack([torch.FloatTensor(ns['image']).permute(2, 0, 1) for _, _, _, ns, _ in batch]).to(device)
                next_state_irs = torch.stack([torch.FloatTensor(ns['irs']) for _, _, _, ns, _ in batch]).to(device)
                dones = torch.FloatTensor([d for _, _, _, _, d in batch]).to(device)
                
                # Calculate target Q-values
                with torch.no_grad():
                    next_q = target_model(next_state_images, next_state_irs).max(1)[0]
                    target_q = rewards + (1 - dones) * wandb.config.gamma * next_q
                
                # Calculate current Q-values
                current_q = model(state_images, state_irs).gather(1, actions.unsqueeze(1))
                
                # Compute loss
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                
                # Optimize model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                wandb.log({"training_loss": loss.item()})
            
            # Update target network
            if total_steps % 1000 == 0:
                target_model.load_state_dict(model.state_dict())
                print(f"🔁 Target network updated at step {total_steps}")
            
            state = next_state

        # Log episode metrics
        duration = time.time() - env.episode_start_time
        wandb.log({
            "episode": episode,
            "total_steps": total_steps,
            "epsilon": epsilon,
            "episode_reward": episode_reward,
            "packages_collected": env.total_packages_collected,
            "episode_duration": duration,
            "average_reward_per_step": episode_reward/(total_steps if total_steps > 0 else 1)
        })
        
        print(f"\nEpisode {episode} Summary:")
        print(f"Duration: {duration:.1f}s")
        print(f"Total steps: {total_steps}")
        print(f"Packages collected: {env.total_packages_collected}")
        print(f"Epsilon: {epsilon:.3f}")
        print(f"Average reward: {episode_reward/(total_steps if total_steps > 0 else 1):.2f}\n")
        
        episode += 1