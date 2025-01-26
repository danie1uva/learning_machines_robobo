# C:\Users\esrio_0v2bwuf\Desktop\Master_AI\Learning Machines\learning_machines_robobo\examples\full_project_setup\catkin_ws\src\learning_machines\src\learning_machines\week_4\push.py

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
import gym
from gym import spaces
from typing import Tuple

from robobo_interface import SimulationRobobo
from robobo_interface import Orientation, Position  # Updated import path

class PushEnv:
    """Custom environment for pushing task with continuous control"""
    
    def __init__(self, rob: SimulationRobobo):
        self.rob = rob
        self.episode_step = 0
        self.max_steps = 300
        
        # Camera parameters
        self.camera_width = 640
        self.camera_height = 480
        self.object_size_threshold = 5000  # Minimum pixel area to consider object detected
        
        # Action space: continuous [left wheel speed, right wheel speed]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: normalized [object_x, object_y, target_x, target_y, IR_front, IR_left, IR_right]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(self):
        """Reset environment and return initial observation"""
        self.rob.stop_simulation()
        self.rob.play_simulation()
        self.episode_step = 0
        
        # Reset robot position and orientation
        self.rob.set_position(Position(0, 0, 0), Orientation(0, 0, 0))
        
        return self._get_observation()

    def _process_image(self, frame):
        """Detect red object and green target areas"""
        # Detect red object
        hsv_red = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv_red, lower_red, upper_red)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect green target
        hsv_green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv_green, lower_green, upper_green)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours_red, contours_green

    def _get_observation(self):
        """Process sensors and camera to create observation vector"""
        # Get camera frame
        frame = self.rob.read_image_front()
        contours_red, contours_green = self._process_image(frame)

        # Process red object
        obj_x, obj_y, obj_present = 0.5, 0.5, 0.0
        if len(contours_red) > 0:
            largest_red = max(contours_red, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > self.object_size_threshold:
                M = cv2.moments(largest_red)
                obj_x = M["m10"] / (M["m00"] + 1e-5) / self.camera_width
                obj_y = M["m01"] / (M["m00"] + 1e-5) / self.camera_height
                obj_present = 1.0

        # Process green target
        target_x, target_y = 0.5, 0.5
        if len(contours_green) > 0:
            largest_green = max(contours_green, key=cv2.contourArea)
            M = cv2.moments(largest_green)
            target_x = M["m10"] / (M["m00"] + 1e-5) / self.camera_width
            target_y = M["m01"] / (M["m00"] + 1e-5) / self.camera_height

        # IR sensors [Front, Left, Right]
        irs = self.rob.read_irs()
        ir_front = min(irs[3], irs[4])  # Front sensors
        ir_left = irs[2]                # Left sensor
        ir_right = irs[5]               # Right sensor

        return np.array([
            obj_x, obj_y, 
            target_x, target_y,
            ir_front / 1000.0,
            ir_left / 1000.0,
            ir_right / 1000.0
        ], dtype=np.float32)

    def step(self, action: np.ndarray):
        """Execute one time step with continuous action"""
        self.episode_step += 1
        
        # Convert action to wheel speeds (-100 to 100)
        left_speed = np.clip(action[0] * 100, -100, 100)
        right_speed = np.clip(action[1] * 100, -100, 100)
        self.rob.move_blocking(left_speed, right_speed, 500)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs)
        
        # Check termination
        done = self.episode_step >= self.max_steps
        if obs[4] < 0.1:  # Check if close to object
            done = True
            reward += 100  # Bonus for reaching object
            
        return obs, reward, done, {}

    def _calculate_reward(self, obs):
        """Complex reward function"""
        # Distance between object and target
        obj_target_dist = np.sqrt((obs[0] - obs[2])**2 + (obs[1] - obs[3])**2)
        
        # Reward components
        progress_reward = (1 - obj_target_dist) * 0.1
        alignment_reward = (1 - abs(obs[0] - 0.5)) * 0.05  # Center object horizontally
        time_penalty = -0.01
        
        # Collision penalty
        collision_penalty = -0.1 if obs[4] < 0.2 else 0
        
        return progress_reward + alignment_reward + time_penalty + collision_penalty

class PPONetwork(nn.Module):
    """PPO Actor-Critic Network"""
    
    def __init__(self, input_size, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)  # Mean and std for each action dimension
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

class PPOAgent:
    """PPO Agent with experience replay"""
    
    def __init__(self, state_dim, action_dim):
        self.policy = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.epsilon = 0.2
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            actor_out, value = self.policy(state)
            
        mean, log_std = actor_out.chunk(2, dim=-1)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        
        return action.numpy()[0], log_prob.numpy(), value.numpy()[0]
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, log_probs, returns, advantages = zip(*random.sample(self.memory, self.batch_size))
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(log_probs))
        returns = torch.FloatTensor(np.array(returns))
        advantages = torch.FloatTensor(np.array(advantages))
        
        actor_out, critic_out = self.policy(states)
        mean, log_std = actor_out.chunk(2, dim=-1)
        std = log_std.exp()
        
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(-1)
        
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = 0.5 * (returns - critic_out.squeeze()).pow(2).mean()
        entropy_loss = -0.01 * dist.entropy().mean()
        
        total_loss = actor_loss + critic_loss + entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
    def train(self, env, episodes=1000):
        best_reward = -np.inf
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, log_prob, value = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                # Calculate advantage
                with torch.no_grad():
                    _, next_value = self.policy(torch.FloatTensor(next_state).unsqueeze(0))
                    
                advantage = reward + self.gamma * (1 - done) * next_value.item() - value.item()
                self.memory.append((state, action, log_prob, value, advantage))
                
                state = next_state
                self.update()
                
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.policy.state_dict(), "best_pusher.pth")
                
            print(f"Episode {episode}, Total Reward: {total_reward}, Best: {best_reward}")

def train_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    agent = PPOAgent(state_dim=7, action_dim=2)
    agent.train(env, episodes=1000)

def run_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    
    policy = PPONetwork(7, 2)
    policy.load_state_dict(torch.load("best_pusher.pth"))
    policy.eval()
    
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            action, _ = policy(torch.FloatTensor(state).unsqueeze(0))
            action = action[0].numpy()
            
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
    print(f"Total reward: {total_reward}")