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
import time

from robobo_interface import SimulationRobobo
from robobo_interface import Orientation, Position

class PushEnv(gym.Env):
    """Custom Gym environment for pushing task with continuous control"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, rob: SimulationRobobo):
        super().__init__()
        self.rob = rob
        self.episode_step = 0
        self.max_steps = 400
        self.last_object_position = None
        self.target_position = None
        self.object_in_target = False
        self.consecutive_target_frames = 0
        
        # Camera parameters
        self.camera_width = 640
        self.camera_height = 480
        self.object_size_threshold = 5000
        self.min_target_area = 2500
        
        # Action space: continuous [left wheel speed, right wheel speed]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: normalized [object_x, object_y, target_x, target_y, IR_front, IR_left, IR_right]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(self):
        """Reset environment and return initial observation"""
        self.rob.stop_simulation()
        self.rob.play_simulation()
        self.episode_step = 0
        self.last_object_position = None
        self.object_in_target = False
        self.consecutive_target_frames = 0
        
        # Reset robot position and orientation
        self.rob.set_position(Position(0, 0, 0), Orientation(0, 0, 0))
        
        # Initialize target position with verification
        self.target_position = self._reliably_detect_target()
        
        # Verify object visibility
        start_time = time.time()
        while time.time() - start_time < 2:  # 2-second timeout
            frame = self.rob.read_image_front()
            contours_red, _ = self._process_image(frame)
            if self._get_valid_object_position(contours_red) is not None:
                break
            time.sleep(0.1)
        
        return self._get_observation()

    def _process_image(self, frame):
        """Robust object detection with improved color ranges"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhanced red detection
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 150, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Enhanced green detection with noise reduction
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((5,5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        return (
            cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        )

    def _reliably_detect_target(self, max_attempts=5):
        """Robust target detection with multiple verification attempts"""
        for _ in range(max_attempts):
            frame = self.rob.read_image_front()
            _, contours_green = self._process_image(frame)
            
            if len(contours_green) > 0:
                largest = max(contours_green, key=cv2.contourArea)
                if cv2.contourArea(largest) > self.min_target_area:
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        return (
                            M["m10"] / M["m00"] / self.camera_width,
                            M["m01"] / M["m00"] / self.camera_height
                        )
            time.sleep(0.1)
        
        # Fallback to center if target not found
        return (0.5, 0.5)

    def _get_valid_object_position(self, contours):
        """Get position with validation checks"""
        if len(contours) == 0:
            return None
            
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / float(h)
        
        # Validation criteria
        if (area < self.object_size_threshold or 
            aspect_ratio < 0.3 or 
            aspect_ratio > 3.0):
            return None
            
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
            
        return (
            M["m10"] / M["m00"] / self.camera_width,
            M["m01"] / M["m00"] / self.camera_height
        )

    def _get_observation(self):
        """Robust observation processing with validation"""
        frame = self.rob.read_image_front()
        contours_red, contours_green = self._process_image(frame)

        # Get validated object position
        obj_pos = self._get_valid_object_position(contours_red)
        if obj_pos is not None:
            self.last_object_position = obj_pos
            self.consecutive_target_frames = 0
        else:
            self.consecutive_target_frames += 1

        # Get IR sensor values with validation
        irs = [x if x is not None else 1000 for x in self.rob.read_irs()]
        ir_front = min(irs[3], irs[4])
        ir_left = irs[2]
        ir_right = irs[5]

        return np.array([
            *(obj_pos if obj_pos is not None else (0.5, 0.5)),
            *self.target_position,
            ir_front / 1000.0,
            ir_left / 1000.0,
            ir_right / 1000.0
        ], dtype=np.float32)

    def _check_success(self, obj_pos):
        """Robust success criteria with multiple checks"""
        if obj_pos is None or self.target_position is None:
            return False
        
        # Position-based check
        position_distance = np.linalg.norm(np.array(obj_pos) - np.array(self.target_position))
        
        # Validate with actual target detection
        frame = self.rob.read_image_front()
        _, contours_green = self._process_image(frame)
        target_visible = any(cv2.contourArea(c) > self.min_target_area for c in contours_green)
        
        # Confirm with multiple consecutive frames
        if position_distance < 0.1 and target_visible:
            self.consecutive_target_frames += 1
        else:
            self.consecutive_target_frames = 0
            
        return self.consecutive_target_frames >= 3

    def step(self, action: np.ndarray):
        self.episode_step += 1
        
        # Convert action to wheel speeds
        left_speed = np.clip(action[0] * 100, -100, 100)
        right_speed = np.clip(action[1] * 100, -100, 100)
        self.rob.move_blocking(left_speed, right_speed, 300)

        obs = self._get_observation()
        done = False
        reward = 0

        # Calculate rewards
        current_obj_pos = obs[0:2]
        
        if self.last_object_position is not None:
            # Reward for movement towards target
            prev_dist = np.linalg.norm(self.last_object_position - self.target_position)
            current_dist = np.linalg.norm(current_obj_pos - self.target_position)
            reward += (prev_dist - current_dist) * 20
            
            # Penalize moving away
            if current_dist > prev_dist:
                reward -= 5
                
            self.last_object_position = current_obj_pos

        # Check for success
        if self._check_success(current_obj_pos):
            reward += 100
            done = True
            self.object_in_target = True
        elif self.episode_step >= self.max_steps:
            done = True
            reward -= 10

        # Proximity reward
        if obs[4] < 0.3:  # Close to object
            reward += 0.5

        # Time penalty
        reward -= 0.1

        return obs, reward, done, {}

    def render(self, mode='human'):
        """Optional rendering implementation"""
        frame = self.rob.read_image_front()
        if mode == 'human':
            cv2.imshow('Push Task', frame)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class PPONetwork(nn.Module):
    """PPO Actor-Critic Network"""
    
    def __init__(self, input_size, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)  # Mean and std
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
    """Enhanced PPO Agent with experience replay and gradient clipping"""
    
    def __init__(self, state_dim, action_dim):
        self.policy = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4, eps=1e-5)
        self.gamma = 0.99
        self.epsilon = 0.2
        self.batch_size = 128
        self.memory = deque(maxlen=20000)
        self.max_grad_norm = 0.5
        
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
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, old_log_probs, returns, advantages = map(torch.FloatTensor, zip(*batch))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
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
        
        total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
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
        env.render()
        
    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    train_push_agent()
    # run_push_agent()