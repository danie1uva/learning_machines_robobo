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

import wandb  # For logging if needed

from robobo_interface import SimulationRobobo
from robobo_interface import Orientation, Position

class PushEnv(gym.Env):
    """Custom Gym environment for pushing task with only non-negative wheel speeds."""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, rob: SimulationRobobo):
        super().__init__()
        self.rob = rob
        
        self.episode_step = 0
        self.max_steps = 100
        
        # Track red box area to reward approaching
        self.last_red_area = 0.0

        # For target detection checks
        self.target_position = None
        self.object_in_target = False
        self.consecutive_target_frames = 0
        
        # Camera parameters
        self.camera_width = 640
        self.camera_height = 480
        self.object_size_threshold = 5000
        self.min_target_area = 2500
        
        # A threshold above which we consider the robot "holding" the red box
        self.red_area_hold_threshold = 15000.0

        # -- Action space: We now only allow [0..1], no negative speeds. --
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # -- Observation space: [obj_x, obj_y, targ_x, targ_y, IR_front, IR_left, IR_right]
        #    IR sensors remain in [0..1000].
        #    We'll define an upper bound for them as 1000 here.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1000.0, 1000.0, 1000.0]),
            dtype=np.float32
        )

    def reset(self):
        """Reset environment and return initial observation."""
        self.rob.stop_simulation()
        self.rob.play_simulation()
        self.episode_step = 0
        self.last_red_area = 0.0
        self.object_in_target = False
        self.consecutive_target_frames = 0

        # Reset robot to scene’s initial position
        initial_pos = self.rob.get_position()
        initial_ori = self.rob.get_orientation()
        self.rob.set_position(initial_pos, initial_ori)

        # Detect green platform
        self.target_position = self._reliably_detect_target()
        
        # Give time to ensure red box is visible
        start_time = time.time()
        while time.time() - start_time < 2:
            frame = self.rob.read_image_front()
            contours_red, _ = self._process_image(frame)
            if self._get_valid_object_position(contours_red) is not None:
                break
            time.sleep(0.1)
        
        return self._get_observation()

    def _process_image(self, frame):
        """Return contours for red, green objects with morphological cleanup."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red detection (two intervals)
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 150, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Green detection
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((5,5), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        contours_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours_green = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        return contours_red, contours_green

    def _get_largest_contour_area(self, contours):
        if len(contours) == 0:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        return cv2.contourArea(largest)

    def _reliably_detect_target(self, max_attempts=5):
        """Try multiple times to find largest green contour above min_target_area."""
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
        # fallback
        return (0.5, 0.5)

    def _get_valid_object_position(self, contours):
        """Return red box center if it meets constraints, else None."""
        if len(contours) == 0:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / float(h)
        if (area < self.object_size_threshold or aspect_ratio < 0.3 or aspect_ratio > 3.0):
            return None
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        return (
            M["m10"] / M["m00"] / self.camera_width,
            M["m01"] / M["m00"] / self.camera_height
        )

    def _get_observation(self):
        """Return: [red_x, red_y, green_x, green_y, ir_front, ir_left, ir_right]."""
        frame = self.rob.read_image_front()
        contours_red, contours_green = self._process_image(frame)

        # Debug prints
        red_area = self._get_largest_contour_area(contours_red)
        if red_area > 0:
            print("[DEBUG] Red box detected (area > 0)")
        if len(contours_green) > 0:
            print("[DEBUG] Green platform detected")

        obj_pos = self._get_valid_object_position(contours_red)
        
        # IR sensor readings in [0..1000], do not normalize
        irs = [x if x is not None else 1000 for x in self.rob.read_irs()]
        # front is (3,4), left=2, right=5 if your robot uses that convention
        ir_front = min(irs[3], irs[4])  
        ir_left = irs[2]
        ir_right = irs[5]

        # store red area for reward
        self.current_red_area = red_area
        
        if obj_pos is not None:
            # reset consecutive frames if we see red box
            self.consecutive_target_frames = 0
            red_x, red_y = obj_pos
        else:
            red_x, red_y = 0.5, 0.5
            self.consecutive_target_frames += 1

        obs = np.array([
            red_x, 
            red_y,
            self.target_position[0], 
            self.target_position[1],
            ir_front,
            ir_left,
            ir_right
        ], dtype=np.float32)
        return obs

    def _check_success(self, red_pos):
        """Success if red box center is near green center for multiple frames."""
        if red_pos is None or self.target_position is None:
            return False
        
        dist = np.linalg.norm(np.array(red_pos) - np.array(self.target_position))
        
        # check if green is visible
        frame = self.rob.read_image_front()
        _, contours_green = self._process_image(frame)
        target_visible = any(cv2.contourArea(c) > self.min_target_area for c in contours_green)
        
        if dist < 0.1 and target_visible:
            self.consecutive_target_frames += 1
        else:
            self.consecutive_target_frames = 0
        return self.consecutive_target_frames >= 3

    def step(self, action: np.ndarray):
        self.episode_step += 1
        
        # 1) Force non-negative speeds, scale to [0..100].
        left_speed = np.clip(action[0], 0.0, 1.0) * 100
        right_speed = np.clip(action[1], 0.0, 1.0) * 100
        
        # Move
        self.rob.move_blocking(left_speed, right_speed, 300)

        obs = self._get_observation()
        done = False
        reward = 0.0

        red_x, red_y = obs[0], obs[1]
        green_x, green_y = obs[2], obs[3]
        ir_front = obs[4]
        ir_left = obs[5]
        ir_right = obs[6]

        # 1) Crash detection: if any IR sensor is > 650 => collision/wall => end
        if (ir_front > 650) or (ir_left > 650) or (ir_right > 650):
            print("[DEBUG] Crash detected via IR > 650 => immediate termination.")
            reward -= 50.0
            done = True

        # 2) Reward for approaching red box (area diff)
        if (self.last_red_area > 0) and (self.current_red_area > 0):
            area_diff = self.current_red_area - self.last_red_area
            reward += area_diff * 0.01
            if area_diff > 0:
                print(f"[DEBUG] Approaching red box: area diff {area_diff:.1f}, partial reward {area_diff*0.01:.2f}")

        # 3) If "holding" the red box, extra reward for bringing it near green
        #    We'll check the difference in distance from last step to this step
        #    If current_red_area big => we are "holding" the box
        if self.last_red_area > 0 and self.current_red_area > 0:
            # We can approximate the old distance by storing red_x etc. from previous step
            # But simpler is to store the last valid center in a variable if desired
            pass

        # We'll do a simpler approach: track old vs new distance if you want:
        # but to keep code short, let's do just area-based approach for "holding."
        
        # 4) Check success: if red box is near green for a few frames
        if not done:
            red_pos = (red_x, red_y)
            if self._check_success(red_pos):
                reward += 100.0
                done = True

        # Step limit
        if self.episode_step >= self.max_steps and not done:
            reward -= 10.0
            done = True

        # Mild penalty for each step
        reward -= 0.1

        self.last_red_area = self.current_red_area
        
        print(f"[DEBUG] Step {self.episode_step}, Reward: {reward:.3f}, Done: {done}")
        return obs, reward, done, {}

    def render(self, mode='human'):
        frame = self.rob.read_image_front()
        if mode == 'human':
            cv2.imshow('Push Task', frame)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

# PPO network and training loop remain the same (slight modifications if you wish).
class PPONetwork(nn.Module):
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
            nn.Linear(64, action_dim*2)  # mean+std for each dimension
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
    def __init__(self, state_dim, action_dim):
        self.policy = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.epsilon = 0.2
        self.batch_size = 128
        self.memory = deque(maxlen=20000)
        self.max_grad_norm = 0.5

    def get_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            actor_out, value = self.policy(state_t)
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
        
        loss = actor_loss + critic_loss + entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def train(self, env, episodes=1000):
        wandb.init(project="push-task", name="PPO_Pusher", resume="allow")
        best_reward = -float("inf")

        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            start_time = time.time()

            while not done:
                action, old_log_prob, value = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                step_count += 1
                with torch.no_grad():
                    _, next_value = self.policy(torch.FloatTensor(next_state).unsqueeze(0))
                advantage = reward + self.gamma * (1 - done) * next_value.item() - value.item()
                self.memory.append((state, action, old_log_prob, value, advantage))
                state = next_state
                self.update()

            episode_time = time.time() - start_time
            wandb.log({
                "episode": ep,
                "episode_reward": total_reward,
                "episode_length": step_count,
                "clip_epsilon": self.epsilon,
                "episode_time_sec": episode_time,
            })

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.policy.state_dict(), "best_pusher.pth")

            print(f"Episode {ep} | Reward: {total_reward:.2f} | Best: {best_reward:.2f}")

def train_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    agent = PPOAgent(state_dim=7, action_dim=2)
    agent.train(env, episodes=500)  # for example

def run_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    policy = PPONetwork(7, 2)
    policy.load_state_dict(torch.load("best_pusher.pth"))
    policy.eval()
    state = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        with torch.no_grad():
            actor_out, _ = policy(torch.FloatTensor(state).unsqueeze(0))
            mean, log_std = actor_out.chunk(2, dim=-1)
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.sample().numpy()[0]
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    train_push_agent()
    # run_push_agent()
