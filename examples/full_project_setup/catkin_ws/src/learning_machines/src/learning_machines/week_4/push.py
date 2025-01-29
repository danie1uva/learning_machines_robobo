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
import time
import wandb
from robobo_interface import SimulationRobobo
from robobo_interface import Orientation, Position
import math

# ----------------------------
# PPO NETWORK
# ----------------------------
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
            nn.Linear(64, action_dim * 2)
        )
        
        # Initialize log_std parameters
        with torch.no_grad():
            action_dim = action_dim
            self.actor[2].weight.data[action_dim:, :] = 0.0
            self.actor[2].bias.data[action_dim:] = torch.log(torch.tensor(0.1))
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)


# ----------------------------
# PPO AGENT
# ----------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.epsilon = 0.2
        self.batch_size = 128
        self.memory = deque(maxlen=20000)
        self.max_grad_norm = 0.5
        self.entropy_coef = 0.1  # Increased entropy for exploration

    def get_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            actor_out, value = self.policy(state_t)
        mean, log_std = actor_out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.numpy()[0], log_prob.numpy(), value.numpy()[0]

    def store_transition(self, s, a, old_log_p, v, advantage):
        self.memory.append((s, a, old_log_p, v, advantage))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, old_log_probs, returns, advantages = map(torch.FloatTensor, zip(*batch))

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_out, critic_out = self.policy(states)
        mean, log_std = actor_out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        dist = Normal(mean, std)

        new_log_probs = dist.log_prob(actions).sum(-1)
        ratio = (new_log_probs - old_log_probs).exp()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = 0.5 * (returns - critic_out.squeeze()).pow(2).mean()
        entropy_loss = -self.entropy_coef * dist.entropy().mean()  # Increased entropy coefficient

        loss = actor_loss + critic_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def train(self, env, episodes=1000):
        wandb.init(project="push-task", name="PPO_Pusher_Improved", config={"episodes": episodes})
        best_reward = -float("inf")

        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            last_positions = deque(maxlen=10)  # Track recent positions for circling detection

            while not done:
                action, old_log_prob, value = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Add position to history for circling detection
                current_pos = env.rob.get_position()
                last_positions.append((current_pos.x, current_pos.y))
                
                total_reward += reward
                step_count += 1

                with torch.no_grad():
                    _, next_value = self.policy(torch.FloatTensor(next_state).unsqueeze(0))
                advantage = reward + self.gamma * (0 if done else next_value.item()) - value.item()

                self.store_transition(state, action, old_log_prob, value, advantage)
                state = next_state
                self.update()

                # Gradually reduce entropy to balance exploration/exploitation
                self.entropy_coef = max(0.01, self.entropy_coef * 0.995)

            wandb.log({
                "episode": ep,
                "episode_reward": total_reward,
                "episode_length": step_count,
                "entropy_coef": self.entropy_coef
            })

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.policy.state_dict(), "best_pusher.pth")

            print(f"[Episode {ep}] Total Reward: {total_reward:.2f} | Best: {best_reward:.2f}")
# ----------------------------
# ENVIRONMENT
# ----------------------------
class PushEnv(gym.Env):
    def __init__(self, rob: SimulationRobobo):
        super().__init__()
        self.rob = rob
        self.episode_step = 0
        self.max_steps = 100
        self.camera_width = 640
        self.camera_height = 480
        self.last_robot_pos = None
        self.last_robot_ori = None
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(13,), dtype=np.float32
        )
        self.position_history = deque(maxlen=20)  # For circling detection

    def reset(self):
        self.rob.stop_simulation()
        self.rob.play_simulation()
        time.sleep(0.5)
        self.episode_step = 0
        self.last_robot_pos = self.rob.get_position()
        self.last_robot_ori = self.rob.get_orientation()
        self.position_history.clear()
        return self._compute_observation()

    def step(self, action):
        self.episode_step += 1
        self.last_robot_pos = self.rob.get_position()
        self.last_robot_ori = self.rob.get_orientation()

        left_speed = float(action[0]) * 100.0
        right_speed = float(action[1]) * 100.0
        self.rob.move_blocking(left_speed, right_speed, 300)

        # Add current position to history
        current_pos = self.rob.get_position()
        self.position_history.append((current_pos.x, current_pos.y))

        # Collision checks
        ir_raw = self.rob.read_irs()
        ir_raw = [x/1000 if x is not None else 1.0 for x in ir_raw]
        
        # Only check IR collision if not close to puck
        if not self._should_ignore_collision():
            if any(val > 0.65 for val in ir_raw):
                obs = self._compute_observation()
                return obs, -50.0, True, {}

        frame = self.rob.read_image_front()
        if self._camera_collision_detected(frame):
            obs = self._compute_observation()
            return obs, -20.0, True, {}

        obs = self._compute_observation()
        reward, done = self._compute_reward_and_done(obs)

        if self.episode_step >= self.max_steps:
            done = True

        return obs, reward, done, {}

    def render(self, mode='human'):
        frame = self.rob.read_image_front()
        cv2.imshow("PushEnv", frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    # ----------------------------
    # REWARD ENGINE
    # ----------------------------

    def _should_ignore_collision(self):
        """Check if we should ignore IR collisions due to being near the puck"""
        puck_close = self._distance_robot_to_puck() < 0.3
        camera_detection = self._detect_red_areas(self.rob.read_image_front()) is not None
        return puck_close or camera_detection

    def _compute_reward_and_done(self, obs):
        reward = -0.1  # Step penalty
        done = False
        reward_components = {}

        # 1) Robot->Puck Proximity
        dist_rp = self._distance_robot_to_puck()
        rp_shaping = 8.0 / (1.0 + dist_rp)  # Increased shaping
        reward += rp_shaping
        reward_components['rp_shaping'] = rp_shaping

        # 2) Camera Red Box Presence (fixed calculation)
        puck_box = obs[5:9]
        actual_w = puck_box[2] * self.camera_width
        actual_h = puck_box[3] * self.camera_height
        puck_area = actual_w * actual_h
        
        # Only reward if actually detecting the puck
        if puck_area > 100:  # Minimum area threshold
            area_shaping = min(puck_area / 15000.0, 3.0)  # Increased reward
            reward += area_shaping
            reward_components['area_shaping'] = area_shaping
        else:
            reward_components['area_shaping'] = 0.0

        # 3) Camera Alignment
        green_box = obs[9:13]
        if puck_box[2] > 0 and green_box[2] > 0:
            puck_cx = puck_box[0] + puck_box[2]/2
            puck_cy = puck_box[1] + puck_box[3]/2
            green_cx = green_box[0] + green_box[2]/2
            green_cy = green_box[1] + green_box[3]/2
            dist_cam = math.sqrt((puck_cx - green_cx)**2 + (puck_cy - green_cy)**2)
            camera_shaping = max(0, 1.5 - dist_cam) * 2.0  # Increased reward
            reward += camera_shaping
            reward_components['camera_shaping'] = camera_shaping

        # 4) Puck->Base Proximity
        dist_gt = self._distance_puck_to_base()
        gt_shaping = 8.0 / (1.0 + dist_gt)  # Increased shaping
        reward += gt_shaping
        reward_components['gt_shaping'] = gt_shaping

        # 5) Success Bonus
        if self.rob.base_detects_food():
            reward += 300.0  # Increased success bonus
            done = True
            reward_components['success'] = 300.0

        # 6) Circling Penalty
        if len(self.position_history) >= 10:
            # Calculate movement variance
            positions = np.array(self.position_history)
            var_x = np.var(positions[:,0])
            var_y = np.var(positions[:,1])
            if var_x < 0.01 and var_y < 0.01:  # Low variance = circling
                circling_penalty = -2.0
                reward += circling_penalty
                reward_components['circling_penalty'] = circling_penalty

        print(f"[REWARD] Components: { {k: round(v, 2) for k, v in reward_components.items()} }")
        return reward, done

    # ----------------------------
    # HELPER METHODS
    # ----------------------------
    def _compute_observation(self):
        ir_raw = [x/1000 if x is not None else 1.0 for x in self.rob.read_irs()]
        chosen_irs = [ir_raw[7], ir_raw[2], ir_raw[4], ir_raw[3], ir_raw[5]]
        
        frame = self.rob.read_image_front()
        puck_box = self._detect_red_areas(frame) or (0,0,0,0)
        green_box = self._detect_green_areas(frame) or (0,0,0,0)
        
        return np.concatenate([
            np.array(chosen_irs, dtype=np.float32),
            self._normalize_box(puck_box),
            self._normalize_box(green_box)
        ])

    # ----------------------------
    # IMPROVED COLOR DETECTION
    # ----------------------------
    def _detect_red_areas(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            if w*h > 100:  # Minimum area threshold
                return (x,y,w,h)
        return None

    def _detect_green_areas(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([40, 50, 50])
        upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(c)
        return None

    def _normalize_box(self, box):
        x, y, w, h = box
        return np.array([
            x/self.camera_width,
            y/self.camera_height,
            w/self.camera_width,
            h/self.camera_height
        ], dtype=np.float32)

    def _camera_collision_detected(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white_ratio = np.mean(gray > 230)
        return white_ratio > 0.7

    def _distance_puck_to_base(self):
        food_pos = self.rob.get_food_position()
        base_pos = self.rob.get_base_position()
        return math.sqrt(
            (food_pos.x - base_pos.x)**2 +
            (food_pos.y - base_pos.y)**2 +
            (food_pos.z - base_pos.z)**2
        )

    def _distance_robot_to_puck(self):
        robot_pos = self.rob.get_position()
        food_pos = self.rob.get_food_position()
        return math.sqrt(
            (robot_pos.x - food_pos.x)**2 +
            (robot_pos.y - food_pos.y)**2 +
            (robot_pos.z - food_pos.z)**2
        )

# ----------------------------
# MAIN EXECUTION
# ----------------------------
def train_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    agent = PPOAgent(state_dim=13, action_dim=2)
    agent.train(env, episodes=500)

def run_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    policy = PPONetwork(13, 2)
    policy.load_state_dict(torch.load("best_pusher.pth"))
    policy.eval()

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        with torch.no_grad():
            actor_out, _ = policy(torch.FloatTensor(state).unsqueeze(0))
            mean, log_std = actor_out.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, min=-20, max=2)
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.sample().numpy()[0]

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        env.render()

    print(f"Final reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    train_push_agent()
    # run_push_agent()
