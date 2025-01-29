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

import wandb  # <-- We keep W&B for logging

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
            nn.Linear(64, action_dim * 2)  # mean + log_std
        )
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

    def store_transition(self, s, a, old_log_p, v, advantage):
        self.memory.append((s, a, old_log_p, v, advantage))

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

        loss = actor_loss + critic_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def train(self, env, episodes=1000):
        # Initialize wandb for logging
        wandb.init(project="push-task", name="PPO_Pusher_GroundTruth", config={"episodes": episodes})

        best_reward = -float("inf")

        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0

            while not done:
                action, old_log_prob, value = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                step_count += 1

                # compute advantage
                with torch.no_grad():
                    _, next_value = self.policy(torch.FloatTensor(next_state).unsqueeze(0))
                advantage = reward + self.gamma * (0 if done else next_value.item()) - value.item()

                self.store_transition(state, action, old_log_prob, value, advantage)
                state = next_state

                # Possibly update PPO
                self.update()

            # Log stats to wandb
            wandb.log({
                "episode": ep,
                "episode_reward": total_reward,
                "episode_length": step_count
            })

            # Save best
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.policy.state_dict(), "best_pusher.pth")

            print(f"[Episode {ep}] Total Reward: {total_reward:.2f} | Best So Far: {best_reward:.2f}")

# ----------------------------
# ENVIRONMENT
# ----------------------------
class PushEnv(gym.Env):
    """
    Push the red /Food object onto the green base.
     - IR sensors in raw [0..1000].
     - Crash if IR>650 => done with penalty.
     - If camera “whites out” => end episode with penalty.
     - Episode limit = 50 steps.
     - Uses ground-truth distances for (robot->puck) and (puck->base) for shaping.
     - Also uses camera bounding boxes for additional shaping for red box + green zone.
    """

    def __init__(self, rob: SimulationRobobo):
        super().__init__()
        self.rob = rob

        self.episode_step = 0
        self.max_steps = 50

        self.camera_width = 640
        self.camera_height = 480

        # We'll keep the robot's last known position/orientation
        self.last_robot_pos = None
        self.last_robot_ori = None

        # 2D action in [0..1], scaled to [0..100] internally
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observations: 5 raw IR + 4 puck box + 4 green box = 13
        # (We rely on GT calls for extra shaping in the reward.)
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(13,), dtype=np.float32
        )

    def reset(self):
        self.rob.stop_simulation()
        self.rob.play_simulation()
        time.sleep(0.5)

        self.episode_step = 0

        self.last_robot_pos = self.rob.get_position()
        self.last_robot_ori = self.rob.get_orientation()

        obs = self._compute_observation()
        return obs

    def step(self, action):
        self.episode_step += 1

        # Save the robot's pos/orient BEFORE we move
        self.last_robot_pos = self.rob.get_position()
        self.last_robot_ori = self.rob.get_orientation()

        left_speed = float(action[0]) * 100.0
        right_speed = float(action[1]) * 100.0
        self.rob.move_blocking(left_speed, right_speed, 300)

        # 1) IR-based crash check
        ir_raw = self.rob.read_irs()
        if any(val is not None and val > 650 for val in ir_raw):
            print("[DEBUG] IR Crash Detected! IR sensor above 650 => done.")
            obs = self._compute_observation()
            return obs, -50.0, True, {}

        # 2) Camera collision => end episode
        frame = self.rob.read_image_front()
        if self._camera_collision_detected(frame):
            print("[DEBUG] Camera collision => big penalty => end episode.")
            obs = self._compute_observation()
            return obs, -20.0, True, {}

        # 3) Normal step
        obs = self._compute_observation()
        reward, done = self._compute_reward_and_done(obs)
        print(f"[DEBUG] Step {self.episode_step}, Reward: {reward:.2f}, Done={done}")

        # 4) Check step limit
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
    # HELPER FUNCTIONS
    # ----------------------------
    def _compute_observation(self):
        # 1) IR sensors in [0..1000]
        ir_raw = self.rob.read_irs()
        ir_processed = [x if (x is not None) else 1000 for x in ir_raw]
        # pick five specific sensors
        chosen_irs = [
            ir_processed[7],
            ir_processed[2],
            ir_processed[4],
            ir_processed[3],
            ir_processed[5]
        ]

        # 2) bounding boxes for red/green in camera
        frame = self.rob.read_image_front()
        puck_box = self._detect_red_areas(frame) or (0,0,0,0)
        green_box = self._detect_green_areas(frame) or (0,0,0,0)
        puck_norm = self._normalize_box(puck_box)
        green_norm = self._normalize_box(green_box)

        obs = np.concatenate([
            np.array(chosen_irs, dtype=np.float32),
            puck_norm,
            green_norm
        ])
        return obs

    def _detect_red_areas(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 50, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 50, 50])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(c)
        return None

    def _detect_green_areas(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_g = np.array([40, 50, 50])
        upper_g = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_g, upper_g)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(c)
        return None

    def _normalize_box(self, box):
        x, y, w, h = box
        return np.array([x, y, w, h], dtype=np.float32)

    def _camera_collision_detected(self, frame):
        """
        If a large portion of the image is bright/white => collision
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white_thresh = 230
        white_ratio = np.mean(gray > white_thresh)
        return (white_ratio > 0.3)

    def _distance_puck_to_base(self):
        """
        Euclidean distance from the /Food position to the /Base
        """
        food_pos = self.rob.get_food_position()   # returns Position(x,y,z)
        base_pos = self.rob.get_base_position()
        dx = food_pos.x - base_pos.x
        dy = food_pos.y - base_pos.y
        dz = food_pos.z - base_pos.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _distance_robot_to_puck(self):
        """
        Euclidean distance from the robot to the /Food
        """
        robot_pos = self.rob.get_position()
        food_pos = self.rob.get_food_position()
        dx = robot_pos.x - food_pos.x
        dy = robot_pos.y - food_pos.y
        dz = robot_pos.z - food_pos.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _compute_reward_and_done(self, obs):
        """
        - Step penalty
        - Reward for robot->puck closeness (GT)
        - Reward for seeing a larger red box area in camera (puck is closer in view)
        - Reward for camera-based puck->green closeness
        - Reward for puck->base closeness (GT)
        - If base_detects_food() => big success
        """
        reward = 0.0
        done = False

        # small step cost
        reward -= 0.1

        # 1) Robot->puck ground-truth shaping
        dist_rp = self._distance_robot_to_puck()
        # e.g. up to +2 if dist=0
        rp_shaping = (1.0 / (1.0 + dist_rp)) * 2.0
        reward += rp_shaping
        print(f"[DEBUG] Robot->Puck dist: {dist_rp:.3f}, rp_shaping={rp_shaping:.2f}")

        # 2) Camera-based red box area shaping (the bigger the box, the closer the puck in view)
        # obs indexes: IR (0..4), puck box (5..8) => (x, y, w, h)
        puck_box = obs[5:9]
        puck_area = puck_box[2] * puck_box[3]
        # scale area to [0..some_max], let's guess 30000 might be near the max in your setup
        # clamp to +1.0 max for shaping
        area_shaping = min(puck_area / 30000.0, 1.0)
        reward += area_shaping
        print(f"[DEBUG] Puck box area: {puck_area:.1f}, area_shaping={area_shaping:.2f}")

        # 3) Camera-based puck->green closeness
        green_box = obs[9:13]  # (x,y,w,h)
        camera_shaping = 0.0
        if puck_box[2] > 0 and green_box[2] > 0:
            puck_cx = puck_box[0] + puck_box[2]/2
            puck_cy = puck_box[1] + puck_box[3]/2
            green_cx = green_box[0] + green_box[2]/2
            green_cy = green_box[1] + green_box[3]/2
            dist_cam = math.sqrt((puck_cx - green_cx)**2 + (puck_cy - green_cy)**2)
            # up to +1 if perfectly overlapped
            camera_shaping = max(0, 100 - dist_cam) * 0.01
            reward += camera_shaping
            print(f"[DEBUG] Camera puck->green dist: {dist_cam:.1f}, camera_shaping={camera_shaping:.2f}")

        # 4) GT shaping: puck->base closeness
        dist_gt = self._distance_puck_to_base()
        gt_shaping = (1.0 / (1.0 + dist_gt)) * 2.0
        reward += gt_shaping
        print(f"[DEBUG] Puck->Base dist: {dist_gt:.3f}, gt_shaping={gt_shaping:.2f}")

        # 5) If base_detects_food() => success
        if self.rob.base_detects_food():
            print("[DEBUG] base_detects_food() => success!")
            reward += 100.0
            done = True

        return reward, done

# ----------------------------
# TRAIN / RUN
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
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.sample().numpy()[0]

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        env.render()

    print(f"Final episode reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    train_push_agent()
    # run_push_agent()
