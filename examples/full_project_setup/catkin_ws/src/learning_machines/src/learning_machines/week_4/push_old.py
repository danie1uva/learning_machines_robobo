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
        actor_out = self.actor(shared_out)
        mean, log_std = actor_out.chunk(2, dim=-1)
        mean = torch.sigmoid(mean)  # Squash mean to [0,1]
        log_std = torch.clamp(log_std, min=-20, max=2)
        return torch.cat([mean, log_std], dim=-1), self.critic(shared_out)


# ----------------------------
# PPO AGENT
# ----------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.lam = 0.95          # GAE-λ parameter
        self.epsilon = 0.2      # PPO clip param
        self.max_grad_norm = 0.5
        self.entropy_coef = 0.01
        self.batch_size = 2048  # For minibatching inside update_policy
        self.n_epochs = 10      # Number of gradient epochs per PPO update

    def get_action(self, state):
        """
        Returns:
          action (np.array): shape (action_dim,)
          log_prob (np.array): shape (1,) or scalar
          value (float): scalar
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            actor_out, value = self.policy(state_t)
        mean, log_std = actor_out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, 0.0, 1.0)  # Force [0,1] range
        log_prob = dist.log_prob(action).sum(-1)
        return (
            action.numpy()[0],
            log_prob.numpy(),
            value.numpy()[0]
        )

    def compute_gae(self, rewards, values, dones):
        """
        Generalized Advantage Estimation (GAE-Lambda).
        
        Args:
          rewards: list of reward floats for each step [r0, r1, ..., r_{T-1}]
          values:  list of value preds for each step *plus one final* V_{T} (so length = T+1)
          dones:   list of booleans for each step [d0, d1, ..., d_{T-1}]

        Returns:
          advantages (list of floats, length T)
          returns    (list of floats, length T), where return_t = advantage_t + value_t
        """
        advantages = [0] * len(rewards)
        gae = 0.0

        # Traverse backwards
        for t in reversed(range(len(rewards))):
            # If done, next_value = 0, else next_value = values[t+1]
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae

        returns = [v + adv for v, adv in zip(values[:-1], advantages)]
        return advantages, returns

    def update_policy(self, states, actions, old_log_probs, returns, advantages):
        """
        Runs multiple epochs of PPO updates on the entire rollout data
        using mini-batches.
        """
        states      = torch.FloatTensor(states)
        actions     = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns     = torch.FloatTensor(returns)
        advantages  = torch.FloatTensor(advantages)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # How many steps in this rollout?
        dataset_size = states.shape[0]
        inds = np.arange(dataset_size)

        for _ in range(self.n_epochs):
            np.random.shuffle(inds)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_inds = inds[start:end]

                # Get mini-batch
                mb_states = states[mb_inds]
                mb_actions = actions[mb_inds]
                mb_old_log = old_log_probs[mb_inds]
                mb_returns = returns[mb_inds]
                mb_advants = advantages[mb_inds]

                # Forward pass
                actor_out, critic_out = self.policy(mb_states)
                mean, log_std = actor_out.chunk(2, dim=-1)
                log_std = torch.clamp(log_std, min=-20, max=2)
                std = log_std.exp()
                dist = Normal(mean, std)

                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1, keepdim=False)
                ratio = torch.exp(new_log_probs - mb_old_log)

                # PPO objectives
                surr1 = ratio * mb_advants
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * mb_advants
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic (value) loss
                critic_loss = 0.5 * (mb_returns - critic_out.squeeze()).pow(2).mean()

                # Entropy bonus
                entropy_loss = -self.entropy_coef * dist.entropy().mean()

                loss = actor_loss + critic_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def train(self, env, episodes=1000):
        wandb.init(project="push-task", name="PPO_Pusher_GAE", config={"episodes": episodes})
        best_reward = -float("inf")

        for ep in range(episodes):
            # Collect an entire episode of transitions
            state = env.reset()
            done = False
            total_reward = 0.0
            transitions = []

            while not done:
                # Step environment
                action, old_log_prob, value = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                total_reward += reward
                transitions.append((state, action, old_log_prob, value, reward, done))

                state = next_state

            # We need one extra "next_value" at the end for GAE
            with torch.no_grad():
                # If done, next_value = 0, else we get from the critic
                if done:
                    next_value = 0.0
                else:
                    _, v = self.policy(torch.FloatTensor(state).unsqueeze(0))
                    next_value = v.item()

            # Separate out the collected episode data
            states      = [t[0] for t in transitions]
            actions     = [t[1] for t in transitions]
            old_logs    = [t[2] for t in transitions]
            values      = [t[3] for t in transitions]
            rewards     = [t[4] for t in transitions]
            dones       = [t[5] for t in transitions]

            # Append the final next_value for easier GAE calculation
            values.append(next_value)

            # --- Compute GAE & returns ---
            advantages, returns = self.compute_gae(rewards, values, dones)

            # --- Update policy on this single episode of data ---
            self.update_policy(states, actions, old_logs, returns, advantages)

            # Logging and model saving
            wandb.log({"episode": ep, "episode_reward": total_reward})
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.policy.state_dict(), "best_pusher.pth")
                wandb.save("best_pusher.pth")

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
        # Set phone tilt to look downwards
        self.rob.set_phone_tilt_blocking(140, 140)
        self.episode_step = 0
        self.last_robot_pos = self.rob.get_position()
        self.last_robot_ori = self.rob.get_orientation()
        self.position_history.clear()
        return self._compute_observation()

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        self.last_action = action
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
        
        # Split sensors into back and front groups
        back_irs = [ir_raw[6]]  #BackC
        front_irs = [ir_raw[4], ir_raw[7], ir_raw[5]]  #FrontC
        print(f"front_irs: {front_irs}, back_irs: {back_irs}")

        # Always check back sensors first
        if any(val > 0.15 for val in back_irs):
            print(f"[COLLISION] Back sensor triggered: {back_irs}")
            obs = self._compute_observation()
            return obs, -0.0, True, {}

        # Conditionally check front sensors
        if not self._should_ignore_front_collision():
            if any(val > 0.25 for val in front_irs):
                print(f"[COLLISION] Front sensor triggered: {front_irs}")
                obs = self._compute_observation()
                return obs, -0.0, True, {}
        # else:
        #     print("[DEBUG] Ignoring front IR collisions due to puck proximity")

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
    def _should_ignore_front_collision(self):
        """Check if we should ignore front IR collisions due to being near the puck"""
        puck_close = self._distance_robot_to_puck() < 0.3
        camera_detection = self._detect_red_areas(self.rob.read_image_front()) is not None
        
        # if puck_close:
        #     print("[DEBUG] Close to puck - ignoring front IR collisions")
        # if camera_detection:
        #     print("[DEBUG] Puck detected in camera - ignoring front IR collisions")
            
        return puck_close or camera_detection

    def _compute_reward_and_done(self, obs):
        # reward = -1  # Step penalty remains
        done = False
        reward_components = {}

        # 1) Robot->Puck Proximity (MAIN PRIORITY)
        dist_rp = self._distance_robot_to_puck()
        rp_shaping = 10.0 / (1.0 + dist_rp)  # Increased base reward
        reward += rp_shaping
        reward_components['rp_shaping'] = rp_shaping

        # 2) Camera Red Box Presence (secondary)
        puck_box = obs[5:9]
        actual_w = puck_box[2] * self.camera_width
        actual_h = puck_box[3] * self.camera_height
        puck_area = actual_w * actual_h
        
        # Updated centering calculation
        if puck_area > 100:
            puck_cx = (puck_box[0] + puck_box[2]/2) * self.camera_width
            puck_cy = (puck_box[1] + puck_box[3]/2) * self.camera_height
            img_center_x = self.camera_width/2
            img_center_y = self.camera_height/2
            
            # Use exponential centering reward
            x_offset = abs(puck_cx - img_center_x)/(self.camera_width/2)
            y_offset = abs(puck_cy - img_center_y)/(self.camera_height/2)
            centering = 2.0 * (1.0 - math.sqrt(x_offset**2 + y_offset**2))
            
            # Increased area vs centering ratio (40/60)
            area_shaping = (0.4 * min(puck_area/15000.0, 2.0)) + (0.6 * centering * 2.5)
            reward += area_shaping
            reward_components['area_shaping'] = area_shaping
        else:
            reward -= 0.5
            reward_components['area_penalty'] = -0.5

        # 3) Forward Motion Incentive (NEW)
        # Get wheel speeds from last action
        left_speed = self.last_action[0] if hasattr(self, 'last_action') else 0
        right_speed = self.last_action[1] if hasattr(self, 'last_action') else 0
        speed_similarity = 1.0 - abs(left_speed - right_speed)
        forward_bonus = speed_similarity * 0.5  # Max 1.2 per step
        reward += forward_bonus
        reward_components['forward_bonus'] = forward_bonus

        # 4) Puck->Base Proximity 
        dist_gt = self._distance_puck_to_base()
        gt_shaping = 10.0 / (1.0 + dist_gt)  # Increased from 10.0
        # Additional bonus for moving puck closer
        if hasattr(self, 'last_puck_distance'):
            distance_delta = self.last_puck_distance - dist_gt
            movement_bonus = 2.0 * distance_delta  # Scale delta for meaningful reward
            reward += movement_bonus
            reward_components['puck_movement'] = movement_bonus
        self.last_puck_distance = dist_gt  # Store for next calculation
        
        reward += gt_shaping
        reward_components['gt_shaping'] = gt_shaping

        # 5) Success Bonus (unchanged)
        if self.rob.base_detects_food():
            reward += 500.0
            done = True
            reward_components['success'] = 500.0
            print("[SUCCESS] Puck delivered to base!")

        # 6) Improved Circling Detection
        if len(self.position_history) >= 5:  # Shorter window
            positions = np.array(self.position_history)
            displacements = np.linalg.norm(positions[:-1] - positions[1:], axis=1)
            total_distance = np.sum(displacements)
            
            if total_distance > 0.5:
                net_displacement = np.linalg.norm(positions[-1] - positions[0])
                circling_ratio = net_displacement / total_distance
                
                # Progressive penalty based on circling severity
                if circling_ratio < 0.3:
                    penalty = -10.0 * (1.0 - (circling_ratio/0.3))
                    reward += penalty
                    reward_components['circling_penalty'] = penalty

        print(f"[REWARD] Components: { {k: round(v, 2) for k, v in reward_components.items()} }")
        print(f"total_reward: {reward:.2f}")
        return reward, done

    # ----------------------------
    # HELPER METHODS
    # ----------------------------
    def _compute_observation(self):
        ir_raw = self.rob.read_irs()
        ir_raw = [x if x is not None else 1.0 for x in self.rob.read_irs()]
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
        lower_red1 = np.array([0, 40, 40])  # Adjusted thresholds
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Improved morphological operations
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            if w*h > 100:
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
            # Change from minAreaRect to boundingRect
            x, y, w, h = cv2.boundingRect(c)
            # print(f"[DEBUG] Green platform detected: {w}x{h} area at ({x},{y})")
            return (x, y, w, h)
        return None

    def _normalize_box(self, box):
        x, y, w, h = box
        return np.array([
            x/self.camera_width,
            y/self.camera_height,
            w/self.camera_width,
            h/self.camera_height
        ], dtype=np.float32)

    def _distance_puck_to_base(self):
        # Get precise positions using simulation API
        food_pos = self.rob.get_food_position()
        base_pos = self.rob.get_base_position()
        
        # Calculate 2D distance (ignore height)
        distance = math.sqrt(
            (food_pos.x - base_pos.x)**2 +
            (food_pos.y - base_pos.y)**2
        )
        # print(f"[DISTANCE] Puck to base: {distance:.2f}m (Δ: {distance - self.last_puck_distance if hasattr(self, 'last_puck_distance') else 0:.2f})")
        return distance

    def _distance_robot_to_puck(self):
        # Get positions from simulation
        robot_pos = self.rob.get_position()
        food_pos = self.rob.get_food_position()
        
        # Calculate 2D distance with height consideration
        distance = math.sqrt(
            (robot_pos.x - food_pos.x)**2 +
            (robot_pos.y - food_pos.y)**2
        )
        # Add small epsilon to avoid division by zero
        return distance + 1e-3

    def _get_robot_heading_vector(self):
        # Calculate heading vector from orientation
        orientation = self.rob.read_orientation()
        yaw = math.radians(orientation.yaw)
        return math.cos(yaw), math.sin(yaw)

# ----------------------------
# MAIN EXECUTION
# ----------------------------
def train_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    
    # Initialize agent with modified parameters
    agent = PPOAgent(
        state_dim=13,
        action_dim=2
    )
    
    # Train with progress tracking
    try:
        agent.train(env, episodes=1999)
    except KeyboardInterrupt:
        print("Training interrupted - saving latest model")
        torch.save(agent.policy.state_dict(), "interrupted_pusher.pth")
        wandb.save("interrupted_pusher.pth")

def run_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    
    # Load best model
    policy = PPONetwork(13, 2)
    policy.load_state_dict(torch.load("best_pusher.pth"))
    policy.eval()

    state = env.reset()
    done = False
    total_reward = 0.0

    # Add performance monitoring
    start_time = time.time()
    success_count = 0
    collision_count = 0

    while not done:
        with torch.no_grad():
            action = policy(torch.FloatTensor(state).unsqueeze(0))[0][0].numpy()
        
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        env.render()

        # Track performance metrics
        if reward >= 500:  # Success condition
            success_count += 1
        if reward <= -0:  # Collision condition
            collision_count += 1

    # Print performance summary
    duration = time.time() - start_time
    print(f"\nPerformance Summary:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Successes: {success_count}")
    print(f"Collisions: {collision_count}")
    print(f"Duration: {duration:.2f} seconds")
    
    env.close()

if __name__ == "__main__":
    train_push_agent()
    # run_push_agent()