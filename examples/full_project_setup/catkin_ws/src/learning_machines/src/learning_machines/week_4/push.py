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
        self.epsilon = 0.2
        self.batch_size = 128
        self.memory = deque(maxlen=20000)
        self.max_grad_norm = 0.5
        self.entropy_coef = 0.01  

    def get_action(self, state):
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

            wandb.log({
                "episode": ep,
                "episode_reward": total_reward,
                "episode_length": step_count
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
        # print(f"[IR] Readings: {ir_raw}")
        
        # Split sensors into back and front groups
        back_irs = [ir_raw[0], ir_raw[1], ir_raw[6]]  # BackL, BackR, BackC
        front_irs = [ir_raw[2], ir_raw[3], ir_raw[4], ir_raw[5], ir_raw[7]]  # FrontL, FrontR, FrontC, FrontRR, FrontLL
        
        # Always check back sensors first
        if any(val > 0.2 for val in back_irs):
            print(f"[COLLISION] Back sensor triggered: {back_irs}")
            obs = self._compute_observation()
            return obs, -100.0, True, {}

        # Conditionally check front sensors
        if not self._should_ignore_front_collision():
            if any(val > 0.2 for val in front_irs):
                print(f"[COLLISION] Front sensor triggered: {front_irs}")
                obs = self._compute_observation()
                return obs, -100.0, True, {}
        # else:
        #     print("[DEBUG] Ignoring front IR collisions due to puck proximity")



        # frame = self.rob.read_image_front()
        # if self._camera_collision_detected(frame):
        #     obs = self._compute_observation()
        #     return obs, -30.0, True, {}

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
        reward = -0.2  # Step penalty remains
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
        
        if puck_area > 100:
            # Reduced centering reward component
            puck_cx = (puck_box[0] + puck_box[2]/2) * self.camera_width
            puck_cy = (puck_box[1] + puck_box[3]/2) * self.camera_height
            img_center_x = self.camera_width/2
            img_center_y = self.camera_height/2
            
            # Normalized centering component (0-1)
            centering = 1.0 - (abs(puck_cx - img_center_x)/(self.camera_width/2))
            
            # Balance area vs centering (60/40 ratio)
            area_shaping = (0.6 * min(puck_area/15000.0, 1.5)) + (0.4 * centering * 1.5)
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
                    penalty = -3.0 * (1.0 - (circling_ratio/0.3))
                    reward += penalty
                    reward_components['circling_penalty'] = penalty

        # print(f"[REWARD] Components: { {k: round(v, 2) for k, v in reward_components.items()} }")
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


    # def _camera_collision_detected(self, frame):
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
    #     # Define white range in HSV (hue doesn't matter for white)
    #     lower_white = np.array([0, 0, 200])
    #     upper_white = np.array([255, 30, 255])
        
    #     mask = cv2.inRange(hsv, lower_white, upper_white)
    #     white_ratio = np.mean(mask > 0)
        
    #     # Lower threshold for collision detection
    #     if white_ratio > 0.95:
    #         print(f"[COLLISION] White-out detected: {white_ratio*100:.1f}% white pixels")
    #         return True
    #     return False

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
        agent.train(env, episodes=2000)
    except KeyboardInterrupt:
        print("Training interrupted - saving latest model")
        torch.save(agent.policy.state_dict(), "interrupted_pusher.pth")

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
        if reward <= -50:  # Collision condition
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