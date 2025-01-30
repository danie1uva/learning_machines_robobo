import cv2
import numpy as np
import gym
from gym import spaces
import time
import wandb
import math
from collections import deque
from robobo_interface import SimulationRobobo
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

# ----------------------------
# KALMAN FILTER IMPLEMENTATION
# ----------------------------
class KalmanFilter:
    def __init__(self, process_noise=0.1, measurement_noise=5):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        self.last_measurement = None
        self.last_prediction = None

    def update(self, measurement):
        if measurement is not None:
            self.last_measurement = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
            self.kf.correct(self.last_measurement)
        self.last_prediction = self.kf.predict()
        return self.last_prediction

# ----------------------------
# ENVIRONMENT WITH FIXES
# ----------------------------
class PushEnv(gym.Env):
    def __init__(self, rob: SimulationRobobo):
        super().__init__()
        self.rob = rob
        self.episode_step = 0
        self.max_steps = 100
        self.camera_width = 640
        self.camera_height = 480
        
        # Fixed action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Correct observation space with proper bounds
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(13,),  # 5 IRS + 4 puck + 4 base
            dtype=np.float32
        )

        # Rest of initialization remains the same
        self.position_history = deque(maxlen=20)
        self.kalman_filter = KalmanFilter()

    def reset(self):
        self.rob.stop_simulation()
        self.rob.play_simulation()
        time.sleep(0.5)
        self.rob.set_phone_tilt_blocking(109, 109)
        self.episode_step = 0
        self.position_history.clear()
        self.kalman_filter = KalmanFilter()
        return self._safe_compute_observation()

    def step(self, action):
        # Scale action from [-1, 1] to [0, 100] for motors
        action = np.clip(action, -1.0, 1.0)
        left_speed = (action[0] + 1) * 50.0  # Convert to 0-100 range
        right_speed = (action[1] + 1) * 50.0
        
        self.rob.move_blocking(left_speed, right_speed, 300)
        self.episode_step += 1

        # Collision checks
        ir_raw = [x/1000 if x else 1.0 for x in self.rob.read_irs()]
        if any(val > 0.15 for val in [ir_raw[6]]):  # Back collision
            return self._safe_compute_observation(), -0.0, True, {}
        if any(val > 0.25 for val in [ir_raw[4], ir_raw[7], ir_raw[5]]) and not self._should_ignore_front_collision():
            return self._safe_compute_observation(), -0.0, True, {}

        obs = self._safe_compute_observation()
        reward, done = self._compute_reward_and_done(obs)
        
        if self.episode_step >= self.max_steps:
            done = True
            
        return obs, reward, done, {}
    
    def _safe_compute_observation(self):
        try:
            return self._compute_observation()
        except Exception as e:
            print(f"Observation error: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _compute_observation(self):
        # Get and normalize IRS values
        ir_raw = [min(max(x, 0.0), 1.0) if x else 1.0 for x in self.rob.read_irs()]
        chosen_irs = [
            min(max(ir_raw[7], 0.0), 1.0),
            min(max(ir_raw[2], 0.0), 1.0),
            min(max(ir_raw[4], 0.0), 1.0),
            min(max(ir_raw[3], 0.0), 1.0),
            min(max(ir_raw[5], 0.0), 1.0)
        ]
        
        # Process camera data
        frame = self.rob.read_image_front()
        puck_box = self._detect_red_areas(frame)
        green_box = self._detect_green_areas(frame)

        # Validate and normalize boxes with clipping
        def safe_normalize(box):
            if box is None or len(box) != 4:
                return np.zeros(4, dtype=np.float32)
            
            x = max(0.0, min(box[0]/self.camera_width, 1.0))
            y = max(0.0, min(box[1]/self.camera_height, 1.0))
            w = max(0.0, min(box[2]/self.camera_width, 1.0))
            h = max(0.0, min(box[3]/self.camera_height, 1.0))
            return np.array([x, y, w, h], dtype=np.float32)

        # Apply Kalman filtering
        if puck_box:
            try:
                puck_center = (puck_box[0] + puck_box[2]/2, puck_box[1] + puck_box[3]/2)
                prediction = self.kalman_filter.update(puck_center)
                puck_box = (
                    float(prediction[0] - puck_box[2]/2),
                    float(prediction[1] - puck_box[3]/2),
                    float(puck_box[2]),
                    float(puck_box[3])
                )
            except Exception as e:
                pass  # Fall back to original detection

        normalized_puck = safe_normalize(puck_box)
        normalized_base = safe_normalize(green_box)
        
        # Combine all observations
        observation = np.concatenate([
            np.array(chosen_irs, dtype=np.float32),
            normalized_puck,
            normalized_base
        ])
        
        # Final clipping to ensure bounds
        return np.clip(observation, 0.0, 1.0)

    def _detect_red_areas(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            if w*h > 100:
                return (x,y,w,h)
        return None

    def _detect_green_areas(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return (x, y, w, h)
        return None

    def _normalize_box(self, box):
        return np.array([
            box[0]/self.camera_width,
            box[1]/self.camera_height,
            box[2]/self.camera_width,
            box[3]/self.camera_height
        ], dtype=np.float32)

    def _distance_puck_to_base(self):
        food_pos = self.rob.get_food_position()
        base_pos = self.rob.get_base_position()
        return math.hypot(food_pos.x - base_pos.x, food_pos.y - base_pos.y)

    def _distance_robot_to_puck(self):
        robot_pos = self.rob.get_position()
        food_pos = self.rob.get_food_position()
        return math.hypot(robot_pos.x - food_pos.x, robot_pos.y - food_pos.y) + 1e-3

    def _should_ignore_front_collision(self):
        return self._distance_robot_to_puck() < 0.3 or self._detect_red_areas(self.rob.read_image_front()) is not None

    def _compute_reward_and_done(self, obs):
        reward = 0
        done = False
        
        puck_box = obs[5:9]
        green_box = obs[9:13]
        puck_area = (puck_box[2] * self.camera_width) * (puck_box[3] * self.camera_height)
        green_area = (green_box[2] * self.camera_width) * (green_box[3] * self.camera_height)

        if puck_area > 100:
            # Distance reward
            dist_rp = self._distance_robot_to_puck()
            reward += 10.0 / (1.0 + dist_rp)
            print("found the puck!")
            
            # Centering bonus
            puck_cx = (puck_box[0] + puck_box[2]/2) * self.camera_width
            puck_cy = (puck_box[1] + puck_box[3]/2) * self.camera_height
            x_offset = abs(puck_cx - self.camera_width/2)/(self.camera_width/2)
            y_offset = abs(puck_cy - self.camera_height/2)/(self.camera_height/2)
            reward += 2.5 * (1.0 - math.hypot(x_offset, y_offset))

            if green_area > 100:
                # Double reward when both visible
                print("found the base!")
                reward *= 2
                # Distance to base reward
                dist_gt = self._distance_puck_to_base()
                reward += 15.0 / (1.0 + dist_gt)

        if self.rob.base_detects_food():
            reward += 500.0
            done = True

        return reward, done

# ----------------------------
# WANDB CALLBACK
# ----------------------------
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                ep_reward = info['episode']['r']
                self.episode_rewards.append(ep_reward)
                wandb.log({
                    "episode_reward": ep_reward,
                    "total_steps": self.num_timesteps
                })
                if ep_reward == max(self.episode_rewards):
                    self.model.save("/root/results/best_model_sb3")
        return True

# ----------------------------
# MAIN TRAINING LOGIC
# ----------------------------
def train_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    env = Monitor(env)
    
    check_env(env, warn=True)
    wandb.init(project="push-task", name="PPO_SB3_Fixed")
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        policy_kwargs={
            "net_arch": dict(pi=[256, 128], vf=[256, 128]),
            "ortho_init": True,
        }
    )
    
    try:
        model.learn(
            total_timesteps=1_000_000,
            callback=WandbCallback(),
            reset_num_timesteps=True
        )
        model.save("/root/results/best_model_sb3")
    except KeyboardInterrupt:
        model.save("/root/results/interrupted_model_sb3")
    finally:
        env.close()

def run_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    
    try:
        model = PPO.load("/root/results/best_model_sb3.zip", env=env)
        obs = env.reset()
        total_reward = 0
        
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done: break
            
        print(f"Total reward: {total_reward}")
    finally:
        env.close()

if __name__ == "__main__":
    train_push_agent()