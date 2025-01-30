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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
# import tensorboard

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
        self.position_history = deque(maxlen=20)

    def reset(self):
        self.rob.stop_simulation()
        self.rob.play_simulation()
        time.sleep(0.5)
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

        current_pos = self.rob.get_position()
        self.position_history.append((current_pos.x, current_pos.y))

        ir_raw = self.rob.read_irs()
        ir_raw = [x/1000 if x is not None else 1.0 for x in ir_raw]
        
        back_irs = [ir_raw[6]]
        front_irs = [ir_raw[4], ir_raw[7], ir_raw[5]]

        if any(val > 0.15 for val in back_irs):
            obs = self._compute_observation()
            return obs, -0.0, True, {}

        if not self._should_ignore_front_collision():
            if any(val > 0.25 for val in front_irs):
                obs = self._compute_observation()
                return obs, -0.0, True, {}

        obs = self._compute_observation()
        reward, done = self._compute_reward_and_done(obs)

        if self.episode_step >= self.max_steps:
            done = True

        return obs, reward, done, {}

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

    def _detect_red_areas(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 40, 40])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 40, 40])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
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
            x, y, w, h = cv2.boundingRect(c)
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
        food_pos = self.rob.get_food_position()
        base_pos = self.rob.get_base_position()
        distance = math.sqrt(
            (food_pos.x - base_pos.x)**2 +
            (food_pos.y - base_pos.y)**2
        )
        return distance

    def _distance_robot_to_puck(self):
        robot_pos = self.rob.get_position()
        food_pos = self.rob.get_food_position()
        distance = math.sqrt(
            (robot_pos.x - food_pos.x)**2 +
            (robot_pos.y - food_pos.y)**2
        )
        return distance + 1e-3

    def _should_ignore_front_collision(self):
        puck_close = self._distance_robot_to_puck() < 0.3
        camera_detection = self._detect_red_areas(self.rob.read_image_front()) is not None
        return puck_close or camera_detection

    def _compute_reward_and_done(self, obs):
        # reward = -1
        done = False

        dist_rp = self._distance_robot_to_puck()
        rp_shaping = 10.0 / (1.0 + dist_rp)
        reward += rp_shaping

        puck_box = obs[5:9]
        actual_w = puck_box[2] * self.camera_width
        actual_h = puck_box[3] * self.camera_height
        puck_area = actual_w * actual_h
        
        if puck_area > 100:
            puck_cx = (puck_box[0] + puck_box[2]/2) * self.camera_width
            puck_cy = (puck_box[1] + puck_box[3]/2) * self.camera_height
            img_center_x = self.camera_width/2
            img_center_y = self.camera_height/2
            
            x_offset = abs(puck_cx - img_center_x)/(self.camera_width/2)
            y_offset = abs(puck_cy - img_center_y)/(self.camera_height/2)
            centering = 2.0 * (1.0 - math.sqrt(x_offset**2 + y_offset**2))
            
            area_shaping = (0.4 * min(puck_area/15000.0, 2.0)) + (0.6 * centering * 2.5)
            reward += area_shaping
        else:
            reward -= 0.5

        if self.rob.base_detects_food():
            reward += 500.0
            done = True

        return reward, done

# ----------------------------
# WANDB CALLBACK
# ----------------------------
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_count += 1
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                
                wandb.log({
                    "episode_reward": episode_reward,
                    "episode": self.episode_count,
                    "total_steps": self.num_timesteps
                })
                
                if episode_reward == max(self.episode_rewards):
                    self.model.save("/root/results/best_model_sb3")
                    wandb.save("/root/results/best_model_sb3.zip")
        return True

# ----------------------------
# MAIN TRAINING LOGIC (UPDATED)
# ----------------------------
def train_push_agent():
    rob = SimulationRobobo()
    env = PushEnv(rob)
    env = Monitor(env)
    
    check_env(env, warn=True)
    
    wandb.init(project="push-task", name="PPO_SB3")
    
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
        },
        # tensorboard_log="/root/results/ppo_tensorboard/"  # Now supported
    )
    
    callbacks = [WandbCallback()]
    
    try:
        model.learn(
            total_timesteps=1_000_000,
            callback=callbacks,
            tb_log_name="ppo",
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
        done = False
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
        print(f"Total reward: {total_reward}")
    finally:
        env.close()

if __name__ == "__main__":
    train_push_agent()
    # run_push_agent()