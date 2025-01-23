# week_3/foraging_env.py
import gym
import numpy as np
import cv2
import time
from gym import spaces
from robobo_interface import IRobobo
from robobo_interface import (
    IRobobo,
    SimulationRobobo,
    HardwareRobobo,
)

class ForagingEnv(gym.Env):
    """Custom Gym environment for foraging task"""
    
    def __init__(self, rob: IRobobo):
        super().__init__()
        self.rob = rob
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'irs': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        })
        
        self.episode_count = 0
        self.package_count = 0
        self.max_steps = 500
        self.current_step = 0
        self.episode_start_time = None

    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.package_count = 0
        self.episode_start_time = time.time()
        
        if self.episode_count % 50 == 0:
            self.rob.stop_simulation()
            self.rob.play_simulation()
            
        self.episode_count += 1
        
        return self._get_observation()

    def _get_observation(self):
        """Process sensors and camera input"""
        irs = self._process_irs(self.rob.read_irs())
        image = self._process_image(self.rob.read_image_front())
        return {'image': image, 'irs': irs}

    def _process_irs(self, irs):
        """Process IR sensor readings"""
        processed = [irs[7], irs[2], irs[4], irs[3], irs[5]]
        return np.clip(processed, 0, 3000) / 3000

    def _process_image(self, image):
        """Resize and normalize image"""
        resized = cv2.resize(image, (64, 64))
        return resized

    def step(self, action):
        """Execute one action step"""
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Execute action
        self._take_action(action)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs)
        
        # Check for done condition
        done |= self._check_collision(obs['irs'])
        
        info = {
            "episode_duration": time.time() - self.episode_start_time,
            "total_packages": self.package_count
        }
        
        return obs, reward, done, info

    def _take_action(self, action):
        """Map action index to movement"""
        action_map = {
            0: (-25, 50, 150),   # Left
            1: (0, 75, 150),     # Left-forward
            2: (0, 0, 100),      # Forward
            3: (75, 0, 150),     # Right-forward
            4: (50, -25, 150)    # Right
        }
        self.rob.move_blocking(*action_map[action])

    def _calculate_reward(self, obs):
        """Custom reward function"""
        reward = 0
        
        # Check for collected package
        if self._detect_package(obs['image']):
            reward += 10
            self.package_count += 1
            
        # Time penalty
        reward -= 0.1
        
        # Collision penalty
        if self._check_collision(obs['irs']):
            reward -= 2
            
        return reward

    def _detect_package(self, image):
        """Detect green packages using CV"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return cv2.countNonZero(mask) > 100

    def _check_collision(self, irs):
        """Check for collision using IR sensors"""
        return any(s > 0.8 for s in irs)