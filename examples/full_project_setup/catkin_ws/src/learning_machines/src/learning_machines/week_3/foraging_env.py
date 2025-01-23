# week_3/foraging_env.py
import gym
import numpy as np
import cv2
import time
from gym import spaces
from robobo_interface import IRobobo

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
        self.package_count = 0  # Packages in current episode
        self.total_packages = 0  # Total collected across all episodes
        self.max_steps = 500
        self.current_step = 0
        self.episode_start_time = None
        self.simulation_reset_interval = 50  # Reset simulation every N episodes

    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.package_count = 0
        self.episode_start_time = time.time()
        
        # Full simulation reset every N episodes
        if self.episode_count % self.simulation_reset_interval == 0:
            self.rob.stop_simulation()
            self.rob.play_simulation()
            self.total_packages = 0  # Reset total count on simulation reset
            print(f"\n=== Simulation Reset ===")
            print(f"Fresh environment with new packages")
        
        self.episode_count += 1
        
        print(f"\nStarting Episode {self.episode_count}")
        print(f"Total packages collected so far: {self.total_packages}")
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
        reward, package_found = self._calculate_reward(obs)
        
        # Check for done condition
        collision = self._check_collision(obs['irs'])
        done |= collision
        
        info = {
            "episode_duration": time.time() - self.episode_start_time,
            "total_packages": self.total_packages,
            "collision": collision,
            "package_found": package_found
        }
        
        # Print immediate feedback
        if package_found:
            print(f"🎁 Package collected! (Total: {self.total_packages})")
        if collision:
            print("💥 Collision detected!")
            
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
        """Custom reward function with detailed tracking"""
        reward = 0
        package_found = False
        
        if self._detect_package(obs['image']):
            reward += 10
            self.package_count += 1
            self.total_packages += 1
            package_found = True
            
        # Time penalty
        reward -= 0.1
        
        if self._check_collision(obs['irs']):
            reward -= 2
            
        return reward, package_found

    def _detect_package(self, image):
        """Detect green packages using CV with validation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        count = cv2.countNonZero(mask)
        
        # Validation check
        if count > 100:
            return True
        return False

    def _check_collision(self, irs):
        """Check for collision using IR sensors with validation"""
        collision_detected = any(s > 0.8 for s in irs)
        return collision_detected