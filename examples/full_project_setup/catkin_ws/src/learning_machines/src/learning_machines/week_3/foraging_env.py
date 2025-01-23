# week_3/foraging_env.py
import gym
import numpy as np
import cv2
import time
from gym import spaces
from robobo_interface import IRobobo, SimulationRobobo

class ForagingEnv(gym.Env):
    """Custom Gym environment for accurate foraging task"""
    
    def __init__(self, rob: IRobobo):
        super().__init__()
        self.rob = rob
        self.action_space = spaces.Discrete(6)  # Added backward movement
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'irs': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        })
        
        # Package tracking
        self.initial_package_count = 8  # Should match your arena setup
        self.current_packages = self.initial_package_count
        self.total_packages_collected = 0
        
        # Episode tracking
        self.episode_count = 0
        self.max_steps = 500
        self.current_step = 0
        self.episode_start_time = None
        self.simulation_reset_interval = 50  # Reset every 50 episodes

    def reset(self):
        """Reset environment with proper package tracking"""
        self.current_step = 0
        self.current_packages = self._get_true_package_count()
        self.episode_start_time = time.time()
        
        # Full simulation reset
        if self.episode_count % self.simulation_reset_interval == 0:
            if isinstance(self.rob, SimulationRobobo):
                self.rob.stop_simulation()
                self.rob.play_simulation()
                self.total_packages_collected = 0  # Reset total count
            print(f"\n=== ENVIRONMENT RESET ===")
            print(f"New episode starts with {self.initial_package_count} packages")
        
        self.episode_count += 1
        print(f"\nStarting Episode {self.episode_count}")
        print(f"Packages remaining: {self.current_packages}")
        print(f"Total collected: {self.total_packages_collected}")
        return self._get_observation()

    def _get_observation(self):
        """Get current environment state"""
        return {
            'image': self._process_image(self.rob.read_image_front()),
            'irs': self._process_irs(self.rob.read_irs())
        }

    def _process_irs(self, irs):
        """Normalize IR sensor readings"""
        return np.clip([irs[7], irs[2], irs[4], irs[3], irs[5]], 0, 3000) / 3000

    def _process_image(self, image):
        """Resize camera image"""
        return cv2.resize(image, (64, 64))

    def step(self, action):
        """Execute action with proper package verification"""
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Store previous package count
        previous_count = self.current_packages
        
        # Execute action
        self._take_action(action)
        
        # Get new state
        obs = self._get_observation()
        
        # Calculate reward
        reward = 0
        self.current_packages = self._get_true_package_count()
        new_collections = previous_count - self.current_packages
        
        if new_collections > 0:
            self.total_packages_collected += new_collections
            reward += 10 * new_collections
            print(f"🎁 Collected {new_collections} package(s)! Total: {self.total_packages_collected}")
        
        # Collision detection
        collision = self._check_collision(obs['irs'])
        if collision:
            reward -= 2
            print("💥 Collision detected!")
            done = True
            
        # Time penalty
        reward -= 0.1
        
        # Episode completion
        done |= self.current_packages == 0
        
        info = {
            "episode_duration": time.time() - self.episode_start_time,
            "packages_remaining": self.current_packages,
            "total_collected": self.total_packages_collected
        }
        
        return obs, reward, done, info

    def _take_action(self, action):
        """Enhanced action map with backward movement"""
        action_map = {
            0: (-25, 50, 150),   # Left
            1: (0, 75, 150),     # Left-forward
            2: (0, 0, 100),      # Forward
            3: (75, 0, 150),     # Right-forward
            4: (50, -25, 150),   # Right
            5: (-100, -100, 200) # Backward
        }
        self.rob.move_blocking(*action_map[action])

    def _get_true_package_count(self):
        """Get actual package count using simulation API"""
        if isinstance(self.rob, SimulationRobobo):
            return self.rob.get_nr_food_collected()
        # Hardware implementation would need different logic
        return self.initial_package_count

    def _check_collision(self, irs):
        """Improved collision detection"""
        return any(s > 0.8 for s in irs)