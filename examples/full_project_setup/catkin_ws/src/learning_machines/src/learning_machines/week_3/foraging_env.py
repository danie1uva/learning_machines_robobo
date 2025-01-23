# week_3/foraging_env.py
import gym
import numpy as np
import cv2
import time
from gym import spaces
from robobo_interface import IRobobo, SimulationRobobo

class ForagingEnv(gym.Env):
    """Custom Gym environment with reliable package collection"""
    
    def __init__(self, rob: IRobobo):
        super().__init__()
        self.rob = rob
        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'irs': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            'proximity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'centered': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        # Package tracking
        self.initial_package_count = 8  # Set this to match your scene's food count
        self.total_packages_collected = 0
        self.last_collected_count = 0
        
        # Episode tracking
        self.episode_count = 0
        self.max_steps = 500
        self.current_step = 0
        self.episode_start_time = None
        self.simulation_reset_interval = 20

    def reset(self):
        """Reset environment with proper package validation"""
        self.current_step = 0
        self.last_collected_count = 0
        self.episode_start_time = time.time()
        
        if self.episode_count % self.simulation_reset_interval == 0:
            if isinstance(self.rob, SimulationRobobo):
                self.rob.stop_simulation()
                self.rob.play_simulation()
                self.total_packages_collected = 0  # Reset total when environment refreshes
            print(f"\n=== ENVIRONMENT RESET ===")
            print(f"New episode starts with {self.initial_package_count} packages")
        
        self.episode_count += 1
        print(f"\nStarting Episode {self.episode_count}")
        print(f"Total collected packages: {self.total_packages_collected}")
        return self._get_observation()

    def _get_observation(self):
        """Get observation with package verification"""
        image = cv2.resize(self.rob.read_image_front(), (64, 64))
        irs = self._process_irs()
        
        # Package detection
        centered, _ = self._detect_centered_package(image)
        proximity = self._get_proximity(irs)
        
        return {
            'image': image,
            'irs': irs,
            'proximity': np.array([proximity], dtype=np.float32),
            'centered': np.array([centered], dtype=np.float32)
        }

    def _process_irs(self):
        """Process IR sensors with validation"""
        raw_irs = self.rob.read_irs()
        return np.clip([raw_irs[7], raw_irs[2], raw_irs[4], raw_irs[3], raw_irs[5]], 0, 3000) / 3000

    def _get_proximity(self, irs):
        """Calculate front proximity score"""
        return max(irs[1], irs[2], irs[3])

    def _detect_centered_package(self, image):
        """Detect centered green packages with size validation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, None
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Check if centered (middle 25% of frame)
        frame_center = image.shape[1] // 2
        box_center = x + w//2
        centered = abs(box_center - frame_center) < image.shape[1] * 0.125
        
        # Size validation (at least 2% of frame area)
        contour_area = cv2.contourArea(largest)
        size_ok = (contour_area / (image.shape[0] * image.shape[1])) > 0.02
        
        return float(centered and size_ok), (x, y, w, h)

    def step(self, action):
        """Execute action with validated package collection"""
        self.current_step += 1
        
        # Store initial collected count
        start_collected = self._get_collected_count()
        
        # Execute action
        self._take_action(action)
        time.sleep(0.5)  # Allow time for collection
        
        # Get new state
        obs = self._get_observation()
        reward = 0
        done = False
        
        # Verify package collection
        current_collected = self._get_collected_count()
        new_collections = current_collected - start_collected
        
        if new_collections > 0:
            self.total_packages_collected += new_collections
            reward += 10 * new_collections
            print(f"🎁 Collected {new_collections} package(s)! Total: {self.total_packages_collected}")
        
        # Collision penalty
        if self._check_collision(obs['irs']):
            reward -= 2
            print("💥 Collision detected!")
            done = True
            
        # Progressive rewards
        reward += obs['proximity'][0] * 0.2  # Approach bonus
        reward += obs['centered'][0] * 0.5    # Centering bonus
        reward -= 0.1  # Time penalty
        
        # Episode termination
        done |= self.current_step >= self.max_steps
        done |= current_collected >= self.initial_package_count
        
        info = {
            "duration": time.time() - self.episode_start_time,
            "total_collected": self.total_packages_collected,
            "new_collections": new_collections
        }
        
        return obs, reward, done, info

    def _take_action(self, action):
        """20 movement actions with varying speeds and durations"""
        # Speed levels: 40, 60, 80, 100
        # Turn ratios: 0.2, 0.4, 0.6, 0.8, 1.0 (straight)
        action_map = {
            # Forward movements (straight)
            0: (100, 100, 150),   # Full speed forward
            1: (80, 80, 200),     # Fast forward
            2: (60, 60, 250),     # Medium forward
            3: (40, 40, 300),     # Slow forward
            
            # Right turns
            4: (100, 80, 100),    # Sharp right
            5: (80, 60, 150),     # Medium right
            6: (60, 40, 200),     # Gentle right
            7: (100, 60, 80),     # Fast sharp right
            8: (80, 40, 120),     # Fast medium right
            
            # Left turns
            9: (80, 100, 100),    # Sharp left
            10: (60, 80, 150),    # Medium left
            11: (40, 60, 200),    # Gentle left
            12: (60, 100, 80),    # Fast sharp left
            13: (40, 80, 120),    # Fast medium left
            
            # Curved movements
            14: (100, 40, 120),   # Extreme right curve
            15: (40, 100, 120),   # Extreme left curve
            16: (80, 20, 150),    # Right pivot
            17: (20, 80, 150),    # Left pivot
            
            # Precision adjustments
            18: (30, 50, 100),    # Small left adjust
            19: (50, 30, 100),    # Small right adjust
        }
        l_speed, r_speed, duration = action_map[action]
        self.rob.move_blocking(l_speed, r_speed, duration)

    def _get_collected_count(self):
        """Get actual collected count from simulation"""
        if isinstance(self.rob, SimulationRobobo):
            return self.rob.get_nr_food_collected()
        # Hardware implementation would need different logic
        return 0

    def _check_collision(self, irs):
        """Validated collision detection"""
        return any(value > 0.85 for value in irs)