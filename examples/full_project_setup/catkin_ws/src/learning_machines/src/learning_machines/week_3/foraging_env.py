# week_3/foraging_env.py
import gym
import numpy as np
import cv2
import time
from gym import spaces
from robobo_interface import IRobobo, SimulationRobobo

class ForagingEnv(gym.Env):
    """Custom Gym environment with full episode management"""
    
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
        self.initial_package_count = 7
        self.total_packages_collected = 0
        self.current_episode_collected = 0
        
        # Episode tracking
        self.episode_count = 0
        self.max_steps = 50  # 75 movements per episode
        self.current_step = 0
        self.episode_start_time = None
        self.simulation_reset_interval = 20

        # Action mapping
        self.action_map = {
            # Forward movements (≥80 speed)
            0: (80, 80, 200),     # Forward
            1: (100, 100, 150),   # Fast forward
            # Backward movements
            2: (-80, -80, 200),   # Backward
            3: (-100, -100, 150), # Fast backward
            # Right turns
            4: (100, 80, 100),    5: (80, 60, 150),    6: (80, 40, 200),
            7: (-80, -100, 100),  8: (-60, -80, 150),
            # Left turns
            9: (80, 100, 100),   10: (60, 80, 150),   11: (40, 80, 200),
            12: (-100, -80, 100),13: (-80, -60, 150),
            # Complex movements
            14: (100, 40, 120),  15: (40, 100, 120),
            16: (-80, -40, 120), 17: (-40, -80, 120),
            18: (20, 80, 150),   19: (80, 20, 150)
        }

    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.current_episode_collected = 0
        self.episode_start_time = time.time()
        
        # Always reset simulation for SimulationRobobo
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
            self.rob.play_simulation()
            time.sleep(1)  # Allow time for simulation reset
            print(f"\n=== SIMULATION RESET ===")
        
        # Reset package counters
        self.current_episode_collected = 0
        if isinstance(self.rob, SimulationRobobo):
            self.total_packages_collected = 0  # Reset total when simulation restarts
        
        self.episode_count += 1
        print(f"\nStarting Episode {self.episode_count}")
        return self._get_observation()

    def _get_observation(self):
        """Get current observation"""
        image = cv2.resize(self.rob.read_image_front(), (64, 64))
        irs = self._process_irs()
        centered, _ = self._detect_centered_package(image)
        return {
            'image': image,
            'irs': irs,
            'proximity': np.array([max(irs[1], irs[2], irs[3])], dtype=np.float32),
            'centered': np.array([centered], dtype=np.float32)
        }

    def _process_irs(self):
        """Process IR sensors into normalized values"""
        raw_irs = self.rob.read_irs()
        return np.clip([raw_irs[7], raw_irs[2], raw_irs[4], raw_irs[3], raw_irs[5]], 0, 3000) / 3000

    def _detect_centered_package(self, image):
        """Detect centered green packages"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return 0.0, None
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        frame_center = image.shape[1] // 2
        box_center = x + w//2
        centered = abs(box_center - frame_center) < image.shape[1] * 0.125
        size_ok = cv2.contourArea(largest)/(image.shape[0]*image.shape[1]) > 0.02
        
        return float(centered and size_ok), (x, y, w, h)

    def step(self, action):
        """Execute action and return new state"""
        self.current_step += 1
        start_collected = self._get_collected_count()
        
        # Execute movement
        self._take_action(action)
        time.sleep(0.5)
        
        obs = self._get_observation()
        reward, done = 0, False
        current_collected = self._get_collected_count()
        new_collections = current_collected - start_collected
        
        # Update package tracking
        if new_collections > 0:
            self.total_packages_collected += new_collections
            self.current_episode_collected += new_collections
            reward += 10 * new_collections
            print(f"🎁 Collected {new_collections} package(s)! Episode total: {self.current_episode_collected}")

        # Collision detection
        action_params = self.action_map[action]
        l_speed, r_speed, _ = action_params
        collision = self._check_collision(obs['irs'], l_speed, r_speed)
        
        if collision:
            reward -= 2
            print("💥 Collision detected!")
            done = True
        else:
            # Movement reward
            if (l_speed > 0 and r_speed > 0) or (l_speed < 0 and r_speed < 0):
                reward += 0.5

        # Additional rewards
        reward += obs['proximity'][0] * 0.2
        reward += obs['centered'][0] * 0.5
        reward -= 0.1  # Time penalty

        # Termination conditions
        done |= self.current_step >= self.max_steps
        done |= self.current_episode_collected >= self.initial_package_count
        
        return obs, reward, done, {
            "duration": time.time() - self.episode_start_time,
            "collected": self.current_episode_collected
        }

    def _take_action(self, action):
        """Execute movement from action map"""
        l_speed, r_speed, duration = self.action_map[action]
        self.rob.move_blocking(l_speed, r_speed, duration)

    def _get_collected_count(self):
        """Get current collection count from simulation"""
        return self.rob.get_nr_food_collected() if isinstance(self.rob, SimulationRobobo) else 0

    def _check_collision(self, irs, l_speed, r_speed):
        """Direction-aware collision detection"""
        if l_speed > 0 and r_speed > 0:  # Forward
            return any(irs[i] > 0.85 for i in [1, 2, 3])
        elif l_speed < 0 and r_speed < 0:  # Backward
            return any(irs[i] > 0.85 for i in [0, 4])
        return any(irs > 0.85)  # Other movements check all sensors