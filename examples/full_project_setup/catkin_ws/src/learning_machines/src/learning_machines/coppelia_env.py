import numpy as np
import gym 
from gym import spaces
import random

from robobo_interface import IRobobo

from robobo_interface.datatypes import Orientation

MAX_SENSOR_VAL = 3000.0  

def clamp_and_normalise(irs):
    """
    Clamps IR readings to [0, MAX_SENSOR_VAL] and normalises to [0, 1].
    """
    clamped = np.clip(irs, 0.0, MAX_SENSOR_VAL)
    return clamped / MAX_SENSOR_VAL

def process_irs(irs):
    """
    Extracts the five relevant IR indices from the Robobo readings.
    Adjust if your sensor mapping is different.
    """
    return [irs[7], irs[2], irs[4], irs[3], irs[5]]

def determine_action(action_idx):
    if action_idx == 0:    # left
        return [0, 75, 300]
    elif action_idx == 1:  # left-forward
        return [0, 75, 150]
    elif action_idx == 2:  # forward
        return [0, 0, 100]
    elif action_idx == 3:  # right-forward
        return [75, 0, 150]
    else:                  # right
        return [75, 0, 300]

def check_collision(state):
    # state is in [0,1] range after normalisation, so compare to 500 / MAX_SENSOR_VAL
    # i.e. 500 => 500 / 3000 = ~0.167
    return max(state) > (500.0 / MAX_SENSOR_VAL)

class CoppeliaSimEnv(gym.Env):
    """
    Example Gym environment interface for CoppeliaSim + Robobo, 
    using normalised sensor values in [0,1].
    """

    def __init__(self, rob: IRobobo):
        """
        Args:
            rob: An instance of your robobo_interface (e.g. SimulationRobobo).
        """
        super().__init__()
        self.rob = rob

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        self.init_pos = rob.get_position()
        self.init_ori = rob.get_orientation()
        self.state = None
        self.done = False

        self.episode_count = 0

    def reset(self):
        """
        Resets the simulation to a known initial state and returns the normalised sensor readings. Every 20 episodes, the 
        robot's pitch is randomly set.
        """

        self.episode_count += 1
        orientation = Orientation(
            roll=self.init_ori.roll,
            pitch=self.init_ori.pitch,
            yaw=self.init_ori.yaw
            )

        if self.episode_count < 10000:
            if self.episode_count % 20 == 0:
                random_int = random.randint(-90, 90)
                orientation.pitch = random_int
        
        elif self.episode_count < 15000 and self.episode_count >= 10000:
            if self.episode_count % 10 == 0:
                random_int = random.randint(-90, 90)
                orientation.pitch = random_int
                
        elif self.episode_count >= 15000:
            if self.episode_count % 5 == 0:
                random_int = random.randint(-90, 90)
                orientation.pitch = random_int

        self.rob.set_position(self.init_pos, orientation)

        irs = self.rob.read_irs()
        raw_state = process_irs(irs)
        self.state = clamp_and_normalise(raw_state)

        self.done = False

        return self.state.astype(np.float32)

    def step(self, action):
        """
        Executes the chosen action, returns (observation, reward, done, info).
        """
        move_cmd = determine_action(action)

        # Perform the move
        self.rob.move_blocking(*move_cmd)
        self.rob.move_blocking(50, 50, 100) 

        # Obtain next state
        irs = self.rob.read_irs()
        raw_next_state = process_irs(irs)
        next_state = clamp_and_normalise(raw_next_state)

        # Compute reward
        movement_reward = 2
        collision = check_collision(next_state)
        collision_penalty = -10 if collision else 0
        proximity_penalty = -2 if max(next_state) > (200.0 / MAX_SENSOR_VAL) else 0
        reward = movement_reward + collision_penalty + proximity_penalty

        # Update internal states
        self.state = next_state
        self.done = collision  # end episode on collision

        info = {}

        return self.state.astype(np.float32), float(reward), self.done, info 
