import numpy as np
import gym
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from robobo_interface import (
    IRobobo,  # Add this to import IRobobo
    SimulationRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
)
from robobo_interface.datatypes import Position, Orientation  # Import required data types

from data_files import FIGURES_DIR, READINGS_DIR

# IRS sensor indices
irs_positions = {
    "BackL": 0,
    "BackR": 1,
    "FrontL": 2,
    "FrontR": 3,
    "FrontC": 4,
    "FrontRR": 5,
    "BackC": 6,
    "FrontLL": 7,
}

class ObstacleAvoidanceEnv(gym.Env):
    """Custom Gym Environment for Robobo obstacle avoidance."""
    def __init__(self):
        super(ObstacleAvoidanceEnv, self).__init__()
        self.robot = SimulationRobobo()
        self.default_time_step = 0.05  # Match the time step from the CoppeliaSim scene

        # Initialize the simulation
        self._initialize_simulation()

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32)  # Wheel speeds
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)  # IRS sensors

    def _initialize_simulation(self):
        """Initialize the simulation environment with default settings."""
        self.robot.stop_simulation()

        # Set simulation time step
        self.robot._sim.setFloatParam(self.robot._sim.floatparam_simulation_time_step, self.default_time_step)
        
        # Set the dynamics engine (1 for Bullet)
        self.robot._sim.setInt32Param(self.robot._sim.intparam_dynamic_engine, 1)

        # Log simulation parameters for debugging
        current_time_step = self.robot._sim.getFloatParam(self.robot._sim.floatparam_simulation_time_step)
        dynamics_engine = self.robot._sim.getInt32Param(self.robot._sim.intparam_dynamic_engine)
        print(f"Simulation time step: {current_time_step}")
        print(f"Dynamics engine: {dynamics_engine}")

        # Start the simulation
        self.robot.play_simulation()


    def reset(self):
        """Reset the environment."""
        self.robot.stop_simulation()
        self._initialize_simulation()

        # Reset the robot's position and orientation
        self.robot.set_position(
            position=Position(x=0, y=0, z=0),
            orientation=Orientation(yaw=0, pitch=0, roll=0),
        )
        return self._get_observation()

    def step(self, action):
        """Take an action, compute reward, and return next state."""
        left_speed, right_speed = action
        self.robot.move_blocking(int(left_speed), int(right_speed), 500)  # Move for 500ms

        obs = self._get_observation()
        collision = self._detect_collision(obs)
        reward = self._calculate_reward(obs, action, collision)

        if collision:
            # Reverse or turn based on which sensor detected the collision
            self._avoid_obstacle(obs)

        done = False  # Episode doesn't terminate automatically
        return obs, reward, done, {}

    def _get_observation(self):
        """Get normalized IRS sensor readings."""
        irs = self.robot.read_irs()
        normalized_irs = np.clip(np.array(irs) / 1000.0, 0, 1)
        return normalized_irs

    def _detect_collision(self, obs):
        """Detect if any sensor exceeds the collision threshold."""
        front_collision = obs[irs_positions["FrontC"]] > self.collision_threshold / 1000.0
        back_collision = obs[irs_positions["BackC"]] > self.collision_threshold / 1000.0
        return front_collision or back_collision

    def _avoid_obstacle(self, obs):
        """Take action to avoid the obstacle based on sensor readings."""
        if obs[irs_positions["FrontC"]] > self.collision_threshold / 1000.0:
            # Detected obstacle in the front, reverse
            self.robot.move_blocking(-20, -20, self.reverse_duration)
        elif obs[irs_positions["BackC"]] > self.collision_threshold / 1000.0:
            # Detected obstacle in the back, move forward
            self.robot.move_blocking(20, 20, self.reverse_duration)

    def _calculate_reward(self, obs, action, collision):
        """Reward function to encourage obstacle avoidance."""
        if collision:
            return -10  # Large penalty for collisions

        forward_motion = (action[0] + action[1]) / 200  # Average speed (normalized)
        proximity_penalty = np.mean(obs[irs_positions["FrontL"]:irs_positions["FrontRR"] + 1])  # Penalty for proximity to obstacles

        reward = forward_motion - proximity_penalty
        return reward

    def render(self, mode="human"):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Stop simulation."""
        self.robot.stop_simulation()


# Training the PPO model
def train_model():
    env = DummyVecEnv([lambda: ObstacleAvoidanceEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(str(READINGS_DIR / "ppo_obstacle_avoidance"))
    print("Model trained and saved!")

# Testing the trained model
def test_model():
    model = PPO.load(str(READINGS_DIR / "ppo_obstacle_avoidance"))
    env = ObstacleAvoidanceEnv()
    obs = env.reset()

    for _ in range(10):
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            print(f"Reward: {reward}, Observation: {obs}")
        obs = env.reset()

def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())

if __name__ == "__main__":
    train_model()  # Uncomment to train
    # test_model()  # Uncomment to test
