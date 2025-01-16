import numpy as np
import gym
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from data_files import FIGURES_DIR, READINGS_DIR
from robobo_interface import SimulationRobobo

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
        self.robot.play_simulation()

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-100, high=100, shape=(2,), dtype=np.float32)  # Wheel speeds
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)  # IRS sensors

    def reset(self):
        """Reset the environment."""
        self.robot.stop_simulation()
        self.robot.play_simulation()
        self.robot.set_position(position=(0, 0, 0), orientation=(0, 0, 0))
        return self._get_observation()

    def step(self, action):
        """Take an action, compute reward, and return next state."""
        left_speed, right_speed = action
        self.robot.move_blocking(int(left_speed), int(right_speed), 500)  # Move for 500ms

        obs = self._get_observation()
        reward = self._calculate_reward(obs, action)
        done = False  # The episode never "ends" automatically
        return obs, reward, done, {}

    def _get_observation(self):
        """Get normalized IRS sensor readings."""
        irs = self.robot.read_irs()
        normalized_irs = np.clip(np.array(irs) / 1000.0, 0, 1)
        return normalized_irs

    def _calculate_reward(self, obs, action):
        """Reward function to encourage forward motion and avoid obstacles."""
        forward_motion = (action[0] + action[1]) / 200  # Average speed (normalized)
        proximity_penalty = np.mean(obs[irs_positions["FrontL"]:irs_positions["FrontRR"] + 1])
        collision_penalty = 0 if all(o < 0.8 for o in obs) else -1  # Harsh penalty for near collisions

        reward = forward_motion - proximity_penalty + collision_penalty
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

if __name__ == "__main__":
    train_model()  # Uncomment to train
    # test_model()  # Uncomment to test
