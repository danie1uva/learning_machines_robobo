import wandb
import gym
import torch
import numpy as np
from datetime import datetime
import time 

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Make sure this import points to your custom environment script
# containing the environment with 360 sweep and step-limit logic.
from .coppelia_env_forage import CoppeliaSimEnv, CoppeliaSimEnvHardware

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo


class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        """
        Called after each environment step in model.learn().
        The Monitor wrapper adds 'episode' info to 'info' dict at the end of an episode.
        """
        for info in self.locals["infos"]:
            if "episode" in info:
                # 'episode' has 'r' (cumulative reward) and 'l' (length)
                wandb.log({
                    "episode_reward": info["episode"]["r"],
                    "episode_length": info["episode"]["l"],
                    "total_timesteps": self.model.num_timesteps
                })
        return True


# Hyperparameters for demonstration; tune as necessary
hyperparams = {
    "algo": "DQN",
    "environment": "CoppeliaSimEnv",
    "learning_rate": 1e-3,          # Reduced to stabilize learning
    "buffer_size": 10000,           # Reduced to match smaller training window
    "learning_starts": 500,         # Keep same; ensures initial exploration
    "batch_size": 64,               # No change; balances memory usage and stability
    "gamma": 0.95,                  # Slightly increased for better long-term reward estimation
    "train_freq": 4,                # More frequent updates to align training with stabilization
    "target_update_interval": 500, # Increase for smoother target network updates
    "exploration_fraction": 0.1,    # Reduce exploration period to reflect faster convergence
    "exploration_final_eps": 0.02,  # Reduce final exploration for better exploitation
    "total_timesteps": 15000,       # Reduced to match stabilization window
    "num_initial_boxes": 7,         # Same; depends on task complexity
    "run_date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
}

def train_dqn_forage(rob: IRobobo):
    """
    Trains a DQN agent on the custom CoppeliaSim Env using Stable Baselines 3,
    logs hyperparameters to Wandb, and saves the final model.
    """
    wandb.init(
        project="forage_dqn",
        config=hyperparams,
        name=hyperparams["run_date"] 
    )

    # Start or resume the simulation, if your robobo interface requires it
    rob.play_simulation()

    # Create the environment with the known number of boxes
    env = CoppeliaSimEnv(rob, num_initial_boxes=wandb.config.num_initial_boxes)
    env = Monitor(env)  # Monitor for logging episodes and rewards

    # Create the DQN model
    model = DQN(
        policy=MlpPolicy,
        env=env,
        learning_rate=wandb.config.learning_rate,
        buffer_size=wandb.config.buffer_size,
        learning_starts=wandb.config.learning_starts,
        batch_size=wandb.config.batch_size,
        gamma=wandb.config.gamma,
        train_freq=wandb.config.train_freq,
        target_update_interval=wandb.config.target_update_interval,
        exploration_fraction=wandb.config.exploration_fraction,
        exploration_final_eps=wandb.config.exploration_final_eps,
        verbose=1
    )

    # WandB callback for logging
    callback = WandbLoggingCallback()

    # Train the model
    model.learn(total_timesteps=wandb.config.total_timesteps, callback=callback)

    # Stop simulation if your interface requires it
    rob.stop_simulation()

    # Save the model locally, then upload to wandb
    model_save_name = f"dqn_model_{wandb.run.name}"
    model.save(model_save_name)
    wandb.save(f"{model_save_name}.zip")

    print("Training complete. Model saved and uploaded to WandB.")


def run_dqn_forage(rob: IRobobo, model_weights_path: str):
    """
    Loads a trained DQN model and runs it in the custom CoppeliaSim Env.
    """
    wandb.init(
        project="my_coppelia_project",
        config=hyperparams,
        name=f"run_{hyperparams['run_date']}" 
    )

    setting = "sim" if isinstance(rob, SimulationRobobo) else "hardware"
    while True: 
        if setting == "sim":
            rob.play_simulation()

        if setting == "sim":
            env = CoppeliaSimEnv(rob, num_initial_boxes=wandb.config.num_initial_boxes)
        else:
            env = CoppeliaSimEnvHardware(rob, num_initial_boxes=wandb.config.num_initial_boxes)

        # Build a model shell to load weights into
        model = DQN(
            policy=MlpPolicy,
            env=env,
            learning_rate=wandb.config.learning_rate,
            buffer_size=wandb.config.buffer_size,
            learning_starts=wandb.config.learning_starts,
            batch_size=wandb.config.batch_size,
            gamma=wandb.config.gamma,
            train_freq=wandb.config.train_freq,
            target_update_interval=wandb.config.target_update_interval,
            exploration_fraction=wandb.config.exploration_fraction,
            exploration_final_eps=wandb.config.exploration_final_eps,
            verbose=1
        )

        # Load the trained weights (PyTorch state_dict or SB3 .zip file)
        # If loading a stable_baselines3 .zip model:
        #   model = DQN.load(model_weights_path, env=env)
        # If loading only the policy state_dict:
        model.policy.load_state_dict(torch.load(model_weights_path, map_location=torch.device("cpu")))

        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
        
        if setting == "sim":
            rob.stop_simulation()

