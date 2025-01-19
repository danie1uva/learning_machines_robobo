import wandb
import gym
import torch
import numpy as np
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from robobo_interface import IRobobo

from .coppelia_env import CoppeliaSimEnv 

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

def train_dqn_with_coppeliasim(rob: IRobobo):
    """
    Trains a DQN agent on the custom CoppeliaSim Env using Stable Baselines 3,
    logs hyperparameters to wandb, and saves the final model as .pth.
    """

    hyperparams = {
        "algo": "DQN",
        "environment": "CoppeliaSimEnv",
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 32,
        "gamma": 0.90,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.1,
        "total_timesteps": 20000,
        "run_date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }

    wandb.init(
        project="my_coppelia_project",
        config=hyperparams,
        name=hyperparams["run_date"] 
    )

    rob.play_simulation()
    env = CoppeliaSimEnv(rob)
    env = Monitor(env)

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

    callback = WandbLoggingCallback()

    model.learn(total_timesteps=wandb.config.total_timesteps, callback=callback)

    rob.stop_simulation()

    model.save(f"dqn_model_{wandb.run.name}")
    wandb.save(f"dqn_model_{wandb.run.name}.zip") 

    print("Training complete. Model uploaded to wandb.")