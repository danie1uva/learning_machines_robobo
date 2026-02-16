import wandb
from datetime import datetime
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Make sure this import points to your updated environment script
# with the discrete action space (7 actions).
from .coppelia_env_push_DQN import CoppeliaSimEnvDQN

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo


class WandbLoggingCallback(BaseCallback):
    """
    Logs episode reward/length and environment info to wandb.
    """
    def __init__(self, stage, verbose=0):
        super().__init__(verbose=verbose)
        self.stage = stage

    def _on_step(self) -> bool:
        # 'infos' typically has "episode" dict at the end of each episode if Monitor is used
        for info in self.locals["infos"]:
            if "episode" in info:
                wandb.log({
                    f"stage_{self.stage}/episode_reward": info["episode"]["r"],
                    f"stage_{self.stage}/episode_length": info["episode"]["l"],
                    "total_timesteps": self.model.num_timesteps
                })
        return True


# Example DQN hyperparameters
hyperparams = {
    "algo": "DQN-2phase",
    "learning_rate": 1e-4,
    "buffer_size": 20000,
    "learning_starts": 500,
    "batch_size": 64,
    "gamma": 0.99,
    "train_freq": 4,
    "target_update_interval": 500,
    "exploration_fraction": 0.2,
    "exploration_final_eps": 0.02,
    "total_timesteps_stage1": 40000,
    "total_timesteps_stage2": 60000,
    "run_date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
}


def train_dqn_two_stage(rob: IRobobo):
    """
    Train a DQN agent in two stages:
      1) Stage 1: Learn to reach the puck
      2) Stage 2: Learn to push the puck into the green zone
    Logs results to wandb. Saves the final model as a zip.
    """
    wandb.init(
        project="coppelia_dqn_two_stage",
        config=hyperparams,
        name=f"dqn_run_{hyperparams['run_date']}"
    )

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # ----------------- STAGE 1: REACH PUCK -----------------
    print("\n--- TRAINING STAGE 1: REACH PUCK ---")
    stage1_env = CoppeliaSimEnvDQN(
        rob,
        stage=1,
        randomize_frequency=5,   # randomise every 5 episodes
        robot_ori_range=45,
        puck_pos_range=0.5
    )
    stage1_env = Monitor(stage1_env)

    model = DQN(
        policy=MlpPolicy,
        env=stage1_env,
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

    stage1_callback = WandbLoggingCallback(stage=1)
    model.learn(total_timesteps=wandb.config.total_timesteps_stage1, callback=stage1_callback)

    # ----------------- STAGE 2: PUSH PUCK -----------------
    print("\n--- TRAINING STAGE 2: PUSH PUCK TO GOAL ---")
    stage2_env = CoppeliaSimEnvDQN(
        rob,
        stage=2,
        randomize_frequency=2,
        robot_ori_range=45,
        puck_pos_range=0.5
    )
    stage2_env = Monitor(stage2_env)

    # Reuse the same model, but set a new environment
    model.set_env(stage2_env)

    stage2_callback = WandbLoggingCallback(stage=2)
    model.learn(total_timesteps=wandb.config.total_timesteps_stage2, callback=stage2_callback)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    # Save final model
    model_save_name = f"dqn_2stage_model_{wandb.run.name}"
    model.save(model_save_name)
    wandb.save(f"{model_save_name}.zip")
    print("Two-stage DQN training complete. Model saved and uploaded to wandb.")


def run_dqn_stage2(rob: IRobobo, model_path: str):
    """
    Load a trained DQN model (presumably after finishing Stage 2)
    and run one episode in Stage 2 environment for evaluation.
    """
    wandb.init(
        project="coppelia_dqn_two_stage_eval",
        name=f"dqn_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # Evaluate in Stage 2 environment
    env = CoppeliaSimEnv(rob, stage=2)
    env = Monitor(env)

    # Load the trained DQN model
    model = DQN.load(model_path, env=env)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    print("Stage 2 evaluation with DQN complete.")
