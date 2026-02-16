import wandb
from datetime import datetime
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from .coppelia_env_push_PPO import CoppeliaSimEnv
from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo

class WandbLoggingCallback(BaseCallback):
    """
    Logs episode reward/length and environment info to wandb.
    """
    def __init__(self, stage, verbose=0):
        super().__init__(verbose=verbose)
        self.stage = stage

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                wandb.log({
                    f"stage_{self.stage}/episode_reward": info["episode"]["r"],
                    f"stage_{self.stage}/episode_length": info["episode"]["l"],
                    "total_timesteps": self.model.num_timesteps
                })
        return True

hyperparams = {
    "algo": "PPO-2phase",
    "learning_rate": 3e-4,
    "n_steps": 2064,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "total_timesteps_stage1": 60000,
    "total_timesteps_stage2": 100000,
    "run_date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
}

def train_ppo_two_stage(rob: IRobobo):
    wandb.init(
        project="coppelia_ppo_two_stage",
        config=hyperparams,
        name=f"ppo_run_{hyperparams['run_date']}"
    )

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # -------------- STAGE 1 --------------
    print("\n--- TRAINING STAGE 1: REACH PUCK ---")
    stage1_env = CoppeliaSimEnv(
        rob,
        stage=1,
        randomize_frequency=5,     # randomise every 5 episode
        robot_ori_range=45,
        puck_pos_range=0.5
    )
    stage1_env = Monitor(stage1_env)

    model = PPO(
        policy=MlpPolicy,
        env=stage1_env,
        learning_rate=wandb.config.learning_rate,
        n_steps=wandb.config.n_steps,
        batch_size=wandb.config.batch_size,
        gamma=wandb.config.gamma,
        gae_lambda=wandb.config.gae_lambda,
        clip_range=wandb.config.clip_range,
        ent_coef=wandb.config.ent_coef,
        vf_coef=wandb.config.vf_coef,
        max_grad_norm=wandb.config.max_grad_norm,
        verbose=1
    )

    stage1_callback = WandbLoggingCallback(stage=1)
    model.learn(total_timesteps=wandb.config.total_timesteps_stage1, callback=stage1_callback)

    # -------------- STAGE 2 --------------
    print("\n--- TRAINING STAGE 2: PUSH PUCK TO GOAL ---")
    stage2_env = CoppeliaSimEnv(
        rob,
        stage=2,
        randomize_frequency=2,
        robot_ori_range=45,
        puck_pos_range=0.5
    )
    stage2_env = Monitor(stage2_env)

    model.set_env(stage2_env)
    stage2_callback = WandbLoggingCallback(stage=2)
    model.learn(total_timesteps=wandb.config.total_timesteps_stage2, callback=stage2_callback)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    # Save final model
    model_save_name = f"ppo_2stage_model_{wandb.run.name}"
    model.save(model_save_name)
    wandb.save(f"{model_save_name}.zip")
    print("Two-stage training complete. Model saved and uploaded to wandb.")

def run_ppo_stage2(rob: IRobobo, model_path: str):
    wandb.init(
        project="coppelia_ppo_two_stage_eval",
        name=f"run_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # If running in sim, start it
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # Evaluate in Stage 2 environment
    env = CoppeliaSimEnv(rob, stage=2)
    env = Monitor(env)

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    print("Stage 2 evaluation complete.")