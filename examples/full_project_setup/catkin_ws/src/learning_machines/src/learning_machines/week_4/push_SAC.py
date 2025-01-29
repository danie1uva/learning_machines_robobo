import wandb
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from collections import deque

from .coppelia_env_push_SAC import CoppeliaSimEnv  # Ensure correct import path
from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo

import wandb
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

class PerformanceBasedRandomizationCallback(BaseCallback):
    """
    Callback to adjust the randomization frequency of the environment based on performance.
    - Monitors the last 100 episodes.
    - Adjusts randomization frequency as follows:
        - No randomization until >=50 successes in last 100 episodes.
        - Then, randomize every 5 episodes until >=70 successes.
        - Then, randomize every 2 episodes until >=80 successes.
        - Stop training once >=80 successes in the last 100 episodes.
    - Logs episodic rewards and lengths to wandb.
    """
    def __init__(self, env, verbose=0):
        super().__init__(verbose=verbose)
        self.env = env
        self.success_history = deque(maxlen=100)
        self.current_stage = 0  # 0: No randomization, 1: Every 5, 2: Every 2
        self.stages = [
            {'success_rate': 50, 'frequency': 5},
            {'success_rate': 70, 'frequency': 2},
            {'success_rate': 80, 'frequency': 2}  # Final stage to stop training
        ]

    def _on_step(self) -> bool:
        # Check if an episode has finished
        infos = self.locals.get('infos', [])
        for info in infos:
            # Monitor handles only one info per step in most cases
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                success = info.get('success', False)

                # Append success to history
                self.success_history.append(success)

                # Calculate current success rate
                success_count = sum(self.success_history)
                success_rate = (success_count / len(self.success_history)) * 100 if len(self.success_history) > 0 else 0

                # Log episodic metrics to wandb
                wandb.log({
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'success': success,
                    'success_rate': success_rate
                })

                # Determine the appropriate stage based on success rate
                for idx, stage in enumerate(self.stages):
                    if success_rate >= stage['success_rate']:
                        if self.current_stage < idx + 1:
                            self.current_stage = idx + 1
                            new_frequency = stage['frequency']
                            self.env.set_randomize_frequency(new_frequency)
                            wandb.log({
                                'randomize_frequency': new_frequency,
                                'current_stage': self.current_stage,
                                'success_rate': success_rate
                            })
                            print(f"Stage {self.current_stage} reached: success_rate={success_rate:.2f}%, "
                                  f"randomize_frequency set to {new_frequency}")

                # Check if final stage is achieved to stop training
                if self.current_stage >= len(self.stages) and success_rate >= self.stages[-1]['success_rate']:
                    print(f"Final stage achieved: success_rate={success_rate:.2f}%. Stopping training.")
                    return False  # Returning False stops training

        return True  # Continue training

def train_sac_dynamic_randomization(rob: IRobobo):
    """
    Trains a SAC agent with dynamic randomization frequency based on performance.
    :param rob: Instance of the robot interface.
    """
    # Initialize Weights & Biases
    wandb.init(
        project="coppelia_sac_dynamic_randomization",
        config={
            "algo": "SAC-DynamicRandomization",
            "learning_rate": 3e-4,
            "buffer_size": 50000,
            "batch_size": 256,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": "auto",
            "max_grad_norm": 0.5,
            "total_timesteps": 300000,  # Initial total timesteps; may stop earlier based on performance
            "run_date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        },
        name=f"sac_dynamic_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    rob.play_simulation()

    # Initialize Environment
    env = CoppeliaSimEnv(
        rob=rob,
        randomize_frequency=0,  # Start with no randomization
        puck_pos_range=0.4
    )
    env = Monitor(env)

    # Initialize SAC Model
    model = SAC(
        policy=MlpPolicy,
        env=env,
        learning_rate=wandb.config.learning_rate,
        buffer_size=wandb.config.buffer_size,
        batch_size=wandb.config.batch_size,
        gamma=wandb.config.gamma,
        tau=wandb.config.tau,
        ent_coef=wandb.config.ent_coef,
        verbose=1
    )

    # Initialize Callback
    randomization_callback = PerformanceBasedRandomizationCallback(env=env)

    # Start Training
    model.learn(
        total_timesteps=wandb.config.total_timesteps,
        callback=randomization_callback
    )

    rob.stop_simulation()

    # Save and log the model
    model_save_name = f"sac_dynamic_model_{wandb.run.name}"
    model.save(model_save_name)
    wandb.save(f"{model_save_name}.zip")
    print("Training complete. Model saved and logged to wandb.")

def run_sac_evaluation(rob: IRobobo, model_path: str):
    """
    Evaluates a trained SAC model on the environment.
    :param rob: Instance of the robot interface.
    :param model_path: Path to the saved SAC model.
    """
    wandb.init(
        project="coppelia_sac_dynamic_randomization_eval",
        name=f"sac_eval_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # Initialize Environment
    env = CoppeliaSimEnv(
        rob=rob,
        randomize_frequency=5,  # Example frequency; adjust as needed
        puck_pos_range=0.4
    )
    env = Monitor(env)

    # Load the trained SAC model
    model = SAC.load(model_path, env=env)

    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        # Use deterministic actions for evaluation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    print(f"Evaluation complete. Total Reward: {total_reward}")
    wandb.log({"evaluation_total_reward": total_reward})
