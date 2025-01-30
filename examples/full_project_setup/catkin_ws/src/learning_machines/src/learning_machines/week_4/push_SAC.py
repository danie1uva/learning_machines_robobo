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

class PerformanceBasedRandomizationCallback(BaseCallback):
    """
    Adjusts the randomization frequency based on performance,
    and requires at least 100 episodes at each stage
    before the next stage can unlock.

    - Monitors the last 100 episodes' success flags.
    - Stages:
        0) No randomization => success_rate >= 50 => Stage 1
        1) randomize_frequency=2 => success_rate >= 80 => Stage 2
        2) randomize_frequency=1 => success_rate >= 90 => Stop training
    - Each time a new stage is reached:
        - success_history is reset
        - episodes_since_stage_update is reset
        - randomize_frequency is updated
    - We only check for stage transitions if:
        - success_rate >= required rate
        - episodes_since_stage_update >= 100
    """

    def __init__(self, env, verbose=0):
        super().__init__(verbose=verbose)
        self.env = env
        self.success_history = deque(maxlen=100)
        self.episodes_since_stage_update = 0
        self.current_stage = 0  # Index in self.stages
        self.stages = [
            {'success_rate': 50, 'frequency': 1},  # after 50% success, randomize every 2 episodes
            {'success_rate': 80, 'frequency': 0},  # after 80% success, randomize every episode
            {'success_rate': 90, 'frequency': 0},  # after 90% success => final stage => training stops
        ]

    def _init_callback(self) -> None:
        # Called once when the training starts
        self.episodes_since_stage_update = 0
        self.current_stage = 0
        self.success_history.clear()  # Start fresh

    def _on_step(self) -> bool:
        # The Monitor wrapper provides 'episode' data in infos when an episode ends
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                success = info.get('success', False)

                # Additional metrics from environment info
                puck_collected_in_episode = info.get('puck_collected_in_episode', False)
                steps_to_puck_coll = info.get('steps_to_puck_collection', None)
                steps_to_goal = info.get('steps_to_final_goal', None)

                # Log the base episode metrics
                wandb.log({
                    'episode_reward': episode_reward,
                    'episode_length': episode_length,
                    'success': success
                })

                # Log puck-collection metrics
                wandb.log({
                    'puck_collected_in_episode': puck_collected_in_episode
                })
                if steps_to_puck_coll is not None:
                    wandb.log({'steps_to_puck_collection': steps_to_puck_coll})
                if steps_to_goal is not None:
                    wandb.log({'steps_to_final_goal': steps_to_goal})

                # Push success into rolling buffer
                self.success_history.append(success)

                # Compute success rate
                success_rate = (sum(self.success_history) / len(self.success_history)) * 100
                wandb.log({'success_rate': success_rate})

                # Another episode done in the current stage
                self.episodes_since_stage_update += 1

                # Check if we can move to next stage
                if self.current_stage < len(self.stages):
                    # Are we above threshold for the current stage?
                    required_rate = self.stages[self.current_stage]['success_rate']
                    if success_rate >= required_rate and self.episodes_since_stage_update >= 100:
                        # Transition to the next stage
                        new_frequency = self.stages[self.current_stage]['frequency']
                        self.env.set_randomize_frequency(new_frequency)
                        wandb.log({
                            'randomize_frequency': new_frequency,
                            'current_stage': self.current_stage + 1,
                            'success_rate': success_rate
                        })
                        print(f"Stage {self.current_stage + 1} reached: "
                              f"success_rate={success_rate:.2f}%, randomize_frequency={new_frequency}")

                        # Move to next stage
                        self.current_stage += 1
                        self.episodes_since_stage_update = 0
                        self.success_history.clear()  # reset rolling buffer
                    # If that was the final stage, check if we should stop
                    if self.current_stage >= len(self.stages):
                        # We are beyond the last stage => training can stop
                        print(f"Final stage: success_rate={success_rate:.2f}%. Stopping training.")
                        return False  # stop training

        return True  # Keep training

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
