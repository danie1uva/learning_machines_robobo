import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import PowerTransformer
import joblib
from robobo_interface import SimulationRobobo
from robobo_interface.datatypes import Position, Orientation
from datetime import datetime
import wandb
import time
import sys
sys.setrecursionlimit(1000000)  # Default is typically 1000


# week_2_sim.py
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

# PPO Hyperparameters
LEARNING_RATE = 0.0003
GAMMA = 0.99
LAMBDA = 0.95
EPSILON_CLIP = 0.1
ENTROPY_BETA = 0.05
EPISODES = 500
MAX_STEPS = 75
BATCH_SIZE = 64
N_EPOCHS = 4

SCALER_PATH = 'software_powertrans_scaler.gz'
if not os.path.exists(SCALER_PATH):
    print("Scaler not found. Generating a new scaler.")
    simulated_irs_data = np.random.rand(1000, 8) * 10
    scaler = PowerTransformer()
    scaler.fit(simulated_irs_data)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
else:
    print(f"Loading existing scaler from {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mean = nn.Linear(64, output_dim)
        self.fc3_std = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc3_mean(x)) * 100
        std = torch.exp(self.fc3_std(x).clamp(-2, 2))
        return mean, std

class PPOValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PPOValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

visited_states = set()

def compute_reward(next_state, action, collision):
    global visited_states
    left_speed, right_speed = action

    # Compute movement magnitude
    movement_magnitude = abs(left_speed) + abs(right_speed)

    # Base rewards
    movement_reward = 50
    speed_reward = movement_magnitude / 2
    smoothness_reward = -abs(left_speed - right_speed)
    collision_penalty = -500 if collision else 0

    # Penalize very small movements
    small_movement_penalty = -50 if movement_magnitude < 20 else 0  # Adjust threshold as needed

    # Reward for exploring new states
    state_hash = tuple(next_state.round(2))  # Hash state with reduced precision
    exploration_reward = 20 if state_hash not in visited_states else 0
    visited_states.add(state_hash)

    reward = (
        movement_reward +
        2 * speed_reward +
        0.05 * smoothness_reward +
        collision_penalty +
        exploration_reward +
        small_movement_penalty
    )
    print(f"Reward components: Movement: {movement_reward}, Speed: {speed_reward}, "
          f"Smoothness: {smoothness_reward}, Collision Penalty: {collision_penalty}, "
          f"Exploration Reward: {exploration_reward}, Small Movement Penalty: {small_movement_penalty}, Total: {reward}")
    return reward



def check_collision(state):
    # Adjust thresholds for normalized sensor values
    coll_FrontLL = state[0] > 200
    coll_FrontL = state[1] > 200
    coll_FrontC = state[2] > 200
    coll_FrontR = state[3] > 200
    coll_FrontRR = state[4] > 200
    coll_BackL = state[5] > 200
    coll_BackC = state[6] > 200
    coll_BackR = state[7] > 200

    collision = any([coll_FrontLL, coll_FrontL, coll_FrontC, coll_FrontR, coll_FrontRR, coll_BackL, coll_BackC, coll_BackR])
    print(f"Normalized sensor readings: {state}, Collision: {collision}")
    return collision

def preprocess_state(scaler, rob):
    irs = rob.read_irs()
    # Normalize the sensor readings using the scaler
    irs_scaled = scaler.transform([irs])[0].tolist()
    return np.array(irs_scaled, dtype=np.float32)

def compute_advantages(rewards, values, dones):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + GAMMA * values[i + 1] * (1 - dones[i]) - values[i]
        gae = delta + GAMMA * LAMBDA * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages

def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, 
               states, actions, log_probs, rewards, advantages):
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.float32)
    log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1)
    advantages = torch.tensor(np.array(advantages), dtype=torch.float32).unsqueeze(-1)

    for _ in range(N_EPOCHS):
        for i in range(0, len(states), BATCH_SIZE):
            idx = slice(i, i + BATCH_SIZE)
            s = states[idx]
            a = actions[idx]
            lp = log_probs[idx]
            adv = advantages[idx]
            r = rewards[idx]

            mean, std = policy_net(s)
            exploration_factor = max(0.1, 1.0 - len(states) / EPISODES)  # Decay exploration over time
            scaled_std = std * exploration_factor

            dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(scaled_std))
            new_log_probs = dist.log_prob(a)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - lp)
            surrogate1 = ratios * adv
            surrogate2 = torch.clamp(ratios, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) * adv

            policy_loss = -torch.min(surrogate1, surrogate2).mean() - ENTROPY_BETA * entropy
            value_loss = nn.MSELoss()(value_net(s), r)

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

def run_ppo_training(rob: SimulationRobobo):
    obs_dim = 8
    act_dim = 2

    policy_net = PPOPolicyNetwork(obs_dim, act_dim)
    value_net = PPOValueNetwork(obs_dim)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)

    wandb.init(project="learning_machines", config={"episodes": EPISODES, "max_steps": MAX_STEPS})

    for episode in range(EPISODES):
        rob.play_simulation()

        # Extract the initial position and orientation from the environment
        initial_position = rob.get_position()
        initial_orientation = rob.get_orientation()

        print(f"Episode {episode}: Initial position: {initial_position}, Initial orientation: {initial_orientation}")

        # Set the robot to its initial position and orientation (if needed)
        rob.set_position(initial_position, initial_orientation)

        states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []
        state = preprocess_state(scaler, rob)
        done = False

        total_reward = 0  # Track the total reward for the episode

        for t in range(MAX_STEPS):
            mean, std = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            exploration_factor = max(0.1, 1.0 - episode / EPISODES)  # Reduce exploration over episodes
            scaled_std = std * exploration_factor

            dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(scaled_std))
            action = dist.sample()
            log_prob = dist.log_prob(action)

            left_speed, right_speed = map(int, action.squeeze().tolist())

            # Clip actions to ensure smooth movement
            left_speed = max(min(left_speed, 100), -100)
            right_speed = max(min(right_speed, 100), -100)
            # if abs(left_speed) + abs(right_speed) < 20:  # Minimum movement threshold
            #     left_speed = np.sign(left_speed) * 20
            #     right_speed = np.sign(right_speed) * 20

            rob.move_blocking(left_speed, right_speed, 250)

            next_state = preprocess_state(scaler, rob)
            collision = check_collision(next_state)
            reward = compute_reward(next_state, action.squeeze().detach().numpy(), collision)

            states.append(state)
            actions.append(action.detach().numpy())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            dones.append(done)
            values.append(value_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item())

            total_reward += reward  # Accumulate reward

            state = next_state

            if collision:
                done = True
                break

        # Log the total reward, round length, and other metrics for the episode
        print(f"Total reward for episode {episode}: {total_reward}")
        wandb.log({
            "episode": episode,
            "total_reward": total_reward,
            "round_length": t + 1,  # Number of steps in the round
            "loss": policy_loss.item() if 'policy_loss' in locals() else 0  # Placeholder for loss if not available
        })

        values.append(0 if done else value_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item())
        advantages = compute_advantages(rewards, values, dones)

        ppo_update(policy_net, value_net, optimizer_policy, optimizer_value,
                states, actions, log_probs, rewards, advantages)

        print(f"Episode {episode} completed")
        rob.stop_simulation()

    torch.save(policy_net.state_dict(), "ppo_policy.pth")
    torch.save(value_net.state_dict(), "ppo_value.pth")
    print("Training complete!")