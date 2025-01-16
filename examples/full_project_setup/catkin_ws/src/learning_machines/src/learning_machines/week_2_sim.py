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
EPSILON_CLIP = 0.2
ENTROPY_BETA = 0.01
EPISODES = 500  # Reduced for testing
MAX_STEPS = 75
BATCH_SIZE = 64
N_EPOCHS = 4

# Ensure scaler exists or create one
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

# Networks
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

# Reward Function
def compute_reward(next_state, action):
    action = np.squeeze(action)  # Remove unnecessary dimensions
    if action.size == 2:  # Two elements
        forward_motion = (action[0] + action[1]) / 200
    elif action.size == 1:  # One element
        forward_motion = action[0] / 200
    else:
        raise ValueError(f"Unexpected action size: {action.size}")
    return forward_motion

# Collision Check
def check_collision(state):
    return any(s > 0.8 for s in state[:8])

# State Preprocessing
def preprocess_state(scaler, rob):
    irs = rob.read_irs()
    irs_scaled = scaler.transform([irs])[0].tolist()
    front_sensors = [irs_scaled[7], irs_scaled[2], irs_scaled[4], irs_scaled[3], irs_scaled[5]]
    back_sensors = [irs_scaled[0], irs_scaled[6], irs_scaled[1]]
    return np.array(front_sensors + back_sensors, dtype=np.float32)

# Advantage Estimation
def compute_advantages(rewards, values, dones):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + GAMMA * values[i + 1] * (1 - dones[i]) - values[i]
        gae = delta + GAMMA * LAMBDA * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return advantages

# PPO Update
def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, 
               states, actions, log_probs, rewards, advantages):
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.float32)
    log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1)  # Add dimension
    advantages = torch.tensor(np.array(advantages), dtype=torch.float32).unsqueeze(-1)  # Add dimension

    for _ in range(N_EPOCHS):
        for i in range(0, len(states), BATCH_SIZE):
            idx = slice(i, i + BATCH_SIZE)
            s = states[idx]
            a = actions[idx]
            lp = log_probs[idx]
            adv = advantages[idx]
            r = rewards[idx]

            mean, std = policy_net(s)
            dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))
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

# PPO Training Loop
def run_ppo_training(rob: SimulationRobobo):
    obs_dim = 8  # 8 IRS sensors
    act_dim = 2  # Left and right wheel speeds

    policy_net = PPOPolicyNetwork(obs_dim, act_dim)
    value_net = PPOValueNetwork(obs_dim)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    optimizer_value = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)

    wandb.init(project="learning_machines", config={"episodes": EPISODES, "max_steps": MAX_STEPS})

    for episode in range(EPISODES):
        rob.play_simulation()
        rob.set_position(Position(0, 0, 0), Orientation(0, 0, 0))

        states, actions, rewards, log_probs, dones, values = [], [], [], [], [], []

        state = preprocess_state(scaler, rob)
        done = False

        for t in range(MAX_STEPS):
            mean, std = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))
            action = dist.sample()
            log_prob = dist.log_prob(action)

            if action.ndim == 1:  # 1D tensor
                left_speed, right_speed = map(int, action.tolist())  # Convert directly to list
            elif action.ndim == 2:  # 2D tensor
                left_speed, right_speed = map(int, action[0].tolist())  # Access first row
            else:
                raise ValueError(f"Unexpected action shape: {action.shape}")



            rob.move_blocking(left_speed, right_speed, 500)

            next_state = preprocess_state(scaler, rob)
            reward = compute_reward(next_state, action.squeeze().detach().numpy())

            states.append(state)
            actions.append(action.detach().numpy())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            dones.append(done)
            values.append(value_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item())

            state = next_state

            if check_collision(next_state):
                done = True
                break

        values.append(0 if done else value_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).item())
        advantages = compute_advantages(rewards, values, dones)

        ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, 
                   states, actions, log_probs, rewards, advantages)

        print(f"Episode {episode} completed")
        rob.stop_simulation()

    torch.save(policy_net.state_dict(), "ppo_policy.pth")
    torch.save(value_net.state_dict(), "ppo_value.pth")
    print("Training complete!")

