import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from datetime import datetime
import wandb

from robobo_interface import (
    IRobobo,
    SimulationRobobo,
)

from robobo_interface.datatypes import (
    Position,
    Orientation,
)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.l1 = nn.Linear(5, num_hidden)
        self.l2 = nn.Linear(num_hidden, 5)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return random.sample(self.memory, len(self.memory))
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Epsilon schedule (global step-based)
def get_epsilon(total_steps, num_steps_till_plateau=1000):
    return max(0.05, 1.0 - (total_steps * 0.95) / num_steps_till_plateau)

# Epsilon-Greedy Policy
class EpsilonGreedyPolicy:
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        with torch.no_grad():
            if np.random.rand() < self.epsilon:
                action = np.random.randint(5)
            else:
                q_vals = self.Q(torch.tensor(obs).float())
                action = q_vals.argmax().item()
        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

# Computes Q-values for the state-action pairs in the batch
def compute_q_vals(Q, states, actions):
    q_actions = Q(states)
    return q_actions.gather(1, actions)

# Computes target values using a target network
def compute_targets(Q_target, rewards, next_states, dones, discount_factor):
    q_vals_next = Q_target(next_states)
    max_q_vals, _ = q_vals_next.max(dim=1, keepdim=True)
    dones = dones.to(dtype=torch.float32)
    targets = rewards + (1 - dones) * discount_factor * max_q_vals
    return targets

# Single training step
def train(Q, Q_target, memory, optimizer, batch_size, discount_factor):
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.tensor(states, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.int64)[:, None]
    next_states = torch.tensor(next_states, dtype=torch.float)
    rewards = torch.tensor(rewards, dtype=torch.float)[:, None]
    dones = torch.tensor(dones, dtype=torch.uint8)[:, None]

    q_val = compute_q_vals(Q, states, actions)
    with torch.no_grad():
        target = compute_targets(Q_target, rewards, next_states, dones, discount_factor)

    loss = F.smooth_l1_loss(q_val, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Check collision
def check_collision(state):
    return max(state) > 500

# Process IR sensors
def process_irs(irs):
    return [irs[7], irs[2], irs[4], irs[3], irs[5]]

# Convert action index to movement command
def determine_action(action_idx):
    if action_idx == 0:    # left
        return [0, 50, 250]
    elif action_idx == 1:  # left-forward
        return [25, 50, 250]
    elif action_idx == 2:  # forward
        return [50, 50, 250]
    elif action_idx == 3:  # right-forward
        return [50, 25, 250]
    else:                  # right
        return [50, 0, 250]

# Moves robot and calculates reward
def move_robobo_and_calc_reward(action, rob, state):
    rob.move_blocking(*action)
    movement_reward = 2
    collision = check_collision(state)
    collision_penalty = -5 if collision else 0
    proximity_penalty = -2 if max(state) > 200 else 0
    move_reward = movement_reward + collision_penalty + proximity_penalty

    log_entry = {
        'proximity_penalty': proximity_penalty,
        'collision_penalty': collision_penalty,
        'move_reward': move_reward
    }
    next_state = process_irs(rob.read_irs())
    return log_entry, collision, next_state

# Main Q-learning function
def run_qlearning_classification(rob: IRobobo):
    print('connected')
    num_hidden = 128
    learning_rate = 0.001
    discount_factor = 0.95
    batch_size = 32
    memory_capacity = 5000
    episodes = 100
    max_steps = 75
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_explore_rate = 0.05
    num_steps_till_plateau = 5000
    run_name = f"{current_date}_hidden{num_hidden}_lr{learning_rate}_gamma{discount_factor}_bs{batch_size}_mem{memory_capacity}_eps{episodes}_steps{max_steps}"

    wandb.init(
        project="learning_machines_DQN",
        name=run_name,
        config={
            "num_hidden": num_hidden,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "batch_size": batch_size,
            "memory_capacity": memory_capacity,
            "episodes": episodes,
            "max_steps": max_steps,
            "date": current_date,
            "final_explore_rate": final_explore_rate,
            "num_steps_till_plateau": num_steps_till_plateau
        }
    )

    Q = QNetwork(num_hidden=num_hidden)
    Q_target = QNetwork(num_hidden=num_hidden)
    Q_target.load_state_dict(Q.state_dict())
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_capacity)
    policy = EpsilonGreedyPolicy(Q, final_explore_rate)

    total_steps = 0
    update_target_interval = 50

    for episode in range(episodes):
        print(f"Episode: {episode}")
        rob.play_simulation()

        raw_sensor_readings = rob.read_irs()
        state = process_irs(raw_sensor_readings)

        total_reward = 0
        episode_length = 0

        for step in range(max_steps):
            # Epsilon depends on total_steps, not local step
            eps = get_epsilon(total_steps, num_steps_till_plateau)
            policy.set_epsilon(eps)

            action_idx = policy.sample_action(state)
            action_cmd = determine_action(action_idx)
            log_entry, collision, next_state = move_robobo_and_calc_reward(action_cmd, rob, state)
            done = collision or (step == max_steps - 1)

            memory.push((state, action_idx, log_entry['move_reward'], next_state, done))
            loss = train(Q, Q_target, memory, optimizer, batch_size, discount_factor)

            total_reward += log_entry['move_reward']
            episode_length += 1
            state = next_state
            total_steps += 1

            # Update target network periodically
            if total_steps % update_target_interval == 0:
                Q_target.load_state_dict(Q.state_dict())

            # Logging
            wandb.log({
                "move_reward": log_entry['move_reward'],
                "loss": loss,
                "episode": episode,
                "step": step,
                "total_steps": total_steps,
                "proximity_penalty": log_entry['proximity_penalty'],
                "collision_penalty": log_entry['collision_penalty'],
                "collision": collision,
                "proximity_to_obstacle": max(state)
            })

            if collision:
                print(f"Collision detected. Ending episode {episode} early.")
                break

        wandb.log({
            "total_reward": total_reward,
            "episode": episode,
            "episode_length": episode_length
        })

        print(f"Total Reward = {total_reward}")
        rob.stop_simulation()

    # Save the model
    model_path = f"{run_name}_trained_qnetwork.pth"
    torch.save(Q.state_dict(), model_path)
    wandb.save(model_path)
    print(f"Model saved to {model_path}")
    print(f"Run details available at: {wandb.run.url}")
