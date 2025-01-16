import random
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from datetime import datetime
import joblib
import os
import time
import pickle
from data_files import FIGURES_DIR, READINGS_DIR, RESULTS_DIR

from robobo_interface import (
    IRobobo,
    SimulationRobobo,
)

from robobo_interface.datatypes import (
    Position,
    Orientation,
)

def get_epsilon(it):
    return max(0.05, 1 - it * 0.95 / 1000)

class QNetwork(nn.Module):
    def __init__(self, num_hidden=128):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(5, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)  # Two outputs for left and right wheels

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))  # Output in [-1, 1]
        return x * 100  # Scale to [-100, 100]

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.pop(0)
            self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)

class EpsilonGreedyPolicy:
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon

    def sample_action(self, state):
        with torch.no_grad():
            if np.random.rand() < self.epsilon:
                return np.random.uniform(-100, 100, size=2)  # Random wheel speeds
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                return self.Q(state_tensor).squeeze(0).cpu().numpy()  # Predicted wheel speeds

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def compute_targets(Q, rewards, next_states, dones, discount_factor):
    next_actions = Q(next_states).detach()
    rewards = rewards.expand_as(next_actions)  
    dones = dones.float()
    targets = rewards + (1 - dones) * discount_factor * next_actions

    return targets 


def train(Q, memory, optimizer, batch_size, discount_factor):
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.float32) 
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

    q_values = Q(states)
    targets = compute_targets(Q, rewards, next_states, dones, discount_factor)

    loss = F.mse_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def get_current_state(scaler, irs):
    front_sensors, _ = scale_and_return_ordered(scaler, irs)
    return front_sensors

def scale_and_return_ordered(scaler, irs):
    irs = scaler.transform([irs])[0].tolist()
    front_sensors = [irs[7], irs[2], irs[4], irs[3], irs[5]]
    back_sensors = [irs[0], irs[6], irs[1]]
    return front_sensors, back_sensors

def eucl_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def move_robobo_and_calc_reward(scaler, action, rob, state):
    left_speed, right_speed = action  
    movement = [left_speed, right_speed, 250]
    rob.move_blocking(*movement)

    speed_reward = (abs(left_speed) + abs(right_speed)) / 2

    smoothness_reward = -abs(left_speed - right_speed)

    max_sensor_reading = max(state)
    collision = max_sensor_reading > 0.9
    collision_penalty = -50 if collision else 0

    if left_speed > 0 and right_speed > 0:
        forward_reward = 5 * min(left_speed, right_speed) / 10  # Encourage forward 
    # elif left_speed < 0 and right_speed < 0:
    #     forward_reward = -2  # penalise backward motion
    else:
        forward_reward = 0

    total_reward = 2 * speed_reward + smoothness_reward + forward_reward + collision_penalty

    log_entry = {
        'speed_reward': speed_reward,
        'smoothness_reward': smoothness_reward,
        'forward_reward': forward_reward,
        'collision_penalty': collision_penalty,
        'max_sensor_reading': max_sensor_reading,
        'total_reward': total_reward,
        'collision': collision
    }

    next_state = get_current_state(scaler, rob.read_irs())

    return total_reward, log_entry, collision, next_state

def run_qlearning_classification(rob: IRobobo):
    print('connected')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #Results directory, logging
    log_dir = RESULTS_DIR / "week2" / "logs" #check if this is the right path
    os.makedirs(log_dir, exist_ok=True)
    model_dir = RESULTS_DIR / "week2" / "models"
    logs_path = os.path.join(log_dir, f"training_logs_{timestamp}.pkl")

    num_hidden = 128
    learning_rate = 0.001
    discount_factor = 0.9
    batch_size = 32
    memory_capacity = 10000

    Q = QNetwork(num_hidden=num_hidden)
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_capacity)
    policy = EpsilonGreedyPolicy(Q, epsilon=0.05)  # Set epsilon to 0.05 for reduced exploration

    scaler = joblib.load('software_powertrans_scaler.gz')
    rob.play_simulation()

    # Track progress
    loss_log = []
    round_rewards = []
    reward_logs = []

    for round in range(10): 
        print(f"Round: {round}")
        rob.play_simulation()
        state = get_current_state(scaler, rob.read_irs())
        total_reward = 0
        round_reward_log = []

        for step in range(400): 
            eps = get_epsilon(step)
            policy.set_epsilon(eps)
            action = policy.sample_action(state)

            reward, log_entry, collision, next_state = move_robobo_and_calc_reward(scaler, action, rob, state)
            round_reward_log.append(log_entry)
            done = collision or (step == 199)

            memory.push((state, action, reward, next_state, done))
            loss = train(Q, memory, optimizer, batch_size, discount_factor)

            if loss is not None:
                loss_log.append(loss)

            total_reward += reward
            state = next_state

            if collision:
                print(f"Collision detected! Ending episode {round} early with penalty.")
                break


        round_rewards.append(total_reward)
        reward_logs.append(round_reward_log)
        print(f"Round {round}: Total Reward = {total_reward}")
        rob.stop_simulation()
        time.sleep(.5)
    
    #use pickle to log
    logs = {
        "loss_log": loss_log,
        "round_rewards": round_rewards,
        "reward_logs": reward_logs
    }
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)

    
    model_path = os.path.join(model_dir,f"trained_qnetwork_{timestamp}.pth")
    # logs_path = f"training_logs_{timestamp}.npz"

    torch.save(Q.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    #logs_path = "training_logs.npz"
    #np.savez(logs_path, loss_log=loss_log, round_rewards=round_rewards, reward_logs=reward_logs)
    

    print(f"Training logs saved to {logs_path}")
