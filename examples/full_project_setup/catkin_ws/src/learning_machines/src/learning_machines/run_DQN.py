import random
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import joblib
import os
import wandb
from datetime import datetime

from robobo_interface import (
    IRobobo,
    SimulationRobobo,
)
from robobo_interface.datatypes import (
    Position,
    Orientation,
)

def get_epsilon(it, num_rounds_till_plateau=5000):
    return max(0.05, 1 - it * 0.95 / num_rounds_till_plateau)

class QNetwork(nn.Module):
    def __init__(self, num_hidden=128):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(8, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)  

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x))  
        return x * 100  

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
                return np.random.uniform(0, 100, size=2) 
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                return self.Q(state_tensor).squeeze(0).cpu().numpy()  

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def compute_q_vals(Q, states, actions):
    q_values = Q(states)
    return q_values.gather(1, actions)

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

    predicted_actions = Q(states)  
    targets = compute_targets(Q, rewards, next_states, dones, discount_factor)

    loss = F.mse_loss(predicted_actions, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def check_collision(state_readings): 
    coll_FrontLL = state_readings[0] > 1  
    coll_FrontL = state_readings[1] > .4
    coll_FrontC = state_readings[2] > 2.0
    coll_FrontR = state_readings[3] > .6
    coll_FrontRR = state_readings[4] > 1.1
    coll_BackL = state_readings[5] > 5.6
    coll_BackC = state_readings[6] > 2.2
    coll_BackR = state_readings[7] > 2.3

    if any([coll_FrontLL, coll_FrontL, coll_FrontC, coll_FrontR, coll_FrontRR, coll_BackL, coll_BackC, coll_BackR]):
        return True
    else:
        return False

def get_current_state(scaler, irs):
    front_sensors, back_sensors = scale_and_return_ordered(scaler, irs)
    return front_sensors + back_sensors

def scale_and_return_ordered(scaler, irs):
    irs = scaler.transform([irs])[0].tolist()
    front_sensors = [irs[7], irs[2], irs[4], irs[3], irs[5]]
    back_sensors = [irs[0], irs[6], irs[1]]
    return front_sensors, back_sensors

def move_robobo_and_calc_reward(scaler, action, rob, state):
    left_speed, right_speed = action  
    movement = [left_speed, right_speed, 250]
    rob.move_blocking(*movement)

    movement_reward = 50

    speed_reward = (abs(left_speed) + abs(right_speed)) / 2

    smoothness_reward = -abs(left_speed - right_speed)

    collision = check_collision(state)

    collision_penalty = -500 if collision else 0


    move_reward = (
        movement_reward
        + 2 * speed_reward
        + .5 * smoothness_reward
        + collision_penalty
    )

    log_entry = {
        'speed_reward': speed_reward,
        'smoothness_reward': smoothness_reward,
        'collision_penalty': collision_penalty,
        'move_reward': move_reward
    }

    next_state = get_current_state(scaler, rob.read_irs())

    return log_entry, collision, next_state

def run_qlearning_classification(rob: IRobobo):
    print('connected')

    num_hidden = 256
    learning_rate = 0.001
    discount_factor = 0.9
    batch_size = 32
    memory_capacity = 1000
    episodes = 5000 
    max_steps = 75
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_explore_rate = .05
    num_steps_till_plateau = 5000
    run_name = f"{current_date}_hidden{num_hidden}_lr{learning_rate}_gamma{discount_factor}_bs{batch_size}_mem{memory_capacity}_eps{episodes}_steps{max_steps}_epsil{final_explore_rate}_rounds_of_exp{num_steps_till_plateau}"

    wandb.init(
        project="learning_machines",
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
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_capacity)
    policy = EpsilonGreedyPolicy(Q, epsilon=final_explore_rate) 
    scaler = joblib.load('software_powertrans_scaler.gz')


    for round in range(episodes):  
        print(f"Round: {round}")

        rob.play_simulation()
        
        if round % 20 == 0:
            validation = True
            pos = Position(-0.875,-0.098,0)
            ori = Orientation(-90, -27, -90)
            rob.set_position(pos, ori)
        
        
        state = get_current_state(scaler, rob.read_irs())
        total_reward = 0
        round_length = 0 

        for step in range(max_steps):
            eps = get_epsilon(step, num_steps_till_plateau)
            policy.set_epsilon(eps)
            action = policy.sample_action(state)

            log_entry, collision, next_state = move_robobo_and_calc_reward(scaler, action, rob, state)
            done = collision or (step == max_steps - 1)

            memory.push((state, action, log_entry['move_reward'], next_state, done))
            loss = train(Q, memory, optimizer, batch_size, discount_factor)


            total_reward += log_entry['move_reward']
            round_length += 1
            state = next_state
            
            # per step logs 
            wandb.log({
                "move_reward": log_entry['move_reward'],
                "loss": loss,
                "round": round,
                "step": step,
                'speed_reward': log_entry['speed_reward'],
                'smoothness_reward': log_entry['smoothness_reward'],
                'collision_penalty': log_entry['collision_penalty'],
                'collision': collision
                })
            
            if collision:
                print(f"Collision detected! Ending episode {round} early with penalty.")
                break


        # episode logs 
        wandb.log({"total_reward": total_reward,
                    "round": round,
                    "round_length": round_length,
                    "validation": validation
                    }) 
        
        validation = False
        
        print(f"Total Reward = {total_reward}")
        rob.stop_simulation()

    # Save the model
    model_path = f"{run_name}_trained_qnetwork.pth"
    torch.save(Q.state_dict(), model_path)
    wandb.save(model_path)  
    print(f"Model saved to {model_path}")
    print(f"Run details available at: {wandb.run.url}")