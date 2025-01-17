import random
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import joblib
import os
import wandb
from datetime import datetime
from collections import namedtuple

from robobo_interface import (
    IRobobo,
    SimulationRobobo,
)
from robobo_interface.datatypes import (
    Position,
    Orientation,
)

# ---------- PPO Hyperparameters ----------
PPOClipParam = 0.2
PPOEpochs = 10
ValueLossCoef = 0.5
EntropyCoef = 0.05
Gamma = 0.95
LambdaGAE = 0.95
MaxGradientNorm = 0.5

CollisionPenalty = -800

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'log_prob', 'value', 'reward', 'done', 'next_state')
)

class ActorCritic(nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(8, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.actor_mean = nn.Linear(num_hidden, 2)
        self.actor_log_std = nn.Parameter(torch.zeros(2))
        self.critic = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def get_action_and_value(self, state):
        x = self.forward(state)
        raw_mean = self.actor_mean(x)

        mean = 100.0 * torch.tanh(raw_mean / 100.0)
        raw_std = F.softplus(self.actor_log_std)
        std = torch.clamp(raw_std, 1e-3, 3.0)

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value = self.critic(x)
        return action, log_prob, value

    def evaluate_actions(self, state, action):
        x = self.forward(state)
        raw_mean = self.actor_mean(x)
        mean = 100.0 * torch.tanh(raw_mean / 100.0)

        raw_std = F.softplus(self.actor_log_std)
        std = torch.clamp(raw_std, 1e-3, 3.0)

        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1).mean()
        value = self.critic(x)
        return log_prob, entropy, value

class PPOMemory:
    def __init__(self):
        self.storage = []

    def push(self, *args):
        self.storage.append(Transition(*args))

    def clear(self):
        self.storage.clear()

    def __len__(self):
        return len(self.storage)

    def sample(self):
        return Transition(*zip(*self.storage))

def check_collision(state_readings):
    coll_FrontLL = state_readings[0] > 1
    coll_FrontL  = state_readings[1] > .4
    coll_FrontC  = state_readings[2] > 2.0
    coll_FrontR  = state_readings[3] > .6
    coll_FrontRR = state_readings[4] > 1.1
    coll_BackL   = state_readings[5] > 5.6
    coll_BackC   = state_readings[6] > 2.2
    coll_BackR   = state_readings[7] > 2.3
    return any([
        coll_FrontLL, coll_FrontL, coll_FrontC, coll_FrontR,
        coll_FrontRR, coll_BackL, coll_BackC, coll_BackR
    ])

def scale_and_return_ordered(scaler, irs):
    if any(np.isnan(irs)) or any(np.isinf(irs)):
        print("Warning: sensor input is NaN/Inf. Clamping to zero.")
        return [0]*5, [0]*3
    irs = scaler.transform([irs])[0].tolist()
    front_sensors = [irs[7], irs[2], irs[4], irs[3], irs[5]]
    back_sensors  = [irs[0], irs[6], irs[1]]
    return front_sensors, back_sensors

def get_current_state(scaler, irs):
    front_sensors, back_sensors = scale_and_return_ordered(scaler, irs)
    state = np.clip(front_sensors + back_sensors, -1000, 1000)
    return state

def distance_shaping_reward(front_sensors):
    # Negative sum of front sensor readings (bigger readings = closer obstacle)
    return -sum(front_sensors) * 2.0

def move_robobo_and_calc_reward(scaler, action, rob, state):
    left_speed, right_speed = action
    movement = [left_speed.item(), right_speed.item(), 250]
    rob.move_blocking(*movement)

    movement_reward = 50
    speed_reward = (abs(left_speed) + abs(right_speed)) / 2
    smoothness_reward = -abs(left_speed - right_speed)
    collision = check_collision(state)
    collision_penalty = CollisionPenalty if collision else 0
    low_speed_penalty = -3 if (abs(left_speed) + abs(right_speed)) < 3.0 else 0

    front_sensors = state[:5]  # [LL, L, C, R, RR]
    shaping = distance_shaping_reward(front_sensors)

    move_reward = (
        movement_reward
        + 2 * speed_reward
        + 0.5 * smoothness_reward
        + collision_penalty
        + low_speed_penalty
        + shaping
    )

    next_state = get_current_state(scaler, rob.read_irs())
    log_entry = {
        'speed_reward': speed_reward.item(),
        'smoothness_reward': smoothness_reward.item(),
        'collision_penalty': collision_penalty,
        'move_reward': move_reward
    }
    return log_entry, collision, next_state

def ppo_update(agent, optimiser, memory, batch_size=32):
    transitions = memory.sample()

    states = torch.tensor(np.array(transitions.state,      dtype=np.float32))
    actions = torch.tensor(np.array(transitions.action,     dtype=np.float32))
    old_log_probs = torch.tensor(np.array(transitions.log_prob, dtype=np.float32)).unsqueeze(-1)
    values = torch.tensor(np.array(transitions.value,      dtype=np.float32)).unsqueeze(-1)
    rewards = torch.tensor(np.array(transitions.reward,     dtype=np.float32)).unsqueeze(-1)
    dones = torch.tensor(np.array(transitions.done,       dtype=np.float32)).unsqueeze(-1)
    next_states = torch.tensor(np.array(transitions.next_state, dtype=np.float32))

    T = len(states)
    if T < batch_size:
        return

    with torch.no_grad():
        _, _, next_value = agent.get_action_and_value(next_states)
    next_value = next_value.squeeze(-1)

    values = values.squeeze(-1)
    rewards = rewards.squeeze(-1)
    dones = dones.squeeze(-1)

    # GAE
    advantages = []
    gae = 0.0
    for i in reversed(range(T)):
        if dones[i] == 1.0:
            delta = rewards[i] - values[i]
            gae = delta
        else:
            delta = rewards[i] + Gamma * next_value[i] - values[i]
            gae = delta + Gamma * LambdaGAE * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = values + advantages

    idxs = np.arange(T)
    for _ in range(PPOEpochs):
        np.random.shuffle(idxs)
        for start in range(0, T, batch_size):
            end = start + batch_size
            batch_idxs = idxs[start:end]

            batch_states     = states[batch_idxs]
            batch_actions    = actions[batch_idxs]
            batch_old_log_p  = old_log_probs[batch_idxs]
            batch_returns    = returns[batch_idxs].unsqueeze(-1)
            batch_advantages = advantages[batch_idxs].unsqueeze(-1)

            adv_std = batch_advantages.std()
            if adv_std < 1e-6:
                continue
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (adv_std + 1e-8)

            log_probs, entropy, state_values = agent.evaluate_actions(batch_states, batch_actions)
            ratio = (log_probs - batch_old_log_p).exp()

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - PPOClipParam, 1.0 + PPOClipParam) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(state_values, batch_returns)
            loss = actor_loss + ValueLossCoef * critic_loss - EntropyCoef * entropy

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MaxGradientNorm)
            optimiser.step()

def run_ppo(rob: IRobobo):
    print('connected')

    learning_rate = 5e-4
    num_hidden = 256
    batch_size = 32
    episodes = 5000
    max_steps = 75

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (
        f"{current_date}_PPO_hidden{num_hidden}_lr{learning_rate}_"
        f"gamma{Gamma}_bs{batch_size}_eps{episodes}_steps{max_steps}"
    )

    wandb.init(
        project="learning_machines",
        name=run_name,
        config={
            "num_hidden": num_hidden,
            "learning_rate": learning_rate,
            "discount_factor": Gamma,
            "batch_size": batch_size,
            "episodes": episodes,
            "max_steps": max_steps,
            "date": current_date,
        }
    )

    agent = ActorCritic(num_hidden=num_hidden)
    optimiser = optim.Adam(agent.parameters(), lr=learning_rate)
    memory = PPOMemory()
    scaler = joblib.load('software_powertrans_scaler.gz')

    for episode in range(episodes):
        print(f"Episode: {episode}")
        rob.play_simulation()

        # Always set to the same position/orientation
        validation = False
        pos = Position(-0.875, -0.098, 0)
        ori = Orientation(-90, -27, -90)
        rob.set_position(pos, ori)

        state = get_current_state(scaler, rob.read_irs())
        total_reward = 0

        for step in range(max_steps):
            st_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = agent.get_action_and_value(st_tensor)

            action_np = action.squeeze(0).numpy()
            log_prob_f = log_prob.item()
            value_f = value.item()

            log_entry, collision, next_state = move_robobo_and_calc_reward(
                scaler, action.squeeze(0), rob, state
            )
            done = collision or (step == max_steps - 1)
            total_reward += log_entry['move_reward']

            memory.push(
                state, action_np, log_prob_f, value_f,
                log_entry['move_reward'], float(done), next_state
            )

            wandb.log({
                "move_reward": log_entry['move_reward'],
                "episode": episode,
                "step": step,
                'speed_reward': log_entry['speed_reward'],
                'smoothness_reward': log_entry['smoothness_reward'],
                'collision_penalty': log_entry['collision_penalty'],
                'collision': collision
            })

            state = next_state
            if done:
                print(f"Ending episode {episode} at step {step}, total_reward={total_reward}")
                break

        wandb.log({
            "total_reward": total_reward,
            "episode": episode,
            "validation": validation
        })

        rob.stop_simulation()

        ppo_update(agent, optimiser, memory, batch_size=batch_size)
        memory.clear()

    model_path = f"{run_name}_ppo_actor_critic.pth"
    torch.save(agent.state_dict(), model_path)
    wandb.save(model_path)
    print(f"Model saved to {model_path}")
    print(f"Run details available at: {wandb.run.url}")
