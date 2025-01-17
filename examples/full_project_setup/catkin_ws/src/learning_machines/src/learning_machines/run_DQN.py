import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import datetime as datetime
import wandb


from robobo_interface import (
    IRobobo,
    SimulationRobobo,
)

from robobo_interface.datatypes import (
    Position,
    Orientation,
)

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(3, num_hidden)
        self.l2 = nn.Linear(num_hidden, 5)

    def forward(self, x):
        
        x = F.relu(self.l1(x))
        x = self.l2(x)
        
        return x

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
        if len(self.memory) < batch_size:
            sample = random.sample(self.memory, len(self.memory))
        else:
            sample = random.sample(self.memory, batch_size)
        return sample

    def __len__(self):
        return len(self.memory)

def get_epsilon(it, num_steps_till_plateau = 1000):
    
    if it <= 1000:
        epsilon = 1 - it * 0.95 / num_steps_till_plateau
    else:
        epsilon = 0.05
    
    return epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        
        with torch.no_grad():
            if np.random.rand() < self.epsilon:
                action = np.random.randint(5)
            else:
                action = np.argmax(self.Q(torch.tensor(obs).float())).item()
        return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """

    q_actions = Q(states)
    return q_actions.gather(1, actions) 
    
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of rewards. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    
    q_vals = Q(next_states) 
    max_q_vals, _ = torch.max(q_vals, dim = 1, keepdim = True) 
    dones = dones.to(dtype=torch.float32) 
    targets = rewards + (1-dones) * discount_factor * max_q_vals
    
    return targets
    

def train(Q, memory, optimizer, batch_size, discount_factor):
    

    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    
    state, action, reward, next_state, done = zip(*transitions)
    
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  
    

    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad(): 
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    loss = F.smooth_l1_loss(q_val, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item() 

def check_collision(state): 
    return max(state) > 800 

def process_irs(irs):
    state = [irs[7], irs[4], irs[5]]
    return state 

def determine_action(int):
    '''
    based on the output of the Q-network, determine the action to take
    '''
    if int == 0:
        # left 
        move = [0, 50, 250]
    
    elif int == 1:
        # left-forward
        move = [25, 50, 250]
    
    elif int == 2:
        # forward
        move = [50, 50, 250]

    elif int == 3:
        # right-forward
        move = [50, 25, 250]

    elif int == 4:
        # right
        move = [50, 0, 250]
    
    return move 


def move_robobo_and_calc_reward(action, rob, state):
    
    rob.move_blocking(*action)

    movement_reward = 50
    collision = check_collision(state)
    collision_penalty = -500 if collision else 0
    proximity_penalty = -100 if max(state) > 500 else 0


    move_reward = (
        movement_reward
        + collision_penalty
        + proximity_penalty
    )

    log_entry = {
        'proximity_penalty': proximity_penalty,
        'collision_penalty': collision_penalty,
        'move_reward': move_reward
    }
    
    next_state = process_irs(rob.read_irs())

    return log_entry, collision, next_state

def run_qlearning_classification(rob: IRobobo):
    print('connected')

    num_hidden = 256
    learning_rate = 0.001
    discount_factor = 0.9
    batch_size = 32
    memory_capacity = 1000
    episodes = 100 
    max_steps = 75
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_explore_rate = .05
    num_steps_till_plateau = 5000
    run_name = f"{current_date}_hidden{num_hidden}_lr{learning_rate}_gamma{discount_factor}_bs{batch_size}_mem{memory_capacity}_eps{episodes}_steps{max_steps}_epsil{final_explore_rate}_rounds_of_exp{num_steps_till_plateau}"

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
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_capacity)
    policy = EpsilonGreedyPolicy(Q, final_explore_rate) 


    for round in range(episodes):  
        print(f"Round: {round}")

        rob.play_simulation()
        
        # if round % 20 == 0:
        #     validation = True
        #     pos = Position(-0.875,-0.098,0)
        #     ori = Orientation(-90, -27, -90)
        #     rob.set_position(pos, ori)
        
        raw_sensor_readings = rob.read_irs()
        state = process_irs(raw_sensor_readings)

        total_reward = 0
        round_length = 0 

        for step in range(max_steps):
            eps = get_epsilon(step, num_steps_till_plateau)
            policy.set_epsilon(eps)
            action = policy.sample_action(state)
            action = determine_action(action)
            log_entry, collision, next_state = move_robobo_and_calc_reward(action, rob, state)
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
                'proximity_penalty': log_entry['proximity_penalty'],
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
                    # "validation": validation
                    }) 
        
        # validation = False
        
        print(f"Total Reward = {total_reward}")
        rob.stop_simulation()

    # Save the model
    model_path = f"{run_name}_trained_qnetwork.pth"
    torch.save(Q.state_dict(), model_path)
    wandb.save(model_path)  
    print(f"Model saved to {model_path}")
    print(f"Run details available at: {wandb.run.url}")