#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
from pickle import FALSE
import random
from select import select

import tensorboard
import gym
import gym_carla
import carla
import numpy as np
from gym_carla.envs.carla_env import CarlaEnv
import pandas as pd

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch import nn
from collections import deque,namedtuple

import glob
import io
import base64
import os
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gym.wrappers import Monitor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class DQN(nn.Module):

    def __init__(self, state_space,action_space):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_space, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25,15),
            nn.ReLU(),
            nn.Linear(15, 8),
            nn.ReLU(),
            nn.Linear(8 ,action_space),
            nn.ReLU()
        )

        self.loss = nn.MSELoss()
        self.learning_rate = 0.001
        self.optimiser = optim.Adam(self.parameters(), self.learning_rate)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self)) 
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)


def update_epsilon(epsilon):
    eps_decay = 0.99975
    epsilon = epsilon*eps_decay
    return epsilon


def select_action(net, state, epsilon):
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')

    sample = random.random()

    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32) # Convert the state to tensor
        net_out = net(state)

    best_action = int(net_out.argmax())
    action_space_dim = net_out.shape[-1]

    if sample < epsilon:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
            action = random.choice(non_optimal_actions)
            
    else:
        action = best_action
            
    net_out.cpu().numpy()

    return action



def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
        
    # Sample from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch)

    # Create tensors for each element of the batch
    states      = torch.tensor([s[0] for s in batch], dtype=torch.float32, device=device)
    actions     = torch.tensor([s[1] for s in batch], dtype=torch.int64, device=device)
    rewards     = torch.tensor([s[3] for s in batch], dtype=torch.float32, device=device)

    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = torch.tensor([s[2] for s in batch if s[2] is not None], dtype=torch.float32, device=device) # the next state can be None if the game has ended
    non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)

    # Compute Q values 
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1).cuda())

    # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
      target_net.eval()
      q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size, device=device)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1)# Set the required tensor shape

    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping 
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()

    return loss

def main():
  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.4, 0.0, 0.4],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.tesla.model3',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'curriculum',  # mode of the task, [random,curriculum ,roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 19,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8.33,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
    'dynamic_weather':False, #Set TRUE for random weather
  }
  env = gym.make('carla-v0', params=params) 
  env.seed(0) # Set a random seed for the environment
  
  state_space_dim = 11
  action_space_dim = env.action_space.n
  # Set random seeds
  torch.manual_seed(0)
  np.random.seed(0)

  gamma = 0.99   #discount factor 
  epsilon = 1
  epsilon_min = 0.01
  replay_memory_capacity = 10000   
  lr = 1e-3 ##Learning rate 
  target_net_update_steps = 10   
  batch_size = 256   
  bad_state_penalty = 0   
  min_samples_for_training = 1000   

  ### Define exploration profile
##  initial_value = 5
  num_iterations = 400
#   exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6) 
#   exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

  # replay memory
  replay_mem = ReplayMemory(replay_memory_capacity)    

  # policy network
  policy_net = DQN(state_space_dim, action_space_dim).to(device)

  # target network with the same weights of the policy network
  target_net = DQN(state_space_dim, action_space_dim).to(device)
  target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

  optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr) # The optimizer will update ONLY the parameters of the policy network

  loss_fn = nn.HuberLoss()  
  #============================================LOOP============================================================================================
  
  from datetime import date
  dis_acc = params.get("discrete_acc")
  dis_steer_ang = params.get("discrete_steer")
  episode_durations=[]
  episode_scores = []
  route_scores=[]
  avg_epsilons = []
  avg_losses = []
  time_steps = 0 
  today  =date.today()
  today_date =  today.strftime("%d/%m/%Y").replace("/", "")
  writer = SummaryWriter(log_dir=f"//home//dh26//Documents//Carla//gym-carla//Models_simple_Curriculum2_{today_date}", flush_secs=30)

  #print(exploration_profile)
  for episode_num in range(num_iterations):  
    print(episode_num)
    obs = env.reset()
    action = 1
    next_state, reward, done, info = env.step(action)
    s= list(next_state.get("state"))
    s_0 = s[0]
    s_1 = s[1]
    s_2 = s[2]
    s_3 = s[3]
    # example [5.23002378e+00 -6.65688906e-01  4.13617835e-08  0.00000000e+00]
    vehicle_front = info.get("vehicle_front")
    position = info.get("position")[0]
    angular_vel = info.get("angular_vel")
    angular_vel_x = angular_vel[0]
    angular_vel_y = angular_vel[1]
    angular_vel_z = angular_vel[2]
    acceleration = info.get("acceleration")
    steer = info.get("steer")
    state = [s_0, s_1, s_2,s_3, vehicle_front, position, angular_vel_x, angular_vel_y, angular_vel_z, acceleration, steer]
    episode_score = 0
    episode_duration = 0
    loss_l = []
    epsilon_l = []
    while done == False:
        epsilon_l.append(epsilon)
        writer.add_scalar("Epsilon_TS", epsilon, time_steps)
        action = select_action(policy_net, state, epsilon)
        if epsilon > epsilon_min:    
            epsilon = update_epsilon(epsilon)
        time_steps +=1
        next_state, reward,done,info = env.step(action)
        episode_score += reward 
        episode_duration += 1
        
        s= next_state.get("state") # example [5.23002378e+00 -6.65688906e-01  4.13617835e-08  0.00000000e+00]
        s_0 = s[0]
        s_1 = s[1]
        s_2 = s[2]
        s_3 = s[3]
        vehicle_front = info.get("vehicle_front")
        position = info.get("position")[0]
        angular_vel = info.get("angular_vel")
        angular_vel_x = angular_vel[0]
        angular_vel_y = angular_vel[1]
        angular_vel_z = angular_vel[2]
        acceleration = info.get("acceleration")
        steer = info.get("steer")
        next_state = [s_0, s_1, s_2, s_3, vehicle_front, position, angular_vel_x, angular_vel_y, angular_vel_z, acceleration, steer]


        # Update the replay memory
        replay_mem.push(state, action, next_state, reward, done)
        
        
        if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
            loss = update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size)
            writer.add_scalar("Loss",loss.item(), time_steps)
            loss_l.append(loss.item())

        if time_steps % 256 == 0:
            print('Updating target network...')
            target_net.load_state_dict(policy_net.state_dict())

        state = next_state


        episode_scores.append(episode_score)
        episode_durations.append(episode_duration)

    route_score = (episode_duration/1000)*100
    print("Route Score = {}%".format(route_score))
    route_scores.append(route_score)

    if episode_num % 256 ==0:
        print('Updating target network...')
        target_net.load_state_dict(policy_net.state_dict())
    
    # if episode_num == 300  or episode_num == 600:
    #     PATH = f"//home//dh26//Documents//Carla//gym-carla//Models//model1_episode_{today_date}_{episode_num}.pth" 
    #     torch.save(policy_net.state_dict(),PATH)
    
    if len(loss_l)>0:
        avg_loss = sum(loss_l)/len(loss_l)
        avg_losses.append(avg_loss)
        writer.add_scalar('Average Loss', avg_loss, episode_num)
    elif len(loss_l) == 0:
        avg_loss = 0
        avg_losses.append(avg_loss)
        writer.add_scalar('Average Loss', avg_loss, episode_num)
    
    if len(epsilon_l)>0:
        avg_epsilon = sum(epsilon_l)/len(epsilon_l)
    elif len(epsilon_l)==0:
        avg_epsilon = 0

    avg_epsilons.append(avg_epsilon)
    # Append episode reward to a list and log stats (every given number of episodes)
    writer.add_scalar("Average_Epsilon", avg_epsilon, episode_num)
    writer.add_scalar('Episode Scores', episode_score, episode_num)
    writer.add_scalar('Episode Duration', episode_duration, episode_num)
    writer.add_scalar('Route Completion %', route_score, episode_num)
    writer.flush()
  writer.close()

  

    # if not episode_num % AGGREGATE_STATS_EVERY or episode == 1:
    #     average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
    #     min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
    #     max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
    #     agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

    #     # Save model, but only when min reward is greater or equal a set value
    #     if min_reward >= MIN_REWARD:
    #         agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

  
#   d = {'Avg. Loss':avg_losses, 'Avg. Epsilon':avg_epsilons, 'Epsiode Score':episode_scores, 'Durations':episode_durations, 'Route Scores':route_scores}  
#   df = pd.DataFrame(d)
#df.to_csv("//home//dh26//Documents//Simple_Model_Random_Metrics2.csv")
  PATH = "//home//dh26//Documents//Carla//gym-carla//Models//Model_simple_Curriculum2.pth" 
  torch.save(policy_net.state_dict(),PATH)        
  env.close()
  #Save
  
  #Load
  #model = torch.load(PATH)
  #model.eval()

if __name__ == '__main__':
    main()
