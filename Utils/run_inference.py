import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from collections import deque
import random
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from cluttere_v8 import ClutteredSceneEnv
from utils_env import compute_features
from typing import Dict, Union, Tuple

import argparse
import datetime
import gym
import numpy as np
import itertools
from sac import SAC
from replay_memory import ReplayMemory
import torch
import numpy as np
from model import QNetwork
# Define function to load the critic model from a checkpoint
def load_critic(critic, checkpoint_path):
    # Load the checkpoint file
    checkpoint = torch.load(checkpoint_path)
    critic.load_state_dict(checkpoint['critic_state_dict'])
    print("Critic model successfully loaded from:", checkpoint_path)
    return critic

# Function to compute argmax action using the critic
def select_best_action(critic, state, env):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    state = state.view(state.size(0), -1)  # Flatten state if necessary (matches forward pass)
    # Store Q-values for each action
    q_values = []
    for action in env.object_ids:  # Iterate over all possible actions
        # Convert action to a PyTorch tensor and expand dimensions
        action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0)  # Batch dimension
        # Pass state and action through the critic
        q1, q2 = critic(state, action_tensor)
        # Take the minimum of Q1 and Q2 to handle overestimation bias
        q_value = torch.min(q1, q2).item()
        # Append the Q-value
        q_values.append(q_value)
    # Find the action with the maximum Q-value
    best_action_index = np.argmax(q_values)
    best_action = env.object_ids[best_action_index]
    return best_action

env = ClutteredSceneEnv(num_cuboids=10, headless=False)
critic = QNetwork(4,10,256)
checkpoint_path = "/home/navin/projects/M2P/Re_M2P/checkpoints/sac_checkpoint_clutter_2000"  # Replace with the actual checkpoint path
critic = load_critic(critic, checkpoint_path)

state = env.reset()
episode_reward = 0
done = False
t=0
while not done:
    t+=1
    action = select_best_action(critic,state,env)
    next_state, reward, done, _ = env.step(action)
    import pdb;pdb.set_trace()
    episode_reward += reward
    state = next_state
#avg_reward += episode_reward