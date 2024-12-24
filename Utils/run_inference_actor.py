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
from cluttered_v7 import ClutteredSceneEnv
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
from model import QNetwork, GaussianPolicy
# Define function to load the critic model from a checkpoint
def load_critic(critic, checkpoint_path):
    # Load the checkpoint file
    checkpoint = torch.load(checkpoint_path)
    critic.load_state_dict(checkpoint['policy_state_dict'])
    print("Actor model successfully loaded from:", checkpoint_path)
    return critic

# Function to compute argmax action using the critic
def select_best_action(actor, state):
    state = torch.FloatTensor(state)
    next_state_action, next_state_log_pi = actor.sample(state)
    return next_state_action

env = ClutteredSceneEnv(num_cuboids=10, headless=False)
actor = GaussianPolicy(4,30,256,30)
checkpoint_path = "/home/navin/projects/M2P/Re_M2P/checkpoints/sac_checkpoint_clutter_2500"  # Replace with the actual checkpoint path
actor = load_critic(actor, checkpoint_path)

state = env.reset()
episode_reward = 0
done = False
t=0
while not done:
    t+=1
    action = select_best_action(actor,state)
    import pdb;pdb.set_trace()
    next_state, reward, done, _ = env.step(action)
    episode_reward += reward
    state = next_state