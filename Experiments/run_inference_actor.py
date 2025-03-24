import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from cluttered_env import ClutteredSceneEnv
from typing import Dict, Union, Tuple
import numpy as np
from sac import SAC
from replay_memory import ReplayMemory
from model import QNetwork, GaussianPolicy
# Define function to load the critic model from a checkpoint
def load_critic(critic, checkpoint_path):
    # Load the checkpoint file
    checkpoint = torch.load(checkpoint_path)
    critic.load_state_dict(checkpoint['policy_state_dict'])
    print("Actor model successfully loaded from:", checkpoint_path)
    return critic

def extract_goal_from_state(state, target_index):
    """Extract goal state using target index."""
    goal = state[target_index]  # Assume state is NxF, extract F-dim goal

    return goal

# Function to compute argmax action using the critic
def select_best_action(actor, state):
    state = torch.FloatTensor(state)
    next_state_action, next_state_log_pi = actor.sample(state)
    return next_state_action    

env = ClutteredSceneEnv(headless=False)
actor = GaussianPolicy(6,10,1024,30)
checkpoint_path = "/home/navin/projects/M2P/ClutterGrasp/ClutterNav/checkpoints/sac_checkpoint_clutter_400"  # Replace with the actual checkpoint path
actor = load_critic(actor, checkpoint_path)

state = env.reset()
episode_reward = 0
done = False
t=0
import pdb;pdb.set_trace()
while not done:
    t+=1
    target_index = env.object_ids.index(env.target)
    goal = extract_goal_from_state(state, target_index).reshape(-1,6)
    state = np.vstack((state, goal))  # Shape (31,6)
    action = select_best_action(actor,state)
    
    next_state, reward, done, _ = env.step(action)
    episode_reward += reward
    state = next_state