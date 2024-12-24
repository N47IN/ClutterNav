import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from collections import deque
import random
from cluttere_v8 import ClutteredSceneEnv
import argparse
import datetime
import gym
import numpy as np
import itertools
from sac import SAC
from replay_memory import ReplayMemory

# Initialize argument parser
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="clutter-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 10 episodes (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='Discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='Target smoothing coefficient (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='Learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter for entropy (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='Random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                    help='Batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='Maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='Hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='Model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=300, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1500, metavar='N',
                    help='Replay buffer size (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='Run on CUDA (default: False)')
args = parser.parse_args()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize WandB
wandb.init(
    project="sac-gc-cluttered-scene",  # Replace with your project name
    config=args,
    name=f"SAC_{args.env_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
)

# Environment setup
env = ClutteredSceneEnv(num_cuboids=10, headless=True)
env.reset()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = SAC(10, 10, args)
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    while not done:
        #import pdb;pdb.set_trace()
        #action = env.target
        if args.start_steps > 16:
            action = np.random.choice(np.asarray(env.object_ids))
        else:
            action = agent.select_action(state)  # Sample action from policy
            
        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                # Log losses to WandB
                wandb.log({
                    'loss/critic_1': critic_1_loss,
                    'loss/critic_2': critic_2_loss,
                    'loss/policy': policy_loss,
                    'loss/entropy_loss': ent_loss,
                    'entropy_temperature/alpha': alpha
                }, step=updates)
                updates += 1

        next_state, reward, done, _ = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        # print(total_numsteps,done)
        episode_reward += reward
        mask = 0
        
            
        memory.push(state, action, reward, next_state, mask)  # Append transition to memory
        state = next_state
    if total_numsteps > args.num_steps:
        break

    # Log episode reward
    wandb.log({'reward/train': episode_reward}, step=i_episode)
    print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {round(episode_reward, 2)}")

    if i_episode % 500 == 0:
        print("hi")
        agent.save_checkpoint(suffix=i_episode)
        
    if i_episode % 50 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes
        wandb.log({'avg_reward/test': avg_reward}, step=i_episode)
        print("----------------------------------------")
        print(f"Test Episodes: {episodes}, Avg. Reward: {round(avg_reward, 2)}")
        print("----------------------------------------")

env.close()
