import torch
import numpy as np
import wandb
from cluttered_env import ClutteredSceneEnv
import argparse
import itertools
from sac_her import SAC  # Ensure this is discrete SAC
from replay_memory import ReplayMemory

def extract_goal(state, target_index):
    """Extract goal state using target index."""
    return state[target_index]

# Argument parser setup
parser = argparse.ArgumentParser(description='SAC + HER with Goal Encoding')
parser.add_argument('--env-name', default="ClutteredScene")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False)
parser.add_argument('--seed', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--num_steps', type=int, default=1000001)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--updates_per_step', type=int, default=1)
parser.add_argument('--start_steps', type=int, default=300)
parser.add_argument('--target_update_interval', type=int, default=1)
parser.add_argument('--replay_size', type=int, default=1500)
parser.add_argument('--her_ratio', type=float, default=0.8)  # HER ratio param
parser.add_argument('--cuda', action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# Initialize Weights & Biases (wandb)
# wandb.init(project="sac-her-goal-encoding", config=vars(args))

# Environment setup
env = ClutteredSceneEnv(headless=False)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent setup (assuming discrete SAC)
agent = SAC(6 * 30, 1, 30, args)  # State size excludes goal, handled internally
memory = ReplayMemory(args.replay_size, args.seed)
total_disturbance = []
num_disturbed = []
num_removed = []
total_numsteps = 0
updates = 0
agent.load_checkpoint("checkpoints/sac_checkpoint_clutter_2500.pth",evaluate=True)

for j in range(4):
    for i in range(100):
        state = env.reset(mode =1)
        target_idx = env.object_ids.index(env.target)
        raw_goal = extract_goal(state, target_idx)
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state,raw_goal,evaluate=True)
            next_state, reward, done,rem = env.step(action)
            next_goal = extract_goal(next_state, target_idx)
            ''' memory.push(
                torch.tensor(state, dtype=torch.float32),
                torch.tensor(raw_goal, dtype=torch.float32),
                action,
                reward,
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(next_goal, dtype=torch.float32),
                done
            ) '''
            state = next_state
            raw_goal = next_goal
            episode_reward += reward
            total_numsteps += 1
        total_disturbance.append(env.disturbance)
        num_disturbed.append(env.num_disturbed)
        num_removed.append(rem)
            # if len(memory) > args.batch_size:
            #     for _ in range(args.updates_per_step):
            #         q_loss, goal_loss, p_loss = agent.update_parameters(memory, args.batch_size, updates)
            #         updates += 1

            #         # Log losses to wandb
            #         # wandb.log({
            #         #     "Q Loss": q_loss,
            #         #     "Goal Encoder Loss": goal_loss,
            #         #     "Policy Loss": p_loss,
            #         #     "Total Steps": total_numsteps
            #         # })

        # Log episode rewards to wandb
        # wandb.log({"Episode Reward": episode_reward, "Episode": i_episode})
        print(f"Episode {i} | Reward: {episode_reward:.2f} | Steps: {total_numsteps}")

        ''' if i % 500 == 0:
            agent.save_checkpoint(suffix=i)

        if total_numsteps > args.num_steps:
            break '''
        
    with open(f'her_eval/greedy_{j}_disturbance.npy', 'wb') as f:
        np.save(f, np.array(total_disturbance))
    with open(f'her_eval/greedy_{j}_num_disturbed.npy', 'wb') as f:
        np.save(f, np.array(num_disturbed))
    with open(f'her_eval/greedy_{j}_num_removed.npy', 'wb') as f:
        np.save(f, np.array(num_removed))


env.close()
# wandb.finish()