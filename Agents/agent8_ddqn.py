import random
import torch
from collections import deque
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse
import datetime
from cluttere_v8 import ClutteredSceneEnv


class BasicBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_dim=4*10, output_dim=10, hidden_dim=1024):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        try:
            x=x.flatten()
            qvals = self.layer3(self.layer2(self.layer1(x)))
        except:
            x = x.view(128, 4*10)
            qvals = self.layer3(self.layer2(self.layer1(x)))
            
        return qvals


class DDQNAgent(nn.Module):
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        super(DDQNAgent, self).__init__()
        self.env = env
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.MSE_loss = nn.MSELoss()
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)
        self.optimizer1 = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.optimizer2 = torch.optim.Adam(
            self.target_model.parameters(), lr=learning_rate)

    def get_action(self, state, eps=0.20):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        offset = 3
        if (np.random.randn() < eps):
            return random.randint(0, 29)
        return action+offset

    # def compute_loss(self, batch):
    #     states, actions, rewards, next_states, dones = batch
    #     states = torch.FloatTensor(states).to(self.device)
    #     actions = torch.LongTensor(actions).to(self.device)
    #     rewards = torch.FloatTensor(rewards).to(self.device)
    #     next_states = torch.FloatTensor(next_states).to(self.device)
    #     dones = torch.FloatTensor(dones)
    #     actions = actions.view(actions.size(0))
    #     dones = dones.view(dones.size(0))

    #     curr_Q1 = self.model.forward(states).gather(1, actions.unsqueeze(-1))
    #     curr_Q2 = self.target_model.forward(
    #         states).gather(1, actions.unsqueeze(-1))

    #     next_Q1 = self.model.forward(next_states)
    #     next_Q2 = self.target_model.forward(next_states)
    #     next_Q = torch.min(
    #         torch.max(self.model.forward(next_states), 1)[0],
    #         torch.max(self.target_model.forward(next_states), 1)[0]
    #     )
    #     next_Q = next_Q.view(next_Q.size(0), 1)
    #     expected_Q = rewards + (1 - dones) * self.gamma * next_Q
    #     loss1 = F.mse_loss(curr_Q1, expected_Q.detach())
    #     loss2 = F.mse_loss(curr_Q2, expected_Q.detach())
    #     return loss1, loss2

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        curr_Q1 = self.model.forward(states)
        curr_Q2 = self.target_model.forward(states)
        next_Q1 = self.model.forward(next_states)
        next_Q2 = self.target_model.forward(next_states)
        next_Q = torch.min(
            torch.max(next_Q1, next_Q2),  # Element-wise max
            next_Q1  # Example operation if you want to keep the same shape
        )

        # next_Q = next_Q.view(next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * next_Q
        loss1 = F.mse_loss(curr_Q1, expected_Q.detach())
        loss2 = F.mse_loss(curr_Q2, expected_Q.detach())

        return loss1, loss2

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss1, loss2 = self.compute_loss(batch)
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
        return loss1, loss2


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    stepcount = 0
    
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.get_action(state)
            # print("len", len(env.object_ids))
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if len(agent.replay_buffer) > batch_size:
                stepcount += 1
                loss1, loss2 = agent.update(batch_size)
                wandb.log({'q1/loss': loss1.item(),
                          'q2/loss': loss2.item()}, step=stepcount)
                wandb.log({'episode_reward': episode_reward}, step=stepcount)
        
        print("Episode " + str(episode) + ": " + str(episode_reward))
        episode_rewards.append(episode_reward)
        
        if episode%500==0:
                
                
                torch.save(agent.state_dict(), f"critic_ep_{episode}.pth")
                

            

    return episode_rewards


parser = argparse.ArgumentParser(description='PyTorch Double DQN args')
parser.add_argument('--env-name', default="ClutteredScene",
                    help='Mujoco Gym environment (default: ClutteredScene)')
args = parser.parse_args()
wandb.init(
    project="ddqn-cluttered-scene",  # Replace with your project name
    config=args,
    name=f"DDQN_{args.env_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
)

MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 128

env = ClutteredSceneEnv(num_cuboids=30, headless=True)
env.reset()
torch.manual_seed(64)
np.random.seed(64)
agent = DDQNAgent(env)
episode_rewards = mini_batch_train(
    env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
env.close()