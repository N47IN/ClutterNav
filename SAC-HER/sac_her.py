import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils_sac import soft_update, hard_update
from her_model import GaussianPolicy, QNetwork, GoalEncoder

class SAC(object):
    def __init__(self, input_dim, q_output_dim, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        # Initialize Goal Encoder
        self.goal_encoder = GoalEncoder(6, 8).to(self.device)
        self.goal_encoder_optim = Adam(self.goal_encoder.parameters(), lr=args.lr)

        self.critic = QNetwork(30*14 +1, q_output_dim, 8).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(30*14 + 1, q_output_dim, 8).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            
            self.policy = GaussianPolicy(6 + 8, 10, args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(input_dim + args.hidden_size, 10, args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
    
    def select_action(self, state, goal, evaluate=False):
        goal_tensor = torch.tensor(goal, dtype=torch.float32).to(self.device)
        encoded_goal = self.goal_encoder(goal_tensor)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        encoded_goal = encoded_goal.repeat(state.shape[0], 1)  # Match state batch size
        state_with_goal = torch.cat((state, encoded_goal), dim=-1)
        # encoded_goal = encoded_goal.unsqueeze(1).repeat(1, state_batch.shape[1], 1)
        if evaluate is False:
            action, log_prob = self.policy.sample(state_with_goal)
        else:
            action, log_prob = self.policy.sample(state_with_goal)
        return action.detach().cpu().numpy()

    def update_parameters(self, memory, batch_size, updates):
        state_batch, goal_batch, action_batch, reward_batch, next_state_batch, next_goal_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        next_goal_batch = torch.FloatTensor(next_goal_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        # Encode goals inside SAC
        encoded_goal = self.goal_encoder(goal_batch)
        encoded_next_goal = self.goal_encoder(next_goal_batch).detach()  # Ensure no in-place modification
        
        encoded_goal = encoded_goal.unsqueeze(1).repeat(1, state_batch.shape[1], 1)  # Ensure shape compatibility
        encoded_next_goal = encoded_next_goal.unsqueeze(1).repeat(1, next_state_batch.shape[1], 1)
        
        state_batch_with_goal = torch.cat((state_batch, encoded_goal), dim=-1)
        next_state_batch_with_goal = torch.cat((next_state_batch, encoded_next_goal), dim=-1)
        
        # Compute goal loss first to prevent graph deletion
        goal_loss = F.mse_loss(encoded_goal, encoded_next_goal)  # No detach here
        
        self.goal_encoder_optim.zero_grad()
        self.critic_optim.zero_grad()
        self.policy_optim.zero_grad()

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy.sample(next_state_batch_with_goal)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch_with_goal, next_state_action)
            next_state_log_pi = next_state_log_pi.unsqueeze(1)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
        
        qf1, qf2 = self.critic(state_batch_with_goal, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # self.critic_optim.zero_grad()
        # qf_loss.backward()
        # self.critic_optim.step()

        pi, log_pi = self.policy.sample(state_batch_with_goal)
        qf1_pi, qf2_pi = self.critic(state_batch_with_goal, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        total_loss = goal_loss + qf_loss + policy_loss
        total_loss.backward()

        # Update all networks
        self.goal_encoder_optim.step()
        self.critic_optim.step()
        self.policy_optim.step()
        # self.policy_optim.zero_grad()
        # policy_loss.backward()
        # self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), goal_loss.item(), policy_loss.item()
    
    def save_checkpoint(self, env_name="clutter", suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = f"checkpoints/sac_checkpoint_{env_name}_{suffix}.pth"
        print(f'Saving models to {ckpt_path}')
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'goal_encoder_state_dict': self.goal_encoder.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'goal_encoder_optimizer_state_dict': self.goal_encoder_optim.state_dict()
        }, ckpt_path)
        
        
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.goal_encoder.load_state_dict(checkpoint['goal_encoder_state_dict'])
            self.critic_target.load_state_dict(
                checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(
                checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(
                checkpoint['policy_optimizer_state_dict'])
            self.goal_encoder_optim.load_state_dict(
                checkpoint['goal_encoder_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.goal_encoder.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.goal_encoder.train()
                self.critic_target.train()
