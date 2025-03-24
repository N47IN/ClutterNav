import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn.utils.spectral_norm as spectral_norm

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GoalEncoder(nn.Module):
    def __init__(self, goal_dim, hidden_dim):
        super(GoalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, goal):
        return self.encoder(goal)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, num_objects, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # Outputs one score per object
        self.score_linear = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # Process each object's state
        # Identify valid rows
        valid_mask = (state.sum(dim=-1) != 0).float().unsqueeze(-1)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        scores = self.score_linear(x)  # Shape: (batch_size, num_objects, 1)
        scores = scores.squeeze(-1)  # Shape: (batch_size, num_objects)
        scores = scores.masked_fill(valid_mask.squeeze(-1) == 0, -1e10)
        probs = F.softmax(scores, dim=-1)
        return probs

    def sample(self, state):
        probs = self.forward(state)  # Get probabilities
        dist = torch.distributions.Categorical(
            probs)  # Create a categorical distribution
        action = dist.sample()  # Sample an action index
        log_prob = dist.log_prob(action)  # Log probability of the action
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, num_inputs, output_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.linear1 = spectral_norm(nn.Linear(num_inputs, hidden_dim))
        self.linear2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.linear3 = spectral_norm(nn.Linear(hidden_dim, output_dim))

        self.linear4 = spectral_norm(nn.Linear(num_inputs, hidden_dim))
        self.linear5 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.linear6 = spectral_norm(nn.Linear(hidden_dim, output_dim))

    def forward(self, state, action):
        state = state.view(state.size(0), -1)
        try:
         action = action.unsqueeze(1)
         xu = torch.cat([state, action], dim=1)
        except:
         action = action.squeeze(1)
         xu = torch.cat([state, action], dim=1)

        x1 = F.leaky_relu(self.linear1(xu))
        x1 = F.leaky_relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.leaky_relu(self.linear4(xu))
        x2 = F.leaky_relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2