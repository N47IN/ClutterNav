import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
#from PointNet import PN2Encoder
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(4*10+1, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(4*10+1, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        # Flatten the state tensor: [256, 30, 12] -> [256, 360]
        state = state.view(state.size(0), -1)

        # Expand the action tensor: [256] -> [256, 1]
        action = action.unsqueeze(-1)

        # Concatenate state and action: [256, 360] + [256, 1] -> [256, 361]
        xu = torch.cat([state, action], dim=1)

        # Q1 forward pass
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # Q2 forward pass
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, num_objects, hidden_dim,action_space=None):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.score_linear = nn.Linear(hidden_dim, 1)  # Outputs one score per object

    def forward(self, state):
        # Process each object's state
        valid_mask = (state.sum(dim=-1) != 0).float().unsqueeze(-1)  # Identify valid rows
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        scores = self.score_linear(x)  # Shape: (batch_size, num_objects, 1)
        scores = scores.squeeze(-1)  # Shape: (batch_size, num_objects)
        scores = scores.masked_fill(valid_mask.squeeze(-1) == 0, -1e10)
        probs = F.softmax(scores, dim=-1)
        return probs

    def sample(self, state):
        probs = self.forward(state)  # Get probabilities
        try:
            dist = torch.distributions.Categorical(probs)  # Create a categorical distribution
        except:
            import pdb;pdb.set_trace()
        action = dist.sample()  # Sample an action index
        log_prob = dist.log_prob(action)  # Log probability of the action
        return action, log_prob

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

''' class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)

class Critic(nn.Module):
    def __init__(
      self, action_shape, hidden_dim,
      encoder_feature_dim
    ):
        super().__init__()
        encoder = PN2Encoder(out_dim=encoder_feature_dim)
        self.encoder = encoder
        self.Q1 = QFunction(
            self.encoder.out_dim, action_shape, hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.out_dim, action_shape, hidden_dim
        )
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        batch_size, num_objects, num_points, _ = obs.shape  # B x 30 x 9 x 3
        flat_obs = obs.view(-1, num_points, 3)  # (B * num_objects) x 9 x 3
        if detach_encoder:
            with torch.no_grad():
                latent_features = self.encoder(flat_obs)  # (B * num_objects) x encoder_feature_dim
        else:
            latent_features = self.encoder(flat_obs)
        latent_features = latent_features.view(batch_size, num_objects, -1)  # B x 30 x encoder_feature_dim
        scene_features = latent_features.view(batch_size, -1)  # B x (30 * encoder_feature_dim)
        q1 = self.Q1(scene_features, action)
        q2 = self.Q2(scene_features, action)
        # Store outputs for debugging/analysis
        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        return q1, q2

    def forward_from_feature(self, feature, action):
        # detach_encoder allows to stop gradient propogation to encoder
        q1 = self.Q1(feature, action)
        q2 = self.Q2(feature, action)
        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        return q1, q2

class Actor(nn.Module):
    def __init__(self,hidden_dim, encoder_feature_dim):
        super(Actor, self).__init__()
        self.encoder = PN2Encoder(out_dim=encoder_feature_dim)
        self.linear1 = nn.Linear(encoder_feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.score_linear = nn.Linear(hidden_dim, 1)  # Outputs one score per object

    def forward(self, state):
        batch_size, num_objects, num_points, _ = state.shape
        flat_state = state.view(-1, num_points, 3)  # (batch_size * num_objects, num_points, 3)
        encoded_features = self.encoder(flat_state)  # (batch_size * num_objects, encoder_feature_dim)
        encoded_features = encoded_features.view(batch_size, num_objects, -1)
        x = F.relu(self.linear1(encoded_features))  # (batch_size, num_objects, hidden_dim)
        x = F.relu(self.linear2(x))
        scores = self.score_linear(x).squeeze(-1)  # (batch_size, num_objects)
        valid_mask = (state.sum(dim=-1).sum(dim=-1) != 0).float()  # (batch_size, num_objects)
        scores = scores.masked_fill(valid_mask == 0, -1e10)
        # Compute softmax probabilities
        probs = F.softmax(scores, dim=-1)  # (batch_size, num_objects)
        return probs

    def sample(self, state):
        probs = self.forward(state)  # Get probabilities
        dist = torch.distributions.Categorical(probs)  # Create a categorical distribution
        action = dist.sample()  # Sample an action index
        log_prob = dist.log_prob(action)  # Log probability of the action
        return action, log_prob
 '''
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
