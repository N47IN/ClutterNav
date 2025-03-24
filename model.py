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

class GaussianPolicy3(nn.Module):
    def __init__(self, obj_feature_dim, hidden_dim):
        super(GaussianPolicy3, self).__init__()
        self.obj_processor = nn.Sequential(
            nn.Linear(obj_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.score_linear = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # Process each object equally
        valid_mask = (state.abs().sum(-1) > 0).float().unsqueeze(-1)
        x = F.relu(self.obj_processor(state))
        x = F.relu(self.obj_processor[1](x))  # Assuming Sequential with two layers
        scores = self.score_linear(x).squeeze(-1)
        
        # Dynamic masking
        scores = scores.masked_fill(valid_mask.squeeze(-1) == 0, -1e10)
        
        return F.softmax(scores, dim=-1)

    def sample(self, state):
        probs = self.forward(state)
        valid_counts = (state.abs().sum(-1) > 0).sum(-1)  # [batch]
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        # Safety clamp for each sample in batch
        action = torch.clamp(action, max=valid_counts-1)
        return action, dist.log_prob(action)
    
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

''' class GaussianPolicy2(nn.Module):
    def __init__(self, obj_feature_dim, hidden_dim, max_objects=30):
        super().__init__()
        self.obj_feature_dim = obj_feature_dim
        self.max_objects = max_objects
        
        # Transformer encoder for permutation invariance
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=obj_feature_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Score predictor per object
        self.score_net = nn.Sequential(
            nn.Linear(obj_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, mask=None):
        # state: [batch, num_objects, obj_feature_dim]
        batch_size, num_objects, _ = state.shape
        
        # Apply transformer with padding mask
        encoded = self.transformer(state, src_key_padding_mask=mask)
        
        # Predict scores
        scores = self.score_net(encoded).squeeze(-1)  # [batch, num_objects]
        
        # Dynamic masking based on actual object count
        valid_objects_mask = torch.ones_like(scores, dtype=torch.bool)
        if num_objects < self.max_objects:
            valid_objects_mask[:, num_objects:] = False
        
        # Combine masks
        if mask is not None:
            valid_objects_mask = valid_objects_mask & (mask.bool())
        
        # Apply final mask
        scores = scores.masked_fill(~valid_objects_mask, -1e9)
        
        # Convert to probabilities
        probs = F.softmax(scores, dim=-1)
        return probs

    def sample(self, state, mask=None):
        probs = self.forward(state, mask)
        valid_count = state.shape[1]  # Actual number of objects
        
        # Create masked distribution
        dist = torch.distributions.Categorical(probs[:, :valid_count])
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs

class QNetwork2(nn.Module):
    def __init__(self, obj_feature_dim, action_dim, hidden_dim):
        super().__init__()
        # Enhanced object encoder
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # From search result [17]
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Action embedding with positional encoding
        self.action_embed = nn.Embedding(num_embeddings=1000, 
                                       embedding_dim=hidden_dim)
        
        # Improved attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Dual Q-heads with layer normalization
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        batch_size, num_objects, _ = state.shape

        # Action processing (convert to valid indices)
        action = action.squeeze().long()
        action = torch.clamp(action, 0, num_objects-1)  # Ensure valid indices

        # Dynamic one-hot encoding
        action_onehot = F.one_hot(action, num_classes=num_objects).float()

        # Projections
        action_embedded = self.action_projector(action_onehot.unsqueeze(1))  # [B, 1, H]
        obj_features = self.obj_encoder(state)  # [B, N, H]

        # Cross-attention with dimension fix
        attn_out, _ = self.cross_attn(
            query=action_embedded,  # [B, 1, H]
            key=obj_features,
            value=obj_features
        )

        # Residual connection
        aggregated = attn_out + action_embedded
        return self.q1(aggregated), self.q2(aggregated) '''


class SafeAction(nn.Module):
    """Ensures action indices stay within valid range"""
    def forward(self, state, raw_action):
        _, num_objects, _ = state.shape
        return torch.clamp(raw_action, max=num_objects-1)

class GaussianPolicy2(nn.Module):
    def __init__(self, obj_feature_dim, hidden_dim):
        super().__init__()
        self.obj_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=obj_feature_dim,
                nhead=4,
                dim_feedforward=hidden_dim
            ), num_layers=2
        )
        self.score_net = nn.Linear(obj_feature_dim, 1)
        self.action_safety = SafeAction()

    def forward(self, state, mask=None):
        batch_size, num_objects, _ = state.shape
        encoded = self.obj_encoder(state)
        scores = self.score_net(encoded).squeeze(-1)
        
        # Dynamic masking
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
        if mask is not None:
            valid_mask = valid_mask & mask.bool()
        scores = scores.masked_fill(~valid_mask, -1e9)
        
        return F.softmax(scores, dim=-1)

    def sample(self, state, mask=None):
        probs = self.forward(state, mask)
        valid_objects = state.shape[1]
        dist = torch.distributions.Categorical(probs[:, :valid_objects])
        action = dist.sample()
        return action, dist.log_prob(action), probs

class QNetwork2(nn.Module):
    def __init__(self, obj_feature_dim, hidden_dim):
        super().__init__()
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_feature_dim, hidden_dim),
            nn.ReLU()
        )
        self.action_projector = nn.Linear(1, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.q1 = nn.Linear(hidden_dim, 1)
        self.q2 = nn.Linear(hidden_dim, 1)
        self.action_safety = SafeAction()

    def forward(self, state, action):
        batch_size, num_objects, _ = state.shape
        
        # Validate action
        action = self.action_safety(state, action.squeeze()).long()
        
        # Process action
        action_onehot = F.one_hot(action, num_classes=num_objects).float()
        action_embedded = self.action_projector(action_onehot.unsqueeze(-1))
        
        # Process state
        obj_features = self.obj_encoder(state)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=action_embedded,
            key=obj_features,
            value=obj_features
        )
        
        # Q-values
        aggregated = attn_out.mean(dim=1)
        return self.q1(aggregated), self.q2(aggregated)

class QNetwork3(nn.Module):
    def __init__(self, obj_feature_dim, hidden_dim, max_objects=30):
        super(QNetwork3, self).__init__()
        self.max_objects = max_objects
        
        # Object-centric processing
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Action embedding handles variable counts
        self.action_embed = nn.Embedding(max_objects, hidden_dim)
        
        # Q-value heads
        self.Q1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.Q2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # Process each object independently
        obj_features = F.relu(self.obj_encoder(state))  # [batch, num_objs, hidden]
        global_state = obj_features.mean(dim=1)        # [batch, hidden]
        action = action.long()  # Ensure it's an integer index
        action_emb = self.action_embed(action)
        # Embed action index
        # action_emb = self.action_embed(action)          # [batch, hidden]
        import pdb;pdb.set_trace()
        # Combine for Q-values
        combined = torch.cat([global_state, action_emb], dim=1)
        return self.Q1(combined), self.Q2(combined)

    
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerQNetwork(nn.Module):
    def __init__(self, feature_dim=6, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        
        # Action-aware object embedding
        self.object_embed = nn.Linear(feature_dim + 1, hidden_dim)  # +1 for action flag
        
        # Transformer encoder (permutation-equivariant processing)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Dual Q-heads with object query attention
        self.q1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, objects, action):
        """
        Args:
            objects: [batch_size, num_objects, 6] (6D features)
            action: [batch_size] (index of object to evaluate)
        Returns:
            q1, q2: [batch_size] (dual Q-values)
        """
        batch_size, num_objects, _ = objects.shape
        
        # Create action mask and embed objects
        
        action_mask = F.one_hot(action.long(), num_objects).float().unsqueeze(-1).squeeze(1) # [B, N, 1]
        # import pdb;pdb.set_trace()
        # print(1)
        x = torch.cat([objects, action_mask], dim=-1)  # [B, N, 7]
        
        # import pdb;pdb.set_trace()
        
        x = self.object_embed(x)  # [B, N, H]
       
        # Transformer processing (global interactions)
        x = self.transformer(x)  # [B, N, H]
        
        # Attention-based querying of action object
        action_emb = x[torch.arange(batch_size), action.long()]  # [B, H]
        
        # Dual Q-values
        q1 = self.q1_head(action_emb).squeeze(-1)
        q2 = self.q2_head(action_emb).squeeze(-1)
        return q1, q2
''' 
class QNetwork(nn.Module):
    def __init__(self, num_inputs, output_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, output_dim)

        self.apply(weights_init_)

    def forward(self, state, action):
        state = state.view(state.size(0), -1)
        
        try:
         action = action.unsqueeze(1)
         xu = torch.cat([state, action], dim=1)
        except:
         action = action.squeeze(1)
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
 '''
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
    
''' class QNetwork(nn.Module):
    def __init__(self, num_inputs, output_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, output_dim)

        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0.01)

    def forward(self, state, action):
        # Explicit dimension handling
        state = state.reshape(state.size(0), -1)
        action = action.reshape(action.size(0), -1)
        xu = torch.cat([state, action], dim=1)

        # Q1 forward
        x1 = F.leaky_relu(self.bn1(self.linear1(xu)), negative_slope=0.01)
        x1 = F.leaky_relu(self.bn2(self.linear2(x1)), negative_slope=0.01)
        x1 = self.linear3(x1)

        # Q2 forward
        x2 = F.leaky_relu(self.bn3(self.linear4(xu)), negative_slope=0.01)
        x2 = F.leaky_relu(self.bn4(self.linear5(x2)), negative_slope=0.01)
        x2 = self.linear6(x2)

        return x1, x2 '''

''' class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        state = state.view(state.size(0), -1)
        state_feat = self.state_encoder(state)
        action_float = action.float()
        action_feat = self.action_encoder(action_float)
        return self.q_head(torch.cat([state_feat, action_feat], dim=-1))
 '''
  
class QNetworkk(nn.Module):
    def __init__(self, num_inputs=12, num_objects=30, output_dim=1, hidden_dim=256):
        super(QNetworkk, self).__init__()

        self.num_objects = num_objects
        self.feature_dim = num_inputs  # Number of features per object

        # Shared feature extractor (Per-object encoding)
        self.shared_fc1 = nn.Linear(num_inputs + 1, hidden_dim)  # +1 for action mask
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Transformer-based aggregation (Permutation invariant)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Q-value branches
        self.q1_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.q2_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def merge_state_action(self, state, action):
        """
        Merges the state tensor with the action by appending a binary action mask.
        - `state`: [B, N, F] (Batch, Num Objects, Features)
        - `action`: [B] (Index of selected object)
        """
        batch_size, num_objects, feature_dim = state.shape  # [B, N, F]

        # Create an action mask (1 for the selected action, 0 otherwise)
        action_mask = torch.zeros((batch_size, num_objects, 1), device=state.device)
        # import pdb;pdb.set_trace()
        action_mask.scatter_(1, action.long().unsqueeze(-1).unsqueeze(-1), 1)  # [B, N, 1]

        # Concatenate action mask with state
        return torch.cat([state, action_mask], dim=-1)  # [B, N, F+1]

    def forward(self, state, action):
        """
        - `state`: [B, N, F] (Batch, Num Objects, Features)
        - `action`: [B] (Index of selected object)
        """
        batch_size, num_objects, feature_dim = state.shape  # [B, N, F]

        # Merge state and action
        xu = self.merge_state_action(state, action)  # [B, N, F+1]

        # Per-object feature encoding
        x = F.leaky_relu(self.shared_fc1(xu))
        x = self.layer_norm(F.leaky_relu(self.shared_fc2(x)))

        # Apply attention-based pooling (Permutation Invariance)
        x, _ = self.attention(x, x, x)  # Self-attention
        x = torch.max(x, dim=1)[0]  # Global Max Pooling -> [B, hidden_dim]

        # Q-value predictions
        q1 = self.q1_fc(x)
        q2 = self.q2_fc(x)

        return q1, q2

class ObjectValuePredictor(nn.Module):
    def __init__(self, obj_feature_dim, hidden_dim):
        super().__init__()
        self.obj_processor = nn.Sequential(
            nn.Linear(obj_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = F.relu(self.obj_processor(state))
        object_values = self.value_predictor(features).squeeze(-1)
        return object_values

class TransformerObjectValuePredictor(nn.Module):
    def __init__(self, obj_feature_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.obj_encoder = nn.Linear(obj_feature_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=4),
            num_layers=num_layers
        )
        self.value_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # state: [batch_size, num_objects, obj_feature_dim]
        x = torch.relu(self.obj_encoder(state))
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)  # (num_objects, batch_size, hidden_dim)
        object_values = self.value_predictor(x).squeeze(-1)
        return object_values
    
class DisturbancePredictor(nn.Module):
    def __init__(self, obj_feature_dim, hidden_dim):
        super().__init__()
        self.obj_processor = nn.Sequential(
            nn.Linear(obj_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.disturbance_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # Process each object's state
        features = F.relu(self.obj_processor(state))  # [batch, num_objs, hidden]
        disturbance_scores = self.disturbance_predictor(features).squeeze(-1)  # [batch, num_objs]
        return disturbance_scores
    

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
        # import pdb;pdb.set_trace()
        action = dist.sample()  # Sample an action index
        log_prob = dist.log_prob(action)  # Log probability of the action
        return action, log_prob

''' class QNetwork(nn.Module):
    def __init__(self, obj_feature_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.action_embed = nn.Embedding(100, hidden_dim)  # Dynamic action embedding
        self.Q1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.Q2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # Ensure action is of correct type
        action = action.long()  # Convert to LongTensor
        
        # Process each object independently
        obj_features = F.relu(self.obj_encoder(state))  # [batch, num_objs, hidden]
        global_state = obj_features.mean(dim=1)        # [batch, hidden]
        
        # Embed action index
        action_emb = self.action_embed(action)          # [batch, hidden]
        # import pdb;pdb.set_trace()
        # Combine for Q-values
        combined = torch.cat([global_state, action_emb], dim=1)
        return self.Q1(combined), self.Q2(combined)

class GaussianPolicy(nn.Module):
    def __init__(self, obj_feature_dim, hidden_dim):
        super(GaussianPolicy, self).__init__()
        self.obj_processor = nn.Sequential(
            nn.Linear(obj_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.score_linear = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # Process each object equally
        valid_mask = (state.abs().sum(-1) > 0).float().unsqueeze(-1)
        x = F.relu(self.obj_processor(state))
        x = F.relu(self.obj_processor[1](x))  # Assuming Sequential with two layers
        scores = self.score_linear(x).squeeze(-1)
        
        # Dynamic masking
        scores = scores.masked_fill(valid_mask.squeeze(-1) == 0, -1e10)
        
        return F.softmax(scores, dim=-1)

    def sample(self, state):
        probs = self.forward(state)
        valid_counts = (state.abs().sum(-1) > 0).sum(-1)  # [batch]
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        # Safety clamp for each sample in batch
        action = torch.clamp(action, max=valid_counts-1)
        return action, dist.log_prob(action) '''


''' class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device) '''


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