import torch
from cluttered_env import ClutteredSceneEnv
from matplotlib import cm
import numpy as np
import logging
from model import QNetwork
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_critic(critic, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    critic.load_state_dict(checkpoint['critic_state_dict'])
    logger.info(f"Critic loaded from: {checkpoint_path}")
    return critic

def compute_clutter_density(positions):
    """Estimate scene complexity using positional variance"""
    return np.mean(np.std(positions, axis=0))


def signed_integrated_saliency(critic, state, target_idx):
    """Gradient calculation with sign preservation using zero-out baseline for each object"""
    state_tensor = torch.tensor(state, dtype=torch.float32)
    total_grad = torch.zeros_like(state_tensor[:, :3])
    
    # Iterate over each object to compute individual contributions
    for obj_idx in range(state_tensor.shape[0]):
        # Create baseline by zeroing out the current object's feature vector
        baseline = state_tensor.clone()
        baseline[obj_idx, :] = 0  # Zero out the current object's features
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32)
        
        # Monte Carlo sampling (10 steps)
        alpha_steps = torch.rand(5)
        obj_grad = torch.zeros_like(state_tensor[:, :3])
        decay = np.exp(-2*np.linalg.norm(state[obj_idx,:3] - state[target_idx,:3]))
        for alpha in alpha_steps:
            interpolated = baseline_tensor + alpha * (state_tensor - baseline_tensor)
            interpolated.requires_grad_(True)
            
            critic.zero_grad()  # Prevent gradient leakage
            
            action_tensor = torch.tensor([[target_idx]], dtype=torch.float32)
            q1, q2 = critic(interpolated.unsqueeze(0), action_tensor)
            q_target = torch.min(q1, q2)
            q_target.backward(retain_graph=True)
            grad_vectors = interpolated.grad[:, :3]
            obj_grad += grad_vectors * (state_tensor - baseline_tensor)[:, :3]
        obj_grad[obj_idx] = decay*obj_grad[obj_idx]
        # Average gradients for the current object
        obj_grad /= alpha_steps.shape[0]
        total_grad += obj_grad  # Accumulate gradients for all objects
    
    # Convert to numpy for further analysis
    signed_grad = total_grad.numpy()
    # signed_grad[target_idx] *= 1.5
    return signed_grad


def visualize_scores_and_gradients(positions, q_values, gradients, object_ids, target_index=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={'projection': '3d'})
    q_values = np.array(q_values)
    # Normalize data for consistent color mapping
    norm_scores = mcolors.Normalize(vmin=min(q_values), vmax=max(q_values))
    norm_gradients = mcolors.Normalize(vmin=min(gradients), vmax=max(gradients))
    cmap = cm.viridis  # Colormap for visualization

    # Extract x, y, z coordinates
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # --- Left Plot: Q-Value Scores ---
    sc1 = axes[0].scatter(x, y, z, c=q_values, cmap=cmap, norm=norm_scores, s=100, edgecolor='k', alpha=0.8)
    for i, obj_id in enumerate(object_ids):
        axes[0].text(x[i], y[i], z[i], str(obj_id), color='black', fontsize=8, ha='center')
    if target_index is not None:
        # Highlight target object
        axes[0].scatter(x[target_index], y[target_index], z[target_index], 
                        color='red', marker='*', s=200, label='Target Object', edgecolor='k')
    cbar1 = plt.colorbar(sc1, ax=axes[0], shrink=0.6, aspect=20)
    cbar1.set_label('Q-Value Score', fontsize=12)
    axes[0].set_title('Q-Value Scores', fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle="--", alpha=0.5)
    if target_index is not None:
        axes[0].legend()

    # --- Right Plot: Saliency (Gradients) ---
    sc2 = axes[1].scatter(x, y, z, c=gradients, cmap=cmap, norm=norm_gradients, s=100, edgecolor='k', alpha=0.8)
    for i, obj_id in enumerate(object_ids):
        axes[1].text(x[i], y[i], z[i], str(obj_id), color='black', fontsize=8, ha='center')
    if target_index is not None:
        # Highlight target object
        axes[1].scatter(x[target_index], y[target_index], z[target_index], 
                        color='red', marker='*', s=200, label='Target Object', edgecolor='k')
    cbar2 = plt.colorbar(sc2, ax=axes[1], shrink=0.6, aspect=20)
    cbar2.set_label('Saliency (Gradient Magnitude)', fontsize=12)
    axes[1].set_title('Saliency Map (Gradients)', fontsize=14, fontweight='bold')
    axes[1].grid(True, linestyle="--", alpha=0.5)
    if target_index is not None:
        axes[1].legend()

    # Shared settings for both plots
    for ax in axes:
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.view_init(elev=25, azim=135)  # Consistent viewing angle

    plt.tight_layout()
    plt.show()
    
def robust_normalize(scores):
    """Winsorized normalization with fallback"""
    sorted_scores = np.sort(scores.flatten())
    q1, q3 = sorted_scores[int(0.25*len(sorted_scores))], sorted_scores[int(0.75*len(sorted_scores))]
    iqr = q3 - q1
    
    if iqr < 1e-6:  # Handle uniform scores
        return np.zeros_like(scores)
    
    # Clip outliers
    clipped = np.clip(scores, q1 - 1.5*iqr, q3 + 1.5*iqr)
    return (clipped - np.min(clipped)) / (np.max(clipped) - np.min(clipped) + 1e-6)

def adaptive_hybrid_selection(q_values, gradients, positions):
    """Dynamic threshold selection with conflict detection"""
    # Adaptive threshold based on clutter
    
    grad_norm = robust_normalize(np.linalg.norm(gradients, axis=1))
    q_norm = robust_normalize(np.array(q_values))
    combined = grad_norm[:len(env.object_ids)] + q_norm
    best_action = np.argmax(combined) 
    return best_action

def get_q(critic, state, env):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    q_values = []
    for action in range(len(env.object_ids)):
        action_tensor = torch.tensor([action], dtype=torch.float32)
        q1, q2 = critic(state, action_tensor)
        q_values.append(torch.min(q1, q2).item())
    return q_values

# Visualization with signed gradients
def visualize_signed_gradients(positions, gradients, object_ids, target_index):
    fig = plt.figure(figsize=(12, 6))
    
    # Position plot
    ax1 = fig.add_subplot(121, projection='3d')
    quiver = ax1.quiver(positions[:,0], positions[:,1], positions[:,2],
                       gradients[:,0], gradients[:,1], gradients[:,2],
                       length=0.1, normalize=True)
    ax1.set_title('Gradient Directions')
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    
    ax2 = fig.add_subplot(122, projection='3d')
    magnitudes = np.linalg.norm(gradients, axis=1)
    sc = ax2.scatter(positions[:,0], positions[:,1], positions[:,2], 
                    c=magnitudes, cmap='viridis')
    if target_index is not None:
        # Highlight target object
        ax2.scatter(x[target_index], y[target_index], z[target_index], 
                        color='red', marker='*', s=200, label='Target Object', edgecolor='k')
    # Magnitude plot
    plt.colorbar(sc, ax=ax2, label='Gradient Magnitude')
    ax2.set_title('Influence Magnitudes')
    
    plt.tight_layout()
    plt.show()

# Main execution flow
env = ClutteredSceneEnv(headless=False)
critic = QNetwork(181, 1, 256)
critic = load_critic(critic, "checkpoints_ral_m100/sac_checkpoint_clutter_100")
episode_reward = 0
history = {'rewards': [], 'conflicts': []}
done = False

total_disturbance = []
num_disturbed = []
num_removed = []

a = None
# import pdb;pdb.set_trace()
for j in range(4):
    for i in range(100):
        state = env.reset(mode=1)
        target_idx = env.object_ids.index(env.target)
        episode_reward = 0
        done = False
        rem=0
        while not done:
            valid_mask = ~np.all(state[:, :3] == 0, axis=1)
            positions = state[valid_mask][:, :3]
            obj_ids = [env.object_ids[i] for i in np.where(valid_mask)[0]]
            target_idx = env.object_ids.index(env.target)
            # q_values = get_q(critic, state, env)
            # gradients = signed_integrated_saliency(critic, state, target_idx)
            # action_idx = adaptive_hybrid_selection(q_values, gradients, positions)
            action = random.choice(env.object_ids)
            next_state, reward, done, rem= env.step(action)
            episode_reward += reward
            state = next_state
        total_disturbance.append(env.disturbance)
        num_disturbed.append(env.num_disturbed)
        num_removed.append(rem)
        print(env.num_disturbed)
        print(i,j)

    with open(f'random_eval/greedy_{j}_disturbance.npy', 'wb') as f:
        np.save(f, np.array(total_disturbance))

    with open(f'random_eval/greedy_{j}_num_disturbed.npy', 'wb') as f:
        np.save(f, np.array(num_disturbed))

    with open(f'random_eval/greedy_{j}_num_removed.npy', 'wb') as f:
        np.save(f, np.array(num_removed))

logger.info(f"Final reward: {episode_reward:.2f} | Conflicts: {history['conflicts']}")
