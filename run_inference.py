import torch
import matplotlib.pyplot as plt
from cluttered_env import ClutteredSceneEnv
from matplotlib import cm
import torch
import numpy as np
from model import QNetwork

def load_critic(critic, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    critic.load_state_dict(checkpoint['critic_state_dict'])
    print("Critic model successfully loaded from:", checkpoint_path)
    return critic

def select_best_actions(critic, state, env, k=3):
    target_id = env.target
    actions = []
    sim_state = torch.tensor(state, dtype=torch.float32).unsqueeze(
        0).view(1, -1)  # Flatten state
    objects = env.object_ids

    # Compute initial Q-values
    q_values = {
        i: torch.min(critic(sim_state, torch.tensor([[i - 3]], dtype=torch.float32))[0],
                     critic(sim_state, torch.tensor([[i - 3]], dtype=torch.float32))[1]).item()
        for i in objects
    }

    selected_ids = set()
    object_reached = False

    while not object_reached:
        top_k_keys = [i for i in sorted(
            q_values, key=q_values.get, reverse=True) if i not in selected_ids][:k]

        if target_id in top_k_keys:
            print("Plan found")
            return actions + [target_id]

        max_id = None
        max_increase = float("-inf")

        for i in top_k_keys:
            if i in selected_ids:
                continue

            sim_state_2 = sim_state.clone()
            sim_state_2[0, i - 3] = 0  # Zero out feature

            # Compute new target Q-value
            new_q_value_target = torch.min(
                critic(sim_state_2, torch.tensor(
                    [[target_id - 3]], dtype=torch.float32))[0],
                critic(sim_state_2, torch.tensor(
                    [[target_id - 3]], dtype=torch.float32))[1]
            ).item()

            increase = new_q_value_target - q_values[target_id]

            if increase > max_increase:
                max_increase = increase
                max_id = i

        if max_id is None:
            break  # Avoid infinite loop

        sim_state[0, max_id - 3] = 0
        actions.append(max_id)
        selected_ids.add(max_id)

        # Update the Q-value for the target after selection
        q_values[target_id] = torch.min(
            critic(sim_state, torch.tensor(
                [[target_id - 3]], dtype=torch.float32))[0],
            critic(sim_state, torch.tensor(
                [[target_id - 3]], dtype=torch.float32))[1]
        ).item()

    return actions

# Function to compute argmax action using the critic
def select_best_action(critic, state, env):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    # state = state.view(state.size(0), -1)  # Flatten state if necessary (matches forward pass)
    # Store Q-values for each action
    q_values = []
    for action in range(len(env.object_ids)):  # Iterate over all possible actions
        # Convert action to a PyTorch tensor and expand dimensions
        action_tensor = torch.tensor([action], dtype=torch.float32)  # Batch dimension
        # Pass state and action through the critic
        # import pdb;pdb.set_trace()
        q1, q2 = critic(state, action_tensor)
        # Take the minimum of Q1 and Q2 to handle overestimation bias
        q_value = torch.min(q1, q2).item()
        # Append the Q-value
        q_values.append(q_value)
    # Find the action with the maximum Q-value
    best_action_index = np.argmax(q_values)
    best_action = env.object_ids[best_action_index]
    return best_action, q_values

# Visualizing function for 3D plot
def visualize_3d_positions_with_scores(positions, scores, object_ids, title="Object Removability Score Visualization"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize the scores for coloring
    norm = plt.Normalize(vmin=min(scores), vmax=max(scores))
    cmap = cm.viridis  # You can change the colormap here

    # Extract the x, y, z coordinates
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Plot the points with colors based on their scores
    scatter = ax.scatter(x, y, z, c=scores, cmap=cmap, norm=norm, s=100)

    # Add text labels for object IDs
    for i, obj_id in enumerate(object_ids):
        ax.text(x[i], y[i], z[i], str(obj_id), color='black', fontsize=10, ha='center')

    # Add color bar for score visualization
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Removability Score')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

# Example initialization
env = ClutteredSceneEnv( headless=False)
critic = QNetwork(181,1, 256)  # Assuming critic is a Q-Network class
checkpoint_path = "/home/navin/projects/M2P/ClutterGrasp/planning/checkpoints_final/sac_checkpoint_clutter_100"
critic = load_critic(critic, checkpoint_path)

state = env.reset()
episode_reward = 0
done = False
t = 0
import pdb;pdb.set_trace()
while not done:
    t += 1
    # np.random.shuffle(state)
    # import pdb;pdb.set_trace()
    action, scores= select_best_action(critic, state, env)
    positions = state[:,:3]  # Assuming positions is a (num_objects, 3) numpy array
    # print(action)
    # import pdb;pdb.set_trace()
    obj_id = env.object_ids
    positions = positions[~np.all(positions == 0, axis=1)]
    # visualize_3d_positions_with_scores(positions, scores,obj_id)
    next_state, reward, done, _ = env.step(action)
    episode_reward += reward
    
    state = next_state
    
    
print("Episode Reward:", episode_reward)