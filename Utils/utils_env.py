import gym
import numpy as np
import pybullet as p
import pybullet_data
import random
import time
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import warnings

# Convert all warnings into exceptions
#warnings.filterwarnings("error", category=RuntimeWarning)

def normalize_features(features):
    feature_matrix = np.array(list(features.values()))
    min_vals = np.min(feature_matrix, axis=0)
    max_vals = np.max(feature_matrix, axis=0)
    range_vals = max_vals - min_vals  # Compute the range
    
    # Log the feature ranges
    #print(f"Feature Matrix:\n{feature_matrix}")
    #print(f"Min Values: {min_vals}")
    #print(f"Max Values: {max_vals}")
    #print(f"Range Values: {range_vals}")

    normalized_features = {}
    for obj_id, feature_list in features.items():
        try:
            normalized_features[obj_id] = (np.array(feature_list) - min_vals) / range_vals
        except Exception as e:
            print(f"Error encountered for Object ID: {obj_id}")
            print(f"Feature list: {feature_list}")
            print(f"Min values: {min_vals}")
            print(f"Max values: {max_vals}")
            print(f"Range values: {range_vals}")
            raise e  # Escalate the error for debugging
    return normalized_features

def compute_pca_shape_features(point_cloud, n_components=6):
    pca = PCA(n_components=n_components)
    pca.fit(point_cloud)
    # Get eigenvalues and eigenvectors
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    # Return eigenvalues and their ratios
    shape_features = list(eigenvalues)  # Eigenvalues (variance along each component)
    # You can also add ratios between the eigenvalues to capture the shape's anisotropy
    for i in range(len(eigenvalues)):
        for j in range(i+1, len(eigenvalues)):
            if eigenvalues[j] > 0:
                shape_features.append(eigenvalues[i] / eigenvalues[j])          
    return shape_features


def compute_features(segmented_clouds, clutter, table_height=0.685):
    object_features = {}
    clutter_features = []
    clutter_mean = np.mean(clutter, axis=0)
    clutter_kdtree = KDTree(clutter)  # For proximity computation
    mean_pos_clutter = np.mean(clutter, axis=0)
    clutter_features.extend(mean_pos_clutter[:3])  # Append x, y, z mean positions

    vertical_distance_clutter = mean_pos_clutter[2] - table_height
    clutter_features.append(vertical_distance_clutter)
    # Distance from mean of clutter (which is the total clutter mean itself)
    dist_from_clutter_mean = 0  # Since we are calculating for the whole clutter
    clutter_features.append(dist_from_clutter_mean)
    # Bounding box dimensions for total clutter
    bbox_clutter = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(clutter))
    bbox_dimensions_clutter = bbox_clutter.get_extent()
    bbox_volume_clutter = np.prod(bbox_dimensions_clutter)
    clutter_features.append(bbox_volume_clutter)
    # Structural density for total clutter
    structural_density_clutter = len(clutter) / bbox_volume_clutter if bbox_volume_clutter > 0 else 0
    #clutter_features.append(structural_density_clutter)
    # Proximity to neighbors in total clutter
    distances_clutter, _ = clutter_kdtree.query(mean_pos_clutter, k=min(len(clutter), len(clutter)) + 100)
    avg_proximity_clutter = 0  # Exclude self-distance
    #clutter_features.append(avg_proximity_clutter)
    # PCA-based shape features for total clutter (e.g., 3 components)
    pca_shape_features_clutter = compute_pca_shape_features(clutter, n_components=3)
    clutter_features.extend(pca_shape_features_clutter)
    # Store the features for the total clutter as a dictionary (use a dummy key, e.g., 'total_clutter')
    object_features['total_clutter'] = clutter_features
    
    all_features = []  # This will hold all features to create the final numpy array
    
    # Compute features for each object in the segmented clouds
    for obj_id, cloud in segmented_clouds.items():
        features_list = []
        point_cloud = cloud["points"]
        # Mean position
        mean_pos = np.mean(point_cloud, axis=0)
        features_list.extend(mean_pos[:3])  # Append x, y, z mean positions
        # Vertical distance from table
        vertical_distance = mean_pos[2] - table_height
        features_list.append(vertical_distance)
        # Distance from mean of clutter
        dist_from_clutter_mean = np.linalg.norm(mean_pos - clutter_mean)
        features_list.append(dist_from_clutter_mean)
        # Bounding box dimensions
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(point_cloud))
        bbox_dimensions = bbox.get_extent()
        bbox_volume = np.prod(bbox_dimensions)
        features_list.append(bbox_volume)
        # Structural density
        structural_density = len(point_cloud) / bbox_volume if bbox_volume > 0 else 0
        #features_list.append(structural_density)
        # Proximity to neighbors
        distances, _ = clutter_kdtree.query(mean_pos, k=min(len(point_cloud), len(clutter)) + 100)
        avg_proximity = np.mean(distances[1:])  # Exclude self-distance
        #features_list.append(avg_proximity)
        # PCA-based shape features (with more components)
        pca_shape_features = compute_pca_shape_features(point_cloud, n_components=3)  # Example with 3 components
        features_list.extend(pca_shape_features)
        # Store the features for the object
        object_features[obj_id] = features_list
        # Collect all object features to create the final array
        all_features.append(features_list)
        ''' if obj_id ==target:
            target_index = len(all_features) -1 '''
            
        

    normalized_features = normalize_features(object_features)
    normalized_feature_matrix = np.array([normalized_features[obj_id] for obj_id in object_features])
    all_feature_matrix = np.array(all_features)
    mean = np.mean(all_feature_matrix, axis=0)  # Column-wise mean
    std = np.std(all_feature_matrix, axis=0)    # Column-wise standard deviation
    # import pdb;pdb.set_trace()
    all_feature_matrix = (all_feature_matrix-mean)/std
    ''' all_feature_matrix = np.hstack([all_feature_matrix, np.zeros((all_feature_matrix.shape[0], 1))])
    if 0 <= target_index < all_feature_matrix.shape[0]:
        all_feature_matrix[target_index, -1] = 1
    else:
        raise ValueError("target_index is out of bounds.") '''

    # import pdb;pdb.set_trace()
    return normalized_features, all_feature_matrix

def compute_features_goal(segmented_clouds, clutter, target, table_height=0.685):
    object_features = {}
    clutter_features = []
    clutter_mean = np.mean(clutter, axis=0)
    clutter_kdtree = KDTree(clutter)  # For proximity computation
    mean_pos_clutter = np.mean(clutter, axis=0)
    clutter_features.extend(mean_pos_clutter[:3])  # Append x, y, z mean positions

    vertical_distance_clutter = mean_pos_clutter[2] - table_height
    # clutter_features.append(vertical_distance_clutter)
    # Distance from mean of clutter (which is the total clutter mean itself)
    dist_from_clutter_mean = 0  # Since we are calculating for the whole clutter
    # clutter_features.append(dist_from_clutter_mean)
    # Bounding box dimensions for total clutter
    bbox_clutter = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(clutter))
    bbox_dimensions_clutter = bbox_clutter.get_extent()
    bbox_volume_clutter = np.prod(bbox_dimensions_clutter)
    # clutter_features.append(bbox_volume_clutter)
    # Structural density for total clutter
    structural_density_clutter = len(clutter) / bbox_volume_clutter if bbox_volume_clutter > 0 else 0
    #clutter_features.append(structural_density_clutter)
    # Proximity to neighbors in total clutter
    distances_clutter, _ = clutter_kdtree.query(mean_pos_clutter, k=min(len(clutter), len(clutter)) + 100)
    avg_proximity_clutter = 0  # Exclude self-distance
    # clutter_features.append(avg_proximity_clutter)
    # PCA-based shape features for total clutter (e.g., 3 components)
    pca_shape_features_clutter = compute_pca_shape_features(clutter, n_components=3)
    # clutter_features.extend(pca_shape_features_clutter)
    # Store the features for the total clutter as a dictionary (use a dummy key, e.g., 'total_clutter')
    object_features['total_clutter'] = clutter_features
    
    all_features = []  # This will hold all features to create the final numpy array
    target_index = None
    # Compute features for each object in the segmented clouds
    for obj_id, cloud in segmented_clouds.items():
        features_list = []
        point_cloud = cloud["points"]
        # Mean position
        mean_pos = np.mean(point_cloud, axis=0)
        features_list.extend(mean_pos[:3])  # Append x, y, z mean positions
        # Vertical distance from table
        vertical_distance = mean_pos[2] - table_height
        # features_list.append(vertical_distance)
        # Distance from mean of clutter
        dist_from_clutter_mean = np.linalg.norm(mean_pos - clutter_mean)
        # features_list.append(dist_from_clutter_mean)
        # Bounding box dimensions
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(point_cloud))
        bbox_dimensions = bbox.get_extent()
        bbox_volume = np.prod(bbox_dimensions)
        # features_list.append(bbox_volume)
        # Structural density
        structural_density = len(point_cloud) / bbox_volume if bbox_volume > 0 else 0
        #features_list.append(structural_density)
        # Proximity to neighbors
        distances, _ = clutter_kdtree.query(mean_pos, k=min(len(point_cloud), len(clutter)) + 100)
        avg_proximity = np.mean(distances[1:])  # Exclude self-distance
        # features_list.append(avg_proximity)
        # PCA-based shape features (with more components)
        pca_shape_features = compute_pca_shape_features(point_cloud, n_components=3)  # Example with 3 components
        # features_list.extend(pca_shape_features)
        # Store the features for the object
        object_features[obj_id] = features_list
        # Collect all object features to create the final array
        all_features.append(features_list)
        if np.isnan(np.asarray(features_list).any()):
            import pdb;pdb.set_trace()
            
        if obj_id ==target:
            # print("yes")
            target_index = len(all_features) -1
            # print(target_index)
        # print(obj_id,target)
        

    normalized_features = normalize_features(object_features)
    # import pdb;pdb.set_trace()
    normalized_feature_matrix = np.array([normalized_features[obj_id] for obj_id in object_features])
    all_feature_matrix = np.array(all_features)
    all_feature_matrix = np.hstack([all_feature_matrix, np.zeros((all_feature_matrix.shape[0], 1))])
    # import pdb;pdb.set_trace()
    
    if target_index is not None:
        if 0 <= target_index < all_feature_matrix.shape[0]:
            all_feature_matrix[target_index, -1] = 1
        else:
            raise ValueError("target_index is out of bounds.") 

    # import pdb;pdb.set_trace()
    return normalized_features, all_feature_matrix

def pad_state(state, max_objects=10):
    """Pad state to the maximum number of objects."""
    padded_state = np.zeros((max_objects, state.shape[1]), dtype=np.float32)
    padded_state[:state.shape[0], :] = state
    if np.isnan(np.asarray(padded_state).any()):
            import pdb;pdb.set_trace()
    return padded_state