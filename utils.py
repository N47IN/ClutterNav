import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.decomposition import PCA


def normalize_features(features):
    feature_matrix = np.array(list(features.values()))
    mean_vals = np.mean(feature_matrix, axis=0)
    std_vals = np.std(feature_matrix, axis=0) + 1e-6  # Avoid division by zero

    normalized_features = {
        obj_id: (np.array(feature_list) - mean_vals) / std_vals
        for obj_id, feature_list in features.items()
    }
    return normalized_features

def compute_pca_shape_features(point_cloud, n_components=6):
    pca = PCA(n_components=n_components)
    pca.fit(point_cloud)
    # Get eigenvalues and eigenvectors
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    # Return eigenvalues and their ratios
    # Eigenvalues (variance along each component)
    shape_features = list(eigenvalues)
    # You can also add ratios between the eigenvalues to capture the shape's anisotropy
    for i in range(len(eigenvalues)):
        for j in range(i+1, len(eigenvalues)):
            if eigenvalues[j] > 0:
                shape_features.append(eigenvalues[i] / eigenvalues[j])
    return shape_features

def compute_features(segmented_clouds, clutter, table_height=0.685):
    object_features = {}
    clutter_features = []

    # Compute features for total clutter
    clutter_mean = np.mean(clutter, axis=0)
    clutter_kdtree = KDTree(clutter)  # For proximity computation
    # Mean position of total clutter
    mean_pos_clutter = np.mean(clutter, axis=0)
    # Append x, y, z mean positions
    clutter_features.extend(mean_pos_clutter[:3])
    # Vertical distance from table for total clutter
    vertical_distance_clutter = mean_pos_clutter[2] - table_height
    clutter_features.append(vertical_distance_clutter)
    # Distance from mean of clutter (which is the total clutter mean itself)
    dist_from_clutter_mean = 0  # Since we are calculating for the whole clutter
    clutter_features.append(dist_from_clutter_mean)
    # Bounding box dimensions for total clutter
    bbox_clutter = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(clutter))
    bbox_dimensions_clutter = bbox_clutter.get_extent()
    bbox_volume_clutter = np.prod(bbox_dimensions_clutter)
    clutter_features.append(bbox_volume_clutter)
    # Structural density for total clutter
    structural_density_clutter = len(
        clutter) / bbox_volume_clutter if bbox_volume_clutter > 0 else 0
    # clutter_features.append(structural_density_clutter)
    # Proximity to neighbors in total clutter
    distances_clutter, _ = clutter_kdtree.query(
        mean_pos_clutter, k=min(len(clutter), len(clutter)) + 100)
    avg_proximity_clutter = 0  # Exclude self-distance
    # clutter_features.append(avg_proximity_clutter)
    # PCA-based shape features for total clutter (e.g., 3 components)
    pca_shape_features_clutter = compute_pca_shape_features(
        clutter, n_components=3)
    # clutter_features.extend(pca_shape_features_clutter)
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
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(point_cloud))
        bbox_dimensions = bbox.get_extent()
        bbox_volume = np.prod(bbox_dimensions)
        features_list.append(bbox_volume)
        # Structural density
        structural_density = len(point_cloud) / \
            bbox_volume if bbox_volume > 0 else 0
        # features_list.append(structural_density)
        # Proximity to neighbors
        distances, _ = clutter_kdtree.query(
            mean_pos, k=min(len(point_cloud), len(clutter)) + 100)
        avg_proximity = np.mean(distances[1:])  # Exclude self-distance
        # features_list.append(avg_proximity)
        # PCA-based shape features (with more components)
        pca_shape_features = compute_pca_shape_features(
            point_cloud, n_components=3)  # Example with 3 components
        # features_list.extend(pca_shape_features)
        # Store the features for the object
        object_features[obj_id] = features_list

        # Collect all object features to create the final array
        all_features.append(features_list)

    # Normalize the features
    normalized_features = normalize_features(object_features)

    # Collect the normalized features for all objects and clutter into a numpy array
    normalized_feature_matrix = np.array(
        [normalized_features[obj_id] for obj_id in object_features])

    # Convert all features into numpy array for the unnormalized list of features
    all_feature_matrix = np.array(all_features)

    return normalized_features, all_feature_matrix

''' def compute_features(segmented_clouds, clutter, table_height=0.685):
    object_features = {}
    clutter_features = []
    all_features = []
    for obj_id, cloud in segmented_clouds.items():
        cloud = np.array(cloud['points'])
        object_features[obj_id] = cloud
        all_features.append(cloud)
    all_feature_matrix = np.array(all_features)
    return all_feature_matrix '''

def pad_state(state, max_objects=30):
    """Pad state to the maximum number of objects."""
    padded_state = np.zeros((max_objects, state.shape[1]), dtype=np.float32)
    padded_state[:state.shape[0], :] = state
    return padded_state