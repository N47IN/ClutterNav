#Vanilla Env

import gym
import numpy as np
import pybullet as p
import pybullet_data
import random
import time 
import open3d as o3d
from gym import spaces
import numpy as np
import cv2 as cv2
from utils_env import compute_features, pad_state


class ClutteredSceneEnv(gym.Env):
    def __init__(self, num_cuboids=50, headless=False):
        super(ClutteredSceneEnv, self).__init__()
        # PyBullet setup
        self.physics_client = p.connect(p.DIRECT if headless else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setGravity(0, 0, -9.81)
        # Environment setup
        self.plane_id = p.loadURDF("plane.urdf")
        self.partial_obs = None
        self.full_obs = None
        self.last_frame = None
        self.last_positions = None
        self.table1_id = p.loadURDF("table/table.urdf", [0, 0, 0], [0, 0, 0, 1])
        self.table2_id = p.loadURDF("table/table.urdf", [1.5, 0, 0], [0, 0, 0, 1])
        self.table1_height = 0.65
        self.table2_height = 0.65
        self.spawn_area_length = 0.6
        self.spawn_area_width = 0.6
        self.object_ids = []
        self.removed_objects = []
        self.target = None
        self.last_com = None
        self.curr_com = None
        self.last_num_visible =None
        self.last_num_points = None

    def reset(self, num_cuboids=30):
        """Reset the environment."""
        for obj_id in self.object_ids + self.removed_objects:
            p.removeBody(obj_id)
        self.object_ids.clear()
        self.removed_objects.clear()
        self.spawn_cluttered_cuboids(num_cuboids)
        self.last_com = None
        self.curr_com = None
        partial, total = self._get_obs()
        _, object_features = compute_features(partial, total)
        global_features = _.get("total_clutter", None)
        state = global_features - object_features
        state = pad_state(state)
        return state

    def spawn_cluttered_cuboids(self, num_objects, max_layers=3):
        """Spawn cuboids efficiently."""
        base_layer_objects = num_objects // max_layers
        layer_height_increment = 0.1

        for layer in range(max_layers):
            for _ in range(base_layer_objects):
                x_pos = random.uniform(-self.spawn_area_length / 6, self.spawn_area_length / 6)
                y_pos = random.uniform(-self.spawn_area_width / 6, self.spawn_area_width / 6)
                length, width, height = np.random.uniform(0.05, 0.1, 3)
                z_pos = self.table1_height + layer * layer_height_increment + height / 2 + 0.1
                body_id = p.createMultiBody(
                    baseMass=1.0,
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[length / 2, width / 2, height / 2]),
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[length / 2, width / 2, height / 2]),
                    basePosition=[x_pos, y_pos, z_pos]
                )
                p.changeDynamics(body_id, -1, lateralFriction=0.5, spinningFriction=0.5, frictionAnchor=True)
                self.object_ids.append(body_id)
            

            # Allow objects to settle for one batch
            for _ in range(50):
                p.stepSimulation()
        self.target = random.choice(self.object_ids)

    def _get_obs(self):
        self.partial_obs, self.full_obs = self.get_segmented_point_clouds()
        return self.partial_obs, self.full_obs

    def get_com(self, exclude_action=None):
        com = {}
        for obj_id in self.object_ids:
            if exclude_action and obj_id == exclude_action:
                continue
            com_pos, _ = p.getBasePositionAndOrientation(obj_id)
            com[obj_id] = np.array(com_pos)  # Store COM positions as NumPy arrays
        return com

    def compute_reward1(self):
        """
        Compute reward based on the number of objects moved compared to the last timestep
        and a simple penalty for shifts.
        """
        # Ensure the necessary variables are initialized
        if self.last_positions is None:
            self.last_positions = {obj_id: p.getBasePositionAndOrientation(obj_id)[0]
                                   for obj_id in self.object_ids}
            self.last_num_moved = 0
            return 0  # No reward on the first step
        # Get the current positions of all objects
        current_positions = {obj_id: p.getBasePositionAndOrientation(obj_id)[0]
                             for obj_id in self.object_ids}
        # Compute the number of objects moved
        num_moved = 0
        for obj_id in self.object_ids:
            last_pos = self.last_positions.get(obj_id, None)
            if last_pos is not None:
                current_pos = current_positions[obj_id]
                if np.linalg.norm(np.array(current_pos) - np.array(last_pos)) > 1e-3:
                    num_moved += 1
        # Reward is based on the number of objects moved
        reward = -num_moved
        # print("Shift penalty:", reward)
        # Update last state variables
        self.last_positions = current_positions
        self.last_num_moved = num_moved
        return reward
    
    def step(self, action, max_settle_steps=1000, check_interval=1):
        if action not in self.object_ids:
            action = random.choice(self.object_ids)
        # Record the initial COM before the action
        self.last_com = self.get_com(action)
        # Remove the selected object
        self.object_ids.remove(action)
        self.removed_objects.append(action)
        # Move the removed object to the second table
        x_pos = random.uniform(1.5 - self.spawn_area_length / 4, 1.5 + self.spawn_area_length / 4)
        y_pos = random.uniform(-self.spawn_area_width / 4, self.spawn_area_width / 4)
        p.resetBasePositionAndOrientation(action, [x_pos, y_pos, self.table2_height + 0.1], [0, 0, 0, 1])
        # Allow the simulation to settle incrementally
        for step in range(max_settle_steps):
            p.stepSimulation()
            # Check every check_interval steps for object disturbances
            if step % check_interval == 0:
                com_positions = self.get_com()  # Current center of mass for all objects
                velocities = {
                    obj_id: p.getBaseVelocity(obj_id) for obj_id in self.object_ids
                }
                #print(f"Step {step}: Observing COM and velocities...")
                for obj_id, velocity in velocities.items():
                    linear, angular = velocity
                    #print(f"Object {obj_id}: Linear Vel {np.linalg.norm(linear):.3f}, Angular Vel {np.linalg.norm(angular):.3f}")
                # If all objects have settled, break the loop
                if self._has_settled():
                    #print(f"Objects settled after {step} steps.")
                    break

        # Update observations after stepping
        partial, total = self._get_obs()
        self.curr_com = self.get_com()
        reward = self.compute_reward1()
        _, object_features = compute_features(partial, total)
        global_features = _.get("total_clutter", None)
        state = global_features - object_features
        state = pad_state(state)
        return state, reward, len(self.object_ids)==15 or len, len(self.object_ids)

    def _has_settled(self, threshold=0.05):
        """
        Check if objects have settled based on their velocities.
        """
        for obj_id in self.object_ids + self.removed_objects:
            linear_velocity, angular_velocity = p.getBaseVelocity(obj_id)
            if np.linalg.norm(linear_velocity) > threshold or np.linalg.norm(angular_velocity) > threshold:
                return False
        return True

    def get_segmented_point_clouds(self):
        """
        Generate a simplified point cloud representation for each object.
        Includes the COM and all 8 corner points of the bounding box.
        """
        object_point_clouds = {}
        global_points = []
        global_colors = []

        for obj_id in self.object_ids:
            # Get COM (Center of Mass)
            com_pos, _ = p.getBasePositionAndOrientation(obj_id)
            com = np.array(com_pos)

            # Get AABB (Axis-Aligned Bounding Box)
            aabb_min, aabb_max = p.getAABB(obj_id)

            # Calculate all 8 corner points of the AABB
            corners = np.array([
                [aabb_min[0], aabb_min[1], aabb_min[2]],
                [aabb_min[0], aabb_max[1], aabb_min[2]],
                [aabb_max[0], aabb_min[1], aabb_min[2]],
                [aabb_max[0], aabb_max[1], aabb_min[2]],
                [aabb_min[0], aabb_min[1], aabb_max[2]],
                [aabb_min[0], aabb_max[1], aabb_max[2]],
                [aabb_max[0], aabb_min[1], aabb_max[2]],
                [aabb_max[0], aabb_max[1], aabb_max[2]]
            ])

            # Combine COM and corners into a single point cloud
            points = np.vstack([com, corners])

            # Assign a dummy color (e.g., gray) for visualization purposes
            colors = np.tile([0.5, 0.5, 0.5], (points.shape[0], 1))

            # Save the point cloud for the object
            object_point_clouds[obj_id] = {
                "points": points,
                "colors": colors
            }

            # Append to global point cloud
            global_points.append(points)
            global_colors.append(colors)

        # Merge global points and colors into a single numpy array
        global_points = np.vstack(global_points)
        global_colors = np.vstack(global_colors)

        return object_point_clouds, global_points
    
    ''' 
    def get_segmented_point_clouds(self, min_points=20):
        width, height = 640, 480
        fov = 60
        aspect = width / height
        near_plane = 0.1
        far_plane = 10.0

        # Camera parameters
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.5, 0, 2],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[1, 0, 0]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near_plane,
            farVal=far_plane
        )

        # Capture RGB-D and segmentation mask
        _, _, rgb_img, depth_img, seg_mask = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER
        )

        # Prepare RGB, depth, and segmentation mask
        rgb_img = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
        depth_img = np.array(depth_img).reshape(height, width)
        seg_mask = np.array(seg_mask).reshape(height, width)

        # Depth to 3D conversion
        z_buffer = depth_img
        depth_img = far_plane * near_plane / (far_plane - (far_plane - near_plane) * z_buffer)
        fx = fy = width / (2 * np.tan(np.deg2rad(fov) / 2))
        cx, cy = width / 2, height / 2
        # Object-wise point clouds
        object_point_clouds = {}
        unique_ids = np.unique(seg_mask)
        for obj_id in unique_ids:
            if obj_id < 0:  # Ignore background (-1)
                continue
            # Mask for current object
            obj_mask = (seg_mask == obj_id)
            # Extract pixel indices for the object
            ys, xs = np.where(obj_mask)
            if len(xs) <= min_points:
                continue  # Skip objects with fewer than min_points
            # Calculate 3D points
            points = []
            colors = []
            for x, y in zip(xs, ys):
                z = depth_img[y, x]
                if near_plane < z < far_plane:
                    px = (x - cx) * z / fx
                    py = (y - cy) * z / fy
                    pz = z
                    points.append([px, py, pz])
                    colors.append(rgb_img[y, x] / 255.0)

            if points:
                angle1 = 0.78
                rotmatx = np.asarray([[1,0,0],[0,np.cos(angle1),-np.sin(angle1)],[0,np.sin(angle1),np.cos(angle1)]])
                angle2 = 1.57
                rotmatz = np.asarray([[np.cos(angle2),-np.sin(angle2),0],[np.sin(angle2),np.cos(angle2),0],[0,0,1],])
                points = points@rotmatx@rotmatz
                points[:,2] = -points[:,2] + 2
                points[:,0] = 1.5 - points[:,0]
                mask = points[:, 2] >= 0.785
                points = points[mask]
                if points.shape[0]==0:

                    continue
                points_world = np.array(points)
                colors = np.array(colors)
                # Save point cloud
                object_point_clouds[obj_id] = {
                    "points": points_world,
                    "colors": colors
                }

        return object_point_clouds
 '''
    def get_point_cloud(self):
        width, height = 640, 480
        fov = 60
        aspect = width / height
        near_plane = 0.1
        far_plane = 10.0
        # Define the camera view matrix and projection matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.5, 0, 2],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[1, 0, 0]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near_plane,
            farVal=far_plane
        )
        # Capture the RGB-D image
        _, _, rgb_img, depth_img, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER
        )

        # Convert depth image to a point cloud
        depth_img = np.array(depth_img).reshape(height, width)  # Reshape depth image
        rgb_img = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]  # RGB image
        z_buffer = depth_img
        depth_img = far_plane * near_plane / (far_plane - (far_plane - near_plane) * z_buffer)
        # Generate point cloud
        fx = fy = width / (2 * np.tan(np.deg2rad(fov) / 2))
        cx, cy = width / 2, height / 2
        points = []
        colors = []
        for i in range(height):
            for j in range(width):
                z = depth_img[i, j]
                if z > near_plane and z < far_plane:
                    x = (j - cx) * z / fx
                    y = (i - cy) * z / fy
                    points.append([x, y, z])
                    colors.append(rgb_img[i, j] / 255.0)
        points = np.array(points)
        colors = np.array(colors)
        angle1 = 0.78
        rotmatx = np.asarray([[1,0,0],[0,np.cos(angle1),-np.sin(angle1)],[0,np.sin(angle1),np.cos(angle1)]])
        angle2 = 1.57
        rotmatz = np.asarray([[np.cos(angle2),-np.sin(angle2),0],[np.sin(angle2),np.cos(angle2),0],[0,0,1],])
        points = points@rotmatx@rotmatz
        points[:,2] = -points[:,2] + 2
        points[:,0] = 1.5 - points[:,0]
        mask = points[:, 2] >= 0.785
        points = points[mask]
        return points, colors
    
    def _has_settled(self, threshold=0.01):
        """Check if objects have settled."""
        return all(
            np.linalg.norm(p.getBaseVelocity(obj_id)[0]) < threshold and
            np.linalg.norm(p.getBaseVelocity(obj_id)[1]) < threshold
            for obj_id in self.object_ids + self.removed_objects
        )

    def close(self):
        p.disconnect()

# Example usage:
if __name__ == "__main__":
    env = ClutteredSceneEnv(num_cuboids=10, headless=False)
    obs = env.reset()
    while True:
        if len(env.object_ids) == 0:
            print("All objects removed!")
            break
        # Randomly select an object to move
        selected_object = random.choice(env.object_ids)
        print(f"Removing object {selected_object}...")
        partial_obs, full_obs, reward, done, _ = env.step(selected_object)


    env.close() 