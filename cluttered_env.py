import gym
import numpy as np
import pybullet as p
import pybullet_data
import random
import numpy as np
import cv2 as cv2
from utils import compute_features, pad_state

class ClutteredSceneEnv(gym.Env):
    def __init__(self,headless=False):
        super(ClutteredSceneEnv, self).__init__()
        # PyBullet setup
        self.physics_client = p.connect(p.DIRECT if headless else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setGravity(0, 0, -9.81)
        # Environment setup
        self.last_num_visible = 0
        self.plane_id = p.loadURDF("plane.urdf")
        self.partial_obs = None
        self.full_obs = None
        self.camera_pose = {
            'eye_position': [1.5, 0, 2.0],
            'target_position': [0.0, 0.0, 0.65],
            'up_vector': [0, 0, 1],
            'fov': 120,
            'aspect': 640/480,
            'near': 0.1,
            'far': 5.0
        }
        self.last_frame = None
        self.last_positions = None
        self.table1_id = p.loadURDF("table/table.urdf", [0, 0, 0], [0, 0, 0, 1])
        self.table2_id = p.loadURDF("table/table.urdf", [1.5, 0, 0], [0, 0, 0, 1])
        self.table1_height = 0.65
        self.table2_height = 0.65
        self.spawn_area_length = 0.6
        self.spawn_area_width = 0.6
        self.object_ids = []
        self.visible = []
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
        self.visible.clear()
        self.removed_objects.clear()
        self.spawn_cluttered_cuboids(num_cuboids)
        self.last_com = None
        self.curr_com = None
        partial, total = self._get_obs()
        _, object_features = compute_features(partial, total)
        global_features = _.get("total_clutter", None)
        state = global_features - object_features
        state = pad_state(state)
        p.changeVisualShape(self.target, -1, rgbaColor=[0, 1, 0, 1])
        return state

    def spawn_cluttered_cuboids(self, num_objects, max_layers=3):
        max_layers = 3
        base_layer_objects = num_objects // max_layers
        layer_height_increment = 0.1
        spawn_area_center = np.array([0, 0])  # Center of the spawn area
        object_positions = []  # To store positions for target selection

        for layer in range(max_layers):
            for _ in range(base_layer_objects):
                
                    
                x_pos = random.uniform(-self.spawn_area_length / 4, self.spawn_area_length / 4)
                y_pos = random.uniform(-self.spawn_area_width / 4, self.spawn_area_width / 4)
                length, width, height = np.random.uniform(0.05, 0.1, 3)
                z_pos = self.table1_height + layer * layer_height_increment + height / 2 + 0.1

                body_id = p.createMultiBody(
                    baseMass=1.0,
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[length / 2, width / 2, height / 2]),
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[length / 2, width / 2, height / 2]),
                    basePosition=[x_pos, y_pos, z_pos])

                p.changeDynamics(body_id, -1, lateralFriction=0.5, spinningFriction=0.5, frictionAnchor=True)
                self.object_ids.append(body_id)
                object_positions.append((body_id, np.array([x_pos, y_pos])))  # Store position
                if layer ==0 and _ == base_layer_objects//2:
                    self.target = body_id
            # Allow objects to settle for one batch
            for _ in range(50):
                p.stepSimulation()

        # Select inner object as target based on distance to the center
        # inner_object = min(object_positions, key=lambda obj: np.linalg.norm(obj[1] - spawn_area_center))
        
    ''' def spawn_cluttered_cuboids(self, num_objects=30, max_layers=5):
        layer_height_increment = 0.1
        cube_size = 0.08  # Uniform cube size for all layers
        base_spacing = cube_size  # No gaps, tightly packed

        layer_config = [
            (4, 1.00, 10),  # 4x3 grid (12 positions) but take 10
            (3, 0.75, 8),   # 3x3 grid (9 positions), take 8
            (3, 0.55, 6),   # 3x2 grid (6 positions)
            (2, 0.40, 4),   # 2x2 grid (4)
            (1, 0.25, 2)    # 1x2 grid (2)
        ]

        objects_spawned = 0
        z_base = self.table1_height + 0.1

        for layer_idx, (grid_size, scale, num_layer_objects) in enumerate(layer_config):
            rows = int(np.ceil(np.sqrt(num_layer_objects)))
            cols = int(np.ceil(num_layer_objects / rows))

            positions = []
            x_offset = (rows - 1) * base_spacing / 2
            y_offset = (cols - 1) * base_spacing / 2

            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= num_layer_objects:
                        break
                    x = (i * base_spacing) - x_offset
                    y = (j * base_spacing) - y_offset
                    positions.append((x, y))

            z_level = z_base + (layer_idx * layer_height_increment)

            for x, y in positions:
                body_id = p.createMultiBody(
                    baseMass=1.2,
                    baseCollisionShapeIndex=p.createCollisionShape(
                        p.GEOM_BOX, 
                        halfExtents=[cube_size/2, cube_size/2, cube_size/2]),
                    baseVisualShapeIndex=p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[cube_size/2, cube_size/2, cube_size/2],
                        rgbaColor=[0.2, 0.6 - (layer_idx*0.1), 0.3, 1]),
                    basePosition=[x, y, z_level + cube_size/2])

                p.changeDynamics(body_id, -1, 
                                lateralFriction=0.6,
                                spinningFriction=0.3,
                                rollingFriction=0.1,
                                contactDamping=0.9,
                                contactStiffness=1e4)

                self.object_ids.append(body_id)
                objects_spawned += 1

            for _ in range(80 - (layer_idx * 15)):
                p.stepSimulation()

        self.target = random.choice(self.object_ids) '''

    ''' def spawn_cluttered_cuboids(self, num_objects=30, height=10):
        cube_size = 0.08
        base_spacing = cube_size
        z_base = self.table1_height + 0.1

        positions = []
        x_offset = 0
        y_offset = 0

        for i in range(height):
            if len(positions) >= num_objects:
                break
            x = 0
            y = 0
            z = z_base + (i * cube_size)
            positions.append((x, y, z))

        for x, y, z in positions:
            body_id = p.createMultiBody(
                baseMass=1.2,
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_BOX, 
                    halfExtents=[cube_size/2, cube_size/2, cube_size/2]),
                baseVisualShapeIndex=p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[cube_size/2, cube_size/2, cube_size/2],
                    rgbaColor=[0.8, 0.2, 0.2, 1]),
                basePosition=[x, y, z])

            p.changeDynamics(body_id, -1, 
                            lateralFriction=0.6,
                            spinningFriction=0.3,
                            rollingFriction=0.1,
                            contactDamping=0.9,
                            contactStiffness=1e4)

            self.object_ids.append(body_id)

        for _ in range(120):
            p.stepSimulation()

        self.target = random.choice(self.object_ids) '''
    
    ''' def spawn_cluttered_cuboids(self, num_objects=30, rows=5, cols=6):
        cube_size = 0.08
        base_spacing = cube_size
        z_base = self.table1_height + 0.1

        positions = []
        x_offset = (cols - 1) * base_spacing / 2
        y_offset = 0

        for i in range(rows):
            for j in range(cols):
                if len(positions) >= num_objects:
                    break
                x = (j * base_spacing) - x_offset
                y = 0
                z = z_base + (i * cube_size)
                positions.append((x, y, z))

        for x, y, z in positions:
            body_id = p.createMultiBody(
                baseMass=1.2,
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_BOX, 
                    halfExtents=[cube_size/2, cube_size/2, cube_size/2]),
                baseVisualShapeIndex=p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[cube_size/2, cube_size/2, cube_size/2],
                    rgbaColor=[0.4, 0.3, 0.8, 1]),
                basePosition=[x, y, z])

            p.changeDynamics(body_id, -1, 
                            lateralFriction=0.6,
                            spinningFriction=0.3,
                            rollingFriction=0.1,
                            contactDamping=0.9,
                            contactStiffness=1e4)

            self.object_ids.append(body_id)

        for _ in range(100):
            p.stepSimulation()

        self.target = random.choice(self.object_ids) '''
     
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

    def compute_reward(self):
        """Compute reward based on disturbance magnitude and target exposure"""
        num_disturbed = 0
        total_displacement = 0.0
        target_visibility = 0

        # Calculate displacement penalties
        for obj_id in self.object_ids:
            if obj_id in self.last_com and obj_id in self.curr_com:
                displacement = np.linalg.norm(
                    self.curr_com[obj_id] - self.last_com[obj_id]
                )
                if displacement > 1e-3:  # 1mm threshold
                    num_disturbed += 1
                    total_displacement += displacement

        # Calculate target visibility reward
        target_points = len(self.partial_obs.get(self.target, {"points": []})["points"])
        max_points = 9  # 8 corners + 1 COM
        target_visibility = target_points / max_points

        # Composite reward with adaptive weighting
        disturbance_penalty = -(num_disturbed + 0.1 * total_displacement)
        visibility_bonus = 2.0 * target_visibility
        reward = disturbance_penalty 
        return reward
    
    def step(self, action, max_settle_steps=1000, check_interval=1):
        # action = self.object_ids[action]
        if action not in self.object_ids:
            action = random.choice(self.object_ids)
        # Record the initial COM before the action
        self.last_com = self.get_com(action)
        self.object_ids.remove(action)
        self.visible.clear()
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

                if self._has_settled():
                    break

        # Update observations after stepping
        partial, total = self._get_obs()
        self.curr_com = self.get_com()
        reward = self.compute_reward()
        _, object_features = compute_features(partial, total)
        global_features = _.get("total_clutter", None)
        state = global_features - object_features
        state = pad_state(state)
        num_remaining = len(list(partial.keys()))
        return state, reward, len(self.object_ids)==5, num_remaining
    
    def _has_settled(self, threshold=0.05):
        for obj_id in self.object_ids + self.removed_objects:
            linear_velocity, angular_velocity = p.getBaseVelocity(obj_id)
            if np.linalg.norm(linear_velocity) > threshold or np.linalg.norm(angular_velocity) > threshold:
                return False
        return True

    def get_segmented_point_clouds(self):
        object_point_clouds = {}
        global_points = []
        global_colors = []
        # Camera parameters
        eye = np.array(self.camera_pose['eye_position'])
        target = np.array(self.camera_pose['target_position'])
        up = np.array(self.camera_pose['up_vector'])
        # Compute camera direction
        cam_dir = target - eye
        cam_dir /= np.linalg.norm(cam_dir)
        fov_rad = np.radians(self.camera_pose['fov'] / 2)  # Half-FOV for cone test
        cos_fov = np.cos(fov_rad)
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
            ''' com[1] = com[1] + 0.15
            com[0] = com[0] + 0.15
            corners[:,1] = corners[:,1] + 0.15
            corners[:,0] = corners[:,0] + 0.15 '''

            # Check if any point is inside the camera's FOV
            to_points = corners - eye
            to_points = to_points / np.linalg.norm(to_points, axis=1, keepdims=True)
            dot_products = np.dot(to_points, cam_dir)

            

            ''' visible_count = 0
            for point in corners:
                ray_test = p.rayTest(eye, point)
                hit = ray_test[0][0]  # Object ID of what was hit
                if hit == obj_id:  
                    visible_count += 1
            
        
            if visible_count < 1:  
                continue   '''
            # print(visible_count)
            self.visible.append(obj_id)
            points = np.vstack([com, corners])
            colors = np.tile([0.5, 0.5, 0.5], (points.shape[0], 1))
            object_point_clouds[obj_id] = {
                "points": points,
                "colors": colors
            }
            # Append to global point cloud
            global_points.append(points)
            global_colors.append(colors)
        # print(len(global_points))
        global_points = np.vstack(global_points)
        global_colors = np.vstack(global_colors)
        return object_point_clouds, global_points

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
    # env.get_point_cloud_from_scene()
    while True:
        if len(env.object_ids) == 0:
            print("All objects removed!")
            break
        env.get_segmented_point_clouds()
        selected_object = random.choice(env.object_ids)
        print(f"Removing object {selected_object}...")
        partial_obs, full_obs, reward,  _ = env.step(selected_object)


    env.close()