import open3d as o3d
import numpy as np
import json
import random
import copy
from collections import deque
from tqdm import tqdm
import argparse
import os
import torch
from plyfile import PlyData
from .constants import INDEX_OBJ, OBJ_INDEX, PRIOR_SCALE

INPUT_PATH="./data/GSCollision/objects"
OUTPUT_PATH="./data/GSCollision/scene_configs"

class Config:

    SPACE_MIN = [-0.9, -0.9, -0.9]
    SPACE_MAX = [0.9, 0.9, 0.9]
    SPACE_SIZE = [1.8, 1.8, 1.8] 
    
    # For collision detection
    SAFETY_MARGIN = 0.1
    
    # Placement parameters
    MAX_ATTEMPTS_PER_OBJECT = 500
    BACKTRACK_DEPTH = 3
    MIN_DISTANCE_FACTOR = 1 
    MAX_DISTANCE_FACTOR = 2.5
    
    VOXEL_MARGIN_FACTOR = 1 

class SceneObject:
    def __init__(self, obj_id, pcd, scale=1.0):
        self.obj_id = obj_id
        self.original_pcd = pcd
        self.scale = scale
        self.position = [0, 0, 0]
        self.scaled_pcd = o3d.geometry.PointCloud()
        scaled_points = np.asarray(pcd.points) * scale
        self.scaled_pcd.points = o3d.utility.Vector3dVector(scaled_points)
        
        self.aabb = self.scaled_pcd.get_axis_aligned_bounding_box()
        self.sidelen = self.aabb.get_max_bound()-self.aabb.get_min_bound()
        
        self.radius = np.linalg.norm(
            (self.aabb.get_max_bound() - self.aabb.get_min_bound()) / 2.0
        )
    
    def set_position(self, position):
        self.position = position
        self.aabb = self.aabb.translate(position)
    
    def get_transformed_pcd(self):
        pcd = copy.deepcopy(self.scaled_pcd)
        pcd.translate(self.position)
        return pcd
    
    def to_dict(self):
        return {
            "id": self.obj_id,
            "position": self.position,
            "scale": self.scale
        }

class SceneState:
    def __init__(self):
        self.objects = []
        self.object_dict = {}
    
    def add_object(self, obj):
        self.objects.append(obj)
        self.object_dict[obj.obj_id] = obj
        
    def remove_last_object(self):
        if not self.objects:
            return None
        
        removed = self.objects.pop()
        self.object_dict.pop(removed.obj_id, None)
                
        return removed
    
    def get_last_objects(self, count):
        return self.objects[-count:] if len(self.objects) >= count else self.objects.copy()

def calc_scale_factor(selected_objects, num_objects):
    max_radius = max(obj.radius for obj in selected_objects)
    
    volume_per_object = (Config.SPACE_SIZE[0] * Config.SPACE_SIZE[1] * Config.SPACE_SIZE[2]) / num_objects
    linear_per_object = volume_per_object ** (1/3)
    
    scale_factor = linear_per_object / (max_radius * 2 * 2)
    return scale_factor

def generate_position_near(base_obj, new_obj, num_obj):
    if num_obj == 2 or num_obj == 3 or num_obj == 4:
        min_x = base_obj.position[0]-0.1
        max_x = base_obj.position[0]+0.1
        min_y = base_obj.position[1]-0.1
        max_y = base_obj.position[1]+0.1
        x=np.random.uniform(min_x,max_x)
        y=np.random.uniform(min_y,max_y)
        z=np.random.uniform(-0.9,0.9)
        return [x, y, z]
    else:
        x = np.random.uniform(base_obj.position[0]-base_obj.sidelen[0]/2-new_obj.sidelen[0]/2,base_obj.position[0]+base_obj.sidelen[0]/2+new_obj.sidelen[0]/2)
        y = np.random.uniform(base_obj.position[1]-base_obj.sidelen[1]/2-new_obj.sidelen[1]/2,base_obj.position[1]+base_obj.sidelen[1]/2+new_obj.sidelen[1]/2)
        z = np.random.uniform(-0.9,0.9)
        return [x, y, z]

def is_in_space(position, radius):
    for i in range(3):
        if position[i] - radius < Config.SPACE_MIN[i] or position[i] + radius > Config.SPACE_MAX[i]:
            return False
    return True

def fast_aabb_collision(new_obj, new_pos, scene_state):
    test_aabb = new_obj.aabb.translate(new_pos)
    
    for obj in scene_state.objects:
        if test_aabb.get_min_bound()[0] > obj.aabb.get_max_bound()[0] or \
            test_aabb.get_max_bound()[0] < obj.aabb.get_min_bound()[0] or \
            test_aabb.get_min_bound()[1] > obj.aabb.get_max_bound()[1] or \
            test_aabb.get_max_bound()[1] < obj.aabb.get_min_bound()[1] or \
            test_aabb.get_min_bound()[2] > obj.aabb.get_max_bound()[2] or \
            test_aabb.get_max_bound()[2] < obj.aabb.get_min_bound()[2]:
            continue
        else:
            return True
    return False

class GPUMemoryManager:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.empty_cache()

def batch_cdist(x, y, batch_size=10000):
    min_dist = float('inf')
    num_x = x.size(0)

    for i in range(0, num_x, batch_size):
        x_batch = x[i: i+batch_size]

        dist_matrix = torch.cdist(x_batch, y)
        batch_min = dist_matrix.min().item()

        if batch_min < min_dist:
            min_dist = batch_min
        
        if min_dist < Config.SAFETY_MARGIN * 0.5:
            break

        del dist_matrix
        torch.cuda.empty_cache()
    
    return min_dist

def exact_collision_check_gpu(new_obj, new_pos, scene_state):
    test_pcd = new_obj.get_transformed_pcd()
    test_pcd.translate(new_pos)
    test_points = torch.tensor(np.asarray(test_pcd.points), dtype=torch.float32).cuda()  # (N_test, 3)
    
    min_dist = float('inf')

    for obj in scene_state.objects:
        obj_pcd = obj.get_transformed_pcd()
        scene_points = torch.tensor(np.asarray(obj_pcd.points), dtype=torch.float32).cuda()  # (M, 3)

        if scene_points.numel() == 0:
            continue
        
        with torch.no_grad():
            obj_min_dist = batch_cdist(test_points, scene_points, batch_size=5000)
        
        min_dist = min(min_dist, obj_min_dist)
        
        if min_dist < Config.SAFETY_MARGIN * 0.5:
            del scene_points
            return True
        
        del scene_points
        torch.cuda.empty_cache()
    
    del test_points
    return min_dist < Config.SAFETY_MARGIN * 0.5

def validate_position(new_obj, new_pos, scene_state):
    if not is_in_space(new_pos, new_obj.radius):
        return False
    if fast_aabb_collision(new_obj, new_pos, scene_state):
        return False
    with GPUMemoryManager():
        if exact_collision_check_gpu(new_obj, new_pos, scene_state):
            return False
    
    return True


class VoxelGrid:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.grid = {}
        self.available_voxels = set()
        
        x_steps = int(Config.SPACE_SIZE[0] / voxel_size)
        y_steps = int(Config.SPACE_SIZE[1] / voxel_size)
        z_steps = int(Config.SPACE_SIZE[2] / voxel_size)
        
        for i in range(x_steps):
            for j in range(y_steps):
                for k in range(z_steps):
                    voxel_id = (i, j, k)
                    center = [
                        Config.SPACE_MIN[0] + (i + 0.5) * voxel_size,
                        Config.SPACE_MIN[1] + (j + 0.5) * voxel_size,
                        Config.SPACE_MIN[2] + (k + 0.5) * voxel_size
                    ]
                    self.grid[voxel_id] = {
                        "center": center,
                        "occupied": False
                    }
                    self.available_voxels.add(voxel_id)
    
    def occupy_voxels(self, position, radius):
        voxels_to_occupy = set()
        radius_voxels = int(np.ceil(radius / self.voxel_size))
        
        # Find nearest voxel
        base_i = int((position[0] - Config.SPACE_MIN[0]) / self.voxel_size)
        base_j = int((position[1] - Config.SPACE_MIN[1]) / self.voxel_size)
        base_k = int((position[2] - Config.SPACE_MIN[2]) / self.voxel_size)
        
        # Label surrounding voxels
        for i in range(base_i - radius_voxels, base_i + radius_voxels + 1):
            for j in range(base_j - radius_voxels, base_j + radius_voxels + 1):
                for k in range(base_k - radius_voxels, base_k + radius_voxels + 1):
                    voxel_id = (i, j, k)
                    if voxel_id in self.grid:
                        self.grid[voxel_id]["occupied"] = True
                        if voxel_id in self.available_voxels:
                            self.available_voxels.remove(voxel_id)
                        voxels_to_occupy.add(voxel_id)
        
        return voxels_to_occupy
    
    def free_voxels(self, voxels):
        for voxel_id in voxels:
            self.grid[voxel_id]["occupied"] = False
            self.available_voxels.add(voxel_id)
    
    def get_random_available_voxel(self):
        if not self.available_voxels:
            return None
        return random.choice(list(self.available_voxels))


def generate_scene_config(object_dict, num_objects, output_path=OUTPUT_PATH):
    # 1. Randomly select objects
    all_keys = list(object_dict.keys()) # ['can', 'soccer', 'cup', ...]
    selected_keys = random.choices(all_keys, k=num_objects)
    # selected_keys = all_keys[5:5+num_objects]
    selected_objects = [object_dict[key] for key in selected_keys]
    
    # 2. Callculate scale factor and scale objects
    scale_factor = 1      #min(calc_scale_factor(selected_objects, num_objects),1)
    scaled_objects = [
        SceneObject(obj.obj_id, obj.original_pcd, scale_factor * obj.scale)
        for obj in selected_objects
    ]

    # 3. Initialize voxel grid
    max_radius = max(obj.radius for obj in scaled_objects)
    voxel_size = max_radius * Config.VOXEL_MARGIN_FACTOR
    voxel_grid = VoxelGrid(voxel_size)
    
    # 4. Initialize scene state
    scene_state = SceneState()
    backtrack_stack = []
    object_queue = deque(scaled_objects)
    
    # 5. Place the first object
    first_obj = object_queue.popleft()
    
    first_pos = [0, 0, 0]
    
    first_obj.set_position(first_pos)
    scene_state.add_object(first_obj)
    occupied_voxels = voxel_grid.occupy_voxels(first_pos, first_obj.radius)
    backtrack_stack.append((first_obj,occupied_voxels))

    # 6. Place other objects recursively
    pbar = tqdm(total=num_objects-1, desc="Placing objects")
    while object_queue:
        obj = object_queue[0]
        
        placed = False
        attempts = 0
        
        while not placed and attempts < Config.MAX_ATTEMPTS_PER_OBJECT:
            if random.random() < 1 and len(scene_state.objects) > 0:
                base_obj = random.choice(scene_state.objects)
                new_pos = generate_position_near(base_obj, obj, num_objects)
            else:
                voxel_id = voxel_grid.get_random_available_voxel()
                if voxel_id:
                    new_pos = voxel_grid.grid[voxel_id]["center"]
                else:
                    new_pos = [
                        random.uniform(Config.SPACE_MIN[0], Config.SPACE_MAX[0]),
                        random.uniform(Config.SPACE_MIN[1], Config.SPACE_MAX[1]),
                        random.uniform(Config.SPACE_MIN[2], Config.SPACE_MAX[2])
                    ]
            
            if validate_position(obj, new_pos, scene_state):
                obj.set_position(new_pos)
                scene_state.add_object(obj)
                occupied_voxels = voxel_grid.occupy_voxels(new_pos, obj.radius)
                backtrack_stack.append((obj, occupied_voxels))
                object_queue.popleft()
                placed = True
                pbar.update(1)
            
            attempts += 1

        if not placed:
            if len(backtrack_stack)<=10: # 3
                pbar.close()
                print("Unable to place object, restarting scene generation...")
                return generate_scene_config(object_dict, num_objects, output_path)
            
            else:
                print("Retrace back to previous state...")
                for _ in range(min(Config.BACKTRACK_DEPTH, len(backtrack_stack))):
                    if backtrack_stack:
                        removed_obj, voxels = backtrack_stack.pop()
                        scene_state.remove_last_object()
                        voxel_grid.free_voxels(voxels)
                        object_queue.appendleft(removed_obj)
                        pbar.update(-1)
    
    pbar.close()
    
    # 7. Check final collision
    if not final_collision_check(scene_state):
        print("Check collision failed, regenerating scene...")
        return generate_scene_config(object_dict, num_objects)
    
    # 8. Generate config dictionary
    count = {key: 0 for key in list(OBJ_INDEX.keys())}
    namestring = ""
    scene_object_idxs = []
    translation = []
    scales = []
    for obj in scene_state.objects:
        id = obj.obj_id
        position = obj.position
        scale = obj.scale
        count[INDEX_OBJ[id]]+=1
        namestring += "_" + INDEX_OBJ[id]
        scene_object_idxs.append(id)
        translation.append(position)
        scales.append(scale)
    
    key = namestring
    value = {
        "scene_object_idxs": scene_object_idxs,
        "translation": translation,
        "scale": scales
        } 
    return key, value

def final_collision_check(scene_state):
    min_distance = float('inf')
    for i in range(len(scene_state.objects)):
        for j in range(i + 1, len(scene_state.objects)):
            dist = np.linalg.norm(
                np.array(scene_state.objects[i].position) - 
                np.array(scene_state.objects[j].position)
            )
            min_distance = min(min_distance, dist)
    
    if min_distance < Config.SAFETY_MARGIN * 0.5:
        print(f"Warning：Least object distance {min_distance:.4f} < {Config.SAFETY_MARGIN * 0.5:.4f}")
        return False
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=INPUT_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--obj_num", type=int, default=0)
    parser.add_argument("--scene_num", type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.output_path,exist_ok=True)

    object_dict = {}
    available_objects = list(INDEX_OBJ.values())
    
    for obj_name in available_objects:
        file_path = os.path.join(args.input_path, obj_name, "point_cloud/iteration_30000/point_cloud.ply")
        plydata = PlyData.read(file_path)
        vertices = plydata['vertex']
        x = vertices['x']
        y = vertices['y']
        z = vertices['z']
        points = np.vstack([x,y,z]).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
            
        object_dict[obj_name] = SceneObject(OBJ_INDEX[obj_name], pcd, scale=PRIOR_SCALE[OBJ_INDEX[obj_name]])
    
    for num_objects in range(args.obj_num, args.obj_num + 1):
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        config_total = {}
        output_path = os.path.join(args.output_path, f"{num_objects}.json")
        print(f"Generating scenes with {num_objects} objects...")
        for i in tqdm(range(args.scene_num), desc="generating config"): # obj_num 2 3: 3000
            key, value = generate_scene_config(object_dict, num_objects, output_path)
            key = f"{i}" + key
            config_total[key] = value

        with open(output_path, 'w') as f:
            json.dump(config_total, f, indent=2)
    
        print(f"Completed! Configs saved into {output_path}.")
    