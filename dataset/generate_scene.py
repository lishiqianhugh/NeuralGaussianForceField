# load two gaussians and combine them and save into a single file

import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from .constants import INDEX_OBJ


def combine_ply_files(input_paths, output_path, translation, scale):
    """
    Combine multiple PLY files into a single PLY file.
    
    Args:
        input_paths (list): List of paths to input PLY files
        output_path (str): Path where the combined PLY file will be saved
    """
    combined_data = {
        'x': [], 'y': [], 'z': [], 'opacity': [],
        'f_dc_0': [], 'f_dc_1': [], 'f_dc_2': []
    }
    assert len(translation) == len(input_paths), "Translation list must match the number of input files."
    all_extra_f_names = set()
    all_scale_names = set()
    all_rot_names = set()

    for path in input_paths:
        ply = PlyData.read(path)
        props = ply.elements[0].properties
        
        all_extra_f_names.update(p.name for p in props if p.name.startswith("f_rest_"))
        all_scale_names.update(p.name for p in props if p.name.startswith("scale_"))
        all_rot_names.update(p.name for p in props if p.name.startswith("rot"))

    extra_f_names = sorted(all_extra_f_names)
    scale_names = sorted(all_scale_names)
    rot_names = sorted(all_rot_names)
    
    # Initialize lists for extra features, scales, and rotations
    for name in extra_f_names:
        combined_data[name] = []
    for name in scale_names:
        combined_data[name] = []
    for name in rot_names:
        combined_data[name] = []
    
    # Combine data from all files
    for i, ply_path in enumerate(input_paths):
        plydata = PlyData.read(ply_path)
        vertex_data = plydata.elements[0]
        
        x=vertex_data['x']
        y=vertex_data['y']
        z=vertex_data['z']
        points=np.column_stack((x,y,z))

        scale_factor = scale[i]
        scaled_points = points * scale_factor

        vertex_data['x'] = scaled_points[:, 0]
        vertex_data['y'] = scaled_points[:, 1]
        vertex_data['z'] = scaled_points[:, 2]

        for name in scale_names:
            vertex_data[name]=np.array(vertex_data[name]) + np.log(scale_factor)

        # add translation to x, y, z
        vertex_data['x'] = np.array(vertex_data['x']) + translation[i][0]
        vertex_data['y'] = np.array(vertex_data['y']) + translation[i][1]
        vertex_data['z'] = np.array(vertex_data['z']) + translation[i][2]
        
        # Append basic attributes
        combined_data['x'].extend(vertex_data['x'])
        combined_data['y'].extend(vertex_data['y'])
        combined_data['z'].extend(vertex_data['z'])
        combined_data['opacity'].extend(vertex_data['opacity'])
        combined_data['f_dc_0'].extend(vertex_data['f_dc_0'])
        combined_data['f_dc_1'].extend(vertex_data['f_dc_1'])
        combined_data['f_dc_2'].extend(vertex_data['f_dc_2'])
        
        # Append extra features
        for name in extra_f_names:
            if name in vertex_data.properties:
                combined_data[name].extend(vertex_data[name])
            else:
                padding = [0.0] * len(vertex_data['x'])
                combined_data[name].extend(padding)
        
        # Append scales
        for name in scale_names:
            combined_data[name].extend(vertex_data[name])
            
        # Append rotations
        for name in rot_names:
            combined_data[name].extend(vertex_data[name])
    # Create vertex element
    vertex_properties = []
    
    # Define property types based on the first file's structure
    property_types = {
        'x': 'float32', 'y': 'float32', 'z': 'float32',
        'opacity': 'float32',
        'f_dc_0': 'float32', 'f_dc_1': 'float32', 'f_dc_2': 'float32'
    }
    
    # Add basic properties
    for name in ['x', 'y', 'z', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2']:
        vertex_properties.append((name, property_types[name]))
    
    # Add extra feature properties
    for name in extra_f_names:
        vertex_properties.append((name, 'float32'))
    
    # Add scale properties
    for name in scale_names:
        vertex_properties.append((name, 'float32'))
    
    # Add rotation properties
    for name in rot_names:
        vertex_properties.append((name, 'float32'))
    
    # Create vertex element
    vertex = np.zeros(len(combined_data['x']), dtype=vertex_properties)

    # Fill vertex data
    for name in combined_data:
        vertex[name] = combined_data[name]
    
    # Create PlyData object and save
    ply = PlyData([PlyElement.describe(vertex, 'vertex')])
    ply.write(output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate combined scenes from JSON configuration")
    parser.add_argument("--input", required=True, help="Path to input JSON scenes file")
    parser.add_argument("--output", required=True, help="Output directory for generated scenes")
    args = parser.parse_args()

    try:
        with open(args.input, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit(1)
    
    print(f"Processing {len(config)} scenes from {args.input}")
    for key in tqdm(config):
        scene_name = key
        scene_object_idxs = config[key]['scene_object_idxs']
        scale = config[key]['scale']
        translation = config[key]['translation']

        pcd_files = [f'./data/GSCollision/objects/{INDEX_OBJ[i]}/point_cloud/iteration_30000/point_cloud.ply' for i in scene_object_idxs]
        output_dir = os.path.join(args.output, scene_name, 'point_cloud', 'iteration_30000')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)
            
        combine_ply_files(pcd_files, os.path.join(output_dir, 'point_cloud.ply'), translation=translation,scale=scale)
        # copy camera file into the new saved file
        camera_file = os.path.join('./data/GSCollision/objects', INDEX_OBJ[scene_object_idxs[0]], 'cameras.json')
        output_camera_file = os.path.join(args.output, scene_name, 'cameras.json')
        # save camera file into the new folder
        with open(camera_file, 'r') as f:
            cameras = json.load(f)
            with open(output_camera_file, 'w') as f:
                json.dump(cameras, f)
        