import os
import torch
import numpy as np
from utils.general_utils import find_indexes
from .constants import INDEX_OBJ, OBJPART, INDEX_OBJ_TEXT

def group_objects(positions, opacity_mask, seq_dir):
    """
    Group point cloud into objects using file name.

    Args:
        positions (torch.Tensor): [N, 3] tensor of point positions.
        opacity_mask (torch.Tensor): [N, 1] tensor indicating which points are visible.

    Returns:
        obj_ids (torch.Tensor): [N, 1] tensor with object IDs assigned to each point.
    """
    if 'segmentation.npy' in os.listdir(seq_dir):
        # For testing, use segmentation file
        seg_path = os.path.join(seq_dir, 'segmentation.npy')
        obj_ids = torch.tensor(np.load(seg_path), dtype=torch.long).unsqueeze(-1)
        # the loaded object ids includes low opacity kernels, so we need to filter them out
        obj_ids = obj_ids[opacity_mask, :]
    else:
        # For training, use predefined object IDs based on sequence directory
        obj_ids = torch.zeros(positions.shape[0], dtype=torch.long).unsqueeze(-1)  # (N, 1)
        seq_dir = seq_dir.replace('mpm', 'scenes').replace('table_', '')
        scene_object_idxs = find_indexes(seq_dir)
        # filter objects to only those in the predefined list and keep the order
        objects = [INDEX_OBJ[i] for i in scene_object_idxs if i in INDEX_OBJ.keys()]
        objects = [obj for obj in objects if obj in OBJPART.keys()]
        current_point_num = 0
        for i in range(0, len(objects)):
            obj_ids[current_point_num : current_point_num + OBJPART[objects[i]]] = i
            current_point_num = current_point_num + OBJPART[objects[i]]
    return obj_ids


def generate_point_prompt(pos, camera, seq_dir):
    objs_id = group_objects(pos, None, seq_dir)
    obj_points = []
    num_objs = objs_id.max().item() + 1

    w, h = camera['width'], camera['height']
    fx = camera['fx']
    fy = camera['fy']
    cx, cy = w / 2, h / 2
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0, 1]], dtype=pos.dtype, device=pos.device)

    R = torch.as_tensor(camera['rotation'], dtype=pos.dtype, device=pos.device)  # (3,3)
    T = torch.as_tensor(camera['position'], dtype=pos.dtype, device=pos.device)  # (3,)

    for i in range(num_objs):
        obj_mask = objs_id == i
        obj_pos = pos[obj_mask.squeeze(-1), :]   # (N_i, 3)
        obj_center = obj_pos.mean(dim=0, keepdim=True)  # (1, 3)

        # World -> Camera
        cam_coord = R.T @ (obj_center.T - T.view(3, 1))  # (3,1)

        uvw = K @ cam_coord                            # (3,1)
        u = uvw[0] / uvw[2]
        v = uvw[1] / uvw[2]

        obj_points.append([u, v])  

    points = torch.tensor(obj_points)
    return points  # shape: (num_objs, 2)

def generate_bbox_prompt(pos, camera, seq_dir):
    objs_id = group_objects(pos, None, seq_dir)
    # visualize_object_seg(pos, objs_id, seq_dir, frame_idx=0)
    bboxes_2d = []
    num_objs = objs_id.max().item() + 1

    w, h = camera['width'], camera['height']
    fx = camera['fx']
    fy = camera['fy']
    cx, cy = w / 2, h / 2
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0, 1]], dtype=pos.dtype, device=pos.device)

    R = torch.as_tensor(camera['rotation'], dtype=pos.dtype, device=pos.device)  # (3,3)
    translation = torch.as_tensor(camera['position'], dtype=pos.dtype, device=pos.device)  # (3,)

    for i in range(num_objs):
        obj_mask = objs_id == i
        obj_pos = pos[obj_mask.squeeze(-1), :]   # (N_i, 3)

        if obj_pos.shape[0] == 0:
            continue

        # --- World -> Camera ---
        cam_coords = R.T @ (obj_pos - translation).T  # (3,N)

        uvw = K @ cam_coords  # (3,N)
        u = uvw[0] / uvw[2]
        v = uvw[1] / uvw[2]

        # 2D bbox
        umin, umax = u.min().item(), u.max().item()
        vmin, vmax = v.min().item(), v.max().item()
        bboxes_2d.append([umin, vmin, umax, vmax])

    return torch.tensor(bboxes_2d)  # shape: (num_objs, 4)

def generate_text_prompt(seq_dir):
    seq_dir = seq_dir.replace('mpm', 'scenes')
    scene_object_idxs = find_indexes(seq_dir)
    texts = [INDEX_OBJ_TEXT[idx] for idx in scene_object_idxs if idx in INDEX_OBJ_TEXT]
    return texts