import os
os.environ['SPCONV_ALGO'] = 'native'
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
# os.environ["OMP_PROC_BIND"] = "false"
# os.environ["ORT_DISABLE_THREAD_AFFINITY"] = "1"
import onnxruntime as ort
import warnings
ort.set_default_logger_severity(4)
warnings.filterwarnings('ignore')

import json
import argparse
import types
import math
from typing import Dict, Tuple, List, Optional, Union, Sequence
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import imageio
import torch_cluster  # for knn

from .reconstruct import prepare_models as prepare_models
from .reconstruct import reconstruction as reconstruction

from utils.general_utils import setup_seed
from utils.camera_utils import extract_camera_matrices
from utils.geometry_utils import farthest_point_sampling_indices
from utils.decode_param import decode_param_json
from utils.camera_utils import get_camera_view
from utils.transformation_utils import get_center_view_worldspace_and_observant_coordinate,matrix3x3_to_dof6,dof6_to_matrix3x3,upsample_gaussian_splatting, transform_gaussians, euler_xyz_to_matrix, align_camera_poses, rotation_quaternion_to_matrix,ensure_spd
from utils.sh_utils import _pad_sh_to_degree2
from gsplat import rasterization

def propagate_covariances_knn(
    new_pos: torch.Tensor,
    ori_pos: torch.Tensor,
    ori_cov3x3: torch.Tensor,
    k: int = 32,
) -> torch.Tensor:
    """
    Propagate 3x3 covariances from original points to new points by KNN weighted average.
    All tensors must be on the same device (cuda) and float32/float16 compatible.

    new_pos: [N_new, 3]
    ori_pos: [N_ori, 3]
    ori_cov3x3: [N_ori, 3, 3]
    returns: [N_new, 3, 3]
    """

    device = new_pos.device
    N_new = new_pos.shape[0]
    if ori_pos.shape[0] == 0:
        # Fallback: return tiny isotropic
        eye = torch.eye(3, device=device, dtype=new_pos.dtype)
        return eye.unsqueeze(0).repeat(N_new, 1, 1) * 1e-5

    # torch_cluster.knn expects (x, y, k) returns edge_index with y->x mapping
    # We need neighbors in ori_pos for each new_pos
    edge_index = torch_cluster.knn(x=ori_pos, y=new_pos, k=min(k, max(1, ori_pos.shape[0])))
    row, col = edge_index  # row: index in new_pos, col: index in ori_pos

    # Simple inverse-distance weights
    diff = new_pos[row] - ori_pos[col]
    dist_sq = (diff * diff).sum(dim=-1) + 1e-12
    w = 1.0 / dist_sq

    # Normalize weights per new point
    weights_sum = torch.zeros(N_new, device=device, dtype=w.dtype)
    weights_sum.index_add_(0, row, w)
    w_norm = w / weights_sum[row].clamp(min=1e-12)

    # Weighted sum of covariance matrices
    cov_agg = torch.zeros(N_new, 3, 3, device=device, dtype=ori_cov3x3.dtype)
    batch_cov = ori_cov3x3[col]
    cov_agg.index_add_(0, row, batch_cov * w_norm.view(-1, 1, 1))

    return ensure_spd(cov_agg)

def upsample_basic(
    downsampled: Dict[str, torch.Tensor],
    ori_pos: torch.Tensor,
    ori_obj_ids: torch.Tensor = None,
    k: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Minimal upsampling on GPU: interpolate positions per frame, and opacity/shs per point.
    Does not upsample covariances/rot; returns rot from original as a placeholder.
    Inputs:
      - downsampled['pos']: [B, T, N_down, 3]
      - downsampled['opacity']: [B, N_down, 1]
      - downsampled['shs']: [B, N_down, 9, 3]
      - ori_pos: [B, N_up, 3]
      - ori_obj_ids: [B, N_up, 1] optional; if provided, do KNN per object id
    Returns:
      - {'pos':[B,T,N_up,3], 'opacity':[B,N_up,1], 'shs':[B,N_up,9,3], 'rot':[B,N_up,3,3]}
    """
    import torch_cluster

    device = downsampled['pos'].device
    dtype = downsampled['pos'].dtype

    B, T, N_down, _ = downsampled['pos'].shape
    N_up = ori_pos.shape[1]

    up = {
        'pos': torch.zeros(B, T, N_up, 3, dtype=dtype, device=device),
        'opacity': torch.zeros(B, N_up, 1, dtype=dtype, device=device),
        'shs': torch.zeros(B, N_up, downsampled['shs'].shape[2], downsampled['shs'].shape[3], dtype=dtype, device=device),
        'rot': None,
    }

    for b in range(B):
        pos_down_first = downsampled['pos'][b, 0]  # [N_down, 3]
        pos_up = ori_pos[b]                        # [N_up, 3]

        if ori_obj_ids is not None:
            unique_obj_ids = torch.unique(ori_obj_ids[b])
            row_all = []
            col_all = []
            for obj_id in unique_obj_ids:
                mask_up = (ori_obj_ids[b] == obj_id).squeeze(-1)
                # For downsampled mask, map via keypoint_indices if exists
                if 'keypoint_indices' in downsampled:
                    kp_idx = downsampled['keypoint_indices'][b]
                    # infer per-downsampled obj via original ids at kp positions
                    down_obj_ids = ori_obj_ids[b][kp_idx].squeeze(-1)
                    mask_down = (down_obj_ids == obj_id)
                else:
                    mask_down = None

                if (mask_down is None) or (not mask_down.any()) or (not mask_up.any()):
                    continue

                edge_index = torch_cluster.knn(
                    x=pos_down_first[mask_down],
                    y=pos_up[mask_up],
                    k=min(k, int(mask_down.sum().item())),
                )
                row_obj, col_obj = edge_index
                global_row = torch.where(mask_up)[0][row_obj]
                global_col = torch.where(mask_down)[0][col_obj]
                row_all.append(global_row)
                col_all.append(global_col)

            if len(row_all) == 0:
                # fallback to global knn
                edge_index = torch_cluster.knn(x=pos_down_first, y=pos_up, k=k)
                row, col = edge_index
            else:
                row = torch.cat(row_all, dim=0)
                col = torch.cat(col_all, dim=0)
        else:
            edge_index = torch_cluster.knn(x=pos_down_first, y=pos_up, k=k)
            row, col = edge_index

        counts = torch.zeros(N_up, 1, device=device, dtype=dtype)
        counts.index_add_(0, row, torch.ones(row.size(0), 1, device=device, dtype=dtype))

        for t in range(T):
            pos_frame_up = torch.zeros(N_up, 3, dtype=dtype, device=device)
            pos_frame_up.index_add_(0, row, downsampled['pos'][b, t, col])
            up['pos'][b, t] = pos_frame_up / counts.clamp(min=1.0)

        opacity_up = torch.zeros(N_up, 1, dtype=dtype, device=device)
        opacity_up.index_add_(0, row, downsampled['opacity'][b][col])
        up['opacity'][b] = opacity_up / counts.clamp(min=1.0)

        SH_dim, SH_dim2 = downsampled['shs'].shape[2], downsampled['shs'].shape[3]
        shs_up = torch.zeros(N_up, SH_dim, SH_dim2, dtype=dtype, device=device)
        counts_expanded = counts.view(N_up, 1, 1).expand(-1, SH_dim, SH_dim2)
        shs_up.index_add_(0, row, downsampled['shs'][b][col])
        up['shs'][b] = shs_up / counts_expanded.clamp(min=1.0)

    return up


def reconstruct_in_memory(
    pretrain_models: Dict[str, object],
    data_path: str,
    seed: int = 0,
    interval: int = 4,
    prompt_type: str = 'bbox',
    real_world: bool = False,
    visualize: bool = False,
) -> Dict:
    """
    Call reconstruct_refine.reconstruction while preventing disk writes of intermediates.
    Returns the result_dict with tensors on GPU where applicable.
    """
    setup_seed(seed)

    # Resolve input path for reconstruction: if directory is provided, pick the mp4 inside
    recon_input = data_path
    if os.path.isdir(data_path):
        # Prefer a file named "+{scene}.mp4", otherwise any single .mp4
        mp4s = [f for f in os.listdir(data_path) if f.endswith('.mp4')]
        if len(mp4s) == 0:
            raise FileNotFoundError(f"No .mp4 found under directory: {data_path}")
        # scene dir is parent of table dir
        scene_dir = os.path.basename(os.path.dirname(os.path.normpath(data_path)))
        preferred = f"+{scene_dir}.mp4"
        if preferred in mp4s:
            recon_input = os.path.join(data_path, preferred)
        else:
            # fallback to first mp4
            recon_input = os.path.join(data_path, mp4s[0])

    # Build a minimal args namespace expected by reconstruction
    args = types.SimpleNamespace()
    args.seed = seed
    args.data_path = recon_input
    args.save_folder = os.path.join("/tmp", "ngff_in_memory")  # harmless dir creation
    args.interval = interval
    args.prompt_type = prompt_type
    args.sd_version = 'sd15'
    args.text_prompt = ''
    args.initial_frame_idx = 0
    args.real_world = real_world
    args.visualize = visualize
    args.refine = False  # not used in reconstruction itself
    args.batch_inference_json = None

    result_dict = reconstruction(args, pretrain_models, save_pt=True)

    # Move heavy tensors to cuda if available
    for k in list(result_dict.keys()):
        v = result_dict[k]
        if torch.is_tensor(v):
            result_dict[k] = v.cuda()
        # nested Gaussians inside result_dict['gaussians'] are custom class with tensors

    g = result_dict['gaussians']
    g.means = g.means.cuda()
    g.covariances = g.covariances.cuda()
    g.harmonics = g.harmonics.cuda()
    g.opacities = g.opacities.cuda()
    g.scales = g.scales.cuda()
    g.rotations = g.rotations.cuda()

    return result_dict


def preprocess_data_in_memory(
    result_dict: Dict,
    data_path: str,
    refine: bool,
    real_world: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, int]:
    """
    Replicate reconstruct_refine.preprocess_data behavior without any writes.
    Returns (sim_gaussians, bg_gaussians, labels) as dictionaries of CUDA tensors.
    """
    device = torch.device('cuda')

    positions = result_dict['gaussians'].means.squeeze(0).to(device)
    covariances = result_dict['gaussians'].covariances.squeeze(0).to(device)
    harmonics = result_dict['gaussians'].harmonics.squeeze(0).to(device)
    opacities = result_dict['gaussians'].opacities.squeeze(0).to(device)
    rotations = result_dict['gaussians'].rotations.squeeze(0).to(device)
    scales = result_dict['gaussians'].scales.squeeze(0).to(device)
    labels = result_dict['segmentation_results'].reshape(-1).to(device)

    # Ensure rotations are 3x3 matrices (convert from quaternion if needed)
    if rotations.dim() == 2 and rotations.shape[-1] == 4:
        rotations = rotation_quaternion_to_matrix(rotations)  # [N, 3, 3]
    elif rotations.dim() == 3 and rotations.shape[-2:] == (3, 3):
        pass
    else:
        # Fallback: identity rotations
        Np = positions.shape[0]
        rotations = torch.eye(3, device=device, dtype=positions.dtype).unsqueeze(0).repeat(Np, 1, 1)

    # Load GT camera positions for alignment
    if real_world:
        parent = '/mnt/nfs_project_a/shared/models/ngff/ngff_perception/datasets/realworld_0919'
        with open(f'{parent}/cameras.json', 'r') as f:
            gt_camera = json.load(f)
        gt_camera_positions = [gt_camera[i]['position'] for i in range(0, len(gt_camera))]
        gt_camera_positions = torch.tensor(gt_camera_positions, device=device)
        center_gt = gt_camera_positions.mean(dim=0)
        gt_camera_positions = (gt_camera_positions - center_gt) * 0.5 + center_gt
        gt_camera_positions[..., 2] -= 1.0
    else:
        parent = os.path.dirname(data_path) if not os.path.isdir(data_path) else data_path
        with open(f'{parent}/cameras.json', 'r') as f:
            gt_camera = json.load(f)
        # interval=4 + skip every 5th -> 20 images -> match reconstruction
        gt_camera_positions = [gt_camera[i]['position'] for i in range(0, len(gt_camera), 4)]
        gt_camera_positions = [gt_camera_positions[idx] for idx in range(0, len(gt_camera_positions)) if idx % 5 != 4]
        gt_camera_positions = torch.tensor(gt_camera_positions, device=device)

    pred_camera_poses = result_dict['camera_poses'].to(device)
    pred_camera_positions = pred_camera_poses[..., :3, 3]

    # Align cameras
    transformation_matrix, scale_factor = align_camera_poses(gt_camera_positions, pred_camera_positions)
    camera_center = pred_camera_poses[:, :3, 3].mean(dim=0)

    R = transformation_matrix[:3, :3]
    T = transformation_matrix[:3, 3]
    S = scale_factor * torch.eye(3, device=R.device, dtype=R.dtype)
    A = S @ R

    transformed_positions = torch.matmul(scale_factor * (positions - camera_center), R.T) + T + camera_center
    transformed_covariances = A @ covariances @ A.transpose(-1, -2)
    transformed_harmonics = harmonics
    transformed_opacities = opacities
    transformed_rotations = rotations
    transformed_scales = scales * scale_factor

    # Build masks for sim (foreground) vs bg, matching refine flag logic
    assert labels.max() % 2 == 0, "Segmentation results must be even per reconstruct_refine convention"
    obj_num = (labels.max() // 2).item()

    if refine:
        sim_area_mask = torch.logical_and(labels != 0, labels <= obj_num)
    else:
        sim_area_mask = torch.logical_and(labels != 0, labels > obj_num)

    if not real_world:
        inside_box_mask = (
            (transformed_positions[..., 2] > -0.98)
            & (transformed_positions[..., 0] > -1)
            & (transformed_positions[..., 0] < 1)
            & (transformed_positions[..., 1] > -1)
            & (transformed_positions[..., 1] < 1)
        )
        out_of_box_mask = torch.logical_and(labels == 0, ~inside_box_mask)
    else:
        out_of_box_mask = labels == 0

    sim_mask = sim_area_mask
    bg_mask = out_of_box_mask

    sim = {
        'pos': transformed_positions[sim_mask],
        'cov': transformed_covariances[sim_mask],
        'rot': rotations[sim_mask],
        'opc': transformed_opacities[sim_mask].reshape(-1, 1),
        'shs': _pad_sh_to_degree2(transformed_harmonics[sim_mask]),
        'labels': labels[sim_mask],
    }
    bg = {
        'pos': transformed_positions[bg_mask],
        'cov': transformed_covariances[bg_mask],
        'rot': rotations[bg_mask],
        'opc': transformed_opacities[bg_mask].reshape(-1, 1),
        'shs': _pad_sh_to_degree2(transformed_harmonics[bg_mask]),
    }

    return sim, bg, labels, obj_num


def _fps_per_object_global_indices(
    pos: torch.Tensor,
    obj_ids: torch.Tensor,
    k_per_object: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run FPS per object on CPU to get global indices and corresponding object ids.
    pos: [N,3] (CUDA)
    obj_ids: [N] (CUDA) with foreground ids (e.g., 1..K), background already removed
    return: (global_indices [K*per_obj], obj_ids_at_indices [K*per_obj]) both on CUDA
    """
    device = pos.device
    pos_cpu = pos.detach().cpu()
    obj_cpu = obj_ids.detach().cpu().unsqueeze(-1)
    kp_idx_cpu = farthest_point_sampling_indices(pos_cpu, obj_cpu, k_per_object)
    obj_ids_kp_cpu = obj_cpu[kp_idx_cpu].squeeze(1)
    return kp_idx_cpu.to(device), obj_ids_kp_cpu.to(device)


def prepare_eval_sample(
    sim_gaussians: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    obj_num: int,
    refine: bool,
    num_keypoints: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build a minimal evaluation sample in memory, analogous to DGSDataset._process_ply_dir.
    Returns dict with fields required by dynamics + upsampling.
    """
    pos_full = sim_gaussians['pos'].to(device)
    cov_full = sim_gaussians['cov'].to(device)
    rot_full = sim_gaussians['rot'].to(device)
    opc_full = sim_gaussians['opc'].to(device)
    shs_full = sim_gaussians['shs'].to(device)

    # Object ids: preserve original semantics (refine: 1..K; non-refine: K+1..2K -> 1..K)
    labels_full = sim_gaussians['labels'] if 'labels' in sim_gaussians else labels[pos_full.new_zeros(pos_full.shape[0], dtype=torch.long)]
    if not refine:
        labels_full = labels_full - int(obj_num)
    obj_ids_all = labels_full  # expected in 1..K

    # FPS per object on CPU, return global indices on CUDA
    kp_idx, obj_ids_kp = _fps_per_object_global_indices(pos_full, obj_ids_all, num_keypoints)

    # Downsampled per-keypoint data (T=1)
    pos_down = pos_full[kp_idx].unsqueeze(0).unsqueeze(0)  # [B=1, T=1, N_down, 3]
    rot_down = rot_full[kp_idx].unsqueeze(0).unsqueeze(0)  # [1,1,N_down,3,3]
    opc_down = opc_full[kp_idx].unsqueeze(0)  # [1, N_down, 1]
    shs_down = shs_full[kp_idx].unsqueeze(0)  # [1, N_down, 9, 3]

    # Original dense data (B=1)
    ori_pos = pos_full.unsqueeze(0)  # [1, N_up, 3]
    ori_cov_mat = cov_full.unsqueeze(0)  # [1, N_up, 3, 3]
    ori_rot = rot_full.unsqueeze(0)  # [1, N_up, 3, 3]
    ori_opacity = opc_full.unsqueeze(0)  # [1, N_up, 1]
    ori_shs = shs_full.unsqueeze(0)  # [1, N_up, 9, 3]
    ori_obj_ids = labels_full.unsqueeze(0).unsqueeze(-1)  # [1, N_up, 1]

    # Convert original 3x3 covariance to DOF6 to match original transform flow
    ori_cov_dof6 = matrix3x3_to_dof6(ori_cov_mat[0]).unsqueeze(0)  # [1, N_up, 6]

    # Object-centric tensors (minimal placeholders consistent with NGFF interface)
    num_objs = int(obj_num)
    T = 1
    N_kp = pos_down.shape[2]
    points = torch.zeros((T, num_objs, num_keypoints, 3), device=device, dtype=pos_full.dtype)
    com = torch.zeros((T, num_objs, 3), device=device, dtype=pos_full.dtype)
    angle = torch.zeros_like(com)
    padding = torch.zeros((num_objs, num_keypoints), device=device, dtype=pos_full.dtype)
    knn = torch.zeros((num_objs, num_keypoints, 8), device=device, dtype=torch.long)  # k placeholder

    # Fill per object from downsampled keypoints
    # For simplicity, uniformly assign sampled kp to objects according to obj_ids_kp order
    # Build mask indices
    obj_ids_kp = obj_ids_kp.to(device)
    for j in range(1, num_objs + 1):
        mask = (obj_ids_kp == j)
        idx_local = torch.where(mask)[0]
        cnt = min(idx_local.shape[0], num_keypoints)
        if cnt == 0:
            continue
        src = pos_full[kp_idx[idx_local[:cnt]]]
        points[0, j-1, :cnt] = src
        padding[j-1, :cnt] = 1.0
        com[0, j-1] = src.mean(dim=0)

    com_vel = torch.zeros_like(com)
    angle_vel = torch.zeros_like(angle)

    data = {
        'pos': pos_down,                      # [1,1,N_down,3]
        # Note: do not provide DOF6 here; we will propagate 3x3 later per frame
        'rot': rot_down,                      # [1,1,N_down,3,3]
        'opacity': opc_down,                  # [1,N_down,1]
        'shs': shs_down,                      # [1,N_down,9,3]
        'keypoint_indices': kp_idx.unsqueeze(0),  # [1, N_down]
        'obj_ids': labels_full[kp_idx].unsqueeze(-1),
        'ori_pos': ori_pos,                   # [1,N_up,3]
        'ori_cov3D_mat': ori_cov_mat,         # [1,N_up,3,3]
        'ori_cov3D': ori_cov_dof6,            # [1,N_up,6]
        'ori_rot': ori_rot,                   # [1,N_up,3,3]
        'ori_opacity': ori_opacity,           # [1,N_up,1]
        'ori_shs': ori_shs,                   # [1,N_up,9,3]
        'points': points.unsqueeze(0),        # [1,T,num_objs,N,3] (add batch dim)
        'com': com.unsqueeze(0),              # [1,T,num_objs,3]
        'angle': angle.unsqueeze(0),          # [1,T,num_objs,3]
        'padding': padding.unsqueeze(0),      # [1,num_objs,N]
        'knn': knn.unsqueeze(0),              # [1,num_objs,N,k]
        'com_vel': com_vel.unsqueeze(0),      # [1,T,num_objs,3]
        'angle_vel': angle_vel.unsqueeze(0),  # [1,T,num_objs,3]
        'ori_obj_ids': ori_obj_ids,           # [1,N_up,1]
    }
    return data


def load_dynamic_model(model_name: str) -> Tuple[nn.Module, Dict]:
    """
    Load dynamics model and its args.json to merge runtime params (paths follow eval.py).
    """
    name = model_name.lower()
    if name == 'ngff':
        weight_path = '/mnt/nfs_project_a/shared/models/ngff/ngff_perception/datasets/GSCollision/exps/ngff/out_2025-09-15-16-26-50/ngff_best.pth'
    elif name == 'gcn':
        weight_path = '/mnt/nfs_project_a/shared/models/ngff/ngff_perception/datasets/GSCollision/exps/gcn/out_2025-09-11-10-51-44/ngff_best.pth'
    elif name == 'pointformer':
        weight_path = '/mnt/nfs_project_a/shared/models/ngff/ngff_perception/datasets/GSCollision/exps/pointformer/out_2025-09-16-17-16-19/ngff_best.pth'
    else:
        raise ValueError(f"Unknown dynamic model: {model_name}")

    args_json_path = os.path.join(weight_path.split('ngff_best.pth')[0], 'args.json')
    with open(args_json_path, 'r') as f:
        args_json = json.load(f)

    # Instantiate model per eval.py
    output_dim = args_json['output_dim']
    hidden_dim = args_json['hidden_dim']
    num_layers = args_json['num_layers']
    num_keypoints = args_json['num_keypoints']

    # Import models lazily
    if name == 'ngff':
        from dynamic_models import NGFFobj
        model = NGFFobj(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                        num_layers=num_layers, num_keypoints=num_keypoints, k=1, mass=args_json['mass'],
                        dt=args_json['dt'], ode_method=args_json['ode_method'], r=0.1, step_size=args_json['step_size'],
                        threshold=args_json['threshold'], rtol=args_json['rtol'], atol=args_json['atol'])
    elif name == 'pointformer':
        from dynamic_models import PointTransformer
        model = PointTransformer(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    elif name == 'gcn':
        from dynamic_models import GCN
        model = GCN(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, r=0.1)
    else:
        raise ValueError(f"Unknown dynamic model: {model_name}")

    sd = torch.load(weight_path, map_location='cuda')
    model.load_state_dict(sd)
    model = model.cuda().eval()
    return model, args_json


def build_cameras_from_config(cfg_path: str, steps: int, resolution: int):
    material_params, bc_params, time_params, preprocessing_params, camera_params = decode_param_json(cfg_path)

    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"]).reshape((1, 3)).cuda()
    )

    viewpoint_center_worldspace, observant_coordinates = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        [],
        1,
        torch.tensor(0),
    )

    current_cameras = []
    for frame in range(steps):
        current_camera, _ = get_camera_view(
            '',
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
            resolution=resolution,
        )
        current_cameras.append(current_camera)
    current_intrinsics, current_extrinsics = extract_camera_matrices(current_cameras)
    return current_intrinsics.cuda(), current_extrinsics.cuda(), time_params


_SPECIAL_VIEW_TOKENS = {"circle360", "circle360_jitter"}


def _parse_views_spec(views_spec: Union[str, Sequence[Union[int, str]]]) -> List[Union[int, str]]:
    if isinstance(views_spec, str):
        tokens: List[Union[int, str]] = [tok.strip() for tok in views_spec.split(',') if tok.strip()]
    elif isinstance(views_spec, Sequence):
        tokens = list(views_spec)
    else:
        raise TypeError(f"Unsupported views specification type: {type(views_spec)}")

    parsed: List[Union[int, str]] = []
    for token in tokens:
        if isinstance(token, int):
            parsed.append(int(token))
            continue

        if isinstance(token, str):
            lowered = token.lower()
            if lowered in _SPECIAL_VIEW_TOKENS:
                parsed.append(lowered)
                continue
            try:
                parsed.append(int(lowered))
                continue
            except ValueError as exc:
                raise ValueError(f"Invalid view token: '{token}'") from exc

        else:
            raise TypeError(f"Unsupported view token type: {type(token)}")

    return parsed


def _build_circle360_jitter_trajectory(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    jitter_ratio: float = 0.01,
    max_keyframes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if extrinsics.dim() != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError("Extrinsics must have shape [T, 4, 4] to build jitter trajectory")
    if intrinsics.dim() != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("Intrinsics must have shape [T, 3, 3] to build jitter trajectory")
    if intrinsics.shape[0] != extrinsics.shape[0]:
        raise ValueError("Intrinsics and extrinsics must have the same number of frames")

    frame_count = extrinsics.shape[0]
    if frame_count <= 1:
        return intrinsics.clone(), extrinsics.clone()

    device = extrinsics.device
    dtype = extrinsics.dtype

    c2w = torch.linalg.inv(extrinsics)
    base_positions = c2w[:, :3, 3]

    mean_radius = base_positions.norm(dim=-1).mean()
    jitter_scale = mean_radius * jitter_ratio

    keyframe_count = min(max_keyframes, frame_count)
    keyframe_indices = torch.linspace(
        0,
        frame_count - 1,
        steps=keyframe_count,
        dtype=torch.float64,
        device=device,
    ).round().to(torch.long)
    keyframe_indices = torch.unique_consecutive(keyframe_indices)
    if keyframe_indices.numel() <= 1:
        return intrinsics.clone(), extrinsics.clone()

    jitter_noise = torch.randn((keyframe_indices.numel(), 3), device=device, dtype=dtype) * jitter_scale
    keyframe_positions = base_positions[keyframe_indices] + jitter_noise

    # Prepare wrap-around for smooth looping
    extended_indices = torch.cat(
        [keyframe_indices, keyframe_indices[:1] + frame_count], dim=0
    )
    extended_positions = torch.cat([keyframe_positions, keyframe_positions[:1]], dim=0)

    interpolated_positions = base_positions.clone()
    for idx in range(extended_indices.numel() - 1):
        start_idx = int(extended_indices[idx].item())
        end_idx = int(extended_indices[idx + 1].item())
        if end_idx <= start_idx:
            continue

        segment_length = end_idx - start_idx
        start_pos = extended_positions[idx]
        end_pos = extended_positions[idx + 1]

        for offset in range(segment_length):
            t = start_idx + offset
            frame = t % frame_count
            alpha = offset / float(segment_length)
            interpolated_positions[frame] = (1.0 - alpha) * start_pos + alpha * end_pos

    c2w_jittered = c2w.clone()
    c2w_jittered[:, :3, 3] = interpolated_positions
    extrinsics_jittered = torch.linalg.inv(c2w_jittered)

    return intrinsics.clone(), extrinsics_jittered


def _prepare_camera_trajectory_for_view(
    view_spec: Union[int, str],
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if intrinsics.shape[0] != extrinsics.shape[0]:
        raise ValueError("Intrinsics and extrinsics must share the same frame count")

    frame_count = intrinsics.shape[0]

    if isinstance(view_spec, int):
        if view_spec < 0 or view_spec >= frame_count:
            raise IndexError(
                f"View index {view_spec} is out of bounds for available cameras: 0-{frame_count - 1}"
            )
        intrinsics_single = intrinsics[view_spec].unsqueeze(0).repeat(frame_count, 1, 1).contiguous()
        extrinsics_single = extrinsics[view_spec].unsqueeze(0).repeat(frame_count, 1, 1).contiguous()
        return intrinsics_single, extrinsics_single

    if isinstance(view_spec, str):
        lowered = view_spec.lower()
        if lowered == "circle360":
            return intrinsics.clone(), extrinsics.clone()
        if lowered == "circle360_jitter":
            return _build_circle360_jitter_trajectory(intrinsics, extrinsics)
        raise ValueError(f"Unsupported view specification: '{view_spec}'")

    raise TypeError(f"Unsupported view specification type: {type(view_spec)}")

def _derive_scene_keys(data_path: str) -> Tuple[str, str, str]:
    """
    Best-effort extraction of (group, scene, table) from a known dataset layout.
    Fallback to ('unknown','unknown','unknown') if not parseable.
    """
    # Accept either a directory path (.../{group}/{scene}/{table}/) or a file inside it
    base = data_path
    if os.path.isfile(base):
        base = os.path.dirname(base)
    parts = base.strip('/').split('/')
    return parts[-3], parts[-2], parts[-1]


@torch.no_grad()
def compute_trajectory_and_cov(
    data: Dict[str, torch.Tensor],
    model: nn.Module,
    model_name: str,
    steps: int,
    cfg_path: str,
    resolution: int,
):
    """
    Compute future trajectory, upsample to dense, and precompute covariances once.
    Returns: (upsampled, cov3D_dof6_seq, intrinsics, extrinsics, time_params)
    """
    intrinsics, extrinsics, time_params = build_cameras_from_config(cfg_path, steps=steps, resolution=resolution)

    points = data['points']
    com = data['com']
    com_vel = data['com_vel']
    angle = data['angle']
    angle_vel = data['angle_vel']
    padding = data['padding']
    knn = data['knn']

    name = model_name.lower()
    if name == 'ngff':
        point_seq, com_seq, angle_seq = model(points[:, 0], com[:, 0], com_vel[:, 0],
                                              angle[:, 0], angle_vel[:, 0], padding, knn,
                                              pred_len=steps - 1, external_forces=None)
        R_seq = euler_xyz_to_matrix(angle_seq)
        future_seq = torch.matmul(point_seq, R_seq.transpose(-2, -1)) + com_seq.unsqueeze(-2)
    elif name in ['pointformer', 'gcn']:
        future_seq = model(points[:, 0], com[:, 0], com_vel[:, 0],
                           angle[:, 0], angle_vel[:, 0], padding, knn,
                           pred_len=steps - 1, external_forces=None)
    else:
        raise ValueError(f"Unknown dynamic model: {model_name}")

    # Merge objects -> [B,T,N,3]
    B, T_pred, num_objs, N, C = future_seq.shape
    mask = padding.unsqueeze(0).unsqueeze(1).reshape(B, 1, num_objs * N, 1).repeat(1, T_pred, 1, C).bool()
    merged = future_seq.reshape(B, T_pred, -1, C)
    merged = merged[mask].reshape(B, T_pred, -1, C)

    # Downsampled dict for upsampling
    downsampled = {
        'pos': merged,
        'cov3D': torch.zeros(1, 1, merged.shape[2], 6, device=merged.device, dtype=merged.dtype),
        'rot': data['rot'],
        'opacity': data['opacity'],
        'shs': data['shs'],
        'keypoint_indices': data['keypoint_indices'],
    }
    upsampled = upsample_gaussian_splatting(downsampled, data['ori_pos'], data['ori_obj_ids'], k=8)

    # Precompute covariances once
    cov3D_dof6_seq = transform_gaussians(upsampled['pos'], data['ori_cov3D'], k=32)

    # Free sources we no longer need
    del future_seq, merged, downsampled
    torch.cuda.empty_cache()

    return upsampled, cov3D_dof6_seq, intrinsics, extrinsics, time_params


@torch.no_grad()
def render_single_view(
    data: Dict[str, torch.Tensor],
    bg: Dict[str, torch.Tensor],
    upsampled: Dict[str, torch.Tensor],
    cov3D_dof6_seq: torch.Tensor,
    intrinsics_traj: torch.Tensor,
    extrinsics_traj: torch.Tensor,
    time_params: Dict,
    resolution: int,
    output_path: str,
):
    fps = int(1.0 / time_params.get("frame_dt", 1/24))
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')
    os.makedirs(output_path, exist_ok=True)
    scene_name = output_path.split('/')[-3]
    writer = imageio.get_writer(os.path.join(output_path, f"{scene_name}.mp4"), fps=fps)
    try:
        steps = upsampled['pos'].shape[1]
        if intrinsics_traj.shape[0] != steps or extrinsics_traj.shape[0] != steps:
            raise ValueError(
                "Camera trajectory length must match rendered steps: "
                f"expected {steps}, got {intrinsics_traj.shape[0]} (K) and {extrinsics_traj.shape[0]} (viewmats)"
            )
        for frame in range(steps):
            current_extrinsic = extrinsics_traj[frame].unsqueeze(0)
            current_intrinsic = intrinsics_traj[frame].unsqueeze(0)

            pos = upsampled['pos'][0, frame].float()
            opacity = data['ori_opacity'][0].float()
            shs = data['ori_shs'][0].float()

            cov3D = dof6_to_matrix3x3(cov3D_dof6_seq[0, frame])

            pos_all = torch.cat([pos, bg['pos']], dim=0)
            cov_all = torch.cat([cov3D, bg['cov']], dim=0)
            opacity_all = torch.cat([opacity, bg['opc']], dim=0)
            shs_all = torch.cat([shs, bg['shs']], dim=0)

            rendering, _, _ = rasterization(
                means=pos_all,
                quats=None,
                scales=None,
                opacities=opacity_all.squeeze(-1),
                colors=shs_all,
                viewmats=current_extrinsic,
                Ks=current_intrinsic,
                width=resolution,
                height=resolution,
                sh_degree=2,
                render_mode="RGB",
                packed=False,
                near_plane=1e-10,
                backgrounds=background.unsqueeze(0).repeat(1, 1),
                radius_clip=0.1,
                covars=cov_all,
                rasterize_mode='classic',
            )
            pred_img = rendering.clamp(0.0, 1.0).mul(255.0).add_(0.5).clamp_(0, 255)
            pred_img = pred_img[0].detach().cpu().numpy().astype(np.uint8)
            writer.append_data(pred_img)
    finally:
        writer.close()


def _all_videos_exist(output_root: str, group: str, scene: str, table: str, models: List[str], refine_flags: List[bool], views: List[int]) -> bool:
    for model_name in models:
        for rf in refine_flags:
            subdir = 'refine' if rf else 'base'
            for v in views:
                out_dir = os.path.join(output_root, f"{model_name}_V", subdir, group, scene, table, f"view_{v}")
                mp4_path = os.path.join(out_dir, f"{scene}.mp4")
                if not os.path.exists(mp4_path):
                    return False
    return True


def run_scene_to_videos(
    data_path: str,
    output_root: str,
    models: List[str],
    views: Union[str, Sequence[Union[int, str]]],
    refine_flags: List[bool],
    cfg_path: str,
    pretrain_models: Dict[str, object],
    dynamic_models: Dict[str, Tuple[nn.Module, Dict]],
    steps: int = 100,
    resolution: int = 256,
    seed: int = 0,
) -> None:
    # Derive keys and skip early if all videos already exist
    group, scene, table = _derive_scene_keys(data_path)

    # if _all_videos_exist(output_root, group, scene, table, models, refine_flags, views):
    #     print(f"Skip {group}/{scene}/{table}: all videos exist")
    #     return

    # Reconstruct once (shared for both refine settings). Reuse cached models if provided
    try:
        result_dict = reconstruct_in_memory(
            pretrain_models=pretrain_models,
            data_path=data_path,
            seed=seed,
            interval=4,
            prompt_type='bbox',
            real_world=False,
            visualize=False,
        )
    except Exception as e:
        print(f"Reconstruct failed for {group}/{scene}/{table}: {e}")
        torch.cuda.empty_cache()
        return

    # For each refine flag, process sequentially to limit peak memory
    parsed_views = _parse_views_spec(views)

    for rf in refine_flags:
        sim_gauss, bg_gauss, labels, obj_num = preprocess_data_in_memory(
            result_dict, data_path=data_path, refine=rf, real_world=False
        )
        # Ensure tensors are on cuda and correct layout
        sim_gauss['shs'] = _pad_sh_to_degree2(sim_gauss['shs']) if sim_gauss['shs'].shape[1] != 9 else sim_gauss['shs']
        bg_gauss['shs'] = _pad_sh_to_degree2(bg_gauss['shs']) if bg_gauss['shs'].shape[1] != 9 else bg_gauss['shs']

        # For each model, expect the dynamic models and args_json to be preloaded in `dynamic_models`
        for model_name in models:
            if model_name not in dynamic_models:
                raise ValueError(f"Dynamic model '{model_name}' not found in provided dynamic_models")
            model, args_json = dynamic_models[model_name]
            num_keypoints = args_json['num_keypoints']

            # Build eval sample
            data = prepare_eval_sample(
                sim_gauss,
                labels,
                obj_num=obj_num,
                refine=rf,
                num_keypoints=num_keypoints,
                device=torch.device('cuda'),
            )

            # Compute trajectory and precompute cov
            print(model_name)
            upsampled, cov3D_dof6_seq, intrinsics, extrinsics, time_params = compute_trajectory_and_cov(
                data=data,
                model=model,
                model_name=model_name,
                steps=steps,
                cfg_path=cfg_path,
                resolution=resolution,
            )

            for view_spec in parsed_views:
                subdir = 'refine' if rf else 'base'
                if isinstance(view_spec, int):
                    view_dir = f"view_{view_spec}"
                else:
                    view_dir = view_spec
                out_dir = os.path.join(
                    output_root,
                    f"{model_name}_V",
                    subdir,
                    group,
                    scene,
                    table,
                    view_dir,
                )
                # # Skip if this mp4 already exists
                # mp4_path = os.path.join(out_dir, f"{scene}.mp4")
                # if os.path.exists(mp4_path):
                #     print(f"Skip existing {mp4_path}")
                #     continue

                frame_count = upsampled['pos'].shape[1]
                intrinsics_traj, extrinsics_traj = _prepare_camera_trajectory_for_view(
                    view_spec=view_spec,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                )
                if intrinsics_traj.shape[0] != frame_count or extrinsics_traj.shape[0] != frame_count:
                    raise ValueError(
                        "Camera trajectory length mismatch: "
                        f"expected {frame_count}, got {intrinsics_traj.shape[0]} and {extrinsics_traj.shape[0]}"
                    )

                render_single_view(
                    data=data,
                    bg={'pos': bg_gauss['pos'], 'cov': bg_gauss['cov'], 'opc': bg_gauss['opc'], 'shs': bg_gauss['shs']},
                    upsampled=upsampled,
                    cov3D_dof6_seq=cov3D_dof6_seq,
                    intrinsics_traj=intrinsics_traj,
                    extrinsics_traj=extrinsics_traj,
                    time_params=time_params,
                    resolution=resolution,
                    output_path=out_dir,
                )

            # Free per-model large tensors before next iteration
            del upsampled, cov3D_dof6_seq, intrinsics, extrinsics, time_params, data
            torch.cuda.empty_cache()

        # Free per-refine large tensors before next refine flag
        del sim_gauss, bg_gauss, labels
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end in-memory reconstruct→simulate→render pipeline')
    parser.add_argument('--data_path', type=str, required=False, help='Input image directory or a video file path for reconstruction')
    parser.add_argument('--output_root', type=str, default='./data/GSCollision/videogen_inference_highres')
    parser.add_argument('--models', type=str, default='NGFF,Pointformer,GCN')
    parser.add_argument('--views', type=str, default='16,36,56,76,96')
    parser.add_argument('--disable_refine', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='./config/dynamic_config.json')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--resolution', type=int, default=448)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_inference_json', type=str, default=None, help='JSON list with entries: {data_path, views}')
    return parser.parse_args()

def main():
    args = parse_args()
    setup_seed(args.seed)

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    refine_flags = [False] if args.disable_refine else [False, True]

    # prepare dynamic models
    dynamic_models: Dict[str, Tuple[nn.Module, Dict]] = {}
    for m in models:
        model_obj, args_json = load_dynamic_model(m)
        dynamic_models[m] = (model_obj, args_json)

    # prepare pretrain models
    pretrain_models = prepare_models()

    if args.batch_inference_json:
        with open(args.batch_inference_json, 'r') as f:
            entries = json.load(f)
        for entry in entries:
            data_path = entry['data_path']
            output_root = args.output_root
            views_raw = entry.get('views', None)

            try:
                run_scene_to_videos(
                    data_path=data_path,
                    output_root=output_root,
                    models=models,
                    views=views_raw if views_raw is not None else args.views,
                    refine_flags=refine_flags,
                    cfg_path=args.config,
                    pretrain_models=pretrain_models,
                    dynamic_models=dynamic_models,
                    steps=args.steps,
                    resolution=args.resolution,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"Error running scene to videos for {data_path}: {e}")
                continue
    else:
        run_scene_to_videos(
            data_path=args.data_path,
            output_root=args.output_root,
            models=models,
            views=args.views,
            refine_flags=refine_flags,
            cfg_path=args.config,
            pretrain_models=pretrain_models,
            dynamic_models=dynamic_models,
            steps=args.steps,
            resolution=args.resolution,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()