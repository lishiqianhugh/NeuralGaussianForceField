import sys
import argparse
import math
import cv2
import torch
import os
import numpy as np
import json
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import time
import imageio

# Gaussian splatting dependencies
from utils.gs_utils import GaussianModel
from utils.camera_utils import Camera as GSCamera
from gsplat import rasterization

# Utils
from dynamic_models import GCN,NGFFobj,PointTransformer
from dataset.constants import generate_ground_plane
from dynamic_models.DGSDataset import DGSDataset
from torch.utils.data import DataLoader

from utils.transformation_utils import transform_gaussians, upsample_gaussian_splatting,euler_xyz_to_matrix,dof6_to_matrix3x3,se3_inverse,apply_cov_rotations,get_center_view_worldspace_and_observant_coordinate
from utils.general_utils import setup_seed, searchForMaxIteration
from utils.camera_utils import extract_camera_matrices,get_camera_view
from utils.gs_utils import load_params_from_gs
from utils.sh_utils import convert_SH
from utils.decode_param import *

from dataset.constants import OBJPART,SCENES


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=2, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--dynamic_model', type=str, default='ngff', choices=['ngff', 'pointformer', 'segno', 'gcn', 'sgnn'], help='Type of dynamic model to use')
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--background_path", type=str, default=None)
    parser.add_argument("--single_view", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=1280)
    parser.add_argument("--use_external_forces", action="store_true", default=False)
    parser.add_argument("--external_forces_config",type=str,default="./data/GSCollision/interactive_generation/config.json")
    args = parser.parse_args()
    setup_seed(seed=args.seed)

    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    #################################
    #   Set loading config path     #
    #################################
    if args.dynamic_model == 'ngff':
        dynamic_model_path = '/mnt/nfs_project_a/shared/models/ngff/ngff_perception/datasets/GSCollision/exps/ngff/out_2025-09-15-16-26-50/ngff_best.pth' # ngff 8 gpus
        # dynamic_model_path = './exps/ngff/out_2025-09-20-10-42-10/ngff_best.pth' # ngff 8 gpus finetune twice
    elif args.dynamic_model == 'gcn':
        dynamic_model_path = '/mnt/nfs_project_a/shared/models/ngff/ngff_perception/datasets/GSCollision/exps/gcn/out_2025-09-11-10-51-44/ngff_best.pth' # gcn
    elif args.dynamic_model == 'pointformer':
        dynamic_model_path = '/mnt/nfs_project_a/shared/models/ngff/ngff_perception/datasets/GSCollision/exps/pointformer/out_2025-09-16-17-16-19/ngff_best.pth' # pointformer
    test_steps = args.steps
    with open(os.path.join(dynamic_model_path.split('ngff_best.pth')[0], 'args.json'), 'r') as f:
        args_json = json.load(f)
    args.__dict__.update(args_json)
    args.k = 1
    args.steps = test_steps
    #################################
    #   Loading initial GS data     #
    #################################
    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)
    # load gaussians
    print("Loading gaussians...")
    gaussians = GaussianModel(2)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    device = "cuda:0"
    data_path = './data/GSCollision/scenes'
    if args.model_path:
        dataset = DGSDataset(None, [args.model_path], preprocessing_params, num_frames=args.steps, num_keypoints=args.num_keypoints, k=args.k, chunk=1, dtype=torch.float32, cache_dir='./data/GSCollision/cache') # set num_keypoints=none if use all points
    else:
        scenes = [os.path.join(data_path, group, scene) for group in ['3_9'] for scene in os.listdir(os.path.join(data_path, group)) if not scene.startswith('.')]
        scenes = scenes[:1]
        dataset = DGSDataset(None, scenes, preprocessing_params, num_frames=args.steps, num_keypoints=args.num_keypoints, k=args.k, chunk=1, dtype=torch.float32, cache_dir='./data/GSCollision/cache') # set num_keypoints=none if use all points
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch of full sequences
    print('Successfully loaded initial frame data')
    #################################
    #    Add ground or background   #
    #################################
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )
    if args.background_path is not None:
        if 'reconstruct' in args.background_path:
            # reconstructed bg; already aligned
            bg_gaussians = load_checkpoint(args.background_path)
            bg_params = load_params_from_gs(bg_gaussians, pipeline)

            bg_pos = bg_params["pos"]
            bg_cov = bg_params["cov3D_precomp"]
            bg_opacity = bg_params["opacity"]
            bg_shs = bg_params["shs"]
            bg_screen_points = bg_params["screen_points"]
            bg_rot = torch.eye(3).repeat(bg_pos.shape[0], 1, 1).to(device=device)
        else:
            # sim bg; need alignment
            bg_gaussians = GaussianModel(2)
            bg_path = f'{args.background_path}/gaussians_feedforward.ply'
            if not os.path.exists(bg_path):
                bg_path = f'{args.background_path}/gaussians_postopt.ply'
            bg_gaussians.load_ply(bg_path)
            bg_params = load_params_from_gs(bg_gaussians, pipeline)

            bg_pos = bg_params["pos"]
            bg_cov = bg_params["cov3D_precomp"]
            bg_opacity = bg_params["opacity"]
            bg_shs = bg_params["shs"]
            bg_screen_points = bg_params["screen_points"]
            bg_rot = torch.eye(3).repeat(bg_pos.shape[0], 1, 1).to(device=device)
            bg_scale = bg_params["scales"]

            # mask low opacity
            opacity_mask = bg_opacity[:, 0] > preprocessing_params["opacity_threshold"]
            bg_pos = bg_pos[opacity_mask, :]
            bg_cov = bg_cov[opacity_mask, :]
            bg_opacity = bg_opacity[opacity_mask, :]
            bg_shs = bg_shs[opacity_mask, :]
            bg_screen_points = bg_screen_points[opacity_mask, :]
            bg_rot = bg_rot[opacity_mask, :]
            bg_scale = bg_scale[opacity_mask, :]

            camera_file = f"{args.background_path}/camera_2999.pt"
            cameras = torch.load(camera_file, map_location=torch.device('cuda'))
            intrinsics = cameras["intrinsics"]
            extrinsics = cameras["extrinsics"]
            poses = se3_inverse(extrinsics)
            pred_camera_positions = poses[:, :3, 3]
            camera_center = pred_camera_positions.mean(dim=0)
            
            bg_rotate = torch.tensor(SCENES[args.background_path.split('/')[-1]]['rotation'], device=device) * math.pi / 180.0
            R = euler_xyz_to_matrix(bg_rotate)
            T = torch.tensor(SCENES[args.background_path.split('/')[-1]]['translation'], device=device)
            scale_factor = SCENES[args.background_path.split('/')[-1]]['scale']
            S = scale_factor * torch.eye(3, device=R.device, dtype=R.dtype) 
            A = S @ R
            bg_pos = torch.matmul(scale_factor*(bg_pos-camera_center), R.T) + T + camera_center #+ torch.tensor([0,0,-0.4], device=device)
            bg_cov = apply_cov_rotations(bg_cov, A.unsqueeze(0))
            # filter the bg points in -1 to 1 xyz
            sim_area_mask = ~((bg_pos[..., 0] > -1) & (bg_pos[..., 0] < 1) & (bg_pos[..., 1] > -1) & (bg_pos[..., 1] < 1) & (bg_pos[..., 2] > -1) & (bg_pos[..., 2] < 1))
            bg_pos = bg_pos[sim_area_mask]
            bg_cov = bg_cov[sim_area_mask]
            bg_opacity = bg_opacity[sim_area_mask]
            bg_shs = bg_shs[sim_area_mask]
            bg_screen_points = bg_screen_points[sim_area_mask]
            bg_rot = bg_rot[sim_area_mask]

    else:
        ground = generate_ground_plane(
            x_range=(-4.0, 4.0),
            y_range=(-4.0, 4.0),
            num_points_x=400,
            num_points_y=400,
            z_value=-0.94,
            frame_num=args.steps,
            device=device
        )

        ground_pos = ground['positions']  # [frames, total_points, 3]
        ground_cov = ground['covariances']  # [frames, total_points, 6]
        ground_rot = ground['rotations']  # [total_points, 3, 3]
        ground_opacity = ground['opacities']  # [total_points, 1]
        ground_shs = ground['shs']  # [total_points, 16, 3]
        ground_screen_points = torch.zeros_like(ground_pos)  # not used
    
    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        [],
        1,
        torch.tensor(0),
    )
    current_cameras = []
    for frame in range(args.steps):
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
            resolution=args.resolution,
        )
        current_cameras.append(current_camera)
    current_intrinsics, current_extrinsics = extract_camera_matrices(current_cameras)
    current_intrinsics = current_intrinsics.to(device)
    current_extrinsics = current_extrinsics.to(device)
    print("Successfully set up camera")
    #################################
    # NGFF simulation on initial GS #
    #################################
    if args.dynamic_model == 'ngff':
        model = NGFFobj(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, num_keypoints=args.num_keypoints, k=args.k, mass=args.mass, 
                        dt=args.dt, ode_method=args.ode_method, r=0.1, step_size=args.step_size, threshold=args.threshold, rtol=args.rtol, atol=args.atol)
    elif args.dynamic_model == 'pointformer':
        model = PointTransformer(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers)
    elif args.dynamic_model == 'gcn':
        model = GCN(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, r=0.1)
    else:
        raise ValueError(f"Unknown dynamic model: {args.dynamic_model}")
    model.load_state_dict(torch.load(dynamic_model_path))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for data in dataloader:
        # put all data variable into device
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        positions = data['pos']
        points = data['points']
        com = data['com']
        com_vel = data['com_vel']
        angle = data['angle']
        angle_vel = data['angle_vel']
        padding = data['padding']
        knn = data['knn']
        opacity = data['opacity']
        shs = data['shs']
        init_screen_points = data['screen_points']  # [num_points, 2]
        keypoint_indices = data['keypoint_indices']
        ori_obj_ids = data['ori_obj_ids'] # [B, N, 1]
        real_imgs = data['real_images']   # [B, T, H, W, 3]

        optimizer.zero_grad()
        with torch.no_grad():
            if args.dynamic_model == 'ngff':
                external_forces = None

                if args.use_external_forces:
                    cfg = json.load(open(args.external_forces_config, 'r'))
                    key = args.model_path.split('/')[-1]
                    lst = cfg[key]
                    external_forces = []
                    for e in lst:
                        ef = {'obj_id': e['obj_id'], 'start': e['start'], 'end': e['end']}
                        if 'force' in e:
                            ef['force'] = torch.tensor(e['force'], device=device).float()
                        if 'torque' in e:
                            ef['torque'] = torch.tensor(e['torque'], device=device).float()
                        external_forces.append(ef)

                point_seq, com_seq, angle_seq = model(points[:,0], com[:,0], com_vel[:,0],
                                                    angle[:,0], angle_vel[:,0], padding, knn,
                                                    pred_len=args.steps-1, external_forces=external_forces)
                # compute point trajectory using translation, rotation, and deformation
                R_seq = euler_xyz_to_matrix(angle_seq)  # (B, T, num_objs, 3, 3)
                future_seq = torch.matmul(point_seq, R_seq.transpose(-2, -1)) + com_seq.unsqueeze(-2)  # (B, T, num_objs, N, 3)
            elif args.dynamic_model in ['pointformer', 'segno', 'gcn', 'sgnn'] :
                assert not args.use_external_forces, "External forces are not supported for this dynamic model"
                future_seq = model(points[:,0], com[:,0], com_vel[:,0],
                                angle[:,0], angle_vel[:,0], padding, knn,
                                pred_len=args.steps-1, external_forces=None)
            else:
                raise ValueError(f"Unknown dynamic model: {args.dynamic_model}")
        
        # set future seq of object 0 to initial frame to keep the object still
        # future_seq[:, :, 1] = future_seq[:, 0:1, 1]
        # merge object points using padding. points: (B, num_objs, N, 3), padding (B, num_objs, N)
        B, T, num_objs, N, C = future_seq.shape
        mask = padding.unsqueeze(1).reshape(B, 1, num_objs * N, 1).repeat(1, T, 1, C).bool()  # (B, 1, num_objs*N, 1)
        future_seq = future_seq.reshape(B, T, -1, C) # (B, T, num_objs*N, C)
        future_seq = future_seq[mask].reshape(B, T, -1, C)  # (B, T, N, C)
        # interpolation
        data['pos'] = future_seq
        upsampled_data = upsample_gaussian_splatting(data, data['ori_pos'], ori_obj_ids, k=8) # the upsample method could be improved
        t3 = time.time()
        upsampled_data['cov3D'] = transform_gaussians(upsampled_data['pos'], data['ori_cov3D'], k=32) # increse k leads to better rendering but cost more time; 32 64 seems best
        upsampled_data['rot'] = data['ori_rot']
        upsampled_data['opacity'] = data['ori_opacity']
        upsampled_data['shs'] = data['ori_shs']

        t4 = time.time()
        print(f"Time taken for cov3D transformation: {t4-t3:.6f}s")
        
        #################################
        #           Render              #
        #################################
        fps = int(1.0 / time_params["frame_dt"])

        # extract scene_name from args.output_path
        if args.use_external_forces:
            scene_name = args.output_path.split('/')[-1]
        else:
            scene_name = args.output_path.split('/')[-3]

        video_writers = {
            "pred": imageio.get_writer(os.path.join(args.output_path, f"{scene_name}.mp4"), fps=fps),
            # "real": imageio.get_writer(os.path.join(args.output_path, "gt.mp4"), fps=fps),
            # "diff": imageio.get_writer(os.path.join(args.output_path, "diff.mp4"), fps=fps),
        }

        for frame in tqdm(range(args.steps)):
            if args.single_view is not None:
                current_camera = current_cameras[args.single_view]
                current_extrinsic = current_extrinsics[args.single_view].unsqueeze(0)
                current_intrinsic = current_intrinsics[args.single_view].unsqueeze(0)
            else:
                current_camera = current_cameras[frame]
                current_extrinsic = current_extrinsics[frame].unsqueeze(0)
                current_intrinsic = current_intrinsics[frame].unsqueeze(0)

            pos = upsampled_data['pos'][0, frame].float()
            cov3D = upsampled_data['cov3D'][0, frame].float()
            rot = upsampled_data['rot'][0].float()
            opacity = upsampled_data['opacity'][0].float()
            shs = upsampled_data['shs'][0].float()

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)

            if args.background_path is None:  
                pos = torch.cat([pos, ground_pos[frame]], dim=0)
                cov3D = torch.cat([cov3D, ground_cov[frame]], dim=0)
                opacity = torch.cat([opacity, ground_opacity], dim=0)
                shs = torch.cat([shs, ground_shs], dim=0)
                colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            else:
                pos = torch.cat([pos, bg_pos], dim=0)     
                cov3D = torch.cat([cov3D, bg_cov], dim=0)      
                opacity = torch.cat([opacity, bg_opacity], dim=0)
                shs = torch.cat([shs, bg_shs], dim=0)
                bg_colors_precomp = convert_SH(bg_shs, current_camera, bg_gaussians, bg_pos, bg_rot)             
                colors_precomp = torch.cat([colors_precomp, bg_colors_precomp], dim=0)
            
            cov3D = dof6_to_matrix3x3(cov3D)
            rendering, _, _ = rasterization(
                means=pos, quats=None, scales=None, opacities=opacity.squeeze(-1), colors=shs,
                viewmats=current_extrinsic, Ks=current_intrinsic,
                width=args.resolution, height=args.resolution, sh_degree=2, render_mode="RGB", packed=False,
                near_plane=1e-10, backgrounds=background.unsqueeze(0).repeat(1, 1),
                radius_clip=0.1, covars=cov3D, rasterize_mode='classic'
            )

            # to numpy uint8
            pred_img = rendering.clamp(0.0, 1.0).mul(255.0).add_(0.5).clamp_(0, 255)
            pred_img = pred_img[0].detach().cpu().numpy().astype(np.uint8)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

            video_writers["pred"].append_data(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))

            # if torch.sum(real_imgs) != 0:
            #     real_img = real_imgs[0, frame].detach().cpu().numpy().astype(np.uint8)
            #     real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
            #     # downsample real_img to match pred_img size if needed
            #     if real_img.shape[0] != pred_img.shape[0] or real_img.shape[1] != pred_img.shape[1]:
            #         real_img = cv2.resize(real_img, (pred_img.shape[1], pred_img.shape[0]), interpolation=cv2.INTER_AREA)
            #     video_writers["real"].append_data(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))

            #     diff_img = np.abs(pred_img.astype(np.int32) - real_img.astype(np.int32)).astype(np.uint8)
            #     video_writers["diff"].append_data(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))

        for writer in video_writers.values():
            writer.close()

        print(f"Videos saved to {args.output_path}")
