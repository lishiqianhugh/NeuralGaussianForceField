import sys
import argparse
import math
import cv2
import torch
import os
import numpy as np
import json
import re
from tqdm import tqdm

# Gaussian splatting dependencies
from utils.gs_utils import GaussianModel,load_params_from_gs
from utils.general_utils import searchForMaxIteration

# Utils
from utils.decode_param import *
from utils.transformation_utils import dof6_to_matrix3x3,euler_xyz_to_matrix,se3_inverse,get_center_view_worldspace_and_observant_coordinate,apply_cov_rotations
from utils.camera_utils import extract_camera_matrices,get_camera_view

from .constants import OBJPART,SCENES,generate_ground_plane
from .generate_prompts import generate_point_prompt, generate_bbox_prompt, generate_text_prompt


from gsplat import rasterization

import imageio
import h5py
import hdf5plugin

torch.manual_seed(42)

from multiprocessing import Pool, cpu_count

ALL_FRAMES = None

def _save_video_worker(task):
    """Save one view's video in a separate process.

    task: (output_subdir, object_name, fps, view_iter)
    Uses global ALL_FRAMES (T,V,H,W,3) uint8 numpy array.
    """
    output_subdir, object_name_local, fps_local, view_iter = task

    os.makedirs(output_subdir, exist_ok=True)

    frames_view = ALL_FRAMES[:, view_iter]  # (T, H, W, 3) RGB uint8
    H, W = frames_view.shape[1], frames_view.shape[2]
    writer = cv2.VideoWriter(
        os.path.join(output_subdir, f"{object_name_local}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_local,
        (W, H),
    )
    for t in range(frames_view.shape[0]):
        frame_bgr = cv2.cvtColor(frames_view[t], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()

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
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--E", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=1280)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--initial_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--background_path", type=str, default=None)
    parser.add_argument("--fixed_view", type=int, default=None, help="Whether to use fixed camera when rendering dynamics, None for none, -1 for all 25 views, others for the index of the view")
    parser.add_argument("--shs_degree", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.initial_path is not None and not os.path.exists(args.initial_path):
        os.makedirs(args.initial_path)
    
    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    material_params["objpart"] = OBJPART

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = GaussianModel(args.shs_degree)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    device = "cuda:0"

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

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]

    height = None
    width = None
    #################################
    #    Add ground or background   #
    #################################
    if args.background_path is not None:
        
        bg_gaussians = GaussianModel(args.shs_degree)
        bg_path = f'{args.background_path}/gaussians_feedforward.ply'
        if not os.path.exists(bg_path):
            bg_path = f'{args.background_path}/gaussians_postopt.ply'

        bg_path_pt = bg_path.replace('.ply', '.pt')
        
        if os.path.exists(bg_path_pt):
            bg_gaussians.load_pt(bg_path_pt)
        else:
            bg_gaussians.load_ply(bg_path)
        
        bg_params = load_params_from_gs(bg_gaussians, pipeline)

        bg_pos = bg_params["pos"]
        bg_cov = bg_params["cov3D_precomp"]
        bg_opacity = bg_params["opacity"]
        bg_shs = bg_params["shs"]
        bg_rot = torch.eye(3).repeat(bg_pos.shape[0], 1, 1).to(device=device)
        bg_scale = bg_params["scales"]

        # mask low opacity
        opacity_mask = bg_opacity[:, 0] > preprocessing_params["opacity_threshold"]
        bg_pos = bg_pos[opacity_mask, :]
        bg_cov = bg_cov[opacity_mask, :]
        bg_opacity = bg_opacity[opacity_mask, :]
        bg_shs = bg_shs[opacity_mask, :]
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
        bg_rot = bg_rot[sim_area_mask]
    else:
        ground = generate_ground_plane(
            x_range=(-4.0, 4.0),
            y_range=(-4.0, 4.0),
            num_points_x=400,
            num_points_y=400,
            z_value=-0.94,
            frame_num=frame_num,
            device=device
        )

        ground_pos = ground['positions']  # [frames, total_points, 3]
        ground_cov = ground['covariances']  # [frames, total_points, 6]
        ground_rot = ground['rotations']  # [total_points, 3, 3]
        ground_opacity = ground['opacities']  # [total_points, 1]
        ground_shs = ground['shs']  # [total_points, 16, 3]
        ground_screen_points = torch.zeros_like(ground_pos)  # not used

    #################################
    #          Load cameras         #
    #################################
    gt_camera = []
    current_cameras = []
    for frame in range(frame_num):
        current_camera, raw_camera = get_camera_view(
            './data/cameras/table/', 
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
            fov_scale=1.5
        )
        current_cameras.append(current_camera)
        gt_camera.append(raw_camera)
    
    # save gt_camera into json
    if args.initial_path:
        with open(os.path.join(args.initial_path, "cameras.json"), "w") as f:
            json.dump(gt_camera, f, indent=4)
    if args.output_path:
        with open(os.path.join(args.output_path, "cameras.json"), "w") as f:
            json.dump(gt_camera, f, indent=4)

    #################################
    #     Render initial scene      #
    #################################
    if args.initial_path:
        with h5py.File(os.path.join(args.model_path, "opacity.h5"), 'r') as f:
            opacity = torch.tensor(f['opacity'][()], dtype=torch.float32, device=device)

        with h5py.File(os.path.join(args.model_path, "shs.h5"), 'r') as f:
            shs = torch.tensor(f['shs'][()], dtype=torch.float32, device=device)
        
        frame_path = os.path.join(args.model_path, f"{0:04d}.h5")
        with h5py.File(frame_path, 'r') as f:
            pos = torch.tensor(f['pos'][()], dtype=torch.float32, device=device)
            cov3D = torch.tensor(f['cov3D'][()].reshape(-1, 6), dtype=torch.float32, device=device)
            rot = torch.tensor(f['rot'][()].reshape(-1, 3, 3), dtype=torch.float32, device=device)
            # scale = torch.tensor(f['scale'][()].reshape(-1, 3, 3), dtype=torch.float32, device=device)
        
        # padding shs into degree args.shs_degree
        shs = torch.cat([shs, torch.zeros((shs.shape[0], (args.shs_degree+1)**2-shs.shape[1], shs.shape[2]), dtype=torch.float32, device=device)], dim=1)
        # generate point and bbox prompt
        point_prompt = generate_point_prompt(pos, gt_camera[0], args.model_path).cpu().numpy()
        bbox_prompt = generate_bbox_prompt(pos, gt_camera[0], args.model_path).cpu().numpy()
        text_prompt = generate_text_prompt(args.model_path)
        
        data = {
            "center_point": point_prompt.tolist(),
            "bbox": bbox_prompt.tolist(),
            "text": text_prompt

        }
        json_path = os.path.join(args.initial_path, "prompts.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        if args.background_path:
            pos = torch.cat([pos, bg_pos], dim=0)         #bg_pos_orig
            cov3D = torch.cat([cov3D, bg_cov], dim=0)              #bg_cov_orig
            opacity = torch.cat([opacity, bg_opacity], dim=0)
            shs = torch.cat([shs, bg_shs], dim=0)
        else:
            pos = torch.cat([pos, ground_pos[0]], dim=0)
            cov3D = torch.cat([cov3D, ground_cov[0]], dim=0)
            opacity = torch.cat([opacity, ground_opacity], dim=0)
            shs = torch.cat([shs, ground_shs], dim=0)

        # Rasterize initial scene for all camera views and cache renderings/depths
        cov3D = dof6_to_matrix3x3(cov3D)

        current_intrinsic, current_extrinsic = extract_camera_matrices(current_cameras)
        current_intrinsic = current_intrinsic.to(device)
        current_extrinsic = current_extrinsic.to(device)

        with torch.no_grad():
            quats = torch.zeros((pos.shape[0], 4), dtype=torch.float32, device=device)
            scales = torch.zeros((pos.shape[0], 3), dtype=torch.float32, device=device)  # ignored by rasterizer
            all_rendering_img = []
            all_rendering_depth = []
            V = current_extrinsic.shape[0]
            mini_batch_size = 5

            for view_iter in range(0, V, mini_batch_size):
                end_view_iter = min(view_iter + mini_batch_size, V)
                rendering, alpha, _ = rasterization(means=pos, quats=quats, scales=scales, opacities=opacity.squeeze(-1), colors=shs,
                                                    viewmats=current_extrinsic[view_iter:end_view_iter], Ks=current_intrinsic[view_iter:end_view_iter],
                                                    width=args.resolution, height=args.resolution, sh_degree=args.shs_degree, render_mode="RGB+D", packed=False,
                                                    near_plane=1e-10, backgrounds=background.unsqueeze(0).repeat(end_view_iter - view_iter, 1),
                                                    radius_clip=0.1, covars=cov3D, rasterize_mode='classic')
                rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1)  # [1,H,W,3], [1,H,W,1]
                rendering_img = rendering_img.clamp(0.0, 1.0)
                all_rendering_img.append(rendering_img)
                all_rendering_depth.append(rendering_depth)

            all_rendering_img = torch.cat(all_rendering_img, dim=0)
            all_rendering_depth = torch.cat(all_rendering_depth, dim=0)
            all_rendering_img = all_rendering_img.clamp(0.0, 1.0)

        # Quantize on GPU to uint8 then move to CPU
        rendering_cpu = (all_rendering_img.mul(255.0).add_(0.5)).clamp_(0, 255).to(torch.uint8).detach().cpu()  # (V, H, W, 3)
        depth = all_rendering_depth.squeeze(-1).detach()  # (V, H, W)
        depth_max = depth.max()
        depth_norm = depth / depth_max
        depth_norm = depth_norm.clamp(0, 1)
        depth_norm = depth_norm.cpu()
        
        # Post-process initial frames: write images and/or compile video
        fps_init = int(1.0 / time_params["frame_dt"])
        object_name_init = args.model_path.split('/')[-1].split('table_')[-1]

        if args.render_img:
            assert args.initial_path is not None
            # rendering_cpu: (V, H, W, 3)
            for view_iter in range(rendering_cpu.shape[0]):
                cv2_img = rendering_cpu[view_iter].numpy()
                # rendering_cpu is RGB uint8; convert to BGR for OpenCV
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
                if height is None or width is None:
                    height = cv2_img.shape[0] // 2 * 2
                    width = cv2_img.shape[1] // 2 * 2
                cv2.imwrite(
                    os.path.join(args.initial_path, f"{view_iter}.png".rjust(8, "0")),
                    cv2_img,
                )

        if args.render_depth:
            # depth_cpu: (V, H, W) -> save as 16-bit grayscale PNG (normalized per-frame)
            assert args.initial_path is not None
            depth_map = depth_norm.numpy()
            depth_16 = (depth_map * 65535.0).round().astype(np.uint16)
            with h5py.File(os.path.join(args.initial_path, "depth.h5"), 'w') as f:
                f.create_dataset("depth", data=depth_16, compression="gzip")
                f.create_dataset("depth_max", data=depth_max.detach().cpu().numpy())
            # for view_iter in range(depth_cpu.shape[0]):
            #     depth_img = depth_cpu[view_iter].clone()
            #     depth_map = depth_img.numpy()
                # normalize to 0..1 then scale to 16-bit
                # depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
                # depth16 = (depth_norm * 65535.0).round().astype(np.uint16)
                # cv2.imwrite(os.path.join(args.initial_path, f"{view_iter:04d}_depth.png"), depth16)

        if args.compile_video:
            # Write initial render video using OpenCV VideoWriter
            assert args.initial_path is not None
            out_path = os.path.join(args.initial_path, f"{object_name_init}.mp4")
            V = rendering_cpu.shape[0]
            H = rendering_cpu.shape[1]
            W = rendering_cpu.shape[2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_init, (W, H))
            for view_iter in range(V):
                frame_np = rendering_cpu[view_iter].numpy()
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            writer.release()

            
    #################################
    #      Render Dynamics          #
    #################################
    if args.output_path:
        # Load global opacity and SHs once
        with h5py.File(os.path.join(args.model_path, "opacity.h5"), 'r') as f:
            opacity_all = torch.tensor(f['opacity'][()], dtype=torch.float32, device=device)
        with h5py.File(os.path.join(args.model_path, "shs.h5"), 'r') as f:
            shs_all = torch.tensor(f['shs'][()], dtype=torch.float32, device=device)

        # Store all frames as uint8 on CPU with shape (T, V, H, W, 3)
        all_frames_u8 = None
        # Store depths as float32 on CPU with shape (T, V, H, W)
        all_frames_depth = None

        for frame in tqdm(range(frame_num)):

            if args.fixed_view is not None:
                if args.fixed_view == -1:
                    current_intrinsic, current_extrinsic = extract_camera_matrices(current_cameras)
                    current_intrinsic = current_intrinsic[::4]
                    current_extrinsic = current_extrinsic[::4]
                    view_idxs = torch.arange(0, current_extrinsic.shape[0]*4, 4)
                else:
                    current_intrinsic, current_extrinsic = extract_camera_matrices(current_cameras[args.fixed_view])
                    view_idxs = [args.fixed_view]
            else:
                current_intrinsic, current_extrinsic = extract_camera_matrices(current_cameras[frame])
                view_idxs = [frame]

            current_intrinsic = current_intrinsic.to(device)
            current_extrinsic = current_extrinsic.to(device)

            # load per-frame gaussian data
            frame_path = os.path.join(args.model_path, f"{frame:04d}.h5")
            with h5py.File(frame_path, 'r') as f:
                pos = torch.tensor(f['pos'][()], dtype=torch.float32, device=device)
                cov3D = torch.tensor(f['cov3D'][()].reshape(-1, 6), dtype=torch.float32, device=device)
                rot = torch.tensor(f['rot'][()].reshape(-1, 3, 3), dtype=torch.float32, device=device)

            pad = (args.shs_degree + 1) ** 2 - shs_all.shape[1]
            if pad > 0:
                shs = torch.cat([shs_all, torch.zeros((shs_all.shape[0], pad, shs_all.shape[2]), dtype=torch.float32, device=device)], dim=1)
            else:
                shs = shs_all

            if args.background_path:
                pos = torch.cat([pos, bg_pos], dim=0)        
                cov3D = torch.cat([cov3D, bg_cov], dim=0)              
                opacity = torch.cat([opacity_all, bg_opacity], dim=0)
                shs = torch.cat([shs, bg_shs], dim=0)
            else:
                pos = torch.cat([pos, ground_pos[frame]], dim=0)
                cov3D = torch.cat([cov3D, ground_cov[frame]], dim=0)
                opacity = torch.cat([opacity_all, ground_opacity], dim=0)
                shs = torch.cat([shs, ground_shs], dim=0)

            cov3D = dof6_to_matrix3x3(cov3D)

            with torch.no_grad():
                quats = torch.zeros((pos.shape[0], 4), dtype=torch.float32, device=device)
                scales = torch.zeros((pos.shape[0], 3), dtype=torch.float32, device=device) # will be ignored, so just set to 0
                all_rendering_img = []
                all_rendering_depth = []
                V = current_extrinsic.shape[0]
                mini_batch_size = 5

                for view_iter in range(0, V, mini_batch_size):
                    end_view_iter = min(view_iter + mini_batch_size, V)
                    rendering, alpha, _ = rasterization(means=pos, quats=quats, scales=scales, opacities=opacity.squeeze(-1), colors=shs,
                                                    viewmats =current_extrinsic[view_iter:end_view_iter], Ks =current_intrinsic[view_iter:end_view_iter], width=args.resolution, height=args.resolution, sh_degree=2, 
                                                    render_mode="RGB", packed=False, # "RGB+D"
                                                    near_plane=1e-10,
                                                    backgrounds=background.unsqueeze(0).repeat(end_view_iter - view_iter, 1),
                                                    radius_clip=0.1,
                                                    covars=cov3D,
                                                    rasterize_mode='classic') # (1, H, W, 3) 
                    # rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1) # [1,H,W,3] [1,H,W,1]
                    rendering_img = rendering
                    all_rendering_img.append(rendering_img)
                    # all_rendering_depth.append(rendering_depth)
                
                all_rendering_img = torch.cat(all_rendering_img, dim=0)
                # all_rendering_depth = torch.cat(all_rendering_depth, dim=0)
                all_rendering_img = all_rendering_img.clamp(0.0, 1.0)   

            # Quantize on GPU to uint8 then move to CPU and store
            frames_u8_gpu = (all_rendering_img.mul(255.0).add_(0.5)).clamp_(0, 255).to(torch.uint8)
            if all_frames_u8 is None:
                Vtmp, Htmp, Wtmp = frames_u8_gpu.shape[0], frames_u8_gpu.shape[1], frames_u8_gpu.shape[2]
                all_frames_u8 = torch.empty((frame_num, Vtmp, Htmp, Wtmp, 3), dtype=torch.uint8)
            frames_u8_cpu = frames_u8_gpu.detach().cpu()
            all_frames_u8[frame].copy_(frames_u8_cpu)
        
        # Post-process: write images and/or compile video from rasterized frames
        fps = int(1.0 / time_params["frame_dt"])
        object_name = args.model_path.split('/')[-1].split('table_')[-1]

        # write individual images if requested
        if args.render_img:
            assert args.output_path is not None
            # save img
            for i in range(all_frames_u8.shape[0]): # [T, V, H, W, 3]
                for view_iter, view_idx in enumerate(view_idxs):
                    if args.fixed_view is not None:
                        output_subdir = os.path.join(args.output_path, f"view_{view_idx}")
                    else:
                        assert len(view_idxs) == 1, "circle360 view index should be 0 or 1"
                        output_subdir = os.path.join(args.output_path, f"circle360")
                    
                    os.makedirs(output_subdir, exist_ok=True)

                    cv2_img = all_frames_u8[i, view_iter].numpy()
                    # all_frames_u8 is RGB uint8; convert to BGR for OpenCV
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
                    if height is None or width is None:
                        height = cv2_img.shape[0] // 2 * 2
                        width = cv2_img.shape[1] // 2 * 2
                    cv2.imwrite(
                        os.path.join(output_subdir, f"{i}.png".rjust(8, "0")),
                        cv2_img,
                    )
                    

        # compile video
        if args.compile_video:
            # Share frames with workers via forked memory; only pass view indices
            ALL_FRAMES = all_frames_u8.numpy()

            tasks = []
            print(view_idxs)
            for view_iter, view_idx in enumerate(view_idxs):
                if args.fixed_view is not None:
                    output_subdir = os.path.join(args.output_path, f"view_{view_idx}")
                else:
                    assert len(view_idxs) == 1, "circle360 view index should be 0 or 1"
                    output_subdir = os.path.join(args.output_path, f"circle360")
                
                os.makedirs(output_subdir, exist_ok=True)
                tasks.append((output_subdir, object_name, fps, view_iter))

            num_workers = min(len(tasks), max(cpu_count() - 1, 1))
            with Pool(processes=num_workers) as pool:
                pool.map(_save_video_worker, tasks)
