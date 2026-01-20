import os
os.environ['SPCONV_ALGO'] = 'native'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
os.environ["OMP_PROC_BIND"] = "false"
os.environ["ORT_DISABLE_THREAD_AFFINITY"] = "1"
import onnxruntime as ort
import warnings
ort.set_default_logger_severity(4)
warnings.filterwarnings('ignore')

import sys
import argparse
import torch
import os
import numpy as np
import json

from utils.general_utils import setup_seed

from pathlib import Path
from scipy.spatial.transform import Rotation as R

import torch
import numpy as np
from PIL import Image

import argparse
import random
import json

from .Pi3_splat.utils.basic import load_images_as_tensor, write_ply, resize_images
from .Pi3_splat.utils.geometry import depth_edge,se3_inverse
from .Pi3_splat.models.pi3_splat import Pi3_splat
from .Pi3_splat.utils.types import Gaussians
from .Pi3_splat.models.gaussians.gaussians import build_covariance


from sam2.build_sam import build_sam2_video_predictor
from .externals.OSEDiff import OSEDiff
from .externals.DiffSplat import DiffSplat
from .externals.superglue.matching_pairs import match_image_lists, prepare_superglue
from .externals.SAM2.sam2_segvideo import segment_image_sequence

from .sim3_pose_estimation import *


from kornia.morphology import erosion
from chamfer_distance import ChamferDistance

from utils.visualize_utils import *
from utils.io_utils import *
from utils.transformation_utils import align_camera_poses

from typing import Dict


def super_resolution_osediff(osediff_model,images):
    images = [osediff_model(image) for image in images]
    return images

def gen_3d_gaussian_from_image(gs_model,image,text_prompt="a_high_quality_3D_asset"):
    g = gs_model.infer(image,text_prompt)
    return g

def prepare_models():
    print(f"Loading model...")
    device = torch.device('cuda')
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    pi3_model = Pi3_splat.from_pretrained(os.path.join(abs_dir, "checkpoints/Pi3_splat/Pi3_splat_wildrgbd_gscollision/checkpoint-final")).to(device).eval()
    gs_model = DiffSplat(checkpoints_dir=os.path.join(abs_dir, "checkpoints/DiffSplat"))
    osediff_model = OSEDiff(checkpoints_dir=os.path.join(abs_dir, "checkpoints/OSEDiff"))
    sam2_video_predictor = build_sam2_video_predictor(config_file=f"/{str(os.path.join(abs_dir, "checkpoints/sam2/sam2.1_hiera_l.yaml"))}", ckpt_path=f"/{str(os.path.join(abs_dir, "checkpoints/sam2/sam2.1_hiera_large.pt"))}", device="cuda")

    superglue_matching = prepare_superglue(
        max_keypoints=1024,
        keypoint_threshold=0.01,
        nms_radius=4,
        sinkhorn_iterations=20,
        match_threshold=0.4,
        superglue='indoor',
        device=device,
    )

    return {
        "pi3": pi3_model,
        "gs": gs_model,
        "osediff": osediff_model,
        "sam2_video_predictor": sam2_video_predictor,
        "superglue": superglue_matching,
    }

def reconstruction(args, pretrain_models, save_pt=False):
    '''Reconstruct the scene from the input images/videos using the Pi3 model.'''
    os.makedirs(args.save_folder, exist_ok=True)

    print(f'Sampling interval: {args.interval}')

    pi3_model = pretrain_models['pi3']
    gs_model = pretrain_models['gs']
    osediff_model = pretrain_models['osediff']
    sam2_video_predictor = pretrain_models['sam2_video_predictor']
    superglue_matching = pretrain_models['superglue']

    device = torch.device('cuda')
    # 1. Load images
    imgs = load_images_as_tensor(args.data_path, interval=args.interval, PIXEL_LIMIT=None).to(device) # (N, 3, H, W)
    if args.real_world:
        assert args.interval == 1, "Interval must be 1 for real world"
    else:
        # default sampling scheme for evaluation:
        assert args.interval == 4, "Interval must be 4 for evaluation"
        imgs = imgs[[idx for idx in range(0, len(imgs)) if idx % 5 != 4]]
        assert len(imgs) == 20, "Number of images must be 20 for evaluation"

    if os.path.isdir(args.data_path):
        assert os.path.exists(os.path.join(args.data_path, 'prompts.json')), "Prompts file does not exist"
        initial_prompts = json.load(open(os.path.join(args.data_path, 'prompts.json'), 'r'))
    else:
        base_dir = os.path.dirname(args.data_path)
        assert os.path.exists(os.path.join(base_dir, 'prompts.json')), "Prompts file does not exist"
        initial_prompts = json.load(open(os.path.join(base_dir, 'prompts.json'), 'r'))

    text_prompts = initial_prompts['text']

    img_list = [(img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8') for img in imgs]

    if args.prompt_type == 'point':
        point_prompts = np.array(initial_prompts['point']) # [M,2]
        segmentation_results = segment_image_sequence(images=img_list,sam2_video_predictor=sam2_video_predictor, point_prompts=point_prompts) # [N,H,W]
    elif args.prompt_type == 'bbox':
        bbox_prompts = np.array(initial_prompts['bbox']) # [M,4]
        segmentation_results = segment_image_sequence(images=img_list,sam2_video_predictor=sam2_video_predictor, bbox_prompts=bbox_prompts) # [N,H,W]

    segmentation_results = torch.from_numpy(np.array(segmentation_results)).to(device) # [N,H,W]

    # 2. Resize imgs and segmentation results for model inference
    imgs_downsampled = resize_images(imgs, PIXEL_LIMIT=201_000)
    segmentation_results_downsampled = resize_images(segmentation_results, PIXEL_LIMIT=201_000)

    print(f"imgs shape is : {imgs.shape}")
    print(f"segmentation_results shape is : {segmentation_results.shape}")
    print(f"imgs_downsampled shape is : {imgs_downsampled.shape}")
    print(f"segmentation_results_downsampled shape is : {segmentation_results_downsampled.shape}")

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            encoder_output,decoder_output = pi3_model.inference(imgs_downsampled[None],require_decoder=True) # Add batch dimension

    # 4. Process mask
    masks = torch.sigmoid(encoder_output['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(encoder_output['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0] # [N,H,W]

    world_points = encoder_output['points'].squeeze(0)
    camera_poses = encoder_output['camera_poses'].squeeze(0) # [N,4,4] OpenCV format camera to world matrics
    gaussians = encoder_output['gaussians'][0]
    extrinsics = encoder_output['extrinsics'].squeeze(0)
    cam_intrinsics = encoder_output['intrinsics'].squeeze(0)

    # 5. Visualize cameras and export camera info
    if args.visualize:
        visualize_cameras_matplotlib(camera_poses=camera_poses, world_points=world_points, scale=0.1, max_points=80000, save_path=os.path.join(args.save_folder, 'camera_visualization.png'))
        export_cam_info(
            extrinsics=extrinsics,
            Ks=cam_intrinsics,
            save_path=os.path.join(args.save_folder, 'camera_info.pt'),
            image_names=[f"frame_{i}" for i in range(camera_poses.shape[0])],
            image_width=imgs.shape[3],
            image_height=imgs.shape[2]
        )
        create_colored_mask_gif(mask_list_tensor=segmentation_results_downsampled,save_path=os.path.join(args.save_folder, 'segmentation.gif'),original_images=imgs_downsampled)

    unique_labels = torch.unique(segmentation_results)
    unique_labels = sorted(unique_labels.tolist())

    # 6. Prepare labels
    object_labels = [int(l) for l in unique_labels if int(l) != 0]

    # Prepare containers for concatenation and add background once
    obj_means_list = []
    obj_covariances_list = []
    obj_harmonics_dc_list = []
    obj_opacities_list = []
    obj_scales_list = []
    obj_rotations_list = []
    obj_labels_list = []

    bg_mask = (segmentation_results_downsampled == 0).flatten()
    bg_means = gaussians.means[bg_mask]
    bg_covs = gaussians.covariances[bg_mask]
    bg_harm = gaussians.harmonics[bg_mask][:, :, :1]
    bg_opac = gaussians.opacities[bg_mask]
    bg_scales = gaussians.scales[bg_mask]
    bg_rots = gaussians.rotations[bg_mask]
    bg_labels = torch.zeros(bg_means.shape[0], dtype=torch.long, device=bg_means.device)

    obj_means_list.append(bg_means)
    obj_covariances_list.append(bg_covs)
    obj_harmonics_dc_list.append(bg_harm)
    obj_opacities_list.append(bg_opac)
    obj_scales_list.append(bg_scales)
    obj_rotations_list.append(bg_rots)
    obj_labels_list.append(bg_labels)

    with torch.amp.autocast('cuda', enabled=False):
        for label_idx, label in enumerate(object_labels):
            print(f"Processing label {label}, text prompt: {text_prompts[label_idx]}")
            infos = []
            for frame_idx, image_np in enumerate(img_list):
                mask = (segmentation_results[frame_idx] == label)
                kernel = torch.ones((3, 3), device=mask.device)
                mask = erosion(mask.float().unsqueeze(0).unsqueeze(0), kernel) # erode the mask for 1 pixel to avoid boundary effect
                mask = mask.squeeze(0).squeeze(0).to(torch.bool)
                if not mask.any().item():
                    continue
                ys, xs = torch.nonzero(mask, as_tuple=True)
                y_min = int(ys.min().item()); y_max = int(ys.max().item()) + 1
                x_min = int(xs.min().item()); x_max = int(xs.max().item()) + 1
                crop_np = image_np[y_min:y_max, x_min:x_max, :]
                crop_mask_np = mask[y_min:y_max, x_min:x_max].detach().to('cpu').numpy().astype(bool)
                composite_np = np.where(crop_mask_np[..., None], crop_np, 255).astype(np.uint8)
                infos.append({'image': Image.fromarray(composite_np, mode='RGB'),
                            'frame_idx': frame_idx,
                            'bbox': (x_min, y_min, x_max, y_max)})
        
            lr_infos = [infos[0]]
            lr_images = [lr_infos[0]['image']]

            hr_images = super_resolution_osediff(osediff_model,lr_images)

            # visualize super-resolution results
            if args.visualize:
                for i, info in enumerate(lr_infos):
                    info['image'].save(os.path.join(args.save_folder, f"original_label_{label}_{i}.png"))
                for i, image in enumerate(hr_images):
                    image.save(os.path.join(args.save_folder, f"upsampled_label_{label}_{i}.png"))
            
            print(f"Generating 3D gaussian for label {label}")

            generated_gs = gen_3d_gaussian_from_image(gs_model,hr_images[0],text_prompt=text_prompts[label_idx])

            generated_gs_opcacity_mask = (generated_gs.opacity > 0).squeeze(-1)

            gs_means = generated_gs.xyz[generated_gs_opcacity_mask]
            gs_scales = generated_gs.scale[generated_gs_opcacity_mask]
            gs_rotations = generated_gs.rotation[generated_gs_opcacity_mask]
            gs_opacities = generated_gs.opacity[generated_gs_opcacity_mask]
            gs_harmonics = generated_gs.rgb[generated_gs_opcacity_mask]
            # calc covariance using scale and rotation
            gs_rotations = gs_rotations / (gs_rotations.norm(dim=-1, keepdim=True) + 1e-8) # [N,4]
            gs_rotations_xyzw = gs_rotations[:, [1, 2, 3, 0]]  # Convert from wxyz to xyzw
            gs_covariances = build_covariance(scale=gs_scales, rotation_xyzw=gs_rotations_xyzw)
            gs_harmonics = gs_harmonics.reshape(-1,3,1)
            gs_harmonics = (gs_harmonics - 0.5) / 0.28209479177387814
            gs_opacities = gs_opacities.reshape(-1) # [N]
            gs_number = gs_means.shape[0]

            generated_gs = Gaussians(
                means=gs_means.unsqueeze(0),
                scales=gs_scales.unsqueeze(0),
                rotations=gs_rotations.unsqueeze(0),
                opacities=gs_opacities.unsqueeze(0),
                covariances=gs_covariances.unsqueeze(0),
                harmonics=gs_harmonics.unsqueeze(0),
            )

            # get index of object gaussians using masks and segmentation
            label_mask = (segmentation_results_downsampled == label) # [N,H,W]
            label_mask = torch.logical_and(masks, label_mask) # [N,H,W]
            label_mask = label_mask.flatten() # [N*H*W]
            
            original_means = gaussians.means[label_mask] # [N,3]
            original_covariances = gaussians.covariances[label_mask] # [N,3,3]
            original_harmonics = gaussians.harmonics[label_mask] # [N,3,(sh+1)^2]
            original_opacities = gaussians.opacities[label_mask] # [N]
            original_scales = gaussians.scales[label_mask] # [N,3]
            original_rotations = gaussians.rotations[label_mask] # [N,4]

            target_means = generated_gs.means.squeeze(0) # [M,3]
            target_covariances = generated_gs.covariances.squeeze(0) # [M,3,3]
            target_harmonics = generated_gs.harmonics.squeeze(0) # [M,3,1]
            target_opacities = generated_gs.opacities.squeeze(0) # [M]
            target_scales = generated_gs.scales.squeeze(0) # [M,3]
            target_rotations = generated_gs.rotations.squeeze(0) # [M,4]

            def get_inlier_mask(means):
                center = means.median(dim=0).values
                d = torch.norm(means - center, dim=1)
                d_med = d.median()
                mad = (d - d_med).abs().median()
                mad = torch.clamp(mad, min=torch.finfo(d.dtype).eps)
                mod_z = 0.6745 * (d - d_med).abs() / mad
                return mod_z <= 3.5

            original_inlier_mask = get_inlier_mask(original_means)
            original_means = original_means[original_inlier_mask]
            original_covariances = original_covariances[original_inlier_mask]
            original_harmonics = original_harmonics[original_inlier_mask]
            original_opacities = original_opacities[original_inlier_mask]
            original_scales = original_scales[original_inlier_mask]
            original_rotations = original_rotations[original_inlier_mask]

            target_inlier_mask = get_inlier_mask(target_means)
            target_means = target_means[target_inlier_mask]
            target_covariances = target_covariances[target_inlier_mask]
            target_harmonics = target_harmonics[target_inlier_mask]
            target_opacities = target_opacities[target_inlier_mask]
            target_scales = target_scales[target_inlier_mask]
            target_rotations = target_rotations[target_inlier_mask]

            num_views = 20
            num_layers = 5
            yaws = torch.linspace(0, 2 * np.pi, num_views).tolist()
            pitch = torch.linspace(0, 0.2*np.pi,num_layers).tolist()
            r = 1.6 # TRELLIS:[-0.5,0.5]r=2  DiffSplat:[-0.4,0.4]r=1.6
            fov = 40
            render_extrinsics_list, render_intrinsics_list = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)

            render_extrinsics = torch.stack(render_extrinsics_list, dim=0).unsqueeze(0).to(device) # [1,num_views,4,4]
            render_intrinsics = torch.stack(render_intrinsics_list, dim=0).unsqueeze(0).to(device)
            
            target_h, target_w = 256, 256
            render_Ks = render_intrinsics.clone()
            render_Ks[..., 0, 0] *= target_w  # fx
            render_Ks[..., 1, 1] *= target_h  # fy
            render_Ks[..., 0, 2] *= target_w  # cx
            render_Ks[..., 1, 2] *= target_h  # cy

            decoded_images = pi3_model.decoder.forward(gaussians=generated_gs, 
                                                extrinsics=render_extrinsics, 
                                                intrinsics=render_Ks, 
                                                masks=torch.ones(1,gs_number,dtype=torch.bool,device=device), 
                                                image_shape=(target_h, target_w)) # [1,num_views,3,H,W]
            
            decoded_images_color = decoded_images.color[0].permute(0,2,3,1).cpu().numpy() # [num_views,H,W,3]

            # Save all decoded images as a single GIF
            if args.visualize:
                images_for_gif = []
                for image in decoded_images_color:
                    img_uint8 = (image * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_uint8)
                    images_for_gif.append(pil_img)

                gif_path = os.path.join(args.save_folder, f"label_{label}_decoded.gif")
                images_for_gif[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images_for_gif[1:],
                    duration=200,
                    loop=0
                )

            # SuperGlue matching between LR images and decoded images
            sg_dir = os.path.join(args.save_folder, f"superglue_label_{label}")

            _best_idx, _best, _best_pairs = match_image_lists(
                matching=superglue_matching,
                image_list_a=lr_images,
                image_list_b=decoded_images_color,
                output_dir=sg_dir,
                resize=[-1],
                resize_float=False,
                viz=args.visualize,
                show_keypoints=False,
                viz_extension='png',
                # pair_mode='product',
            )

            H_orig, W_orig = imgs.shape[2], imgs.shape[3]
            H_ds, W_ds = imgs_downsampled.shape[2], imgs_downsampled.shape[3]

            all_pa = []
            all_pb = []

            for best in _best_pairs:
                ia, ib = best['ia'], best['ib']
                mk0 = torch.from_numpy(best['mkpts0']).float().to(device)  # crop A coords
                mk1 = torch.from_numpy(best['mkpts1']).float().to(device)  # decoded B coords

                # crop->orig->downsampled
                x_min, y_min, x_max, y_max = lr_infos[ia]['bbox']
                mk0 = torch.stack([
                    (mk0[:, 0] + x_min) * (W_ds / W_orig),
                    (mk0[:, 1] + y_min) * (H_ds / H_orig)
                ], dim=-1).to(device)

                # ----- A side: per-pair global nearest-to-ray -----
                frame_idx_src = lr_infos[ia]['frame_idx']
                Ka_pix = cam_intrinsics[frame_idx_src]
                Ta_c2w = camera_poses[frame_idx_src]
                pa_w = nearest_points_on_rays_to_cloud(
                    pixel_xy=mk0,
                    K_pixels=Ka_pix,
                    T_c2w=Ta_c2w,
                    point_cloud_world=original_means,
                )

                # ----- B side: per-pair global nearest-to-ray -----
                Kb_pix = render_Ks[0, ib]
                T_w2c_B = render_extrinsics[0, ib]
                T_c2w_B = se3_inverse(T_w2c_B)
                pb_w = nearest_points_on_rays_to_cloud(
                    pixel_xy=mk1,
                    K_pixels=Kb_pix,
                    T_c2w=T_c2w_B,
                    point_cloud_world=target_means,
                )

                all_pa.append(pa_w)
                all_pb.append(pb_w)

            all_pa = torch.cat(all_pa, dim=0)
            all_pb = torch.cat(all_pb, dim=0)

            # visualize correspondences with clouds (target vs original) before alignment
            if args.visualize:
                vis_dir = os.path.join(args.save_folder, 'correspondences')
                visualize_pointclouds_and_correspondences(
                    points_a=target_means,
                    points_b=original_means,
                    corr_a=all_pa,
                    corr_b=all_pb,
                    out_dir=vis_dir,
                    basename=f'correspondences_label_{label}',
                    point_radius=0.001,
                    corr_radius=0.01,
                    max_points=80000,
                    max_corr=3000,
                )

            num_ransac_iters = 256
            subset = min(6, all_pa.shape[0])
            best_loss = float('inf')
            best_init = None

            # scale from bounding-box extent ratio (target -> original)
            ext_o = original_means.max(dim=0).values - original_means.min(dim=0).values
            ext_t = target_means.max(dim=0).values - target_means.min(dim=0).values
            s_bbox = (torch.linalg.norm(ext_o) / torch.clamp(torch.linalg.norm(ext_t), min=1e-8))

            def _estimate_rt_kabsch_batched(src: torch.Tensor, tgt: torch.Tensor):
                # src/tgt: [K, S, 3]
                mu_s = src.mean(dim=1)
                mu_t = tgt.mean(dim=1)
                X = src - mu_s.unsqueeze(1)
                Y = tgt - mu_t.unsqueeze(1)
                H = torch.matmul(X.transpose(1, 2), Y)  # [K, 3, 3]
                U, _, Vt = torch.linalg.svd(H)
                R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
                det = torch.det(R)
                mask = det < 0
                if mask.any():
                    Vt[mask, -1, :] *= -1
                    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
                t = mu_t - torch.matmul(R, mu_s.unsqueeze(-1)).squeeze(-1)
                return R, t

            # Parallel RANSAC
            Ncorr = all_pa.shape[0]
            weights = torch.ones(num_ransac_iters, Ncorr, device=all_pa.device)
            idx = torch.multinomial(weights, num_samples=subset, replacement=False)  # [K, subset]
            a_s_batch = all_pa[idx]  # [num_ransac_iters, subset, 3]
            b_s_batch = all_pb[idx]  # [num_ransac_iters, subset, 3]

            # Pre-scale source by s_bbox, then estimate R and t via Kabsch on scaled correspondences
            R_batch, t_batch = _estimate_rt_kabsch_batched(src=s_bbox * b_s_batch, tgt=a_s_batch)

            # Compute Chamfer loss for each hypothesis in parallel
            chamfer_fn = ChamferDistance()
            x_batch = torch.einsum('kij,mj->kmi', R_batch, target_means)  # [K, M, 3]
            x_batch = s_bbox * x_batch + t_batch.unsqueeze(1)
            y_batch = original_means.unsqueeze(0).expand(num_ransac_iters, -1, -1)
            dist1, dist2 = chamfer_fn(x_batch, y_batch)
            losses = dist1.mean(dim=1) + dist2.mean(dim=1)

            best_k = torch.argmin(losses)
            best_loss = float(losses[best_k].item())
            best_R = R_batch[best_k]  
            best_t = t_batch[best_k]
            best_init = {'scale': s_bbox, 'R': best_R, 't': best_t}

            print(f"Initialize pose estimation with RANSAC init with loss: {best_loss}, scale(bbox): {s_bbox}, R: {best_R}, t: {best_t}")

            # Prepare anchor correspondences from RANSAC best hypothesis: use the sampled subset that produced best_k
            anchor_src = a_s_batch[best_k]
            anchor_tgt = b_s_batch[best_k]

            scale_factor, T_sim3 = estimate_sim3_chamfer_gd(
                source_points=target_means,
                target_points=original_means,
                max_iterations=500,
                lr=1e-2,
                num_samples=None,
                bidirectional=True,
                init_guess=best_init,
                anchor_src=anchor_src,
                anchor_tgt=anchor_tgt,
                anchor_weight=0,
                lr_rot=1e-2,
                lr_scale=1e-2,
                lr_trans=1e-2,
                convergence_tol=1e-7,
            )
            

            R = T_sim3[:3, :3]
            t = T_sim3[:3, 3]

            transformed_target_means = (scale_factor * (target_means @ R.T)) + t  # [M,3]

            A = scale_factor * R
            transformed_target_covariances = A @ target_covariances @ A.transpose(-1, -2)  # [M,3,3]

            # labels for objects
            labels_obj = torch.full((transformed_target_means.shape[0],), int(label), dtype=torch.long, device=transformed_target_means.device)

            obj_means_list.append(transformed_target_means)
            obj_covariances_list.append(transformed_target_covariances)
            obj_harmonics_dc_list.append(target_harmonics)
            obj_opacities_list.append(target_opacities)
            obj_scales_list.append(target_scales)
            obj_rotations_list.append(target_rotations)
            obj_labels_list.append(labels_obj)

            # append original, for debug
            obj_means_list.append(original_means)
            obj_covariances_list.append(original_covariances)
            obj_harmonics_dc_list.append(original_harmonics[:,:,:1])
            obj_opacities_list.append(original_opacities)
            obj_scales_list.append(original_scales)
            obj_rotations_list.append(original_rotations)
            obj_labels_list.append(torch.full((original_means.shape[0],), int(label+max(object_labels)), dtype=torch.long, device=original_means.device))

            # Report results
            dist_1_mean, dist_2_mean = ChamferDistance()(transformed_target_means.unsqueeze(0), original_means.unsqueeze(0))
            final_chamfer_loss = dist_1_mean.mean() + dist_2_mean.mean()
            print(f"[Label {label}] Sim3 scale: {float(scale_factor):.6f}")
            print(f"[Label {label}] Sim3 matrix (target -> original):\n{T_sim3}")
            print(f"[Label {label}] Alignment Chamfer distance: {final_chamfer_loss.item():.6e}")

            if args.visualize:
                write_ply(xyz=original_means, rgb=torch.clamp(original_harmonics[:,:,0] * 0.28209479177387814 + 0.5, 0.0, 1.0), path=os.path.join(args.save_folder, f"label_{label}_original.ply"))
                write_ply(xyz=transformed_target_means, rgb=torch.clamp(target_harmonics[:,:,0] * 0.28209479177387814 + 0.5, 0.0, 1.0), path=os.path.join(args.save_folder, f"label_{label}_refine.ply"))

        # Concatenate background with all objects and export before releasing models
        obj_means = torch.cat(obj_means_list, dim=0) # [N,3]
        obj_covs = torch.cat(obj_covariances_list, dim=0) # [N,3,3]
        obj_harm_dc = torch.cat(obj_harmonics_dc_list, dim=0)
        obj_opac = torch.cat(obj_opacities_list, dim=0)
        obj_scales = torch.cat(obj_scales_list, dim=0)
        obj_rots = torch.cat(obj_rotations_list, dim=0)
        obj_labels = torch.cat(obj_labels_list, dim=0)

        edited_gaussians = Gaussians(
            means=obj_means.unsqueeze(0),
            covariances=obj_covs.unsqueeze(0),
            harmonics=obj_harm_dc.unsqueeze(0),
            opacities=obj_opac.unsqueeze(0),
            scales=obj_scales.unsqueeze(0),
            rotations=obj_rots.unsqueeze(0),
        )

        # recalculate the scales and rotations
        edited_gaussians.calc_world_scale_and_rot()

        if args.visualize:
            export_pt(gaussians=edited_gaussians,
                    pt_path=os.path.join(args.save_folder, 'edited_gaussians.pt'),
                    labels=obj_labels)

            # export_pt(gaussians=gaussians,
            #         pt_path=os.path.join(args.save_folder, 'gaussians.pt'),
            #         depth_conf=torch.sigmoid(encoder_output['conf'][..., 0]),labels=segmentation_results_downsampled)

            # write_ply(
            #     xyz=world_points,
            #     rgb=imgs_downsampled,
            #     path=os.path.join(args.save_folder, 'points.ply'),
            # )

        if args.visualize:
            rendered_img = decoder_output.color
            rendered_depth = decoder_output.depth
            pred_depth = encoder_output['local_points'][...,2]

            editted_decoder_output = pi3_model.decoder.forward(gaussians=edited_gaussians, 
                                                    extrinsics=encoder_output['extrinsics'], 
                                                    intrinsics=encoder_output['intrinsics'], 
                                                    masks=torch.ones(1,edited_gaussians.means.shape[1],dtype=torch.bool,device=device), 
                                                    image_shape=(448, 448))

            create_gif_from_tensor(
                    tensor_pred=rendered_img[0].detach().cpu(),
                    tensor_gt=imgs.detach().cpu(),
                    depth_pred=rendered_depth[0].detach().cpu(),
                    depth_gt=pred_depth[0].detach().cpu(),
                    output_path=os.path.join(args.save_folder, 'rendered.gif'),
                    fps=4
                )

            create_refine_comparison(
                    tensor_pred=rendered_img[0].detach().cpu(),
                    tensor_refined=editted_decoder_output.color[0].detach().cpu(),
                    tensor_gt=imgs.detach().cpu(),
                    output_path=os.path.join(args.save_folder, 'rendered.gif'),
                    fps=4
                )

        result_dict = dict(
            view_num=len(imgs), # Number of input images
            # world_points=world_points, # [N,H,W,3]
            gaussians=edited_gaussians,
            camera_poses=camera_poses, # [N,4,4]
            conf_masks=masks, # [N,H,W]
            segmentation_results=obj_labels, # [N,H,W]
        )

        # save result_dict as a torch file
        if save_pt:
            torch.save(result_dict, os.path.join(args.save_folder, 'reconstruction_results.pt'))
    
    return result_dict

def preprocess_data(result_dict,args):
    positions = result_dict['gaussians'].means.squeeze(0)  # [num_points, 3]
    covariances = result_dict['gaussians'].covariances.squeeze(0) 
    harmonics = result_dict['gaussians'].harmonics.squeeze(0) 
    opacities = result_dict['gaussians'].opacities.squeeze(0) 
    rotations = result_dict['gaussians'].rotations.squeeze(0) 
    scales = result_dict['gaussians'].scales.squeeze(0) 
    segmentation_results = result_dict['segmentation_results'].reshape(-1) # [num_points]

    if args.real_world:
        # put cameras away from origin and shift z
        parent = '/mnt/nfs_project_a/shared/models/ngff/ngff_perception/datasets/realworld_0919' # real world
        with open(f'{parent}/cameras.json', 'r') as file:
            gt_camera = json.load(file)
        gt_camera_positions = [gt_camera[i]['position'] for i in range(0,len(gt_camera),args.interval*10)][:result_dict['view_num']] # real world 10 views
        gt_camera_positions = torch.tensor(gt_camera_positions).to(positions.device) # [87, 3]
        center_gt_camera = gt_camera_positions.mean(dim=0)
        gt_camera_positions = (gt_camera_positions - center_gt_camera) * 0.5 + center_gt_camera # scale 0.5 z-1.0 works; 0.4 z-1.25
        gt_camera_positions[...,2] -= 1.0
    else:
        # Load sim camera
        parent = os.path.dirname(args.data_path) if not os.path.isdir(args.data_path) else args.data_path
        with open(f'{parent}/cameras.json', 'r') as file:
            gt_camera = json.load(file)
        gt_camera_positions = [gt_camera[i]['position'] for i in range(0,len(gt_camera),args.interval)] # sim 100 views
        gt_camera_positions = [gt_camera_positions[idx] for idx in range(0, len(gt_camera_positions)) if idx % 5 != 4]
        gt_camera_positions = torch.tensor(gt_camera_positions).to(positions.device) # [87, 3]
    
    pred_camera_poses = result_dict['camera_poses']  # [87, 4, 4]
    # Load predicted camera
    pred_camera_positions = pred_camera_poses[..., :3, 3]  # [87, 3]
    # Align the predicted camera to the sim camera
    transformation_matrix, scale_factor = align_camera_poses(gt_camera_positions, pred_camera_positions) # [4, 4]
    camera_center = pred_camera_poses[:, :3, 3].mean(dim=0)  # [3]
    # Transform points to align with sim coordinate using the estimated transformation matrix
    R = transformation_matrix[:3, :3] 
    T = transformation_matrix[:3, 3]
    S = scale_factor * torch.eye(3, device=R.device, dtype=R.dtype) 
    A = S @ R
    transformed_positions = torch.matmul(scale_factor*(positions-camera_center), R.T) + T + camera_center
    transformed_covariances = A @ covariances @ A.T
    transformed_harmonics = harmonics  # [num_points, 9, 3], keep the same
    transformed_opacities = opacities  # [num_points, 1], keep the same
    transformed_rotations = rotations  # [num_points, 3, 3], keep the same
    transformed_scales = scales * scale_factor # [num_points, 3], keep the same

    # Filter out points in the box of -1 to 1 (xyz)
    assert segmentation_results.max() % 2 == 0, "Segmentation results must be even"
    obj_num = segmentation_results.max() // 2

    # sim_area_mask = (segmentation_results != 0)
    if args.refine:
        sim_area_mask = torch.logical_and(segmentation_results != 0, segmentation_results <= obj_num)
    else:
        sim_area_mask = torch.logical_and(segmentation_results != 0, segmentation_results > obj_num)
    
    sim_positions = transformed_positions[sim_area_mask]  # [num_points, 3]
    sim_covariances = transformed_covariances[sim_area_mask]  # [num_points, 3, 3]
    sim_harmonics = transformed_harmonics[sim_area_mask]  # [num_points, 9, 3]
    sim_opacities = transformed_opacities[sim_area_mask]  # [num_points, 1]
    sim_rotations = transformed_rotations[sim_area_mask]  # [num_points, 3, 3]
    sim_scales = transformed_scales[sim_area_mask]  # [num_points, 3]
    sim_segmentation_results = segmentation_results[sim_area_mask]  # [num_points]

    # Save out of box data

    # out_of_box_mask = ~sim_area_mask
    # filter out points with z > 0.01 and x y in [-1,1]

    if not args.real_world:
        inside_box_mask = (
            (transformed_positions[..., 2] > -0.98) &
            (transformed_positions[..., 0] > -1) & (transformed_positions[..., 0] < 1) &
            (transformed_positions[..., 1] > -1) & (transformed_positions[..., 1] < 1)
        )

        out_of_box_mask = torch.logical_and(segmentation_results == 0, ~inside_box_mask)
    else:
        out_of_box_mask = (segmentation_results == 0)
    
    out_of_box_positions = transformed_positions[out_of_box_mask]  # [num_points, 3]
    out_of_box_covariances = transformed_covariances[out_of_box_mask]  # [num_points, 3, 3]
    out_of_box_harmonics = transformed_harmonics[out_of_box_mask]  # [num_points, 9, 3]
    out_of_box_opacities = transformed_opacities[out_of_box_mask]  # [num_points, 1]
    out_of_box_rotations = transformed_rotations[out_of_box_mask]  # [num_points, 3, 3]
    out_of_box_scales = transformed_scales[out_of_box_mask]  # [num_pts, 3]

    if args.real_world:
        # align min_sim_area_mask_z to z=-1
        min_sim_area_mask_z = sim_positions[...,2].min() 
        sim_positions[...,2] -= (min_sim_area_mask_z + 1.0)
        out_of_box_positions[...,2] -= (min_sim_area_mask_z + 1.0)

    # save the sim gaussians into ply file
    if args.refine:
        sim_prefix = 'sim_refine'
    else:
        sim_prefix = 'sim'

    export_ply(means=sim_positions,
                scales=torch.log(sim_scales),
                rotations=sim_rotations,
                harmonics=sim_harmonics,
                opacities=torch.log(sim_opacities) - torch.log1p(-sim_opacities), 
                path=Path(f'{args.save_folder}/{sim_prefix}/point_cloud/iteration_30000/point_cloud.ply'),
                save_sh_dc_only=False)
    # save the out of box gaussians into ply file
    if not os.path.exists(f'{args.save_folder}/bg/point_cloud/iteration_30000/point_cloud.ply'):
        export_ply(means=out_of_box_positions,
                    scales=torch.log(out_of_box_scales),
                    rotations=out_of_box_rotations,
                    harmonics=out_of_box_harmonics,
                    opacities=torch.log(out_of_box_opacities) - torch.log1p(-out_of_box_opacities), 
                    path=Path(f'{args.save_folder}/bg/point_cloud/iteration_30000/point_cloud.ply'),
                    save_sh_dc_only=False)

    # save cameras.json into the same directory as the ply file
    cameras_path = Path(args.save_folder) / sim_prefix / 'cameras.json'
    # copy the cameras.json file
    os.system(f'cp {parent}/cameras.json {cameras_path}')
    # save segmentation_result
    segmentation_path = Path(args.save_folder) / sim_prefix / 'segmentation.npy'
    np.save(segmentation_path, sim_segmentation_results.cpu().numpy())

    sim_gaussians = Gaussians(
        means=sim_positions.unsqueeze(0),
        covariances=sim_covariances.unsqueeze(0),
        harmonics=sim_harmonics.unsqueeze(0),
        opacities=sim_opacities.unsqueeze(0),
        scales=sim_scales.unsqueeze(0),
        rotations=sim_rotations.unsqueeze(0),
    )
    bg_gaussians = Gaussians(
        means=out_of_box_positions.unsqueeze(0),
        covariances=out_of_box_covariances.unsqueeze(0),
        harmonics=out_of_box_harmonics.unsqueeze(0),
        opacities=out_of_box_opacities.unsqueeze(0),
        scales=out_of_box_scales.unsqueeze(0),
        rotations=out_of_box_rotations.unsqueeze(0),
    )
    labels = sim_segmentation_results
    return sim_gaussians, bg_gaussians, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_folder", type=str, default='examples',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=4,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--prompt_type", type=str, default='bbox',choices=['point', 'bbox'],
                    help="Prompt type for grounding.")
    parser.add_argument("--sd_version", type=str, choices=['sd15','sd35'],default='sd15')
    parser.add_argument("--real_world", action='store_true', default=False, help="Whether the input is real-world data.")
    parser.add_argument("--visualize", action="store_true", default=False, help="Whether to visualize intermediate results")
    parser.add_argument("--refine", action="store_true", default=False, help="Whether to refine the reconstruction results")
    parser.add_argument("--batch_inference_json",type=str,default=None,help="Path to the batch inference json file.")
                        
    args = parser.parse_args()
    setup_seed(seed=args.seed)

    #################################
    #  Reconstruct initial GS data  #
    #################################
    if args.batch_inference_json is not None:
        with open(args.batch_inference_json, 'r') as f:
            batch_inference_data = json.load(f)
        for data in batch_inference_data:
            args.data_path = data['data_path']
            args.save_folder = data['save_folder']

            try:
                if os.path.exists(os.path.join(args.save_folder, 'reconstruction_results.pt')):
                    print(f"Loading existing reconstruction results from {os.path.join(args.save_folder, 'reconstruction_results.pt')}")
                    result_dict = torch.load(os.path.join(args.save_folder, 'reconstruction_results.pt'), weights_only=False)
                else:
                    pretrain_models = prepare_models()
                    result_dict = reconstruction(args, pretrain_models, save_pt=True)
                
                preprocess_data(result_dict,args)

            except Exception as e:
                print(f"Error processing {args.data_path}: {e}")
                continue
    else:
        if os.path.exists(os.path.join(args.save_folder, 'reconstruction_results.pt')):
            print(f"Loading existing reconstruction results from {os.path.join(args.save_folder, 'reconstruction_results.pt')}")
            result_dict = torch.load(os.path.join(args.save_folder, 'reconstruction_results.pt'), weights_only=False)
        else:
            pretrain_models = prepare_models()
            result_dict = reconstruction(args, pretrain_models, save_pt=True)
        
        preprocess_data(result_dict,args)