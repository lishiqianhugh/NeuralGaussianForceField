import sys
import argparse
import torch
import os
import numpy as np
import json
from tqdm import tqdm

# Utils
from utils.decode_param import *
from utils.gs_utils import GaussianModel,load_params_from_gs
from utils.camera_utils import Camera as GSCamera
from utils.general_utils import searchForMaxIteration,find_indexes
from utils.transformation_utils import generate_rotation_matrices,apply_rotations,apply_cov_rotations,shift2center111, \
    transform2origin,get_center_view_worldspace_and_observant_coordinate,apply_inverse_rotations,undotransform2origin,undoshift2center111,apply_inverse_cov_rotations

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp
wp.init()
wp.config.verify_cuda = True

from dataset.particle_filling import *
from dataset.constants import OBJPART,INDEX_OBJ

import taichi as ti
ti.init(arch=ti.cuda, device_memory_GB=10.0)

torch.manual_seed(42)

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
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--initial_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--background_path", type=str, default=None)
    parser.add_argument("--single_view", action="store_true")
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
    indexs = find_indexes(args.model_path)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path, args.shs_degree)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]

    reconstruct = False
    if 'segmentation.npy' in os.listdir(args.model_path):
        reconstruct = True
        # use reconstructed scene
        seg_path = os.path.join(args.model_path, 'segmentation.npy')
        obj_ids = torch.tensor(np.load(seg_path), dtype=torch.long, device=mask.device).unsqueeze(-1)
        obj_ids = obj_ids[mask, :]
        # reoder the object ids to put the same object together
        obj_ids, reoder_indexs = torch.sort(obj_ids, dim=0)
        reoder_indexs = reoder_indexs.squeeze(-1)
        init_pos = init_pos[reoder_indexs, :]
        init_cov = init_cov[reoder_indexs, :]
        init_opacity = init_opacity[reoder_indexs, :]
        init_screen_points = init_screen_points[reoder_indexs, :]
        init_shs = init_shs[reoder_indexs, :]
        # create a dictionary to map object ids to object names
        INDEX_OBJ = {}
        for i in torch.unique(obj_ids):
            INDEX_OBJ[i.item()] = f"object_{i.item()}"
        material_params["objpart"] = {}
        material_params["E"] = {}
        for j, index in enumerate(INDEX_OBJ.keys()):
            material_params["objpart"][INDEX_OBJ[index]] = int(torch.sum(obj_ids == index).item())
            material_params["E"][INDEX_OBJ[index]] = json.loads(args.E)[j]
        indexs = INDEX_OBJ.keys()
    
    # rorate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(
            init_pos,
            "./log/init_particles.ply",
        )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    # select a sim area and save params of unslected particles
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    # filling_params = preprocessing_params["particle_filling"]
    filling_params = None

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )

    expanded_objects = []
    for obj_idx in indexs:
        expanded_objects.append(INDEX_OBJ[obj_idx])
    material_params["objects"] = expanded_objects
    mpm_solver.set_parameters_dict(material_params)

    # set up initial velocity
    mpm_init_vel = torch.zeros_like(mpm_init_pos).to(device=device)
    # mpm_init_vel[:material_params["objpart"][material_params["objects"][0]], 0] = 1.0
    mpm_solver.import_particle_v_from_torch(mpm_init_vel)


    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    mpm_solver.finalize_mu_lam()

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
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )


    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None
            
    #################################
    #           Simulation          #
    #################################
    if args.output_path:
        for frame in tqdm(range(frame_num)):
            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)

            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)
                screen_points = init_screen_points

            if args.save_h5:
                # save all the variables required for rendering into one .h5 
                if frame == 0:
                    shs_h5 = h5py.File(os.path.join(args.output_path, "shs.h5"), "w")
                    shs_h5.create_dataset("shs", data=shs.detach().cpu().numpy(), dtype="f2")
                    shs_h5.close()
                    opacity_h5 = h5py.File(os.path.join(args.output_path, "opacity.h5"), "w")
                    opacity_h5.create_dataset("opacity", data=opacity.detach().cpu().numpy(), dtype="f2")
                    opacity_h5.close()
                gs_h5 = h5py.File(os.path.join(args.output_path, f"{frame:04d}.h5"), "w")
                gs_h5.create_dataset("pos", data=pos.detach().cpu().numpy(), dtype="f2")
                gs_h5.create_dataset("cov3D", data=cov3D.detach().cpu().numpy(), dtype="f2")
                gs_h5.create_dataset("rot", data=rot.detach().cpu().numpy(), dtype="f2")
                gs_h5.close()

            for step in range(step_per_frame):
                mpm_solver.p2g2p(frame, substep_dt, device=device)
