import argparse
import torch
import viser
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from gsplat.rendering import rasterization
from nerfview import CameraState, RenderTabState, apply_float_colormap
from .gsplat_viewer import GsplatViewer, GsplatRenderTabState
from utils.io_utils import load_ply_gaussian_splats,load_pt_gaussian_splats

def save_filtered_ply(save_path, means, quats, scales, opacities, colors):
    """
    Save filtered Gaussian splats into a PLY file in the same format as load_ply_gaussian_splats.
    
    Args:
        save_path (str or Path):
        means (Tensor): [N,3]
        quats (Tensor): [N,4]
        scales (Tensor): [N,3] (after exp)
        opacities (Tensor): [N]
        colors (Tensor): [N, coeff_cnt+1, 3] (SH+DC)
    """
    means = means.cpu().numpy()
    quats = quats.cpu().numpy()
    scales = scales.cpu().numpy()
    opacities = opacities.cpu().numpy()
    colors = colors.cpu().numpy()

    N, coeff_cnt_plus1, _ = colors.shape
    coeff_cnt = coeff_cnt_plus1 - 1

    dtype_list = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), 
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    # f_rest_i
    for i in range(coeff_cnt * 3):
        dtype_list.append((f"f_rest_{i}", "f4"))

    dtype_list.extend([
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ])

    vertex_data = np.empty(N, dtype=dtype_list)


    vertex_data["x"], vertex_data["y"], vertex_data["z"] = means[:, 0], means[:, 1], means[:, 2]
    vertex_data["nx"], vertex_data["ny"], vertex_data["nz"] = 0.0, 0.0, 0.0 
    
    # DC
    vertex_data["f_dc_0"], vertex_data["f_dc_1"], vertex_data["f_dc_2"] = colors[:, 0, 0], colors[:, 0, 1], colors[:, 0, 2]

    # rest
    if coeff_cnt > 0:
        rest_flat = colors[:, 1:, :].reshape(N, coeff_cnt * 3)
        for i in range(coeff_cnt * 3):
            vertex_data[f"f_rest_{i}"] = rest_flat[:, i]

    vertex_data["opacity"] = torch.logit(torch.from_numpy(opacities)).numpy()  # 存回logit空间
    vertex_data["scale_0"], vertex_data["scale_1"], vertex_data["scale_2"] = np.log(scales[:, 0]), np.log(scales[:, 1]), np.log(scales[:, 2])
    vertex_data["rot_0"], vertex_data["rot_1"], vertex_data["rot_2"], vertex_data["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    ply_el = PlyElement.describe(vertex_data, "vertex")
    PlyData([ply_el], text=False).write(str(save_path))

    print(f"Saved filtered PLY with {N} Gaussians → {save_path}")

import torch
import math

def quat_mul(q1, q2):
    """ q' = q1 * q2"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_from_axis_angle(axis: str, degrees: float, device="cuda"):
    theta = math.radians(degrees)
    half = theta / 2
    c, s = math.cos(half), math.sin(half)

    if axis.lower() == "x":
        q = torch.tensor([c, s, 0, 0], dtype=torch.float32, device=device)
    elif axis.lower() == "y":
        q = torch.tensor([c, 0, s, 0], dtype=torch.float32, device=device)
    elif axis.lower() == "z":
        q = torch.tensor([c, 0, 0, s], dtype=torch.float32, device=device)
    else:
        raise ValueError("axis must be 'x', 'y' or 'z'")
    return q

def rotate_gaussians(means, quats, axis="z", degrees=0.0):
    device = quats.device
    rot_q = quat_from_axis_angle(axis, degrees, device=device)  # (4,)
    rot_q = rot_q.expand(quats.shape[0], -1)  # (N, 4)

    rotated_quats = quat_mul(rot_q, quats)

    if means is not None:
        w, x, y, z = rot_q[0]
        R = torch.tensor([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ], dtype=torch.float32, device=device)
        rotated_means = means @ R.T
    else:
        rotated_means = means

    return rotated_means, rotated_quats

def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Gaussian splats from .pt or .ply
    if args.ply_path.lower().endswith('.pt'):
        means, quats, scales, opacities, colors, sh_degree, labels, depth_conf = load_pt_gaussian_splats(args.ply_path, device)
    else:
        means, quats, scales, opacities, colors, sh_degree = load_ply_gaussian_splats(args.ply_path, device)
        labels, depth_conf = None, None

    # Keep originals for runtime filtering
    base_means = means
    base_quats = quats
    base_scales = scales
    base_opacities = opacities
    base_colors = colors
    base_labels = labels
    base_depth_conf = depth_conf

    print(f"Number of Gaussians: {len(base_means)}")
    
    # Create viewer render function
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):

        assert isinstance(render_tab_state, GsplatRenderTabState)
        
        # Get render dimensions
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        
        # Get camera parameters
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device) # [3,3]
        viewmat = c2w.inverse()
        
        # Render mode mapping
        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }
        
        # Render the scene
        # Create background tensor
        backgrounds = torch.tensor([render_tab_state.backgrounds], device=device) / 255.0
        
        # Apply depth_conf top-percent filtering if available
        if base_depth_conf is not None:
            percent = getattr(render_tab_state, 'top_conf_percent', 100.0)
            percent = float(percent)
            percent = max(0.0, min(100.0, percent))
            if percent < 100.0 and base_depth_conf.numel() > 0:
                N_total = base_means.shape[0]
                k = max(1, int(N_total * (percent / 100.0)))
                # Top-k by depth_conf
                vals, idx = torch.topk(base_depth_conf, k, largest=True, sorted=False)
                means_f = base_means[idx]
                quats_f = base_quats[idx]
                scales_f = base_scales[idx]
                opacities_f = base_opacities[idx]
                colors_f = base_colors[idx]
                labels_f = base_labels[idx] if base_labels is not None else None
            else:
                means_f = base_means
                quats_f = base_quats
                scales_f = base_scales
                opacities_f = base_opacities
                colors_f = base_colors
                labels_f = base_labels
        else:
            means_f = base_means
            quats_f = base_quats
            scales_f = base_scales
            opacities_f = base_opacities
            colors_f = base_colors
            labels_f = base_labels

        # Apply label filtering if labels available and some are selected
        if labels_f is not None and hasattr(render_tab_state, 'selected_labels'):
            selected = list(render_tab_state.selected_labels)
            if len(selected) > 0:
                sel_mask = torch.zeros(labels_f.shape, dtype=torch.bool, device=labels_f.device)
                for lbl in selected:
                    sel_mask |= (labels_f == int(lbl))
                means_f = means_f[sel_mask]
                quats_f = quats_f[sel_mask]
                scales_f = scales_f[sel_mask]
                opacities_f = opacities_f[sel_mask]
                colors_f = colors_f[sel_mask]

        render_colors, render_alphas, info = rasterization(
            means=means_f,
            quats=quats_f,
            scales=scales_f,
            opacities=opacities_f,
            colors=colors_f,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],        # [1, 3, 3]
            width=width,
            height=height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            backgrounds=backgrounds,
        )
        
        # Update statistics
        render_tab_state.total_gs_count = len(base_means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()
        
        # Process render output based on mode
        if render_tab_state.render_mode == "rgb":
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
        
        return renders
    
    # Start viewer server
    server = viser.ViserServer(host="0.0.0.0", port=args.port, verbose=False)
    viewer = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path("."),
        mode="rendering",
    )
    # Initialize label filter UI if labels exist
    if base_labels is not None:
        try:
            uniq = torch.unique(base_labels.detach().cpu()).tolist()
            viewer.set_available_labels(uniq)
        except Exception:
            pass
    
    print(f"Viewer running on http://0.0.0.0:{args.port}")
    print("Press Ctrl+C to exit...")
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PLY Gaussian Splats in web browser")
    parser.add_argument("--ply_path", type=str, help="Path to the PLY file")
    parser.add_argument("--port", type=int, default=7860, help="Port for the viewer server")
    
    args = parser.parse_args()
    main(args)