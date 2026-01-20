import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
import io
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
from typing import *

def create_colored_mask_gif(mask_list_tensor, save_path, original_images=None):

    if original_images is not None and isinstance(original_images, torch.Tensor):
        original_images = original_images.permute(0, 2, 3, 1).cpu().numpy()

    mask_numpy = mask_list_tensor.cpu().numpy()
    unique_labels = np.unique(mask_numpy)
    np.random.seed(42)
    color_map = {label: ([0, 0, 0] if label == 0 else np.random.rand(3).tolist()) for label in unique_labels}
    
    frames = []
    for frame_idx, mask_frame in enumerate(mask_numpy):
        H, W = mask_frame.shape
        colored_mask = np.zeros((H, W, 3))
        for label, color in color_map.items():
            colored_mask[mask_frame == label] = color
        
        fig = plt.figure(figsize=(24, 5), dpi=300)
        
        ax1 = plt.subplot(1, 4, 1)
        original_img = original_images[frame_idx].copy()
        ax1.imshow(original_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(colored_mask)
        ax2.set_title('Colored Masks')
        ax2.axis('off')
        
        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(original_img)

        ax3.imshow(colored_mask, alpha=0.5)
        ax3.set_title('Overlay Mask')
        ax3.axis('off')
    
        ax4 = plt.subplot(1, 4, 4)
        ax4.axis('off')
        ax4.set_title('Legend')
        
        sorted_labels = sorted(unique_labels)
        for i, label in enumerate(sorted_labels):
            col = i % 3 
            row = i // 3 
            
            x = 0.05 + col * 0.32
            y = 0.9 - row * 0.08
            
            color = color_map[label]
            label_text = f"Obj {label}" if label != 0 else "BG"
            
            rect = Rectangle((x, y-0.02), 0.08, 0.04, facecolor=color, edgecolor='black')
            ax4.add_patch(rect)
            ax4.text(x + 0.1, y, label_text, fontsize=9, verticalalignment='center')
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
    
    if frames:
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=500, loop=0)
        print(f"Colored mask GIF saved to {save_path}")

def create_gif_from_tensor(tensor_pred, tensor_gt, depth_pred, depth_gt, output_path, fps=8):
    """
    Create a GIF from video tensors with 2x2 layout using turbo colormap for depth
    
    Args:
        tensor_pred: predicted RGB images (T, C, H, W)
        tensor_gt: ground truth RGB images (T, C, H, W)
        depth_pred: rendered/predicted depth maps (T, H, W) or (T, 1, H, W)
        depth_gt: alternative depth maps (e.g., point z-coordinates) (T, H, W) or (T, 1, H, W)
        output_path: path to save the GIF
        fps: frames per second
    """
    
    def normalize_depth_for_colormap(depth_pred, depth_gt):
        """Normalize depth for colormap visualization using the same scale"""
        # Convert to numpy if needed
        if torch.is_tensor(depth_pred):
            depth_pred = depth_pred.detach().cpu().numpy()
        if torch.is_tensor(depth_gt):
            depth_gt = depth_gt.detach().cpu().numpy()
            
        # Handle shape (T, 1, H, W) -> (T, H, W)
        if depth_pred.ndim == 4 and depth_pred.shape[1] == 1:
            depth_pred = depth_pred.squeeze(1)
        if depth_gt.ndim == 4 and depth_gt.shape[1] == 1:
            depth_gt = depth_gt.squeeze(1)
            
        # Calculate global min/max across both depth maps and all frames
        global_min = min(np.min(depth_pred), np.min(depth_gt))
        global_max = max(np.max(depth_pred), np.max(depth_gt))
        
        # Normalize both depth maps using the same global scale
        if global_max > global_min:
            depth_pred_norm = (depth_pred - global_min) / (global_max - global_min)
            depth_gt_norm = (depth_gt - global_min) / (global_max - global_min)
        else:
            depth_pred_norm = depth_pred
            depth_gt_norm = depth_gt
        
        return depth_pred_norm, depth_gt_norm, global_min, global_max
    
    images = []
    T, C, H, W = tensor_pred.shape
    
    # Normalize depth maps using the same scale
    depth_pred_norm, depth_gt_norm, global_min, global_max = normalize_depth_for_colormap(depth_pred, depth_gt)
    
    for t in range(T):
        # Create 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f'Frame {t+1}', fontsize=16)
        
        # Top-left: Predicted RGB
        frame_pred = tensor_pred[t].permute(1, 2, 0).cpu().numpy()
        if frame_pred.shape[2] == 1:
            frame_pred = frame_pred.squeeze(2)
        frame_pred = np.clip(frame_pred, 0, 1)
        axes[0, 0].imshow(frame_pred)
        axes[0, 0].set_title('Predicted RGB')
        axes[0, 0].axis('off')
        
        # Top-right: Ground Truth RGB
        frame_gt = tensor_gt[t].permute(1, 2, 0).cpu().numpy()
        if frame_gt.shape[2] == 1:
            frame_gt = frame_gt.squeeze(2)
        frame_gt = np.clip(frame_gt, 0, 1)
        axes[0, 1].imshow(frame_gt)
        axes[0, 1].set_title('Ground Truth RGB')
        axes[0, 1].axis('off')
        
        # Bottom-left: Rendered Depth
        im1 = axes[1, 0].imshow(depth_pred_norm[t], cmap='turbo', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Rendered Depth (range: {global_min:.2f}-{global_max:.2f})')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        axes[1, 0].axis('off')
        
        # Bottom-right: Point Depth (Z-coordinate)
        im2 = axes[1, 1].imshow(depth_gt_norm[t], cmap='turbo', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Point Depth (range: {global_min:.2f}-{global_max:.2f})')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        pil_image = Image.open(buf)
        images.append(pil_image)
        plt.close(fig)
    
    duration = int(1000 / fps)
    images[0].save(output_path, save_all=True, append_images=images[1:], 
                   duration=duration, loop=0)
    print(f"Created GIF with {len(images)} frames at {output_path}")

def create_refine_comparison(tensor_pred, tensor_refined, tensor_gt, output_path, fps=8):
    """
    Create a GIF from video tensors with 1x3 layout using turbo colormap for depth
    
    Args:
        tensor_pred: predicted RGB images (T, C, H, W)
        tensor_refined: refined RGB images (T, C, H, W)
        tensor_gt: ground truth RGB images (T, C, H, W)
        output_path: path to save the GIF
        fps: frames per second
    """
    images = []
    T, C, H, W = tensor_pred.shape
    
    for t in range(T):
        # Create 1x3 subplot figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 12))
        
        # Top-left: Predicted RGB
        frame_pred = tensor_pred[t].permute(1, 2, 0).cpu().numpy()
        if frame_pred.shape[2] == 1:
            frame_pred = frame_pred.squeeze(2)
        frame_pred = np.clip(frame_pred, 0, 1)
        axes[0].imshow(frame_pred)
        axes[0].set_title('Predicted RGB')
        axes[0].axis('off')

        # Top-middle: Refined RGB
        frame_refined = tensor_refined[t].permute(1, 2, 0).cpu().numpy()
        if frame_refined.shape[2] == 1:
            frame_refined = frame_refined.squeeze(2)
        frame_refined = np.clip(frame_refined, 0, 1)
        axes[1].imshow(frame_refined)
        axes[1].set_title('Refined RGB')
        axes[1].axis('off')

        # Top-right: Ground Truth RGB
        frame_gt = tensor_gt[t].permute(1, 2, 0).cpu().numpy()
        if frame_gt.shape[2] == 1:
            frame_gt = frame_gt.squeeze(2)
        frame_gt = np.clip(frame_gt, 0, 1)
        axes[2].imshow(frame_gt)
        axes[2].set_title('Ground Truth RGB')
        axes[2].axis('off')

        
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        pil_image = Image.open(buf)
        images.append(pil_image)
        plt.close(fig)
    
    duration = int(1000 / fps)
    images[0].save(output_path, save_all=True, append_images=images[1:], 
                   duration=duration, loop=0)
    print(f"Created GIF with {len(images)} frames at {output_path}")



def visualize_cameras_matplotlib(
    camera_poses,
    world_points=None,
    scale=0.1,
    max_points=100000,
    elev=20,
    azim=-60,
    save_path=None,
):
    """
    Visualize cameras (as axes) in the world coordinate frame, optionally with world points.

    Args:
        camera_poses: (N, 4, 4) camera-to-world matrices (OpenCV: camera -> world).
        world_points: (..., 3) optional point cloud in world frame.
        scale: length of axis lines drawn for each camera.
        max_points: maximum number of points to scatter for speed.
        elev, azim: view angles for 3D plot.
        save_path: if provided, save the figure to this path.
        show: whether to show the figure interactively (ignored on headless servers).
    """
    # to numpy
    if isinstance(camera_poses, torch.Tensor):
        camera_poses_np = camera_poses.detach().cpu().numpy()
    else:
        camera_poses_np = np.asarray(camera_poses)

    assert camera_poses_np.ndim == 3 and camera_poses_np.shape[1:] == (4, 4), "camera_poses must be (N,4,4)"
    N = camera_poses_np.shape[0]

    pts_np = None
    if world_points is not None:
        if isinstance(world_points, torch.Tensor):
            pts_np = world_points.detach().cpu().numpy()
        else:
            pts_np = np.asarray(world_points)
        if pts_np.ndim > 2:
            pts_np = pts_np.reshape(-1, 3)
        if pts_np.shape[-1] != 3:
            raise ValueError("world_points must have last dim = 3")
        # remove NaNs/Infs
        mask_finite = np.all(np.isfinite(pts_np), axis=-1)
        pts_np = pts_np[mask_finite]
        # subsample for speed
        if pts_np.shape[0] > max_points:
            idx = np.random.choice(pts_np.shape[0], size=max_points, replace=False)
            pts_np = pts_np[idx]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # plot points first (light gray)
    if pts_np is not None and pts_np.size > 0:
        ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], s=0.2, c='#b0b0b0', alpha=0.5, depthshade=False)

    # draw each camera as colored axes at its center
    centers = []
    for i in range(N):
        T_wc = camera_poses_np[i]
        R = T_wc[:3, :3]
        t = T_wc[:3, 3]
        centers.append(t)

        x_end = t + scale * R @ np.array([1.0, 0.0, 0.0])
        y_end = t + scale * R @ np.array([0.0, 1.0, 0.0])
        z_end = t + scale * R @ np.array([0.0, 0.0, 1.0])

        ax.plot([t[0], x_end[0]], [t[1], x_end[1]], [t[2], x_end[2]], color='r', linewidth=1)
        ax.plot([t[0], y_end[0]], [t[1], y_end[1]], [t[2], y_end[2]], color='g', linewidth=1)
        ax.plot([t[0], z_end[0]], [t[1], z_end[1]], [t[2], z_end[2]], color='b', linewidth=1)

    centers = np.array(centers)
    if centers.shape[0] > 1:
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color='#333333', linewidth=1, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title('Cameras in World Coordinates')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def visualize_camera_trajectory(extrinsics,
                                Ks=None,
                                out_dir: str = 'output',
                                basename: str = 'camera_trajectory',
                                views: Optional[List[Tuple[float, float]]] = None,
                                arrow_length: float = 0.2,
                                point_size: int = 0.01,
                                point_cloud: Optional[torch.Tensor] = None,
                                point_cloud_color: str = '#2ca02c',
                                draw_image_axes: bool = True,
                                axes_length: float = 0.15,
                                forward_color: str = 'red',
                                right_color: str = 'green',
                                up_color: str = 'blue',
                                ) -> None:
    """
    Visualize camera trajectory and orientations produced by
    `yaw_pitch_r_fov_to_extrinsics_intrinsics`.

    Assumes `extrinsics` is an iterable (or tensor) of 4x4 world->camera transforms
    (T_w2c). The function computes camera centers by inverting each transform
    and draws camera positions, trajectory lines and orientation arrows (camera
    forward axis) from the camera centers. Saves one image per view in `views`.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert extrinsics to numpy array of shape [N,4,4]
    if isinstance(extrinsics, torch.Tensor):
        extr_np = extrinsics.detach().cpu().numpy()
    else:
        extr_np = np.array([e.detach().cpu().numpy() if hasattr(e, 'detach') else np.array(e) for e in extrinsics])

    if extr_np.ndim != 3 or extr_np.shape[1:] != (4, 4):
        raise ValueError('extrinsics must be a list/tensor of 4x4 matrices')

    N = extr_np.shape[0]
    centers = []
    fronts = []

    for i in range(N):
        T_w2c = extr_np[i]
        # invert to get camera-to-world
        T_c2w = np.linalg.inv(T_w2c)
        center = T_c2w[:3, 3]
        R = T_c2w[:3, :3]
        # camera forward (optical) axis in world coords (z-axis)
        front = R[:, 2]
        centers.append(center)
        fronts.append(front)

    centers = np.stack(centers, axis=0)
    fronts = np.stack(fronts, axis=0)

    # Convert optional point cloud to numpy
    pc_np = None
    if point_cloud is not None:
        if isinstance(point_cloud, torch.Tensor):
            pc_np = point_cloud.detach().cpu().numpy()
        else:
            pc_np = np.array(point_cloud)

    # Scene bounds and scaling for consistent marker sizes
    xyz_min = centers.min(axis=0)
    xyz_max = centers.max(axis=0)
    center_scene = 0.5 * (xyz_min + xyz_max)
    extent = xyz_max - xyz_min
    max_range = float(max(extent.max(), 1e-6))

    if views is None:
        views = [(20, 30)]

    for idx, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        if pc_np is not None and pc_np.shape[0] > 0:
            ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], c=point_cloud_color, s=0.001, depthshade=False, alpha=0.6)

        # Camera centers
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='#1f77b4', s=point_size, depthshade=False)
        # Trajectory line
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], c='#ff7f0e', linewidth=1)
        # Orientation arrows (front direction)
        ax.quiver(
            centers[:, 0], centers[:, 1], centers[:, 2],
            fronts[:, 0], fronts[:, 1], fronts[:, 2],
            length=arrow_length, normalize=True, color=forward_color, linewidth=0.5
        )

        if draw_image_axes:
            # rights and ups are columns of rotation R in camera-to-world
            # we already computed 'fronts' from R[:,2]; compute rights and ups similarly
            rights = []
            ups = []
            for i in range(N):
                T_w2c = extr_np[i]
                T_c2w = np.linalg.inv(T_w2c)
                R = T_c2w[:3, :3]
                rights.append(R[:, 0])
                ups.append(R[:, 1])
            rights = np.stack(rights, axis=0)
            ups = np.stack(ups, axis=0)

            ax.quiver(
                centers[:, 0], centers[:, 1], centers[:, 2],
                rights[:, 0], rights[:, 1], rights[:, 2],
                length=axes_length, normalize=True, color=right_color, linewidth=0.5
            )

            ax.quiver(
                centers[:, 0], centers[:, 1], centers[:, 2],
                ups[:, 0], ups[:, 1], ups[:, 2],
                length=axes_length, normalize=True, color=up_color, linewidth=0.5
            )

        # Label camera centers with indices
        label_offset_scale = max_range * 0.02 + arrow_length
        for j in range(N):
            c = centers[j]
            f = fronts[j]
            off = f * label_offset_scale
            ax.text(c[0] + off[0], c[1] + off[1], c[2] + off[2], str(j), color='black', fontsize=8)

        # axis equal
        ax.set_xlim(center_scene[0] - 0.5 * max_range, center_scene[0] + 0.5 * max_range)
        ax.set_ylim(center_scene[1] - 0.5 * max_range, center_scene[1] + 0.5 * max_range)
        ax.set_zlim(center_scene[2] - 0.5 * max_range, center_scene[2] + 0.5 * max_range)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        out_path = os.path.join(out_dir, f"{basename}_view_{idx:02d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

def visualize_pointclouds_and_correspondences(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    corr_a: Optional[torch.Tensor],
    corr_b: Optional[torch.Tensor],
    out_dir: str,
    basename: str,
    point_radius: float = 0.001,
    corr_radius: float = 0.01,
    max_points: int = 50000,
    max_corr: int = 2048,
    views: Optional[List[Tuple[float, float]]] = None,
) -> None:
    """
    Visualize two 3D point clouds in the same coordinate frame and their correspondences.

    - points_a: [Na,3], colored blue
    - points_b: [Nb,3], colored green
    - corr_a, corr_b: [Nc,3], correspondence pairs (red), connected with red lines

    The visual marker sizes are derived from world-space radii (point_radius, corr_radius)
    scaled by the scene size so the result is visible in static images.
    """
    os.makedirs(out_dir, exist_ok=True)

    device = points_a.device
    def _to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().to('cpu').numpy()

    # Subsample large clouds for rendering speed
    def _subsample(x: torch.Tensor, k: int) -> torch.Tensor:
        if x is None:
            return None
        if x.shape[0] <= k:
            return x
        idx = torch.randperm(x.shape[0], device=x.device)[:k]
        return x[idx]

    pa = _subsample(points_a, max_points)
    pb = _subsample(points_b, max_points)
    ca = corr_a
    cb = corr_b

    # Compute scene bounds for axis equal and size scaling
    all_pts = [pa, pb]
    
    all_np = [
        _to_np(t) for t in all_pts if t is not None and t.shape[0] > 0
    ]
    if len(all_np) == 0:
        return
    concat = np.concatenate(all_np, axis=0)
    xyz_min = concat.min(axis=0)
    xyz_max = concat.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    extent = (xyz_max - xyz_min)
    max_range = float(max(extent.max(), 1e-6))

    if views is None:
        # views = [(20, 30), (20, 90), (20, 150), (20, 210), (20, 270), (60, 30)]
        views = [(20,30)]

    pa_np = _to_np(pa)
    pb_np = _to_np(pb)
    ca_np = _to_np(ca) if ca is not None else None
    cb_np = _to_np(cb) if cb is not None else None

    for i, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pa_np[:, 0], pa_np[:, 1], pa_np[:, 2], c='#1f77b4', s=0.001, depthshade=False, label='A') # color: blue
        ax.scatter(pb_np[:, 0], pb_np[:, 1], pb_np[:, 2], c='#2ca02c', s=0.001, depthshade=False, label='B') # color: green

        if ca_np is not None and cb_np is not None:
            ax.scatter(ca_np[:, 0], ca_np[:, 1], ca_np[:, 2], c='#d62728', s=0.01, depthshade=False)
            ax.scatter(cb_np[:, 0], cb_np[:, 1], cb_np[:, 2], c='#d62728', s=0.01, depthshade=False)
            # draw lines
            for j in range(ca_np.shape[0]):
                ax.plot(
                    [ca_np[j, 0], cb_np[j, 0]],
                    [ca_np[j, 1], cb_np[j, 1]],
                    [ca_np[j, 2], cb_np[j, 2]],
                    c='#d62728', linewidth=0.5,
                )

        # axis equal
        ax.set_xlim(center[0] - 0.5 * max_range, center[0] + 0.5 * max_range)
        ax.set_ylim(center[1] - 0.5 * max_range, center[1] + 0.5 * max_range)
        ax.set_zlim(center[2] - 0.5 * max_range, center[2] + 0.5 * max_range)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Keep legend light to avoid clutter
        # ax.legend(loc='upper right')

        out_path = os.path.join(out_dir, f"{basename}_view_{i:02d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    """
    Generate extrinsics/intrinsics for combinations of yaws and pitchs.

    - `yaws`: scalar or list of yaw angles (radians) to sweep around the vertical axis.
    - `pitchs`: scalar or list of elevation angles (radians). Each pitch will produce a full
      circle of cameras by sweeping all yaws.
    - `rs`: scalar or list (we use the first value for all combinations if list provided).
    - `fovs`: scalar or list (degrees) - converted to radians internally.

    The active (default) sweep is around the Y axis (XZ plane). Pitch is treated as
    elevation above the XZ plane: for yaw=theta and pitch=phi the camera origin is
      x = r * cos(phi) * cos(theta)
      y = r * sin(phi)
      z = r * cos(phi) * sin(theta)

    The commented examples for Z- and X-axis sweeps have been updated to use the same
    "pitch as elevation" convention (left commented for reference).
    """
    # Normalize inputs to lists
    yaws_list = yaws if isinstance(yaws, list) else [yaws]
    pitch_list = pitchs if isinstance(pitchs, list) else [pitchs]

    # rs and fovs: support scalar or list; if list provided use first element for now
    r_val = float(rs[0]) if isinstance(rs, list) else float(rs)
    fov_val = float(fovs[0]) if isinstance(fovs, list) else float(fovs)

    extrinsics = []
    intrinsics = []

    thetas = [torch.tensor(float(y)).cuda() for y in yaws_list]
    phis = [torch.tensor(float(p)).cuda() for p in pitch_list]
    fov_rad = torch.deg2rad(torch.tensor(fov_val)).cuda()

    # # Example: sweep around Z-axis (XY plane), pitch as elevation -> (x,y,z):
    # # x = r*cos(phi)*cos(theta), y = r*cos(phi)*sin(theta), z = r*sin(phi)
    # # for theta in thetas, phi in phis: orig = torch.tensor([r*torch.cos(phi)*torch.cos(theta), r*torch.cos(phi)*torch.sin(theta), r*torch.sin(phi)])

    # # Example: sweep around X-axis (YZ plane), pitch as elevation -> (x,y,z):
    # # x = r*torch.sin(phi), y = r*torch.cos(phi)*torch.cos(theta), z = r*torch.cos(phi)*torch.sin(theta)

    # Active: sweep around Y-axis (XZ plane). For each pitch, draw a full circle of yaws.
    for phi in phis:
        for theta in thetas:
            # spherical -> Cartesian with elevation phi above XZ plane
            x = r_val * torch.cos(phi) * torch.cos(theta)
            y = r_val * torch.sin(phi)
            z = r_val * torch.cos(phi) * torch.sin(theta)
            orig = torch.tensor([x, y, z]).float().cuda()
            extr = extrinsics_look_at(
                orig,
                torch.tensor([0, 0, 0]).float().cuda(),
                torch.tensor([0, 1, 0]).float().cuda(),
            )
            intr = intrinsics_from_fov_xy(fov_rad, fov_rad)
            extrinsics.append(extr)
            intrinsics.append(intr)

    return extrinsics, intrinsics

def yaw_pitch_r_fov_to_extrinsics_intrinsics_trellis(yaws, pitchs, rs, fovs):
    """
    Trellis variant that matches the iteration behavior of
    `yaw_pitch_r_fov_to_extrinsics_intrinsics`: iterate over pitches (outer)
    and yaws (inner), use a single `r` and `fov` (if scalars provided), and
    always return lists of extrinsics/intrinsics (do not collapse scalar inputs).
    The produced camera origins follow the same spherical sampling semantics as
    the non-trellis function but are mapped into the trellis Z-up axes.
    """
    # Normalize inputs to lists for iteration (do not collapse on return)
    yaws_list = yaws if isinstance(yaws, list) else [yaws]
    pitch_list = pitchs if isinstance(pitchs, list) else [pitchs]

    # rs and fovs: if lists provided we use the first element as in the old function
    r_val = float(rs[0]) if isinstance(rs, list) else float(rs)
    fov_val = float(fovs[0]) if isinstance(fovs, list) else float(fovs)

    extrinsics = []
    intrinsics = []

    thetas = [torch.tensor(float(y)).cuda() for y in yaws_list]
    phis = [torch.tensor(float(p)).cuda() for p in pitch_list]
    fov_rad = torch.deg2rad(torch.tensor(fov_val)).cuda()

    # Match old outer/inner looping: for each pitch, sweep all yaws
    for phi in phis:
        for theta in thetas:
            # spherical -> Cartesian in old Y-up convention
            x_old = r_val * torch.cos(phi) * torch.cos(theta)
            y_old = r_val * torch.sin(phi)
            z_old = r_val * torch.cos(phi) * torch.sin(theta)

            orig = torch.tensor([x_old, z_old, y_old]).float().cuda()

            extr = extrinsics_look_at(
                orig,
                torch.tensor([0, 0, 0]).float().cuda(),
                torch.tensor([0, 0, 1]).float().cuda(),
            )
            intr = intrinsics_from_fov_xy(fov_rad, fov_rad)
            extrinsics.append(extr)
            intrinsics.append(intr)

    return extrinsics, intrinsics



def visualize_object_seg(pos, obj_ids, seq_dir, frame_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pos = pos.detach().cpu().numpy()
    obj_ids = np.asarray(obj_ids).reshape(-1)

    unique_ids = np.unique(obj_ids)
    for oid in unique_ids:
        mask = obj_ids == oid
        ax.scatter(
            pos[mask, 0],
            pos[mask, 1],
            pos[mask, 2],
            s=1,
            label=f"Object {oid}"
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title(f"Object IDs for {seq_dir} frame {frame_idx}")

    ax.legend(loc='best', markerscale=5)
    plt.savefig(f"obj_ids_{frame_idx}.png", dpi=300)
    plt.close(fig)


def extrinsics_look_at(
    eye: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor
) -> torch.Tensor:
    """
    Get OpenCV extrinsics matrix looking at something

    Args:
        eye (torch.Tensor): [..., 3] the eye position
        look_at (torch.Tensor): [..., 3] the position to look at
        up (torch.Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (torch.Tensor): [..., 4, 4], extrinsics matrix
    """
    N = eye.shape[0]
    z = look_at - eye
    x = torch.cross(-up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    # x = torch.cross(y, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    R = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(R, eye[..., None])
    ret = torch.zeros((N, 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = R
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.
    return ret

def intrinsics_from_focal_center(
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix

    Args:
        focal_x (float | torch.Tensor): focal length in x axis
        focal_y (float | torch.Tensor): focal length in y axis
        cx (float | torch.Tensor): principal point in x axis
        cy (float | torch.Tensor): principal point in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    N = fx.shape[0]
    ret = torch.zeros((N, 3, 3), dtype=fx.dtype, device=fx.device)
    zeros, ones = torch.zeros(N, dtype=fx.dtype, device=fx.device), torch.ones(N, dtype=fx.dtype, device=fx.device)
    ret = torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1).unflatten(-1, (3, 3))
    return ret

def intrinsics_from_fov_xy(
        fov_x: Union[float, torch.Tensor],
        fov_y: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix from field of view in x and y axis

    Args:
        fov_x (float | torch.Tensor): field of view in x axis
        fov_y (float | torch.Tensor): field of view in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    focal_x = 0.5 / torch.tan(fov_x / 2)
    focal_y = 0.5 / torch.tan(fov_y / 2)
    cx = cy = 0.5
    return intrinsics_from_focal_center(focal_x, focal_y, cx, cy)