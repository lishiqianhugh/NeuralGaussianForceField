import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
import io
import torch
import numpy as np
from plyfile import PlyData
from PIL import Image
from torchvision import transforms
from plyfile import PlyData, PlyElement
from videogen.Pi3_splat.utils.types import Gaussians
from jaxtyping import Float
from torch import Tensor
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Optional, Tuple, Union

@torch.no_grad()
def export_pt(gaussians: Gaussians , pt_path, labels=None, feature_vectors=None, depth_conf=None):
    """
    Save gaussians to .pt without a batch dimension. Shapes on disk:
    - means:       (N, 3)
    - scales:      (N, 3) [stored in log-space]
    - rotations:   (N, 4) normalized quats, wxyz
    - opacities:   (N,)   [stored as logits]
    - harmonics:   (N, K, 3)
    - covariances: optional (N, 3, 3)
    Optional:
    - labels: (N,)
    - feature_vectors: (num_labels+1, C)
    - depth_conf: (N,)
    """
    def _to_cpu_no_batch(x):
        x = x.detach().cpu()
        return x.squeeze(0) if x.dim() >= 2 and x.shape[0] == 1 else x

    # Means (N,3)
    means = _to_cpu_no_batch(gaussians.means)

    # Scales -> store as log(scales) (N,3)
    scales = _to_cpu_no_batch(gaussians.scales)
    scales = torch.log(scales)

    # Rotations normalized (N,4)
    rotations = _to_cpu_no_batch(gaussians.rotations)
    rotations = rotations / torch.linalg.norm(rotations, dim=-1, keepdim=True).clamp_min(1e-8)

    # Opacities -> store as logits (N,)
    opacities = _to_cpu_no_batch(gaussians.opacities)
    eps = 1e-6
    opacities = opacities.clamp(min=eps, max=1 - eps)
    opacities = torch.log(opacities) - torch.log1p(-opacities)
    opacities = opacities.reshape(-1)

    # Harmonics ensure (N,K,3)
    harmonics = _to_cpu_no_batch(gaussians.harmonics)
    if harmonics.dim() == 3 and harmonics.shape[-1] != 3 and harmonics.shape[-2] == 3:
        # (N,3,K) -> (N,K,3)
        harmonics = harmonics.permute(0, 2, 1).contiguous()
    if harmonics.dim() == 2 and harmonics.shape[-1] == 3:
        # DC only -> (N,1,3)
        harmonics = harmonics.unsqueeze(1)

    # Optional covariances -> (N,3,3)
    covariances = _to_cpu_no_batch(gaussians.covariances)

    cpu_dict = {
        'means': means,
        'scales': scales,
        'rotations': rotations,
        'opacities': opacities,
        'harmonics': harmonics,
        'covariances': covariances,
    }

    if labels is not None:
        lbl = labels.detach().cpu()
        lbl = lbl.squeeze(0) if lbl.dim() >= 2 and lbl.shape[0] == 1 else lbl
        cpu_dict['labels'] = lbl.reshape(-1)
    if feature_vectors is not None:
        # (num_labels+1, C)
        fv = feature_vectors.detach().cpu()
        fv = fv.squeeze(0) if fv.dim() >= 2 and fv.shape[0] == 1 else fv
        cpu_dict['feature_vectors'] = fv
    if depth_conf is not None:
        dc = depth_conf.detach().cpu()
        dc = dc.squeeze(0) if dc.dim() >= 2 and dc.shape[0] == 1 else dc
        cpu_dict['depth_conf'] = dc.reshape(-1)

    torch.save(cpu_dict, pt_path)

    n = means.shape[0]
    print(f"Saved {n} Gaussians to {pt_path}")

@torch.no_grad()
def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"], # save in log space (viewer will exp)
    rotations: Float[Tensor, "gaussian 4"], # save in wxyz format
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"], # save in logit space (viewer will sigmoid)
    path: Path,
    save_sh_dc_only: bool = True,
):
    def construct_list_of_attributes(num_rest: int) -> list[str]:
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(3):
            attributes.append(f"f_dc_{i}")
        for i in range(num_rest):
            attributes.append(f"f_rest_{i}")
        attributes.append("opacity")
        for i in range(3):
            attributes.append(f"scale_{i}")
        for i in range(4):
            attributes.append(f"rot_{i}")
        return attributes

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()

    # Since current model use SH_degree = 4,
    # which require large memory to store, we can only save the DC band to save memory.
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0 if save_sh_dc_only else f_rest.shape[1])]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = [
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        f_dc.detach().cpu().contiguous().numpy(),
        f_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.detach().cpu().numpy(),
        rotations,
    ]
    if save_sh_dc_only:
        attributes.pop(3)

    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    print(f"Saved {means.shape[0]} Gaussians to {path}")

def load_pt_gaussian_splats(pt_path: str, device: torch.device):
    """Load Gaussian Splats stored by export_pt() with optional batch dim.

    Accepts either (N, ...) or (1, N, ...) and squeezes axis 0 if present.
    Returns: (means, quats, scales, covars, opacities, colors, sh_degree, labels, depth_conf)
    """

    data = torch.load(pt_path, map_location=device)

    def sq0(t):
        t = t.to(device).squeeze(0)
        if t.is_floating_point():
            t = t.float()
        return t

    # Core fields
    means = sq0(data['means'])                           # (N,3)
    scales = torch.exp(sq0(data['scales']))              # (N,3)
    quats = sq0(data['rotations'])                       # (N,4)
    if quats.dim() == 1:
        quats = quats.view(-1, 4)
    quats = quats / torch.norm(quats, dim=-1, keepdim=True).clamp_min(1e-8)
    opacities = torch.sigmoid(sq0(data['opacities']).reshape(-1))  # (N,)

    colors = sq0(data['harmonics'])
    if colors.dim() == 3 and colors.shape[-1] != 3 and colors.shape[-2] == 3:
        colors = colors.permute(0, 2, 1).contiguous()  # (N,K,3)
    if colors.dim() == 2 and colors.shape[-1] == 3:
        colors = colors.unsqueeze(1)                    # (N,1,3)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1) if colors.shape[1] > 1 else 0

    labels = data.get('labels', None)
    if labels is not None:
        labels = sq0(labels).reshape(-1).long()

    depth_conf = data.get('depth_conf', None)
    if depth_conf is not None:
        depth_conf = sq0(depth_conf).reshape(-1).float()

    return means, quats, scales, opacities, colors, sh_degree, labels, depth_conf

def load_ply_gaussian_splats(ply_path: str, device: torch.device):
    """Load Gaussian Splat parameters from a PLY file."""
    plydata = PlyData.read(ply_path)
    vertices = plydata["vertex"]
    
    # Extract positions
    means = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    means = torch.from_numpy(means).float().to(device)
    
    # Extract spherical harmonics (only DC component for simplicity)
    sh_dc = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1)
    sh_dc = torch.from_numpy(sh_dc).float().to(device)
    num_rest = sum(1 for key in vertices.data.dtype.names if key.startswith("f_rest_"))
    # If there are rest coefficients, extract them
    if num_rest > 0:
        # Flattened (N, num_rest) where num_rest = 3 * coeff_cnt, order: R0,G0,B0,R1,G1,B1...
        sh_rest_flat = np.stack([vertices[f"f_rest_{i}"] for i in range(num_rest)], axis=1)
        sh_rest_flat = torch.from_numpy(sh_rest_flat).float().to(device)  # (N, num_rest)
        coeff_cnt = num_rest // 3
        # Arrange to coefficient-major (N, coeff_cnt, 3)
        sh_rest = (sh_rest_flat.view(-1, 3, coeff_cnt).permute(0, 2, 1))
        colors = torch.cat([sh_dc.unsqueeze(1), sh_rest], dim=1)  # (N, coeff_cnt+1, 3)
    else:
        colors = sh_dc.unsqueeze(1)  # (N,1,3)
    
    opacities = torch.from_numpy(vertices["opacity"].copy()).float().to(device)
    opacities = torch.sigmoid(opacities)
    
    # Extract scales
    scales = np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1)
    scales = torch.from_numpy(scales).float().to(device)
    # If scales are stored as log values, convert them
    scales = torch.exp(scales)
    
    quats = np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1)
    quats = torch.from_numpy(quats).float().to(device)
    # Normalize quaternions
    quats = quats / torch.norm(quats, dim=1, keepdim=True)
    
    # Calculate SH degree
    sh_degree = int(np.sqrt(colors.shape[1]) - 1) if colors.shape[1] > 1 else 0
    
    print(f"Loaded {means.shape[0]} Gaussians from {ply_path}")
    print(f"SH degree: {sh_degree}")
    
    return means, quats, scales, opacities, colors, sh_degree

def rotate_target_dim_to_last_axis(x, target_dim=3):
    shape = x.shape
    axis_to_move = -1
    # Iterate backwards to find the first occurrence from the end 
    # (which corresponds to the last dimension of size 3 in the original order).
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] == target_dim:
            axis_to_move = i
            break

    # 2. If the axis is found and it's not already in the last position, move it.
    if axis_to_move != -1 and axis_to_move != len(shape) - 1:
        # Create the new dimension order.
        dims_order = list(range(len(shape)))
        dims_order.pop(axis_to_move)
        dims_order.append(axis_to_move)
        
        # Use permute to reorder the dimensions.
        ret = x.transpose(*dims_order)
    else:
        ret = x

    return ret

def write_ply(
    xyz,
    rgb=None,
    path='output.ply',
) -> None:
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()

    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()

    if rgb is not None and rgb.max() > 1:
        rgb = rgb / 255.

    xyz = rotate_target_dim_to_last_axis(xyz, 3)
    xyz = xyz.reshape(-1, 3)

    if rgb is not None:
        rgb = rotate_target_dim_to_last_axis(rgb, 3)
        rgb = rgb.reshape(-1, 3)
    
    if rgb is None:
        min_coord = np.min(xyz, axis=0)
        max_coord = np.max(xyz, axis=0)
        normalized_coord = (xyz - min_coord) / (max_coord - min_coord + 1e-8)
        
        hue = 0.7 * normalized_coord[:,0] + 0.2 * normalized_coord[:,1] + 0.1 * normalized_coord[:,2]
        hsv = np.stack([hue, 0.9*np.ones_like(hue), 0.8*np.ones_like(hue)], axis=1)

        c = hsv[:,2:] * hsv[:,1:2]
        x = c * (1 - np.abs( (hsv[:,0:1]*6) % 2 - 1 ))
        m = hsv[:,2:] - c
        
        rgb = np.zeros_like(hsv)
        cond = (0 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 1)
        rgb[cond] = np.hstack([c[cond], x[cond], np.zeros_like(x[cond])])
        cond = (1 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 2)
        rgb[cond] = np.hstack([x[cond], c[cond], np.zeros_like(x[cond])])
        cond = (2 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 3)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), c[cond], x[cond]])
        cond = (3 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 4)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), x[cond], c[cond]])
        cond = (4 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 5)
        rgb[cond] = np.hstack([x[cond], np.zeros_like(x[cond]), c[cond]])
        cond = (5 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 6)
        rgb[cond] = np.hstack([c[cond], np.zeros_like(x[cond]), x[cond]])
        rgb = (rgb + m)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb * 255), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def export_cam_info(
    extrinsics: torch.Tensor,
    Ks: torch.Tensor,
    save_path: Union[str, Path],
    image_names: Optional[List[str]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
):
    """
    Export camera information to a .pt file.

    Output format:
        {
            'intrinsics': torch.Tensor [N, 3, 3],
            'extrinsics': torch.Tensor [N, 4, 4],  # world-to-camera
            'image_names': List[str],              # optional
            'image_width': int,                    # optional
            'image_height': int                    # optional
        }

    Notes:
    - Intrinsics matrix (3x3):
        [fx,  0, cx]
        [ 0, fy, cy]
        [ 0,  0,  1]
    - Extrinsics matrix (4x4, world-to-camera):
        [R11, R12, R13, tx]
        [R21, R22, R23, ty]
        [R31, R32, R33, tz]
        [ 0,   0,   0,  1]
    """

    # Ensure tensors are on CPU and with expected shapes
    if extrinsics.dim() == 4:
        extrinsics = extrinsics[0]
    if Ks.dim() == 4:
        Ks = Ks[0]

    extrinsics = extrinsics.detach().cpu().float()  # [N, 4, 4]
    Ks = Ks.detach().cpu().float()                    # [N, 3, 3]
    
    # Convert cam-to-world to world-to-camera

    data: Dict[str, Union[torch.Tensor, List[str], int]] = {
        'intrinsics': Ks,
        'extrinsics': extrinsics,
    }

    if image_names is not None and len(image_names) > 0:
        data['image_names'] = image_names
    if image_width is not None:
        data['image_width'] = int(image_width)
    if image_height is not None:
        data['image_height'] = int(image_height)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, save_path)
    print(f"Saved camera info for {Ks.shape[0]} views to {save_path}")
