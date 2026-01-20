from typing import *
from torch import Tensor

import torch
import torch.nn.functional as tF


def normalize_normals(normals: Tensor, C2W: Tensor, i: int = 0) -> Tensor:
    """Normalize a batch of multi-view `normals` by the `i`-th view.

    Inputs:
        - `normals`: (B, V, 3, H, W)
        - `C2W`: (B, V, 4, 4)
        - `i`: the index of the view to normalize by

    Outputs:
        - `normalized_normals`: (B, V, 3, H, W)
    """
    _, _, R, C = C2W.shape  # (B, V, 4, 4)
    assert R == C == 4
    _, _, CC, _, _ = normals.shape  # (B, V, 3, H, W)
    assert CC == 3

    dtype = normals.dtype
    normals = normals.clone().float()
    transform = torch.inverse(C2W[:, i, :3, :3])  # (B, 3, 3)

    return torch.einsum("brc,bvchw->bvrhw", transform, normals).to(dtype)  # (B, V, 3, H, W)


def normalize_C2W(C2W: Tensor, i: int = 0, norm_radius: float = 0.) -> Tensor:
    """Normalize a batch of multi-view `C2W` by the `i`-th view.

    Inputs:
        - `C2W`: (B, V, 4, 4)
        - `i`: the index of the view to normalize by
        - `norm_radius`: the normalization radius

    Outputs:
        - `normalized_C2W`: (B, V, 4, 4)
    """
    _, _, R, C = C2W.shape  # (B, V, 4, 4)
    assert R == C == 4

    device, dtype = C2W.device, C2W.dtype
    C2W = C2W.clone().float()

    if abs(norm_radius) > 0.:
        radius = torch.norm(C2W[:, i, :3, 3], dim=1)  # (B,)
        C2W[:, :, :3, 3] *= (norm_radius / radius.unsqueeze(1).unsqueeze(2))

    # The `i`-th view is normalized to a canonical matrix as the reference view
    transform = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, norm_radius],
        [0, 0, 0, 1]  # canonical c2w in OpenGL world convention
    ], dtype=torch.float32, device=device) @ torch.inverse(C2W[:, i, ...])  # (B, 4, 4)

    return (transform.unsqueeze(1) @ C2W).to(dtype)  # (B, V, 4, 4)


def unproject_depth(depth_map: Tensor, C2W: Tensor, fxfycxcy: Tensor) -> Tensor:
    """Unproject depth map to 3D world coordinate.

    Inputs:
        - `depth_map`: (B, V, H, W)
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `xyz_world`: (B, V, 3, H, W)
    """
    device, dtype = depth_map.device, depth_map.dtype
    B, V, H, W = depth_map.shape

    depth_map = depth_map.reshape(B*V, H, W).float()
    C2W = C2W.reshape(B*V, 4, 4).float()
    fxfycxcy = fxfycxcy.reshape(B*V, 4).float()
    K = torch.zeros(B*V, 3, 3, dtype=torch.float32, device=device)
    K[:, 0, 0] = fxfycxcy[:, 0]
    K[:, 1, 1] = fxfycxcy[:, 1]
    K[:, 0, 2] = fxfycxcy[:, 2]
    K[:, 1, 2] = fxfycxcy[:, 3]
    K[:, 2, 2] = 1

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")  # OpenCV/COLMAP camera convention
    y = y.to(device).unsqueeze(0).repeat(B*V, 1, 1) / (H-1)
    x = x.to(device).unsqueeze(0).repeat(B*V, 1, 1) / (W-1)
    # NOTE: To align with `plucker_ray(bug=False)`, should be:
    # y = (y.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / H
    # x = (x.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / W
    xyz_map = torch.stack([x, y, torch.ones_like(x)], axis=-1) * depth_map[..., None]
    xyz = xyz_map.view(B*V, -1, 3)

    # Get point positions in camera coordinate
    xyz = torch.matmul(xyz, torch.transpose(torch.inverse(K), 1, 2))
    xyz_map = xyz.view(B*V, H, W, 3)

    # Transform pts from camera to world coordinate
    xyz_homo = torch.ones((B*V, H, W, 4), device=device)
    xyz_homo[..., :3] = xyz_map
    xyz_world = torch.bmm(C2W, xyz_homo.reshape(B*V, -1, 4).permute(0, 2, 1))[:, :3, ...].to(dtype)  # (B*V, 3, H*W)
    xyz_world = xyz_world.reshape(B, V, 3, H, W)
    return xyz_world


def plucker_ray(h: int, w: int, C2W: Tensor, fxfycxcy: Tensor, bug: bool = True) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """Get Plucker ray embeddings.

    Inputs:
        - `h`: image height
        - `w`: image width
        - `C2W`: (B, V, 4, 4)
        - `fxfycxcy`: (B, V, 4)

    Outputs:
        - `plucker`: (B, V, 6, `h`, `w`)
        - `ray_o`: (B, V, 3, `h`, `w`)
        - `ray_d`: (B, V, 3, `h`, `w`)
    """
    device, dtype = C2W.device, C2W.dtype
    B, V = C2W.shape[:2]

    C2W = C2W.reshape(B*V, 4, 4).float()
    fxfycxcy = fxfycxcy.reshape(B*V, 4).float()

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")  # OpenCV/COLMAP camera convention
    y, x = y.to(device), x.to(device)
    if bug:  # BUG !!! same here: https://github.com/camenduru/GRM/blob/master/model/visual_encoder/vit_gs.py#L85
        y = y[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) / (h - 1)
        x = x[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) / (w - 1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    else:
        y = (y[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) + 0.5) / h
        x = (x[None, :, :].expand(B*V, -1, -1).reshape(B*V, -1) + 0.5) / w
        x = (x - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # (B*V, h*w, 3)
    ray_d = torch.bmm(ray_d, C2W[:, :3, :3].transpose(1, 2))  # (B*V, h*w, 3)
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # (B*V, h*w, 3)
    ray_o = C2W[:, :3, 3][:, None, :].expand_as(ray_d)  # (B*V, h*w, 3)

    ray_o = ray_o.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)
    ray_d = ray_d.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)
    plucker = torch.cat([torch.cross(ray_o, ray_d, dim=2).to(dtype), ray_d.to(dtype)], dim=2)

    return plucker, (ray_o, ray_d)


def orbit_camera(
    elevs: Tensor, azims: Tensor, radius: Optional[Tensor] = None,
    is_degree: bool = True,
    target: Optional[Tensor] = None,
    opengl: bool=True,
) -> Tensor:
    """Construct a camera pose matrix orbiting a target with elevation & azimuth angle.

    Inputs:
        - `elevs`: (B,); elevation in (-90, 90), from +y to -y is (-90, 90)
        - `azims`: (B,); azimuth in (-180, 180), from +z to +x is (0, 90)
        - `radius`: (B,); camera radius; if None, default to 1.
        - `is_degree`: bool; whether the input angles are in degree
        - `target`: (B, 3); look-at target position
        - `opengl`: bool; whether to use OpenGL convention

    Outputs:
        - `C2W`: (B, 4, 4); camera pose matrix
    """
    device, dtype = elevs.device, elevs.dtype

    if radius is None:
        radius = torch.ones_like(elevs)
    assert elevs.shape == azims.shape == radius.shape
    if target is None:
        target = torch.zeros(elevs.shape[0], 3, device=device, dtype=dtype)

    if is_degree:
        elevs = torch.deg2rad(elevs)
        azims = torch.deg2rad(azims)

    x = radius * torch.cos(elevs) * torch.sin(azims)
    y = - radius * torch.sin(elevs)
    z = radius * torch.cos(elevs) * torch.cos(azims)

    camposes = torch.stack([x, y, z], dim=1) + target  # (B, 3)
    R = look_at(camposes, target, opengl=opengl)  # (B, 3, 3)
    C2W = torch.cat([R, camposes[:, :, None]], dim=2)  # (B, 3, 4)
    C2W = torch.cat([C2W, torch.zeros_like(C2W[:, :1, :])], dim=1)  # (B, 4, 4)
    C2W[:, 3, 3] = 1.
    return C2W


def look_at(camposes: Tensor, targets: Tensor, opengl: bool = True) -> Tensor:
    """Construct batched pose rotation matrices by look-at.

    Inputs:
        - `camposes`: (B, 3); camera positions
        - `targets`: (B, 3); look-at targets
        - `opengl`: whether to use OpenGL convention

    Outputs:
        - `R`: (B, 3, 3); normalized camera pose rotation matrices
    """
    device, dtype = camposes.device, camposes.dtype

    if not opengl:  # OpenCV convention
        # forward is camera -> target
        forward_vectors = tF.normalize(targets - camposes, dim=-1)
        up_vectors = torch.tensor([0., 1., 0.], device=device, dtype=dtype)[None, :].expand_as(forward_vectors)
        right_vectors = tF.normalize(torch.cross(forward_vectors, up_vectors), dim=-1)
        up_vectors = tF.normalize(torch.cross(right_vectors, forward_vectors), dim=-1)
    else:
        # forward is target -> camera
        forward_vectors = tF.normalize(camposes - targets, dim=-1)
        up_vectors = torch.tensor([0., 1., 0.], device=device, dtype=dtype)[None, :].expand_as(forward_vectors)
        right_vectors = tF.normalize(torch.cross(up_vectors, forward_vectors), dim=-1)
        up_vectors = tF.normalize(torch.cross(forward_vectors, right_vectors), dim=-1)

    R = torch.stack([right_vectors, up_vectors, forward_vectors], dim=-1)
    return R
