from typing import *
from torch import Tensor

import os
import numpy as np
from plyfile import PlyData, PlyElement
import torch
import kiui

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

class Camera:
    def __init__(self,
        C2W: Tensor, fxfycxcy: Tensor, h: int, w: int,
        znear: float = 0.01, zfar: float = 100.,
    ):
        self.fxfycxcy = fxfycxcy.clone().float()
        self.C2W = C2W.clone().float()
        self.W2C = self.C2W.inverse()

        self.znear = znear
        self.zfar = zfar
        self.h = h
        self.w = w

        fx, fy, cx, cy = self.fxfycxcy[0], self.fxfycxcy[1], self.fxfycxcy[2], self.fxfycxcy[3]
        self.tanfovX = 1 / (2 * fx)  # `tanHalfFovX` actually
        self.tanfovY = 1 / (2 * fy)  # `tanHalfFovY` actually
        self.fovX = 2 * torch.atan(self.tanfovX)
        self.fovY = 2 * torch.atan(self.tanfovY)
        self.shiftX = 2 * cx - 1
        self.shiftY = 2 * cy - 1

        def getProjectionMatrix(znear, zfar, fovX, fovY, shiftX, shiftY):
            tanHalfFovY = torch.tan((fovY / 2))
            tanHalfFovX = torch.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = torch.zeros(4, 4, device=fovX.device)

            z_sign = 1

            P[0, 0] = 2 * znear / (right - left)
            P[1, 1] = 2 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left) + shiftX
            P[1, 2] = (top + bottom) / (top - bottom) + shiftY
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        self.world_view_transform = self.W2C.transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.fovX, self.fovY, self.shiftX, self.shiftY).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.C2W[:3, 3]


class GaussianModel:
    def __init__(self):
        self.xyz = None
        self.rgb = None
        self.scale = None
        self.rotation = None
        self.opacity = None

        self.sh_degree = 0

    def set_data(self, xyz: Tensor, rgb: Tensor, scale: Tensor, rotation: Tensor, opacity: Tensor):
        self.xyz = xyz
        self.rgb = rgb
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        return self

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "GaussianModel":
        self.xyz = self.xyz.to(device, dtype)
        self.rgb = self.rgb.to(device, dtype)
        self.scale = self.scale.to(device, dtype)
        self.rotation = self.rotation.to(device, dtype)
        self.opacity = self.opacity.to(device, dtype)
        return self

    def save_ply(self, path: str, opacity_threshold: float = 0., compatible: bool = True):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.xyz.detach().cpu().numpy()
        f_dc = self.rgb.detach().cpu().numpy()
        rgb = (f_dc * 255.).clip(0., 255.).astype(np.uint8)
        opacity = self.opacity.detach().cpu().numpy()
        scale = self.scale.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        # Filter out points with low opacity
        mask = (opacity > opacity_threshold).squeeze()
        xyz = xyz[mask]
        f_dc = f_dc[mask]
        opacity = opacity[mask]
        scale = scale[mask]
        rotation = rotation[mask]
        rgb = rgb[mask]

        # Invert activation to make it compatible with the original ply format
        if compatible:
            def inverse_sigmoid(x):
                return torch.log(x/(1-x))
            # opacity = kiui.op.inverse_sigmoid(torch.from_numpy(opacity)).numpy()
            opacity = inverse_sigmoid(torch.from_numpy(opacity)).numpy()
            scale = torch.log(torch.from_numpy(scale) + 1e-8).numpy()
            f_dc = (torch.from_numpy(f_dc) - 0.5).numpy() / 0.28209479177387814

        dtype_full = [(attribute, "f4") for attribute in self._construct_list_of_attributes()]
        dtype_full.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, opacity, scale, rotation, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path: str, compatible: bool = True):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
        f_dc = np.stack((
            np.asarray(plydata.elements[0]["f_dc_0"]),
            np.asarray(plydata.elements[0]["f_dc_1"]),
            np.asarray(plydata.elements[0]["f_dc_2"]),
        ), axis=1)
        opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        scale = np.stack((
            np.asarray(plydata.elements[0]["scale_0"]),
            np.asarray(plydata.elements[0]["scale_1"]),
            np.asarray(plydata.elements[0]["scale_2"]),
        ), axis=1)
        rotation = np.stack((
            np.asarray(plydata.elements[0]["rot_0"]),
            np.asarray(plydata.elements[0]["rot_1"]),
            np.asarray(plydata.elements[0]["rot_2"]),
            np.asarray(plydata.elements[0]["rot_3"]),
        ), axis=1)

        self.xyz = torch.from_numpy(xyz).float()
        self.rgb = torch.from_numpy(f_dc).float()
        self.opacity = torch.from_numpy(opacity).float()
        self.scale = torch.from_numpy(scale).float()
        self.rotation = torch.from_numpy(rotation).float()

        if compatible:
            self.opacity = torch.sigmoid(self.opacity)
            self.scale = torch.exp(self.scale)
            self.rgb = 0.28209479177387814 * self.rgb + 0.5

    def _construct_list_of_attributes(self):
        l = ["x", "y", "z"]
        for i in range(self.rgb.shape[1]):
            l.append(f"f_dc_{i}")
        l.append("opacity")
        for i in range(self.scale.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self.rotation.shape[1]):
            l.append(f"rot_{i}")
        return l


def render(
    pc: GaussianModel,
    height: int,
    width: int,
    C2W: Tensor,
    fxfycxcy: Tensor,
    znear: float = 0.01,
    zfar: float = 100.,
    bg_color: Union[Tensor, Tuple[float, float, float]] = (1., 1., 1.),
    scaling_modifier: float = 1.,
    render_dn: bool = False,
):
    viewpoint_camera = Camera(C2W, fxfycxcy, height, width, znear, zfar)

    if not isinstance(bg_color, Tensor):
        bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=C2W.device)
    else:
        bg_color = bg_color.to(C2W.device, dtype=torch.float32)

    pc = pc.to(dtype=torch.float32)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.h),
        image_width=int(viewpoint_camera.w),
        tanfovx=viewpoint_camera.tanfovX,
        tanfovy=viewpoint_camera.tanfovY,
        kernel_size=0.,  # cf. Mip-Splatting; not used
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # cf. RaDe-GS
        require_depth=render_dn,
        require_coord=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    image, _, _, _, depth, _, alpha, normal = rasterizer(  # not used: radii, coord, mcoord, mdepth
        means3D=pc.xyz,
        means2D=torch.zeros_like(pc.xyz, dtype=torch.float32, device=pc.xyz.device),
        shs=None,
        colors_precomp=pc.rgb,
        opacities=pc.opacity,
        scales=pc.scale,
        rotations=pc.rotation,
        cov3D_precomp=None,
    )

    return {
        "image": image,
        "alpha": alpha,
        "depth": depth,
        "normal": normal,
    }
