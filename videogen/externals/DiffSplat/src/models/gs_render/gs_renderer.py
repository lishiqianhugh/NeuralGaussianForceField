from typing import *
from torch import Tensor

import torch
from einops import rearrange, repeat

from src.models.gs_render.deferred_bp import deferred_bp
from src.models.gs_render.gs_util import GaussianModel, render
from src.options import Options
from src.utils import unproject_depth


class GaussianRenderer:
    def __init__(self, opt: Options):
        self.opt = opt

        self.scale_activation = lambda x: \
            self.opt.scale_min * x + self.opt.scale_max * (1. - x)  # [0, 1] -> [s_min, s_max]

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def render(self,
        model_outputs: Dict[str, Tensor],
        input_C2W: Tensor, input_fxfycxcy: Tensor,
        C2W: Tensor, fxfycxcy: Tensor,
        height: Optional[float] = None,
        width: Optional[float] = None,
        bg_color: Tuple[float, float, float] = (1., 1., 1.),
        scaling_modifier: float = 1.,
        opacity_threshold: float = 0.,
        input_normalized: bool = True,
        in_image_format: bool = True,
    ):
        if not in_image_format:
            assert height is not None and width is not None
            assert "xyz" in model_outputs  # depth must be in image format

        rgb, scale, rotation, opacity = model_outputs["rgb"], model_outputs["scale"], model_outputs["rotation"], model_outputs["opacity"]
        depth = model_outputs.get("depth", None)
        xyz = model_outputs.get("xyz", None)
        # Only one of `depth` and `xyz` should be None
        assert (depth is not None or xyz is not None) and not (depth is not None and xyz is not None)

        # Rendering resolution could be different from input resolution
        H = height if height is not None else rgb.shape[-2]
        W = width if width is not None else rgb.shape[-1]

        # Reshape for rendering
        if in_image_format:
            rgb = rearrange(rgb, "b v c h w -> b (v h w) c")
            scale = rearrange(scale, "b v c h w -> b (v h w) c")
            rotation = rearrange(rotation, "b v c h w -> b (v h w) c")
            opacity = rearrange(opacity, "b v c h w -> b (v h w) c")

        # Prepare XYZ for rendering
        if xyz is None:
            if input_normalized:
                depth = depth + torch.norm(input_C2W[:, :, :3, 3], p=2, dim=2, keepdim=True)[..., None, None]  # [-1, 1] -> image plane + [-1, 1]
            xyz = unproject_depth(depth.squeeze(2), input_C2W, input_fxfycxcy)  # [-1, 1]
        xyz = xyz + model_outputs.get("offset", torch.zeros_like(xyz))
        if in_image_format:
            xyz = rearrange(xyz, "b v c h w -> b (v h w) c")

        # From [-1, 1] to valid values
        if input_normalized:
            rgb = rgb * 0.5 + 0.5  # [-1, 1] -> [0, 1]
            scale = self.scale_activation(scale * 0.5 + 0.5)  # [-1, 1] -> [0, 1] -> [s_min, s_max]
            rotation = rotation  # not changed; already L2 normalized
            opacity = opacity * 0.5 + 0.5  # [-1, 1] -> [0, 1]

        # Filter by opacity
        opacity = (opacity > opacity_threshold) * opacity

        (B, V), device = C2W.shape[:2], C2W.device  # `HR`/`WR` meight be different from `H`/`W`
        images = torch.zeros(B, V, 3, H, W, dtype=torch.float32, device=device)
        alphas = torch.zeros(B, V, 1, H, W, dtype=torch.float32, device=device)
        depths = torch.zeros(B, V, 1, H, W, dtype=torch.float32, device=device)
        normals = torch.zeros(B, V, 3, H, W, dtype=torch.float32, device=device)

        pcs = []
        for i in range(B):
            pcs.append(GaussianModel().set_data(xyz[i], rgb[i], scale[i], rotation[i], opacity[i]))

        if self.opt.render_type == "defered":
            images, alphas, depths, normals = deferred_bp(
                xyz, rgb, scale, rotation, opacity,
                H, W, C2W, fxfycxcy,
                self.opt.deferred_bp_patch_size, GaussianModel(),
                self.opt.znear, self.opt.zfar,
                bg_color,
                scaling_modifier,
                self.opt.coord_weight > 0. or self.opt.normal_weight > 0. or \
                    self.opt.vis_coords or self.opt.vis_normals,  # whether render depth & normal
            )
        else:  # default
            for i in range(B):
                pc = pcs[i]
                for j in range(V):
                    render_results = render(
                        pc, H, W, C2W[i, j], fxfycxcy[i, j],
                        self.opt.znear, self.opt.zfar,
                        bg_color,
                        scaling_modifier,
                        self.opt.coord_weight > 0. or self.opt.normal_weight > 0. or \
                            self.opt.vis_coords or self.opt.vis_normals,  # whether render depth & normal
                    )
                    images[i, j] = render_results["image"]
                    alphas[i, j] = render_results["alpha"]
                    depths[i, j] = render_results["depth"]
                    normals[i, j] = render_results["normal"]

        if not isinstance(bg_color, Tensor):
            bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=device)
        bg_color = repeat(bg_color, "c -> b v c h w", b=B, v=V, h=H, w=W)

        coords = (unproject_depth(depths.squeeze(2), C2W, fxfycxcy)
            * 0.5 + 0.5) * alphas + (1. - alphas) * bg_color
        normals_ = (torch.einsum("bvrc,bvchw->bvrhw", C2W[:, :, :3, :3], normals)
            * 0.5 + 0.5) * alphas + (1. - alphas) * bg_color

        return {
            "image": images,
            "alpha": alphas,
            "coord": coords,
            "normal": normals_,
            "raw_depth": depths,
            "raw_normal": normals,
            "pc": pcs,
        }
