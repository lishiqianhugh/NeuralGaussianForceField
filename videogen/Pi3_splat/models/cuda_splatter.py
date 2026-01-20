from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float, Bool
from torch import Tensor
import torchvision

from ..utils.types import Gaussians
from math import sqrt 
from gsplat import rasterization

@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]
    masks: Bool[Tensor, "batch view height width"] | None
    depth: Float[Tensor, "batch view height width"] | None
    alpha: Float[Tensor, "batch view height width"] | None

class DecoderSplattingCUDA(nn.Module):    
    def __init__(
        self,background_color=[1.0,1.0,1.0]
    ) -> None:
        super().__init__()
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )

    def rendering_fn(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        masks: Bool[Tensor, "batch view H W"],
        image_shape: tuple[int, int],
    ) -> DecoderOutput:
        B, V, _, _  = intrinsics.shape
        H, W = image_shape
        rendered_imgs, rendered_depths, rendered_alphas = [], [], []
        xyzs, opacitys, rotations, scales, features = gaussians.means, gaussians.opacities, gaussians.rotations, gaussians.scales, gaussians.harmonics.permute(0, 1, 3, 2).contiguous()
        covariances = gaussians.covariances
        for i in range(B):
            xyz_i = xyzs[i].float()
            feature_i = features[i].float()
            covar_i = covariances[i].float()
            scale_i = scales[i].float()
            rotation_i = rotations[i].float()
            opacity_i = opacitys[i].squeeze().float()

            mask_i = masks[i].flatten()

            if mask_i.sum() > 0:
                xyz_i = xyz_i[mask_i]
                feature_i = feature_i[mask_i]
                covar_i = covar_i[mask_i]
                scale_i = scale_i[mask_i]
                rotation_i = rotation_i[mask_i]
                opacity_i = opacity_i[mask_i]

            test_w2c_i = extrinsics[i].float() # (V, 4, 4)
            test_intr_i = intrinsics[i].float() # (V, 3, 3)

            # test_intr_i_normalized = intrinsics[i].float()
            # # Denormalize the intrinsics into standred format
            # test_intr_i = test_intr_i_normalized.clone()
            # test_intr_i[:, 0] = test_intr_i_normalized[:, 0] * W
            # test_intr_i[:, 1] = test_intr_i_normalized[:, 1] * H
            sh_degree = (int(sqrt(feature_i.shape[-2])) - 1)

            rendering_list = []
            rendering_depth_list = []
            rendering_alpha_list = []
            for j in range(V):
                rendering, alpha, _ = rasterization(xyz_i, rotation_i, scale_i, opacity_i, feature_i,
                                                test_w2c_i[j:j+1], test_intr_i[j:j+1], W, H, sh_degree=sh_degree, 
                                                render_mode="RGB+D", packed=False,
                                                near_plane=1e-10,
                                                backgrounds=self.background_color.unsqueeze(0).repeat(1, 1),
                                                radius_clip=0.1,
                                                covars=covar_i,
                                                rasterize_mode='classic') # (V, H, W, 3) 
                rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1) # [V,H,W,3] [V,H,W,1]
                rendering_img = rendering_img.clamp(0.0, 1.0)
                rendering_list.append(rendering_img.permute(0, 3, 1, 2))
                rendering_depth_list.append(rendering_depth)
                rendering_alpha_list.append(alpha)
            rendered_depths.append(torch.cat(rendering_depth_list, dim=0).squeeze())
            rendered_imgs.append(torch.cat(rendering_list, dim=0))
            rendered_alphas.append(torch.cat(rendering_alpha_list, dim=0).squeeze())
        return DecoderOutput(torch.stack(rendered_imgs),masks,torch.stack(rendered_depths), torch.stack(rendered_alphas))

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        masks: Bool[Tensor, "batch view H W"],
        image_shape: tuple[int, int],
    ) -> DecoderOutput:
        
        return self.rendering_fn(gaussians, extrinsics, intrinsics,masks, image_shape)

