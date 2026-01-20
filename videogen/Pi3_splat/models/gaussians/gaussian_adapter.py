from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...utils.sh_rotation import rotate_sh
from ...utils.geometry import homogenize_points
from .gaussians import build_covariance, quaternion_to_matrix, matrix_to_quaternion

from ...utils.types import Gaussians

class GaussianAdapter(nn.Module):

    def __init__(self,sh_degree=2):
        super().__init__()

        self.sh_degree = sh_degree
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        local_means: Float[Tensor, "*#batch 3"],
        opacities: Float[Tensor, "*#batch"],
        local_raw_gaussians: Float[Tensor, "*#batch _"],
        camera_poses: Float[Tensor, "*#batch 4 4"],
        eps: float = 1e-8,
    ) -> Gaussians: # local means,opacities,scales,rotations and sh

        B, N, H, W, _ = local_means.shape

        global_means = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_means))[..., :3]

        local_scales, local_rotations, sh = local_raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)
        
        local_scales = 0.001 * F.softplus(local_scales)
        local_scales = local_scales.clamp_max(0.3)
        
        # Normalize the quaternion features to yield a valid quaternion.
        local_rotations = local_rotations / (local_rotations.norm(dim=-1, keepdim=True) + eps)

        local_covariances = build_covariance(local_scales, local_rotations) # local_rotations is optimized to xyzw
        
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        c2w_rotations = camera_poses[...,:3,:3] # B, N, 3, 3
        # local_covariances # B, N, H, W, 3, 3
        # global_covariances = c2w_rotations @ local_covariances @ c2w_rotations.transpose(-1, -2)
        
        global_covariances = torch.einsum('bnij,bnhwjk,bnlk->bnhwil', c2w_rotations, local_covariances, c2w_rotations)

        # flatten N,H,W
        global_means = global_means.reshape(B,-1,3)
        global_covariances = global_covariances.reshape(B,-1,3,3)
        sh = sh.reshape(B,-1,3,self.d_sh)
        opacities = opacities.reshape(B,-1)
        local_scales = local_scales.reshape(B,-1,3)
        local_rotations = local_rotations.reshape(B,-1,4)
        
        return Gaussians(
            means=global_means.float(),
            covariances=global_covariances.float(),
            harmonics=sh.float(),
            opacities=opacities.float(),
            scales=local_scales.float(), # computing global scales need SVD, while the rasterization process only needs cov
            rotations=local_rotations.float(),
        )
        
    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh