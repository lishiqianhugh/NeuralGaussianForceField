from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor
import torch
from ..models.gaussians.gaussians import matrix_to_quaternion

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]
    # levels: Float[Tensor, "batch gaussian"]

    def clone(self):
        """Create a deep copy of the Gaussians object with cloned tensors."""
        return Gaussians(
            means=self.means.clone(),
            covariances=self.covariances.clone(),
            harmonics=self.harmonics.clone(),
            opacities=self.opacities.clone(),
            scales=self.scales.clone(),
            rotations=self.rotations.clone()
        )

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.means.shape[0]

    def __getitem__(self, idx):
        """
        Slice by the batch dimension and return a new Gaussians.

        - If `idx` is an int, the batch dimension is preserved (size 1),
          so downstream code expecting a batch dimension keeps working.
        - If `idx` is a slice / list / tensor mask, it's applied to the batch dim.
        """
        # Normalize integer index to a slice to preserve the batch dimension
        # if isinstance(idx, int):
        #     if idx < 0:
        #         idx = len(self) + idx
        #     idx = slice(idx, idx + 1)

        return Gaussians(
            means=self.means[idx],
            covariances=self.covariances[idx],
            harmonics=self.harmonics[idx],
            opacities=self.opacities[idx],
            scales=self.scales[idx],
            rotations=self.rotations[idx],
        )

    @torch.no_grad()
    def calc_world_scale_and_rot(self):
        """
        Recalculate scales and rotations according to covariances as sometimes Gaussians's scales and rotations are in local coords

        支持两种协方差张量形状:
        - [B, N, 3, 3]: 带 batch 维度
        - [N, 3, 3]: 无 batch 维度
        """
        max_mini_batch = 100_0000

        device = self.covariances.device
        dtype = self.covariances.dtype

        # 分支一: 无 batch 维度 [N, 3, 3]
        if self.covariances.ndim == 3:
            N = self.covariances.shape[0]

            out_scales = torch.empty((N, 3), device=device, dtype=dtype)
            out_rots = torch.empty((N, 4), device=device, dtype=dtype)

            start = 0
            while start < N:
                end = min(start + max_mini_batch, N)
                cov_chunk = self.covariances[start:end]  # [M, 3, 3]

                # 特征分解 (对称实矩阵)
                evals, evecs = torch.linalg.eigh(cov_chunk)  # evals:[M,3], evecs:[M,3,3]
                evals = torch.clamp(evals, min=1e-8)

                # 按特征值从大到小排序，使第一轴为最大尺度
                order = torch.argsort(evals, dim=-1, descending=True)  # [M,3]
                evals_sorted = torch.gather(evals, 1, order)  # [M,3]
                order_col = order.unsqueeze(1).expand(-1, 3, -1)  # [M,3,3]
                evecs_sorted = torch.gather(evecs, 2, order_col)  # [M,3,3]

                # 保证右手坐标系: 若 det(R) < 0 则翻转第三列
                detR = torch.det(evecs_sorted)  # [M]
                neg = detR < 0
                if neg.any():
                    evecs_sorted[neg, :, 2] = -evecs_sorted[neg, :, 2]

                # 尺度 = 协方差的特征值开方
                scales_chunk = torch.sqrt(evals_sorted)  # [M,3]
                # 旋转 = 主轴矩阵 -> 四元数 (wxyz)
                quats_chunk = matrix_to_quaternion(evecs_sorted)  # [M,4]

                out_scales[start:end] = scales_chunk
                out_rots[start:end] = quats_chunk
                start = end

            # 写回到对象属性
            self.scales = out_scales
            self.rotations = out_rots

            return

        # 分支二: 带 batch 维度 [B, N, 3, 3]
        if self.covariances.ndim == 4:
            B, N = self.covariances.shape[0], self.covariances.shape[1]

            out_scales = torch.empty((B, N, 3), device=device, dtype=dtype)
            out_rots = torch.empty((B, N, 4), device=device, dtype=dtype)

            for b in range(B):
                cov_b = self.covariances[b]  # [N, 3, 3]
                start = 0
                while start < N:
                    end = min(start + max_mini_batch, N)
                    cov_chunk = cov_b[start:end]  # [M, 3, 3]

                    # 特征分解 (对称实矩阵)
                    evals, evecs = torch.linalg.eigh(cov_chunk)  # evals:[M,3], evecs:[M,3,3]
                    evals = torch.clamp(evals, min=1e-8)

                    # 按特征值从大到小排序，使第一轴为最大尺度
                    order = torch.argsort(evals, dim=-1, descending=True)  # [M,3]
                    evals_sorted = torch.gather(evals, 1, order)  # [M,3]
                    order_col = order.unsqueeze(1).expand(-1, 3, -1)  # [M,3,3]
                    evecs_sorted = torch.gather(evecs, 2, order_col)  # [M,3,3]

                    # 保证右手坐标系: 若 det(R) < 0 则翻转第三列
                    detR = torch.det(evecs_sorted)  # [M]
                    neg = detR < 0
                    if neg.any():
                        evecs_sorted[neg, :, 2] = -evecs_sorted[neg, :, 2]

                    # 尺度 = 协方差的特征值开方
                    scales_chunk = torch.sqrt(evals_sorted)  # [M,3]
                    # 旋转 = 主轴矩阵 -> 四元数 (wxyz)
                    quats_chunk = matrix_to_quaternion(evecs_sorted)  # [M,4]

                    out_scales[b, start:end] = scales_chunk
                    out_rots[b, start:end] = quats_chunk
                    start = end

            # 写回到对象属性
            self.scales = out_scales
            self.rotations = out_rots
            return

