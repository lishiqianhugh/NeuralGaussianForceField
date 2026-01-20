import numpy as np
import torch
import torch.nn.functional as F

def se3_inverse(T):
    """
    Computes the inverse of a batch of SE(3) matrices.
    T: Tensor of shape (..., 4, 4) - can handle arbitrary batch dimensions
    """
    # Store original shape for later restoration
    original_shape = T.shape
    
    # Handle single matrix case
    if len(T.shape) == 2:
        T = T[None]
        unseq_flag = True
    else:
        unseq_flag = False
    
    # Flatten all batch dimensions except the last two (4x4 matrix)
    batch_shape = T.shape[:-2]
    batch_size = np.prod(batch_shape) if batch_shape else 1
    T_flat = T.reshape(batch_size, 4, 4)

    if torch.is_tensor(T):
        R = T_flat[..., :3, :3]
        t = T_flat[..., :3, 3].unsqueeze(-1)
        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)
        
        # Create bottom row [0, 0, 0, 1] for each matrix
        bottom_row = torch.zeros(batch_size, 1, 4, device=T.device, dtype=T.dtype)
        bottom_row[..., 3] = 1
        
        T_inv_flat = torch.cat([
            torch.cat([R_inv, t_inv], dim=-1),
            bottom_row
        ], dim=-2)
    else:
        R = T_flat[..., :3, :3]
        t = T_flat[..., :3, 3, np.newaxis]

        R_inv = np.swapaxes(R, -2, -1)
        t_inv = -R_inv @ t

        bottom_row = np.zeros((batch_size, 1, 4), dtype=T.dtype)
        bottom_row[..., 3] = 1

        top_part = np.concatenate([R_inv, t_inv], axis=-1)
        T_inv_flat = np.concatenate([top_part, bottom_row], axis=-2)

    # Reshape back to original batch dimensions
    if unseq_flag:
        T_inv = T_inv_flat[0]
    else:
        T_inv = T_inv_flat.reshape(*batch_shape, 4, 4)
    
    return T_inv

def get_pixel(H, W):
    # get 2D pixels (u, v) for image_a in cam_a pixel space
    u_a, v_a = np.meshgrid(np.arange(W), np.arange(H))
    # u_a = np.flip(u_a, axis=1)
    # v_a = np.flip(v_a, axis=0)
    pixels_a = np.stack([
        u_a.flatten() + 0.5, 
        v_a.flatten() + 0.5, 
        np.ones_like(u_a.flatten())
    ], axis=0)
    
    return pixels_a

def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, z_far=0, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)
    if z_far > 0:
        valid_mask = valid_mask & (depthmap < z_far)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    # assert camera_intrinsics[0, 1] == 0.0
    # assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    # Invalid any depth > 80m
    valid_mask = valid_mask
    return X_cam, valid_mask

def homogenize_points(
    points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def get_gt_warp(depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode = 'bilinear', relative_depth_error_threshold = 0.05, H = None, W = None):
    
    if H is None:
        B,H,W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(
                    -1 + 1 / n, 1 - 1 / n, n, device=depth1.device
                )
                for n in (B, H, W)
            ],
            indexing = 'ij'
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, smooth_mask = False, return_relative_depth_error = False, depth_interpolation_mode = "bilinear", relative_depth_error_threshold = 0.05):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    if depth_interpolation_mode == "combined":
        # Inspired by approach in inloc, try to fill holes from bilinear interpolation by nearest neighbour interpolation
        if smooth_mask:
            raise NotImplementedError("Combined bilinear and NN warp not implemented")
        valid_bilinear, warp_bilinear = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "bilinear",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        valid_nearest, warp_nearest = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "nearest-exact",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        nearest_valid_bilinear_invalid = (~valid_bilinear).logical_and(valid_nearest) 
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp
        
        
    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode = depth_interpolation_mode, align_corners=False)[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    # nonzero_mask = kpts0_depth != 0
    # Sample depth, get calculable_mask on depth > 0
    nonzero_mask = kpts0_depth > 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]
    
    relative_depth_error = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs()
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = (-relative_depth_error/smooth_mask).exp()
    valid_mask = nonzero_mask * covisible_mask * consistent_mask
    if return_relative_depth_error:
        return relative_depth_error, w_kpts0
    else:
        return valid_mask, w_kpts0


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def opencv_camera_to_plucker(poses, K, H, W):
    device = poses.device
    B = poses.shape[0]

    pixel = torch.from_numpy(get_pixel(H, W).astype(np.float32)).to(device).T.reshape(H, W, 3)[None].repeat(B, 1, 1, 1)         # (3, H, W)
    pixel = torch.einsum('bij, bhwj -> bhwi', torch.inverse(K), pixel)
    ray_directions = torch.einsum('bij, bhwj -> bhwi', poses[..., :3, :3], pixel)

    ray_origins = poses[..., :3, 3][:, None, None].repeat(1, H, W, 1)

    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    plucker_ray = torch.cat([ray_directions, plucker_normal], dim=-1)

    return plucker_ray


def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge


def solve_camera_intrinsics_batch(
    camera_coords: torch.Tensor,
    mask: torch.Tensor = None,
    fixed_cx_cy: bool = True,
    min_points: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """根据相机坐标点估计相机内参矩阵。

    新增特性:
    1. 可选 mask, 只使用高置信度 / 有效像素估计内参。
    2. 支持将 cx, cy 固定为图像中心 (自由度只剩 fx, fy)。
    3. 支持任意批处理维度，只要最后三个维度是 (H, W, 3)。

    参数:
        camera_coords: (..., H, W, 3)  相机坐标下的 3D 点 (Xc, Yc, Zc)，与像素一一对应。
        mask: (..., H, W) bool/float，可选。True/1 表示该像素参与估计。
        fixed_cx_cy: 若为 True，则强制 cx=W/2, cy=H/2，仅最小二乘求 fx, fy。
        min_points: 有效点少于该值时退化为使用全部像素。
        eps: 数值稳定用。

    返回:
        K: (..., 3, 3)
    """
    assert camera_coords.ndim >= 3, "camera_coords shape 至少应为 (...,H,W,3)"
    assert camera_coords.shape[-1] == 3, "最后一个维度应为 3 (相机坐标)"
    
    # 获取原始形状信息
    original_shape = camera_coords.shape
    batch_shape = original_shape[:-3]  # 除了最后 (H,W,3) 的所有维度
    H, W = original_shape[-3], original_shape[-2]
    
    # 展平所有批处理维度
    batch_size = np.prod(batch_shape) if batch_shape else 1
    camera_coords_flat = camera_coords.reshape(batch_size, H, W, 3)
    
    # 处理 mask
    if mask is not None:
        assert mask.shape == original_shape[:-1], f"mask shape {mask.shape} 应匹配 camera_coords shape 的前面维度 {original_shape[:-1]}"
        mask_flat = mask.reshape(batch_size, H, W)
    else:
        mask_flat = None
    
    device = camera_coords.device
    N = H * W

    # 展平 3D 点
    points_3d = camera_coords_flat.reshape(batch_size, N, 3)
    Xc, Yc, Zc = points_3d.unbind(-1)

    # 避免除以 0
    valid_depth = (Zc.abs() > eps)

    # 计算像素坐标 (u,v)
    v_grid, u_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )  # (H,W)
    u_flat = u_grid.reshape(1, N).expand(batch_size, N)
    v_flat = v_grid.reshape(1, N).expand(batch_size, N)

    if mask_flat is not None:
        # 处理 mask
        if mask_flat.dtype != torch.bool:
            # 非 bool 当作权重 >0 判定有效，同时保留权重
            weights = mask_flat.reshape(batch_size, N).clamp_min(0).to(torch.float32)
            mask_bool = weights > 0
        else:
            mask_bool = mask_flat
            weights = mask_bool.to(torch.float32)
        mask_bool = mask_bool.reshape(batch_size, N) & valid_depth
        weights = weights.reshape(batch_size, N) * mask_bool.to(torch.float32)
        # 若有效点太少，则放弃 mask
        need_fallback = (mask_bool.sum(dim=1) < min_points)
        if need_fallback.any():
            fallback_ids = torch.nonzero(need_fallback, as_tuple=False).squeeze(-1)
            for bi in fallback_ids.tolist():
                mask_bool[bi] = valid_depth[bi]
                weights[bi] = valid_depth[bi].to(torch.float32)
    else:
        mask_bool = valid_depth
        weights = mask_bool.to(torch.float32)

    # 归一化系数
    Xn = (Xc / (Zc + eps))
    Yn = (Yc / (Zc + eps))

    if fixed_cx_cy:
        # 固定主点
        cx_fixed = W / 2.0
        cy_fixed = H / 2.0

        # 最小二乘: fx = sum w * x * (u-cx) / sum w * x^2
        du = (u_flat - cx_fixed)
        dv = (v_flat - cy_fixed)

        wX = weights * mask_bool
        num_fx = (wX * Xn * du).sum(dim=1)
        den_fx = (wX * Xn * Xn).sum(dim=1).clamp_min(eps)
        fx = num_fx / den_fx

        wY = weights * mask_bool
        num_fy = (wY * Yn * dv).sum(dim=1)
        den_fy = (wY * Yn * Yn).sum(dim=1).clamp_min(eps)
        fy = num_fy / den_fy

        cx = torch.full_like(fx, cx_fixed)
        cy = torch.full_like(fy, cy_fixed)
    else:
        # 回退到同时估计 fx,cx,fy,cy 的线性最小二乘（原实现）
        A = torch.zeros(batch_size, 2 * N, 4, device=device, dtype=torch.float32)
        b = torch.zeros(batch_size, 2 * N, device=device, dtype=torch.float32)

        # 仅对有效点赋值，其他保持 0
        mb2 = mask_bool
        # u = fx * Xn + cx
        A[:, 0::2, 0] = (Xn * mb2).reshape(batch_size, N)
        A[:, 0::2, 1] = mb2.reshape(batch_size, N)
        b[:, 0::2] = (u_flat * mb2).reshape(batch_size, N)
        # v = fy * Yn + cy
        A[:, 1::2, 2] = (Yn * mb2).reshape(batch_size, N)
        A[:, 1::2, 3] = mb2.reshape(batch_size, N)
        b[:, 1::2] = (v_flat * mb2).reshape(batch_size, N)

        solution = torch.linalg.lstsq(A, b, driver='gels').solution
        fx = solution[:, 0]
        cx = solution[:, 1]
        fy = solution[:, 2]
        cy = solution[:, 3]

    # 构建 K
    K_flat = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)
    K_flat[:, 0, 0] = fx
    K_flat[:, 1, 1] = fy
    K_flat[:, 0, 2] = cx
    K_flat[:, 1, 2] = cy
    K_flat[:, 2, 2] = 1.0
    
    # 重新整形回原始批处理维度
    if batch_shape:
        K = K_flat.reshape(*batch_shape, 3, 3)
    else:
        K = K_flat[0]  # 单个样本的情况
    
    return K