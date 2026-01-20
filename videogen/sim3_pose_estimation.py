import torch
from chamfer_distance import ChamferDistance

def _random_subsample(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if points.shape[0] <= max_points:
        return points
    indices = torch.randperm(points.shape[0], device=points.device)[:max_points]
    return points[indices]

def nearest_points_on_rays_to_cloud(
    pixel_xy: torch.Tensor,
    K_pixels: torch.Tensor,
    T_c2w: torch.Tensor,
    point_cloud_world: torch.Tensor,
) -> torch.Tensor:
    """
    For each 2D pixel coordinate, cast a ray into 3D using intrinsics/extrinsics
    and return the nearest world-space point from the provided point cloud to that ray.

    Args:
        pixel_xy: [N,2] pixel coordinates (x,y) in pixels.
        K_pixels: [3,3] intrinsics matrix in pixel units.
        T_c2w: [4,4] camera-to-world transform.
        point_cloud_world: [M,3] world-space points to search.

    Returns:
        [N,3] nearest world-space points for each ray.
    """
    assert pixel_xy.dim() == 2 and pixel_xy.shape[1] == 2, "pixel_xy must be [N,2]"

    fx, fy = K_pixels[0, 0], K_pixels[1, 1]
    cx, cy = K_pixels[0, 2], K_pixels[1, 2]

    # Build normalized camera-frame rays
    dirs_c = torch.stack([
        (pixel_xy[:, 0] - cx) / fx,
        (pixel_xy[:, 1] - cy) / fy,
        torch.ones_like(pixel_xy[:, 0])
    ], dim=-1)
    dirs_c = dirs_c / torch.norm(dirs_c, dim=-1, keepdim=True)

    R_c2w = T_c2w[:3, :3]
    C_w = T_c2w[:3, 3]
    dirs_w = dirs_c @ R_c2w.transpose(0, 1)

    diff = point_cloud_world - C_w
    nearest_points = []
    for i in range(dirs_w.shape[0]):
        si = (diff @ dirs_w[i])
        perp = diff - si.unsqueeze(-1) * dirs_w[i]
        di = (perp * perp).sum(-1)
        nearest_points.append(point_cloud_world[torch.argmin(di)])
    return torch.stack(nearest_points, dim=0)


def estimate_sim3_chamfer_gd(source_points: torch.Tensor,
                             target_points: torch.Tensor,
                             max_iterations: int = 200,
                             lr: float = 1e-2,
                             num_samples = None,
                             bidirectional: bool = True,
                             clamp_scale = None,
                             convergence_tol: float = 1e-7,
                             init_guess: dict | None = None,
                             anchor_src: torch.Tensor | None = None,
                             anchor_tgt: torch.Tensor | None = None,
                             anchor_weight: float = 0.0,
                             lr_rot: float | None = None,
                             lr_scale: float | None = None,
                             lr_trans: float | None = None,
                             ):
    """
    Optimize Sim(3) (s,R,t) by minimizing Chamfer distance between transformed source and target.
    Returns (scale, T) where T contains pure rotation and translation (scale returned separately).
    """
    def _skew(v: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0, -v[2], v[1]],
                             [v[2], 0.0, -v[0]],
                             [-v[1], v[0], 0.0]], dtype=v.dtype, device=v.device)

    def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
        theta = torch.linalg.norm(omega)
        if theta < 1e-9:
            K = _skew(omega)
            return torch.eye(3, dtype=omega.dtype, device=omega.device) + K
        axis = omega / theta
        K = _skew(axis)
        s = torch.sin(theta)
        c = torch.cos(theta)
        return torch.eye(3, dtype=omega.dtype, device=omega.device) + s * K + (1.0 - c) * (K @ K)

    def _so3_log(R: torch.Tensor) -> torch.Tensor:
        # Map rotation matrix to axis-angle (approx for small angles)
        cos_theta = torch.clamp((torch.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = torch.arccos(cos_theta)
        if theta < 1e-9:
            return torch.zeros(3, dtype=R.dtype, device=R.device)
        w = torch.tensor([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1],
        ], dtype=R.dtype, device=R.device) / (2.0 * torch.sin(theta))
        return theta * w

    # Use external ChamferDistance implementation to avoid explicit cdist graphs

    src = source_points.detach()
    tgt = target_points.detach()

    if num_samples is None:
        num_samples = min(src.shape[0], tgt.shape[0])

    src = src[torch.isfinite(src).all(dim=1)]
    tgt = tgt[torch.isfinite(tgt).all(dim=1)]

    if src.shape[0] < 3 or tgt.shape[0] < 3:
        T = torch.eye(4, dtype=src.dtype, device=src.device)
        return torch.tensor(1.0, dtype=src.dtype, device=src.device), T

    # Subsample for optimization
    src_s = _random_subsample(src, num_samples)
    tgt_s = _random_subsample(tgt, num_samples)

    # Initialize parameters
    if init_guess is not None:
        s0 = init_guess.get('scale', torch.tensor(1.0, dtype=src.dtype, device=src.device))
        R0 = init_guess.get('R', torch.eye(3, dtype=src.dtype, device=src.device))
        t0 = init_guess.get('t', torch.zeros(3, dtype=src.dtype, device=src.device))
        s0 = s0.to(dtype=src.dtype, device=src.device)
        R0 = R0.to(dtype=src.dtype, device=src.device)
        t0 = t0.to(dtype=src.dtype, device=src.device)
    else:
        s0 = torch.tensor(1.0, dtype=src.dtype, device=src.device)
        R0 = torch.eye(3, dtype=src.dtype, device=src.device)
        t0 = torch.zeros(3, dtype=src.dtype, device=src.device)

    omega = _so3_log(R0).detach().clone().requires_grad_(True)
    log_s = torch.log(torch.clamp(s0.detach(), min=torch.finfo(src.dtype).eps)).detach().clone().requires_grad_(True)
    t_param = t0.detach().clone().requires_grad_(True)

    if lr_rot is None:
        lr_rot = lr * 0.01
    if lr_scale is None:
        lr_scale = lr * 0.1
    if lr_trans is None:
        lr_trans = lr
    optimizer = torch.optim.Adam([
        {'params': omega, 'lr': lr_rot},
        {'params': log_s, 'lr': lr_scale},
        {'params': t_param, 'lr': lr_trans},
    ])

    chamfer_fn = ChamferDistance()

    prev_loss = None
    for iter in range(max_iterations):
        optimizer.zero_grad()
        s = torch.exp(log_s)
        R = _so3_exp(omega)
        x = s * (src_s @ R.T) + t_param

        xb = x.unsqueeze(0)
        yb = tgt_s.unsqueeze(0)
        dist1, dist2 = chamfer_fn(xb, yb)
        if bidirectional:
            loss = dist1.mean() + dist2.mean()
        else:
            loss = dist1.mean()

        if anchor_src is not None and anchor_weight > 0.0:
            a = anchor_src.float()
            b = anchor_tgt.float()
            a_trans = s * (a @ R.T) + t_param
            anchor_loss = ((a_trans - b)**2).sum(dim=1).mean()
            loss = loss + anchor_weight * anchor_loss

        loss.backward()
        optimizer.step()

        # if iter % 50 == 0:
        #     print(f"Iteration {iter} loss: {loss.item()}")

        if clamp_scale is not None:
            min_s, max_s = clamp_scale
            min_log = torch.log(torch.tensor(min_s, dtype=log_s.dtype, device=log_s.device))
            max_log = torch.log(torch.tensor(max_s, dtype=log_s.dtype, device=log_s.device))
            with torch.no_grad():
                log_s.clamp_(min=min_log.item(), max=max_log.item())

        if prev_loss is not None:
            rel = torch.abs(prev_loss - loss) / torch.clamp(prev_loss, min=1e-12)
            if rel < convergence_tol:
                break
        
        prev_loss = loss.detach()

    # Compose final outputs
    with torch.no_grad():
        s = torch.exp(log_s)
        R = _so3_exp(omega)
        t = t_param
        T = torch.eye(4, dtype=src.dtype, device=src.device)
        T[:3, :3] = R
        T[:3, 3] = t
    
    return s, T
