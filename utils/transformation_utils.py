import torch
import numpy as np
from torch_cluster import knn
from .camera_utils import *

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def transform2origin(position_tensor):
    min_pos = torch.min(position_tensor, 0)[0]
    max_pos = torch.max(position_tensor, 0)[0]
    max_diff = torch.max(max_pos - min_pos)
    original_mean_pos = (min_pos + max_pos) / 2.0
    scale = 1.0
    original_mean_pos = original_mean_pos.to(device="cuda")
    new_position_tensor = position_tensor

    return new_position_tensor, scale, original_mean_pos


def undotransform2origin(position_tensor, scale, original_mean_pos):
    return position_tensor

def generate_rotation_matrix(degree, axis):
    cos_theta = torch.cos(degree / 180.0 * 3.1415926)
    sin_theta = torch.sin(degree / 180.0 * 3.1415926)
    if axis == 0:
        rotation_matrix = torch.tensor(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
        )
    elif axis == 1:
        rotation_matrix = torch.tensor(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
    elif axis == 2:
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
    else:
        raise ValueError("Invalid axis selection")
    return rotation_matrix.cuda()


def generate_rotation_matrices(degrees, axises):
    assert len(degrees) == len(axises)

    matrices = []

    for i in range(len(degrees)):
        matrices.append(generate_rotation_matrix(degrees[i], axises[i]))

    return matrices


def apply_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix.T)
    return rotated


def apply_cov_rotation(cov_tensor, rotation_matrix):
    # rotated = torch.matmul(cov_tensor, rotation_matrix.T)
    rotated = rotation_matrix @ cov_tensor @ rotation_matrix.T
    return rotated


def get_mat_from_upper(upper_mat):
    upper_mat = upper_mat.reshape(-1, 6)
    mat = torch.zeros((upper_mat.shape[0], 9), device="cuda")
    mat[:, :3] = upper_mat[:, :3]
    mat[:, 3] = upper_mat[:, 1]
    mat[:, 4] = upper_mat[:, 3]
    mat[:, 5] = upper_mat[:, 4]
    mat[:, 6] = upper_mat[:, 2]
    mat[:, 7] = upper_mat[:, 4]
    mat[:, 8] = upper_mat[:, 5]

    return mat.view(-1, 3, 3)


def get_upper_from_mat(mat):
    mat = mat.view(-1, 9)
    upper_mat = torch.zeros((mat.shape[0], 6), device="cuda")
    upper_mat[:, :3] = mat[:, :3]
    upper_mat[:, 3] = mat[:, 4]
    upper_mat[:, 4] = mat[:, 5]
    upper_mat[:, 5] = mat[:, 8]

    return upper_mat


def apply_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        position_tensor = apply_rotation(position_tensor, rotation_matrices[i])
    return position_tensor


def apply_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        cov_tensor = apply_cov_rotation(cov_tensor, rotation_matrices[i])
    return get_upper_from_mat(cov_tensor)


def shift2center111(position_tensor):
    tensor111 = torch.tensor([2.0, 2.0, 1.0], device="cuda")
    return position_tensor + tensor111


def undoshift2center111(position_tensor):
    tensor111 = torch.tensor([2.0, 2.0, 1.0], device="cuda")
    return position_tensor - tensor111


def apply_inverse_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix)
    return rotated


def apply_inverse_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        position_tensor = apply_inverse_rotation(position_tensor, R)
    return position_tensor


def apply_inverse_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        cov_tensor = apply_cov_rotation(cov_tensor, R.T)
    return get_upper_from_mat(cov_tensor)


# input must be (n,3) tensor on cuda
def undo_all_transforms(input, rotation_matrices, scale_origin, original_mean_pos):
    return apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(input), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )


def get_center_view_worldspace_and_observant_coordinate(
    mpm_space_viewpoint_center,
    mpm_space_vertical_upward_axis,
    rotation_matrices,
    scale_origin,
    original_mean_pos,
):
    viewpoint_center_worldspace = undo_all_transforms(
        mpm_space_viewpoint_center, rotation_matrices, scale_origin, original_mean_pos
    )
    mpm_space_up = mpm_space_vertical_upward_axis + mpm_space_viewpoint_center
    worldspace_up = undo_all_transforms(
        mpm_space_up, rotation_matrices, scale_origin, original_mean_pos
    )
    world_space_vertical_axis = worldspace_up - viewpoint_center_worldspace
    viewpoint_center_worldspace = np.squeeze(
        viewpoint_center_worldspace.clone().detach().cpu().numpy(), 0
    )
    vertical, h1, h2 = generate_local_coord(
        np.squeeze(world_space_vertical_axis.clone().detach().cpu().numpy(), 0)
    )
    observant_coordinates = np.column_stack((h1, h2, vertical))

    return viewpoint_center_worldspace, observant_coordinates

def kabsch(P, Q):
    """
    Computes optimal rotation matrices using the Kabsch algorithm.

    Args:
        P: torch.Tensor of shape (T, N, 3), predicted points
        Q: torch.Tensor of shape (T, N, 3), target (canonical) points

    Returns:
        R: torch.Tensor of shape (T, 3, 3), optimal rotation matrices
    """
    # Subtract centroids
    P_centered = P - P.mean(dim=1, keepdim=True)  # (T, N, 3)
    Q_centered = Q - Q.mean(dim=1, keepdim=True)  # (T, N, 3)

    # Compute covariance matrix H = P^T * Q
    H = torch.matmul(P_centered.transpose(1, 2), Q_centered).to(P_centered.dtype)  # (T, 3, 3)

    # Compute SVD of H
    U, S, Vt = torch.linalg.svd(H)  # U: (T, 3, 3), Vt: (T, 3, 3)

    # Compute rotation matrix R = V * U^T
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)).to(P_centered.dtype)  # (T, 3, 3)

    # Handle reflection (when det(R) < 0)
    det_R = torch.linalg.det(R)
    mask = det_R < 0
    if mask.any():
        Vt[mask, 2, :] *= -1
        R[mask] = torch.matmul(Vt[mask].transpose(-2, -1), U[mask].transpose(-2, -1)).to(P_centered.dtype)

    return R

def align_camera_poses(gt_positions, pred_positions):
    # Compute centroids
    gt_centroid = gt_positions.mean(dim=0)
    pred_centroid = pred_positions.mean(dim=0)
    
    # Center the points
    gt_centered = gt_positions - gt_centroid
    pred_centered = pred_positions - pred_centroid
    # estimate radius
    gt_radius = torch.mean(torch.norm(gt_centered, dim=1))
    pred_radius = torch.mean(torch.norm(pred_centered, dim=1))

    # scale pred_centered to match gt_centered radius
    scale_factor = gt_radius / pred_radius
    pred_centered = pred_centered * scale_factor
    
    R = kabsch(pred_centered.unsqueeze(0), gt_centered.unsqueeze(0))[0]  # Use Kabsch algorithm for rotation
    T = gt_centroid - pred_centroid
    
    # Construct the transformation matrix
    transformation_matrix = torch.eye(4, device=gt_positions.device, dtype=gt_positions.dtype)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    return transformation_matrix, scale_factor

def rotation_matrix_to_quaternion(R):
    """
    Convert a batch of rotation matrices to quaternions.
    
    Args:
        R: torch.Tensor of shape (T, 3, 3)
    
    Returns:
        torch.Tensor of shape (T, 4) in (w, x, y, z) format
    """
    T = R.shape[0]
    quats = torch.empty((T, 4), dtype=R.dtype, device=R.device)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Create masks
    cond1 = trace > 0
    cond2 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    cond3 = R[:, 1, 1] > R[:, 2, 2]

    # Case 1: trace > 0
    S = torch.sqrt(trace[cond1] + 1.0) * 2  # S=4*qw
    quats[cond1, 0] = 0.25 * S
    quats[cond1, 1] = (R[cond1, 2, 1] - R[cond1, 1, 2]) / S
    quats[cond1, 2] = (R[cond1, 0, 2] - R[cond1, 2, 0]) / S
    quats[cond1, 3] = (R[cond1, 1, 0] - R[cond1, 0, 1]) / S

    # Case 2: R[0,0] is the largest diagonal term
    cond = ~cond1 & cond2
    S = torch.sqrt(1.0 + R[cond, 0, 0] - R[cond, 1, 1] - R[cond, 2, 2]) * 2  # S=4*qx
    quats[cond, 0] = (R[cond, 2, 1] - R[cond, 1, 2]) / S
    quats[cond, 1] = 0.25 * S
    quats[cond, 2] = (R[cond, 0, 1] + R[cond, 1, 0]) / S
    quats[cond, 3] = (R[cond, 0, 2] + R[cond, 2, 0]) / S

    # Case 3: R[1,1] is the largest diagonal term
    cond = ~cond1 & ~cond2 & cond3
    S = torch.sqrt(1.0 + R[cond, 1, 1] - R[cond, 0, 0] - R[cond, 2, 2]) * 2  # S=4*qy
    quats[cond, 0] = (R[cond, 0, 2] - R[cond, 2, 0]) / S
    quats[cond, 1] = (R[cond, 0, 1] + R[cond, 1, 0]) / S
    quats[cond, 2] = 0.25 * S
    quats[cond, 3] = (R[cond, 1, 2] + R[cond, 2, 1]) / S

    # Case 4: R[2,2] is the largest diagonal term
    cond = ~cond1 & ~cond2 & ~cond3
    S = torch.sqrt(1.0 + R[cond, 2, 2] - R[cond, 0, 0] - R[cond, 1, 1]) * 2  # S=4*qz
    quats[cond, 0] = (R[cond, 1, 0] - R[cond, 0, 1]) / S
    quats[cond, 1] = (R[cond, 0, 2] + R[cond, 2, 0]) / S
    quats[cond, 2] = (R[cond, 1, 2] + R[cond, 2, 1]) / S
    quats[cond, 3] = 0.25 * S

    # Normalize quaternion to ensure unit length
    quats = quats / quats.norm(dim=1, keepdim=True)

    return quats

def rotation_quaternion_to_matrix(q):
    """
    Convert a quaternion tensor to a rotation matrix.
    
    Args:
        q: Tensor of shape (..., 4), representing a quaternion in (w, x, y, z) format.
        
    Returns:
        Tensor of shape (..., 3, 3), representing the corresponding rotation matrices.
    """
    # Normalize the quaternion to ensure it's a unit quaternion
    q = torch.nn.functional.normalize(q, dim=-1)

    w, x, y, z = q.unbind(-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    rot = torch.stack([
        torch.stack([ww + xx - yy - zz, 2 * (xy - wz),     2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz),     ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy),     2 * (yz + wx),     ww - xx - yy + zz], dim=-1)
    ], dim=-2)

    return rot

def euler_xyz_to_matrix(euler):
    """
    Converts Euler angles (XYZ order) to rotation matrices.
    euler: (B, T, num_objs, 3) in radians
    returns: (B, T, num_objs, 3, 3) rotation matrices
    """
    x, y, z = euler[..., 0], euler[..., 1], euler[..., 2]

    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)

    ones = torch.ones_like(cx)
    zeros = torch.zeros_like(cx)

    # Rotation matrices around X axis
    R_x = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cx, -sx], dim=-1),
        torch.stack([zeros, sx, cx], dim=-1)
    ], dim=-2)

    # Rotation matrices around Y axis
    R_y = torch.stack([
        torch.stack([cy, zeros, sy], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sy, zeros, cy], dim=-1)
    ], dim=-2)

    # Rotation matrices around Z axis
    R_z = torch.stack([
        torch.stack([cz, -sz, zeros], dim=-1),
        torch.stack([sz, cz, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)

    # Combined rotation matrix: R = R_z @ R_y @ R_x
    R = torch.matmul(R_z, torch.matmul(R_y, R_x))  # (B, T, num_objs, 3, 3)
    return R

def dof6_to_matrix3x3(cov3D_dof6):
    """
    Convert 6-DOF covariance representation to 3x3 matrices (vectorized)
    
    Args:
        cov3D_dof6: Tensor of shape [N, 6] or [B, N, 6] representing covariance in 6-DOF format
        
    Returns:
        cov3D_matrices: Tensor of shape [N, 3, 3] or [B, N, 3, 3] representing covariance matrices
    """
    device = cov3D_dof6.device
    if len(cov3D_dof6.shape) == 2:
        N = cov3D_dof6.shape[0]
        cov3D_matrices = torch.zeros((N, 3, 3), device=device)
    elif len(cov3D_dof6.shape) == 3:
        B, N = cov3D_dof6.shape[:2]
        cov3D_matrices = torch.zeros((B, N, 3, 3), device=device)
    else:
        raise ValueError("Input tensor must be of shape [N, 6] or [B, N, 6]")
    
    # Fill the matrices using the 6-DOF parameters (vectorized)
    cov3D_matrices[..., 0, 0] = cov3D_dof6[..., 0]  # a
    cov3D_matrices[..., 0, 1] = cov3D_dof6[..., 1]  # b
    cov3D_matrices[..., 1, 0] = cov3D_dof6[..., 1]  # b
    cov3D_matrices[..., 1, 1] = cov3D_dof6[..., 3]  # c
    cov3D_matrices[..., 0, 2] = cov3D_dof6[..., 2]  # d
    cov3D_matrices[..., 2, 0] = cov3D_dof6[..., 2]  # d
    cov3D_matrices[..., 1, 2] = cov3D_dof6[..., 4]  # e
    cov3D_matrices[..., 2, 1] = cov3D_dof6[..., 4]  # e
    cov3D_matrices[..., 2, 2] = cov3D_dof6[..., 5]  # f
    
    return cov3D_matrices

def matrix3x3_to_dof6(cov3D_matrices):
    """
    Convert 3x3 covariance matrices to 6-DOF representation (vectorized)
    
    Args:
        cov3D_matrices: Tensor of shape [N, 3, 3] or [B, N, 3, 3] representing covariance matrices
        
    Returns:
        cov3D_dof6: Tensor of shape [N, 6] or [B, N, 6] representing covariance in 6-DOF format
    """
    device = cov3D_matrices.device
    if len(cov3D_matrices.shape) == 3:
        N = cov3D_matrices.shape[0]
        cov3D_dof6 = torch.zeros((N, 6), device=device)
    elif len(cov3D_matrices.shape) == 4:
        B, N = cov3D_matrices.shape[:2]
        cov3D_dof6 = torch.zeros((B, N, 6), device=device)
    else:
        raise ValueError("Input tensor must be of shape [N, 3, 3] or [B, N, 3, 3]")
    # Extract the 6 parameters from the symmetric matrices (vectorized)
    cov3D_dof6[..., 0] = cov3D_matrices[..., 0, 0]  # a
    cov3D_dof6[..., 1] = cov3D_matrices[..., 0, 1]  # b (or matrices[..., 1, 0])
    cov3D_dof6[..., 2] = cov3D_matrices[..., 0, 2]  # c
    cov3D_dof6[..., 3] = cov3D_matrices[..., 1, 1]  # d (or matrices[..., 2, 0])
    cov3D_dof6[..., 4] = cov3D_matrices[..., 1, 2]  # e (or matrices[..., 2, 1])
    cov3D_dof6[..., 5] = cov3D_matrices[..., 2, 2]  # f
    
    return cov3D_dof6

def upsample_gaussian_splatting(downsampled_data, original_positions, ori_obj_ids=None, k=3):
    """
    Perform upsampling interpolation for Gaussian splatting data, including position, covariance,
    rotation matrix, opacity, and spherical harmonic coefficients.
    
    Args:
        downsampled_data: dict - dictionary containing:
            'pos': [B, T, N_down, 3] - positions of downsampled points
            'cov3D': [B, T, N_down, 6] - 3D covariances
            'rot': [B, T, N_down, 3, 3] - rotation matrices
            'opacity': [B, N_down, 1] - opacity values
            'shs': [B, N_down, SH_dim, 3] - spherical harmonic coefficients
            'keypoint_indices': [B, N_down] - indices of keypoints in the original point cloud (if available)
        original_positions: [B, N_up, 3] - positions of the original (upsampled) point cloud, 
            typically from the first frame
        ori_obj_ids: [B, N_up, 1] - object IDs for each upsampled point, 
            used to ensure KNN search is restricted to points from the same object
        k: int - number of neighbors for KNN query
        
    Returns:
        upsampled_data: dict - dictionary containing all upsampled attributes
    """
    device = downsampled_data['pos'].device
    dtype = downsampled_data['pos'].dtype
    
    B, T, N_down, _ = downsampled_data['pos'].shape
    N_up = original_positions.shape[1]
    
    upsampled_data = {
        'pos': torch.zeros(B, T, N_up, 3, dtype=dtype, device=device),
        'cov3D': torch.zeros(B, N_up, 6, dtype=dtype, device=device),
        'rot': torch.zeros(B, N_up, 3, 3, dtype=dtype, device=device),
        'opacity': torch.zeros(B, N_up, 1, dtype=dtype, device=device),
        'shs': torch.zeros(B, N_up, downsampled_data['shs'].shape[2], downsampled_data['shs'].shape[3], dtype=dtype, device=device)
    }
    
    for b in range(B):
        pos_down_first = downsampled_data['pos'][b, 0]   # (N_down, 3)
        pos_up = original_positions[b]                   # (N_up, 3)
        
        # If object IDs are provided, perform KNN within each object
        if ori_obj_ids is not None:
            unique_obj_ids = torch.unique(ori_obj_ids[b])
            row_all = []
            col_all = []
            
            for obj_id in unique_obj_ids:
                # Get mask for the current object ID
                mask_down = downsampled_data.get('obj_ids', None)
                if mask_down is not None:
                    # Ensure mask is 1D
                    mask_down = (mask_down[b] == obj_id).squeeze(-1)
                else:
                    # If object IDs are missing in downsampled data, try mapping via keypoint indices
                    if 'keypoint_indices' in downsampled_data:
                        indices = downsampled_data['keypoint_indices'][b]
                        mask_down = (ori_obj_ids[b][indices] == obj_id).squeeze(-1)
                    else:
                        continue  # Skip if object IDs cannot be determined
                
                # Ensure 1D mask for upsampled points
                mask_up = (ori_obj_ids[b] == obj_id).squeeze(-1)
                
                # Skip if object has no points in downsampled or upsampled data
                if not mask_down.any() or not mask_up.any():
                    continue
                
                # Perform KNN for the current object
                pos_down_obj = pos_down_first[mask_down]  # Downsampled points for this object
                pos_up_obj = pos_up[mask_up]  # Upsampled points for this object
                
                if pos_down_obj.shape[0] == 0 or pos_up_obj.shape[0] == 0:
                    continue
                
                # Run KNN and get edge indices
                edge_index_obj = knn(pos_down_obj, pos_up_obj, k=min(k, pos_down_obj.shape[0]))
                row_obj, col_obj = edge_index_obj
                
                # Convert local indices to global indices
                global_row = torch.where(mask_up)[0][row_obj]
                global_col = torch.where(mask_down)[0][col_obj]
                
                row_all.append(global_row)
                col_all.append(global_col)
            
            if len(row_all) == 0:
                continue
            row = torch.cat(row_all, dim=0)
            col = torch.cat(col_all, dim=0)
        else:
            edge_index = knn(pos_down_first, pos_up, k=k)
            row, col = edge_index
        
        # Interpolation
        counts = torch.zeros(N_up, 1, device=device)
        counts.index_add_(0, row, torch.ones_like(row, dtype=dtype, device=device).unsqueeze(-1))
        
        for t in range(T):
            pos_frame_up = torch.zeros(N_up, 3, dtype=dtype, device=device)
            pos_frame_up.index_add_(0, row, downsampled_data['pos'][b, t, col])
            upsampled_data['pos'][b, t] = pos_frame_up / counts.clamp(min=1.0)
        
        cov_frame_up = torch.zeros(N_up, 6, dtype=dtype, device=device)
        cov_frame_up.index_add_(0, row, downsampled_data['cov3D'][b, 0, col])
        upsampled_data['cov3D'][b] = cov_frame_up / counts.clamp(min=1.0)
        
        rot_frame_up = torch.zeros(N_up, 9, dtype=dtype, device=device)
        rot_frame_up.index_add_(0, row, downsampled_data['rot'][b, 0, col].reshape(-1, 9))
        upsampled_data['rot'][b] = (rot_frame_up / counts.clamp(min=1.0)).reshape(N_up, 3, 3)
        
        opacity_up = torch.zeros(N_up, 1, dtype=dtype, device=device)
        opacity_up.index_add_(0, row, downsampled_data['opacity'][b][col])
        upsampled_data['opacity'][b] = opacity_up / counts.clamp(min=1.0)
        
        SH_dim, SH_dim2 = downsampled_data['shs'].shape[2], downsampled_data['shs'].shape[3]
        shs_up = torch.zeros(N_up, SH_dim, SH_dim2, dtype=dtype, device=device)
        counts_expanded = counts.view(N_up, 1, 1).expand(-1, SH_dim, SH_dim2)
        shs_up.index_add_(0, row, downsampled_data['shs'][b][col])
        upsampled_data['shs'][b] = shs_up / counts_expanded.clamp(min=1.0)
        
        if 'keypoint_indices' in downsampled_data:
            upsampled_data['keypoint_indices'] = torch.arange(N_up, device=device)
    
    return upsampled_data

def transform_gaussians(positions_sequence, initial_cov3D, k=8, chunk_size=100_000):
    """
    Parallelized version of Gaussian transformation using initial frame as reference.

    Args:
        positions_sequence: Tensor of shape [B, T, N, 3]
        initial_cov3D: Tensor of shape [B, N, 6]
        k: Number of nearest neighbors
        chunk_size: Maximum number of Gaussians processed at once along N dimension.
            Setting chunk_size >= N yields identical behavior to the original
            unchunked implementation while smaller values reduce peak memory.

    Returns:
        Tensor of shape [B, T, N, 6] with 3D covariances over time
    """
    B = positions_sequence.shape[0]  # Batch size
    T = positions_sequence.shape[1]
    N = positions_sequence.shape[2]  # Number of points
    print(f"There are {T} frames and {N} points")
    if chunk_size is None:
        chunk_size = N
    else:
        chunk_size = int(chunk_size)
    chunk_size = max(1, min(chunk_size, N))
    device = positions_sequence.device

    # Prepare output tensor for covariances
    cov3D_sequence = torch.zeros(B, T, N, 6, device=device)

    for b in range(B):
        # Stack into a single tensor for batching: [T, N, 3]
        positions = positions_sequence[b]  # [T, N, 3]
        initial_positions = positions[0]  # [N, 3]

        # Get k-NN indices from initial frame
        _, col = knn(initial_positions, initial_positions, k=k + 1)  # Include self
        neighbor_idx = col.view(N, k + 1)  # Shape: (N, k+1)
        neighbor_idx = neighbor_idx[:, 1:]  # Remove self, shape: [N, k]

        # Convert initial 6-DOF covariances to 3x3 once
        cov_matrices_full = dof6_to_matrix3x3(initial_cov3D[b])  # [N, 3, 3]

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_slice = slice(start, end)
            chunk_len = end - start
            chunk_neighbor_idx = neighbor_idx[chunk_slice]  # [chunk_len, k]

            # Reference-centered neighbor vectors from initial frame (chunk)
            ref_neighbors = initial_positions[chunk_neighbor_idx]  # [chunk_len, k, 3]
            ref_centered = ref_neighbors - initial_positions[chunk_slice].unsqueeze(1)  # [chunk_len, k, 3]
            ref_centered_T = ref_centered.transpose(1, 2).unsqueeze(0).expand(T, -1, -1, -1)  # [T, chunk_len, 3, k]

            # Current positions and their neighbors for the same chunk
            chunk_positions = positions[:, chunk_slice]  # [T, chunk_len, 3]
            positions_neighbors = positions[:, chunk_neighbor_idx]  # [T, chunk_len, k, 3]
            positions_centered = positions_neighbors - chunk_positions.unsqueeze(2)  # [T, chunk_len, k, 3]

            # Compute H matrices and SVD
            H = torch.matmul(ref_centered_T, positions_centered)  # [T, chunk_len, 3, 3]
            U, _, V = torch.svd(H)
            R = torch.matmul(V, U.transpose(-1, -2))

            # Correct for reflections using the same logic as the unchunked version
            det = torch.det(R)
            correction = torch.zeros((T, chunk_len, 3, 3), device=device, dtype=R.dtype)
            correction[..., 0, 0] = 1.0
            correction[..., 1, 1] = 1.0
            sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
            correction[..., 2, 2] = sign
            R = torch.matmul(R, correction)

            # Rotate covariances: R @ cov @ R^T
            cov_chunk = cov_matrices_full[chunk_slice].unsqueeze(0).expand(T, -1, -1, -1)  # [T, chunk_len, 3, 3]
            R_T = R.transpose(-1, -2)
            rotated_cov = torch.matmul(R, torch.matmul(cov_chunk, R_T))  # [T, chunk_len, 3, 3]

            # Convert back to 6-DOF
            cov3D_sequence[b, :, chunk_slice] = matrix3x3_to_dof6(rotated_cov)  # [T, chunk_len, 6]

    return cov3D_sequence


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

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

def ensure_spd(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Make symmetric positive definite by symmetrizing and clamping eigenvalues.
    cov: [*, 3, 3]
    """
    # Symmetrize
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    # Eigen clamp
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.clamp(eigvals, min=eps)
    cov_spd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
    return cov_spd
