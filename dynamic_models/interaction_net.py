import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transformation_utils import euler_xyz_to_matrix
import matplotlib.pyplot as plt
import os
from tools.monitor import timeit, memoryit
from torch_cluster import radius as cluster_radius

class IN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=3, num_layers=3, threshold=1e-2, num_keypoints=1024, k=8):
        super(IN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.threshold = threshold # threshold for intersection and boundary detection
        
        self.pointnet = PointNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        self.state_embedding = self.make_layers(input_dim*4, hidden_dim, num_blocks=num_layers, final_out=False)  # For com, angle, com_vel, angle_vel
        self.brunch = self.make_layers(hidden_dim*2, hidden_dim*2, num_blocks=num_layers, final_out=False)
        self.trunk = self.make_layers(hidden_dim, hidden_dim*2, num_blocks=num_layers, final_out=False)
        self.output = self.make_layers(hidden_dim*2, hidden_dim*2, output_dim=output_dim*3, num_blocks=1, final_out=True)
        self.boundary_net = self.make_layers(hidden_dim*2, hidden_dim*2, output_dim=output_dim*3, num_blocks=num_layers, final_out=True)
        self.stress_net = StressNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_keypoints=num_keypoints, k=k)

    def make_layers(self, input_dim, hidden_dim, output_dim=None, num_blocks=3, final_out=False):
        layers = []

        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_blocks - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())

        # Final output layer
        if final_out and output_dim is not None:
            layers.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def _calc_intersection(self, point_objects, padding_mask, threshold=0.001):
        '''
        input:
            point_objects: (B, num_obj, num_point, 3)
            padding_mask: (B, num_obj, num_point, 1), bool or 0/1, True for valid points
            threshold: radius threshold for considering overlap
        output:
            intersection_matrix: (B, num_obj, num_obj, 1), whether two objects have any overlap
            overlap_mask: (B, num_obj, num_obj, num_point), per-point mask indicating a point overlaps any point in the other object
        '''
        bs, num_obj, num_point, _ = point_objects.shape
        device = point_objects.device

        valid_mask = padding_mask.squeeze(-1).bool()  # (B, O, P)
        # Flatten valid points
        pos_flat = point_objects.reshape(-1, 3)  # (B*O*P, 3)
        valid_flat = valid_mask.reshape(-1)  # (B*O*P)

        pos_valid = pos_flat[valid_flat]  # (M, 3)

        b_idx = torch.arange(bs, device=device).view(bs, 1, 1).expand(bs, num_obj, num_point).reshape(-1)[valid_flat]
        o_idx = torch.arange(num_obj, device=device).view(1, num_obj, 1).expand(bs, num_obj, num_point).reshape(-1)[valid_flat]
        p_idx = torch.arange(num_point, device=device).view(1, 1, num_point).expand(bs, num_obj, num_point).reshape(-1)[valid_flat]

        # Range search within each batch, exclude same object pairs later
        row, col = cluster_radius(pos_valid, pos_valid, r=threshold, batch_x=b_idx, batch_y=b_idx, max_num_neighbors=64)

        # Exclude self-edges and same-object pairs
        neq_obj = o_idx[row] != o_idx[col]
        neq_self = row != col
        keep = neq_obj & neq_self
        row = row[keep]
        col = col[keep]

        # Build point-level overlap mask (B, O, O, P)
        overlap_point_mask = torch.zeros((bs, num_obj, num_obj, num_point), dtype=torch.bool, device=device)
        if row.numel() > 0:
            overlap_point_mask[b_idx[row], o_idx[row], o_idx[col], p_idx[row]] = True

        intersection_matrix = overlap_point_mask.any(dim=-1, keepdim=True).float()  # (B, O, O, 1)

        return intersection_matrix, overlap_point_mask

    def _calc_boundary(self, point_objects, padding_mask, boundary=[1, -1, 1, -1, 1, -1], threshold=0.001):
        '''
        input:
            point_objects: (B, num_obj, num_point, 3)
            padding_mask: (B, num_obj, num_point, 1)
            boundary: [x_min, x_max, y_min, y_max, z_min, z_max]
        output:
            boundary_matrix: (B, num_obj, 1) -- whether any point of object touches boundary
            overlap_mask: (B, num_obj, num_point, 1) -- per-point boundary overlap mask
        '''
        bs, num_obj, num_point, _ = point_objects.shape
        device = point_objects.device

        # Parse boundaries
        x_max, x_min, y_max, y_min, z_max, z_min = boundary

        # Distances from all 6 boundaries
        dists = torch.stack([
            torch.abs(point_objects[..., 0] - x_min),
            torch.abs(point_objects[..., 0] - x_max),
            torch.abs(point_objects[..., 1] - y_min),
            torch.abs(point_objects[..., 1] - y_max),
            torch.abs(point_objects[..., 2] - z_min),
            # torch.abs(point_objects[..., 2] - z_max),
        ], dim=-1)  # (B, num_obj, num_point, 6)

        # Minimum distance to any boundary
        min_dist = torch.min(dists, dim=-1)[0]  # (B, num_obj, num_point)

        # Mask padding
        valid_mask = padding_mask.squeeze(-1).bool()  # (B, num_obj, num_point)
        min_dist[~valid_mask] = threshold + 1  # force padded points to be outside threshold

        # Point-wise overlap with boundary
        overlap_mask = (min_dist < threshold) & valid_mask  # (B, num_obj, num_point)
        boundary_matrix = overlap_mask.any(dim=-1).unsqueeze(-1).float()  # (B, num_obj, 1)
        # if overlap_mask.sum() != 0:
        #     print(f"Boundary mask sum: {overlap_mask.sum()}")

        return boundary_matrix, overlap_mask.unsqueeze(-1).float()
    
    def forward(self, points, ori_pos, points_vel, com, angle, com_vel, angle_vel, padding, knn, boundary, features=None):
        '''
        points: (B, num_objs, N, 3)
        ori_pos: (B, num_objs, N, 3) — original pos for each point
        points_vel: (B, num_objs, N, 3) — local velocity of each point
        com: (B, num_objs, 3) — center of mass for each object
        angle: (B, num_objs, 3) — orientation angles for each object
        com_vel: (B, num_objs, 3) — velocity of the center of mass for each object
        angle_vel: (B, num_objs, 3) — angular velocity for each object
        padding: (B, num_objs, N) — padding mask for each object
        knn: (B, num_objs, N, K) — k-nearest neighbors for each point
        '''
        import time
        bs, num_objs, N, C = points.shape
        t2 = time.time()
        # input into pointnet to extract object-level feature
        geom_features = self.pointnet(points)  # (B, num_objs, hidden_dim)
        brach_geom_features = geom_features.unsqueeze(2).repeat(1, 1, num_objs, 1) # (bs, num_objs, num_objs, hidden_dim)
        trunk_geom_features = geom_features.unsqueeze(1).repeat(1, num_objs, 1, 1) # (bs, num_objs, num_objs, hidden_dim)
        t3 = time.time()
        states = torch.cat([com, angle, com_vel, angle_vel], dim=-1) # (B, num_objs, 12)
        ground = torch.zeros_like(states)
        ground[..., 2] = boundary[-1]
        state_features = self.state_embedding(states-ground) # (B, num_objs, hidden_dim)
        branch_states = states.unsqueeze(2).repeat(1, 1, num_objs, 1) # (bs, num_objs, num_objs, 12)
        trunk_states = states.unsqueeze(1).repeat(1, num_objs, 1, 1) # (bs, num_objs, num_objs, 12)
        # Equivariant state features
        branch_states -= trunk_states
        # trunk_states -= trunk_states
        brach_states_features = self.state_embedding(branch_states) # (bs, num_objs, num_objs, hidden_dim)
        # trunk_states_features = self.state_embedding(trunk_states) # (bs, num_objs, num_objs, hidden_dim)
        t4 = time.time()
        branch_features = torch.cat([brach_geom_features, brach_states_features], dim=-1) # (bs, num_objs, num_objs, hidden_dim*2)
        # trunk_features = torch.cat([trunk_geom_features, trunk_states_features], dim=-1) # (bs, num_objs, num_objs, hidden_dim*2)
        branch_out = self.brunch(branch_features)
        trunk_out = self.trunk(trunk_geom_features)
        object_output = self.output(branch_out * trunk_out) # (bs, num_objs, num_objs, output_dim*2)
        t5 = time.time()
        global_points = torch.matmul(points, euler_xyz_to_matrix(angle).transpose(-2, -1)) + com.unsqueeze(-2)   # (B, num_objs, N, 3), rotate points to global frame
        # remove uncontact force and self force
        intersection_matrix, overlap_mask = self._calc_intersection(global_points, padding.unsqueeze(-1).bool(), threshold=self.threshold) # (bs, num_objs, num_objs, 1)
        object_output = object_output * intersection_matrix # (bs, num_objs, num_objs, output_dim*2)
        t6 = time.time()
        output = object_output.sum(dim=2) # (bs, num_objs, output_dim*2)
        # add boundary force
        boundary_matrix, boundary_mask = self._calc_boundary(global_points, padding.unsqueeze(-1).bool(), boundary=boundary, threshold=self.threshold) # (bs, 2, 1)
        t7 = time.time()
        boundary_features = torch.cat([geom_features, state_features], dim=-1) # (bs, num_objs, hidden_dim*2)
        boundary_output = self.boundary_net(boundary_features) # (bs, num_objs, output_dim*2)
        boundary_output = boundary_output * boundary_matrix # (bs, num_objs, output_dim*2)
        output = output + boundary_output # (bs, num_objs, output_dim*2)
        t8 = time.time()

        forces = output[..., :self.output_dim]
        torques = output[..., self.output_dim:self.output_dim*2]
        # stresses = self.stress_net(points, ori_pos, points_vel, object_output[..., -self.output_dim:], boundary_output[..., -self.output_dim:], overlap_mask, boundary_mask) # (bs, num_objs, N, output_dim)
        stresses = self.stress_net(points, ori_pos, geom_features, 
                                   angle, points_vel, 
                                   object_output[..., -self.output_dim:], 
                                   boundary_output[..., -self.output_dim:], 
                                   overlap_mask, boundary_mask, knn) # (bs, num_objs, N, output_dim)

        # print(f"PointNet: {t3-t2:.4f}s, State Embedding: {t4-t3:.4f}s, DeepONet: {t5-t4:.4f}s, Intersection: {t6-t5:.4f}s, Boundary: {t7-t6:.4f}s, Boundary Force: {t8-t7:.4f}s")
        return forces, torques, stresses


class PointNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=3, num_layers=4):
        super(PointNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Point-wise embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Hierarchical feature extraction
        self.extraction_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.extraction_blocks.append(
                PointwiseBlock(hidden_dim, hidden_dim)
            )
        
        # Global feature aggregation
        self.global_features = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, objects):
        '''
        points: (B, num_objs, N, C)
        '''
        bs, num_obj, num_points, C = objects.shape
        point_features = self.input_embedding(objects)
        # Hierarchical feature extraction
        for block in self.extraction_blocks:
            point_features = block(point_features)
        # Global feature aggregation to get object-level features
        geom_features = torch.max(point_features, dim=-2, keepdim=True)[0]
        geom_features = self.global_features(geom_features)
        geom_features = geom_features.reshape(bs, num_obj, self.hidden_dim)
        
        return geom_features

class PointwiseBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PointwiseBlock, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
        
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.mlp(x)
        return out + identity
    

class StressNet(nn.Module):
    '''
    A simple stress prediction network that takes in point velocities and forces and outputs stress values.
    '''
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=3, num_keypoints=1024, k=8):
        super(StressNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim*2 + output_dim*2 , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, centered_objects, ori_pos, geom_features, angle, points_vel, object_output, boundary_output, overlap_mask=None, boundary_mask=None, knn=None):
        '''
        centered_objects: (B, num_objs, N, 3) — points centered around the center of mass
        points_vel: (B, num_objs, N, 3) — point velocities
        ori_pos: (B, num_objs, N, 3) — original pos for each point
        angle: (B, num_objs, 3) — orientation angles for each object
        object_output: (B, num_objs, num_objs, output_dim*2) — forces and torques from the interaction network
        boundary_output: (B, num_objs, output_dim*2) — boundary forces and torques
        overlap_mask: (B, num_objs, num_objs, N), optional point-level mask; True if a point overlaps any point in the other object
        boundary_mask: (B, num_objs, N, 1), optional mask for boundary points
        stresses: (B, num_objs, N, output_dim) — predicted stresses
        '''
        bs, num_objs, N, C = points_vel.shape
        # assign output to valid points
        if overlap_mask is not None:
            # overlap_mask is (B, num_objs, num_objs, N)
            object_output = object_output.unsqueeze(-2).expand(-1, -1, -1, N, -1) * overlap_mask.unsqueeze(-1)  # (B, num_objs, num_objs, N, output_dim*2)
            object_output = object_output.sum(2) # (B, num_objs, N, output_dim*2)
        if boundary_mask is not None:
            boundary_output = boundary_output.unsqueeze(-2).expand(-1, -1, N, -1) * boundary_mask # (B, num_objs, N, output_dim*2)
        # rotate object_output and boundary_output back to local frame
        object_output = object_output.reshape(bs, num_objs, -1, self.output_dim)  # (B, num_objs, N*2, output_dim)
        boundary_output = boundary_output.reshape(bs, num_objs, -1, self.output_dim)  # (B, num_objs, N*2, output_dim)
        object_output = torch.matmul(object_output, euler_xyz_to_matrix(-angle).transpose(-2, -1))  # (B, num_objs, N*2, output_dim)
        boundary_output = torch.matmul(boundary_output, euler_xyz_to_matrix(-angle).transpose(-2, -1))  # (B, num_objs, N*2, output_dim)
        object_output = object_output.reshape(bs, num_objs, N, -1)  # (B, num_objs, N, output_dim*2)
        boundary_output = boundary_output.reshape(bs, num_objs, N, -1)  # (B, num_objs, N, output_dim*2)
        # concatenate velocities and forces
        input_features = torch.cat([(centered_objects - ori_pos)*100, points_vel*100, object_output, boundary_output], dim=-1)
        # apply the stress network
        features = self.feature_layer(input_features)
        global_feature = torch.max(features, dim=-2, keepdim=True)[0]  # (B, num_objs, 1, hidden_dim)
        # concat global feature to each point
        point_features = torch.cat([features, global_feature.repeat(1, 1, N, 1)], dim=-1)  # (B, num_objs, N, 2*hidden_dim)
        stresses = self.output_layer(point_features) # (B, num_objs, N, output_dim)

        return stresses # (B, num_objs, N, output_dim)



