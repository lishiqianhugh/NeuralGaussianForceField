import numpy as np
from typing import NamedTuple
import torch
import torch.nn.functional as F

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def farthest_point_sampling_indices(points, object_ids, k_per_object):
    """
    Farthest Point Sampling to select k points for each unique object ID
    
    Args:
        points: Tensor of shape (N, 3) containing point coordinates
        object_ids: Tensor of shape (N,) containing object ID for each point
        k_per_object: Number of points to sample per object
        
    Returns:
        Tensor of indices of sampled points
    """
    unique_ids = torch.unique(object_ids)
    all_indices = []
    for obj_id in unique_ids:
        # Get points belonging to this object
        obj_mask = object_ids == obj_id
        obj_points = points[obj_mask.squeeze(1)]
        obj_indices = torch.nonzero(obj_mask)[:, 0]
        
        N, _ = obj_points.shape
        # If object has fewer points than k, take all points
        if N <= k_per_object:
            # padding to k_per_object
            obj_indices = F.pad(obj_indices, (0, k_per_object - N), mode='constant', value=obj_indices[0])
            all_indices.append(obj_indices)
            continue
            
        # Perform FPS on this object's points
        centroids = torch.zeros(k_per_object, dtype=torch.long)
        distance = torch.ones(N) * 1e10
        farthest = torch.randint(0, N, (1,), dtype=torch.long)
        
        for i in range(k_per_object):
            centroids[i] = farthest
            centroid = obj_points[farthest, :].squeeze(0)
            dist = torch.sum((obj_points - centroid) ** 2, dim=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance)
        
        # Map local indices back to global indices
        sampled_indices = obj_indices[centroids]
        all_indices.append(sampled_indices)

    # Concatenate all sampled indices
    return torch.cat(all_indices)
