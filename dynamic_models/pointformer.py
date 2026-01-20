import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTransformer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=3, num_heads=1, num_layers=4, dropout=0.1):
        super(PointTransformer, self).__init__()
    
        # Positional encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Point embedding layer
        self.feature_embedding = nn.Linear(input_dim*4, hidden_dim)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection
        self.point_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Normalization layers
        self.norm_input = nn.LayerNorm(hidden_dim)
        self.norm_output = nn.LayerNorm(hidden_dim)
    
    def step(self, points, padding):
        B, num_objs, num_keypoints, _ = points.shape
        # Create positional encoding based on point positions
        pos_encoding = self.pos_encoder(points) # Shape: [B, num_objs, num_keypoints, hidden_dim]

        # Apply transformer to model object interactions
        key_padding_mask = ~padding.reshape(B, num_objs*num_keypoints).bool()
        x = self.transformer_encoder(pos_encoding.reshape(B, num_objs * num_keypoints, -1), src_key_padding_mask=key_padding_mask)
        x = self.norm_output(x)
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        
        # Project to output dimension
        res_points = self.point_output(x)
        res_points = res_points.reshape(B, num_objs, num_keypoints, -1)
        return res_points
    
    def forward(self, points, com, com_vel, angle, angle_vel, padding, knn, pred_len=10, external_forces=None):
        points_seq = [points]
        for _ in range(pred_len):
            res_points = self.step(points, padding)
            points = points + res_points * padding.unsqueeze(-1) * 1e-1
            points_seq.append(points)
            
        points_seq = torch.stack(points_seq, dim=1)  # Shape: [B, steps, num_objs, num_keypoints, output_dim]

        return points_seq