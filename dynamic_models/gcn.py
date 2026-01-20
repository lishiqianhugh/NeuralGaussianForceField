import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, radius_graph
from torch_geometric.data import Data

# Step 1: Define the Graph Network Layer (same as before)
class GraphNetworkLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphNetworkLayer, self).__init__(aggr='add')  # Aggregation method: sum
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Apply GCNConv layer (message passing)
        x = self.conv(x, edge_index)
        return x

# Step 2: Define the full Graph Network (same as before)
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, r=0.1):
        super(GCN, self).__init__()
        self.GCN = []
        # Define the layers
        for i in range(num_layers):
            if i == 0:
                self.GCN.append(GraphNetworkLayer(input_dim*2, hidden_dim))

            else:
                self.GCN.append(GraphNetworkLayer(hidden_dim, hidden_dim))
        self.GCN = nn.ModuleList(self.GCN)
        self.fc = nn.Linear(hidden_dim, output_dim*2)  # Output pos and vel vector for each node
        self.r = r  # Radius for edge construction

    def forward(self, points, com, com_vel, angle, angle_vel, padding, knn, pred_len=10, external_forces=None):
        # edge_index = radius_graph(points, r=self.r)
        batch_size, num_objs, num_keypoints, feature_dim = points.shape
        num_nodes = num_objs * num_keypoints
        velociteis = points.clone() * 0
        device = points.device
        
        # Concatenate points and velocities
        states = torch.cat([points, velociteis], dim=-1)
        traj = [states]
        for _ in range(pred_len):
            # Reshape inputs to handle batches in torch_geometric format
            batch_pos = states[...,:3].reshape(batch_size * num_nodes, -1)  # [batch_size*num_nodes, 3]
            
            # Create batch tensor for torch_geometric
            batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
            
            # Construct edge index using radius_graph
            edge_index = radius_graph(batch_pos, r=self.r, batch=batch_idx)
            src, dst = edge_index
            valid_edges = padding.reshape(batch_size * num_nodes).bool()[src] & padding.reshape(batch_size * num_nodes).bool()[dst]
            edge_index = edge_index[:, valid_edges]
            
            x = states.reshape(batch_size * num_nodes, -1)
            # Forward pass through the network
            for layer in self.GCN:
                x = F.relu(layer(x, edge_index))
            res = self.fc(x)  # Predict next state residual
            res = res.reshape(batch_size, num_objs, num_keypoints, -1)
            states = states + res * padding.unsqueeze(-1)
            traj.append(states)
        traj = torch.stack(traj, dim=1)
        return traj[...,:3]