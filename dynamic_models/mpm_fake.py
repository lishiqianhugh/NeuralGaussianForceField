import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .DGSDataset import DGSDataset

class MPMSimulator(nn.Module):
    def __init__(self, data_path, groups, sample_num, steps, num_keypoints, chunk, dtype, batch_size, cache_dir):
        super(MPMSimulator, self).__init__()
        self.data_path = data_path
        self.groups = groups
        self.sample_num = sample_num
        self.steps = steps
        self.num_keypoints = num_keypoints
        self.chunk = chunk
        self.dtype = dtype
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        scenes = [
            os.path.join(self.data_path, group, scene)
            for group in self.groups
            for scene in os.listdir(os.path.join(self.data_path, group))
            if not scene.startswith('.')
        ]
        scenes = scenes[:self.sample_num]

        self.dataset = DGSDataset(
            scenes,
            num_frames=self.steps,
            num_keypoints=self.num_keypoints,
            chunk=self.chunk,
            dtype=self.dtype,
            cache_dir=self.cache_dir
        )
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def forward(self, *args, **kwargs):
        try:
            batch = next(self._iter)
        except (AttributeError, StopIteration):
            self._iter = iter(self.dataloader)
            batch = next(self._iter)
        # if has nan
        if torch.isnan(batch['points']).any():
            print("Warning: NaN values found in the batch points.")
            batch['points'][torch.isnan(batch['points'])] = 0.0

        # shape: [B, T, num_obj, N, 3] or [B, T, N, 3]
        return batch['points'].to('cuda' if torch.cuda.is_available() else 'cpu')
