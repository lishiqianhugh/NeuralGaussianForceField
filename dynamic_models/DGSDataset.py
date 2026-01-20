import torch
from torch.utils.data import Dataset
import h5py
import json
import numpy as np
import os
import torch.nn.functional as F
from utils.transformation_utils import rotation_quaternion_to_matrix
from utils.visualize_utils import visualize_object_seg
from utils.gs_utils import load_params_from_gs
from utils.general_utils import find_indexes
from utils.gs_utils import GaussianModel
from dataset.constants import INDEX_OBJ, OBJPART
from utils.general_utils import searchForMaxIteration
from utils.geometry_utils import farthest_point_sampling_indices
import multiprocessing
import cv2
from tqdm import tqdm

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

def load_checkpoint(model_path, sh_degree=2, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


class DGSDataset(Dataset):
    def __init__(self, sequence_dirs, ply_dirs=None, preprocessing_params=None,
                 num_frames=100, num_keypoints=1024, k=8, chunk=1, view=0,
                 dtype=torch.float32, device='cpu', use_cache=True, 
                 cache_dir="cache"):
        self.sequence_dirs = sequence_dirs
        self.ply_dirs = ply_dirs
        self.num_frames = num_frames
        self.num_keypoints = num_keypoints
        self.k = k
        self.chunk = chunk
        self.view = view
        self.dtype = dtype
        self.device = device
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.data_cache = []

        max_workers = 1
        print(f"Using {max_workers} / {multiprocessing.cpu_count()} workers for loading sequences.")

        if ply_dirs and preprocessing_params: 
            for ply_dir in tqdm(self.ply_dirs):
                data = self._process_ply_dir(ply_dir, preprocessing_params)
                self.data_cache.append(data)
        else: 
            for seq_dir in tqdm(self.sequence_dirs):
                cache_path = self._get_cache_path(seq_dir)
                if self.use_cache and os.path.exists(cache_path):
                    data = torch.load(cache_path, map_location=self.device)
                    if data['pos'].shape[0] < self.num_frames:
                        print(f"Warning: {seq_dir} has only {data['pos'].shape[0]} frames, expected {self.num_frames}. Reprocessing...")
                        data = self._process_seq_dir(seq_dir)
                        if self.use_cache:
                            torch.save(data, cache_path)
                else:
                    data = self._process_seq_dir(seq_dir)
                    if self.use_cache:
                        torch.save(data, cache_path)
                # only use num_frames data
                for key in ['pos','cov3D','rot','points','com','com_vel','angle','angle_vel']:
                    if torch.is_tensor(data[key]) and data[key].shape[0] >= self.num_frames:
                        data[key] = data[key][:self.num_frames]
                data['path'] = seq_dir
                self.data_cache.append(data)

    def _get_cache_path(self, path):
        base = os.path.basename(path.rstrip("/"))
        return os.path.join(self.cache_dir, f"{base}.pt")
    
    def _process_ply_dir(self, ply_dir, preprocessing_params):
        positions = []
        covariances = []
        rotations = []
        keypoint_indices = None
        return_dict = {}
        # Load initial scene data
        gaussians = load_checkpoint(ply_dir)
        pipeline = PipelineParamsNoparse()
        pipeline.compute_cov3D_python = True

        # init the scene
        print(f"Initializing scene and pre-processing for {ply_dir}...")
        params = load_params_from_gs(gaussians, pipeline)

        init_pos = params["pos"].to(self.device)
        init_cov = params["cov3D_precomp"].to(self.device)
        init_rot = params["rotations"].to(self.device)
        init_screen_points = params["screen_points"].to(self.device)  # (N, 2)
        init_opacity = params["opacity"].to(self.device)  # (N, 1)
        init_shs = params["shs"].to(self.device)  # (N, 9)
        # throw away low opacity kernels
        opacity_mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
        init_pos = init_pos[opacity_mask, :]
        init_cov = init_cov[opacity_mask, :]
        init_rot = init_rot[opacity_mask, :]
        init_screen_points = init_screen_points[opacity_mask, :]
        init_opacity = init_opacity[opacity_mask, :]
        init_shs = init_shs[opacity_mask, :]

        # init_rot from 4 quaternions to rotation matrices 3, 3
        if init_rot.shape[-1] == 4:  # quaternion
            init_rot = rotation_quaternion_to_matrix(init_rot)  # (N, 3, 3)
        # original data
        return_dict['ori_pos'] = init_pos  # (N, 3)
        return_dict['ori_cov3D'] = init_cov
        return_dict['ori_rot'] = init_rot
        return_dict['screen_points'] = init_screen_points # (N, 2)
        return_dict['ori_opacity'] = init_opacity
        return_dict['ori_shs'] = init_shs  # (N, 9)
        obj_ids = self.group_objects(init_pos, opacity_mask, ply_dir)
        return_dict['ori_obj_ids'] = obj_ids
                    
        # downsampled data
        if self.num_keypoints is not None and init_pos.shape[0] >= self.num_keypoints:
            keypoint_indices = farthest_point_sampling_indices(init_pos, obj_ids, self.num_keypoints)
        elif self.num_keypoints is None:
            keypoint_indices = torch.arange(0, init_pos.shape[0], dtype=torch.long)
        else:
            raise ValueError(f"Not enough points in {ply_dir}")
        return_dict['keypoint_indices'] = keypoint_indices  # (N, 1)
        positions.append(init_pos[keypoint_indices].unsqueeze(0))
        covariances.append(init_cov[keypoint_indices].unsqueeze(0))
        rotations.append(init_rot[keypoint_indices].unsqueeze(0))

        return_dict['obj_ids'] = obj_ids[keypoint_indices]  # (N, 1)
        return_dict['pos'] = torch.cat(positions, dim=0)  # (1, N, 3)
        return_dict['cov3D'] = torch.cat(covariances, dim=0) # (1, N, 6)
        return_dict['rot'] = torch.cat(rotations, dim=0)  # (1, N, 3, 3)
        return_dict['opacity'] = init_opacity[keypoint_indices]
        return_dict['shs'] = init_shs[keypoint_indices]  # (N, 9)
        # visualize_object_seg(init_pos[keypoint_indices], obj_ids[keypoint_indices], ply_dir, 0)
        self.get_object_info(return_dict)
        return_dict['real_images'] = self.load_real_images(ply_dir)
        return_dict['path'] = ply_dir
        if return_dict['real_images'] is None:
            return_dict['real_images'] = torch.zeros((self.num_frames, 1280, 1280, 3), dtype=self.dtype)
        return return_dict
    
    def _process_seq_dir(self, seq_dir):
        # Load global attributes (opacity, shs)
        with h5py.File(os.path.join(seq_dir, "opacity.h5"), 'r') as f:
            opacity = torch.tensor(f['opacity'][()], dtype=self.dtype)

        with h5py.File(os.path.join(seq_dir, "shs.h5"), 'r') as f:
            shs = torch.tensor(f['shs'][()], dtype=self.dtype)

        positions = []
        covariances = []
        rotations = []
        keypoint_indices = None
        return_dict = {}

        last_frame = self.num_frames - 1
        for frame_idx in range(self.num_frames):
            if frame_idx > last_frame:
                frame_idx = last_frame
            frame_path = os.path.join(seq_dir, f"{frame_idx:04d}.h5")
            if not os.path.exists(frame_path):
                last_frame = frame_idx - 1
                frame_path = os.path.join(seq_dir, f"{last_frame:04d}.h5")
            with h5py.File(frame_path, 'r') as f:
                pos = torch.tensor(f['pos'][()], dtype=self.dtype)
                cov3D = torch.tensor(f['cov3D'][()].reshape(-1, 6), dtype=self.dtype)
                rot = torch.tensor(f['rot'][()].reshape(-1, 3, 3), dtype=self.dtype)

            if frame_idx == 0:
                obj_ids = self.group_objects(pos, None, seq_dir)
                if self.num_keypoints is not None and pos.shape[0] >= self.num_keypoints:
                    keypoint_indices = farthest_point_sampling_indices(pos, obj_ids, self.num_keypoints)
                elif self.num_keypoints is None:
                    keypoint_indices = torch.arange(0, pos.shape[0], dtype=torch.long)
                else:
                    raise ValueError(f"Not enough points in frame {frame_idx} of sequence {seq_dir}")

                sampled_opacity = opacity[keypoint_indices]
                sampled_shs = shs[keypoint_indices]
                return_dict['obj_ids'] = obj_ids[keypoint_indices]
                # visualize_object_seg(pos, obj_ids, seq_dir, frame_idx)

            positions.append(pos[keypoint_indices].unsqueeze(0))
            covariances.append(cov3D[keypoint_indices].unsqueeze(0))
            rotations.append(rot[keypoint_indices].unsqueeze(0))
        
        return_dict['pos'] = torch.cat(positions, dim=0) # (T, N, 3)
        return_dict['cov3D'] = torch.cat(covariances, dim=0)
        return_dict['rot'] = torch.cat(rotations, dim=0)
        return_dict['opacity'] = sampled_opacity
        return_dict['shs'] = sampled_shs
        return_dict['keypoint_indices'] = keypoint_indices
        self.get_object_info(return_dict)
        return return_dict

    def get_object_info(self, return_dict):
        # object centric pos
        num_objs = len(torch.unique(return_dict['obj_ids']))
        return_dict['points'] = torch.zeros((return_dict['pos'].shape[0], num_objs, self.num_keypoints, 3), dtype=self.dtype) # (T, num_objs, N, 3)
        return_dict['com'] = torch.zeros((return_dict['pos'].shape[0], num_objs, 3), dtype=self.dtype) # (T, num_objs, 3)
        return_dict['angle'] = torch.zeros_like(return_dict['com'])  # (T, N, 3)
        return_dict['padding'] = torch.ones((num_objs, self.num_keypoints), dtype=self.dtype) # (num_objs, N)
        return_dict['knn'] = torch.zeros((num_objs, self.num_keypoints, self.k), dtype=torch.long)
        for j in range(num_objs):
            # extract object points
            index = return_dict['obj_ids'] == torch.unique(return_dict['obj_ids'])[j]
            p = return_dict['pos'][:, index.squeeze(-1), :]  # (T, N, 3)
            return_dict['com'][:, j] = p.mean(dim=1) # (T, 3)
            
            num_points = p.shape[1]
            # padding to N points
            if num_points < self.num_keypoints:
                p = F.pad(p, (0, 0, 0, self.num_keypoints - num_points), mode='constant', value=0)
                # knn = F.pad(knn, (0, 0, 0, self.num_keypoints - num_points), mode='constant', value=0)
                return_dict['padding'][j][num_points:] = 0
            return_dict['points'][:, j] = p
            return_dict['knn'][j] = torch.zeros((self.num_keypoints, self.k), dtype=torch.long)

        return_dict['com_vel'] = return_dict['com'][1:] - return_dict['com'][:-1]
        return_dict['com_vel'] = torch.cat((torch.zeros_like(return_dict['com'][0:1]), return_dict['com_vel']), dim=0)
        return_dict['angle_vel'] = torch.zeros_like(return_dict['com_vel'])  # (T, N, 3)

    def group_objects(self, positions, opacity_mask, seq_dir):
        """
        Group point cloud into objects using file name.

        Args:
            positions (torch.Tensor): [N, 3] tensor of point positions.
            opacity_mask (torch.Tensor): [N, 1] tensor indicating which points are visible.

        Returns:
            obj_ids (torch.Tensor): [N, 1] tensor with object IDs assigned to each point.
        """
        if 'segmentation.npy' in os.listdir(seq_dir):
            # For testing, use segmentation file
            seg_path = os.path.join(seq_dir, 'segmentation.npy')
            obj_ids = torch.tensor(np.load(seg_path), dtype=torch.long).unsqueeze(-1)
            # the loaded object ids includes low opacity kernels, so we need to filter them out
            obj_ids = obj_ids[opacity_mask, :]
        else:
            # For training, use predefined object IDs based on sequence directory
            obj_ids = torch.zeros(positions.shape[0], dtype=torch.long).unsqueeze(-1)  # (N, 1)
            seq_dir = seq_dir.replace('mpm', 'scenes').replace('table_', '')
            scene_object_idxs = find_indexes(seq_dir)
            # filter objects to only those in the predefined list and keep the order
            objects = [INDEX_OBJ[i] for i in scene_object_idxs if i in INDEX_OBJ.keys()]
            objects = [obj for obj in objects if obj in OBJPART.keys()]
            current_point_num = 0
            for i in range(0, len(objects)):
                obj_ids[current_point_num : current_point_num + OBJPART[objects[i]]] = i
                current_point_num = current_point_num + OBJPART[objects[i]]
        
        return obj_ids

    def load_real_images(self, ply_dir):
        if '/sim' in ply_dir:
            ply_dir = '/'.join(ply_dir.split('/')[:-1])
        bg_name = ply_dir.split('/')[-1]
        scene_name = ply_dir.split('/')[-2]
        group_name = ply_dir.split('/')[-3]
        image_dir = f'./data/GSCollision/dynamic/{group_name}/{scene_name}/{bg_name}/view_{self.view}'
        video_path = os.path.join(image_dir, f'{scene_name}.mp4')
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} does not exist.")
            return None
        cap = cv2.VideoCapture(video_path)
        real_images = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(min(self.num_frames, frame_count)):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.resize(frame, (256, 256))
            frame = torch.tensor(frame, dtype=self.dtype)
            real_images.append(frame.unsqueeze(0))
        cap.release()
        real_images = torch.cat(real_images, dim=0)
        
        return real_images
    
    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]
    
class LazyDGSDataset(Dataset):
    def __init__(self, sequence_dirs, ply_dirs=None, preprocessing_params=None,
                 num_frames=100, num_keypoints=1024, k=8, chunk=1,
                 dtype=torch.float32, device='cpu', cache=False):
        self.sequence_dirs = sequence_dirs
        self.ply_dirs = ply_dirs
        self.preprocessing_params = preprocessing_params
        self.num_frames = num_frames
        self.num_keypoints = num_keypoints
        self.k = k
        self.chunk = chunk
        self.dtype = dtype
        self.device = device
        self.cache = cache 
        self._cache_dict = {} 

        print(f"[DGSDataset] lazy init with {len(sequence_dirs) if sequence_dirs else len(ply_dirs)} samples")

    def __len__(self):
        return len(self.sequence_dirs) if self.sequence_dirs else len(self.ply_dirs)

    def __getitem__(self, idx):
        if self.cache and idx in self._cache_dict:
            return self._cache_dict[idx]

        if self.ply_dirs and self.preprocessing_params:
            sample = self._process_ply_dir(self.ply_dirs[idx], self.preprocessing_params)
        else:
            sample = self._process_seq_dir(self.sequence_dirs[idx])

        if self.cache:
            self._cache_dict[idx] = sample

        return sample

    def _process_ply_dir(self, ply_dir, preprocessing_params):
        positions = []
        covariances = []
        rotations = []
        keypoint_indices = None
        return_dict = {}
        # Load initial scene data
        gaussians = load_checkpoint(ply_dir)
        pipeline = PipelineParamsNoparse()
        pipeline.compute_cov3D_python = True

        # init the scene
        print(f"Initializing scene and pre-processing for {ply_dir}...")
        params = load_params_from_gs(gaussians, pipeline)

        init_pos = params["pos"].to(self.device)
        init_cov = params["cov3D_precomp"].to(self.device)
        init_rot = params["rotations"].to(self.device)
        init_screen_points = params["screen_points"].to(self.device)  # (N, 2)
        init_opacity = params["opacity"].to(self.device)  # (N, 1)
        init_shs = params["shs"].to(self.device)  # (N, 9)
        # throw away low opacity kernels
        opacity_mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
        init_pos = init_pos[opacity_mask, :]
        init_cov = init_cov[opacity_mask, :]
        init_opacity = init_opacity[opacity_mask, :]
        init_screen_points = init_screen_points[opacity_mask, :]
        init_shs = init_shs[opacity_mask, :]

        # init_rot from 4 quaternions to rotation matrices 3, 3
        if init_rot.shape[-1] == 4:  # quaternion
            init_rot = rotation_quaternion_to_matrix(init_rot)  # (N, 3, 3)
        # original data
        return_dict['ori_pos'] = init_pos  # (N, 3)
        return_dict['ori_cov3D'] = init_cov
        return_dict['ori_rot'] = init_rot
        return_dict['screen_points'] = init_screen_points # (N, 2)
        return_dict['ori_opacity'] = init_opacity
        return_dict['ori_shs'] = init_shs  # (N, 9)
        obj_ids = self.group_objects(init_pos, opacity_mask, ply_dir)
 
        # downsampled data
        if self.num_keypoints is not None and init_pos.shape[0] >= self.num_keypoints:
            keypoint_indices = farthest_point_sampling_indices(init_pos, obj_ids, self.num_keypoints)
        elif self.num_keypoints is None:
            keypoint_indices = torch.arange(0, init_pos.shape[0], dtype=torch.long)
        else:
            raise ValueError(f"Not enough points in {ply_dir}")
        return_dict['keypoint_indices'] = keypoint_indices  # (N, 1)
        positions.append(init_pos[keypoint_indices].unsqueeze(0))
        covariances.append(init_cov[keypoint_indices].unsqueeze(0))
        rotations.append(init_rot[keypoint_indices].unsqueeze(0))

        return_dict['obj_ids'] = obj_ids[keypoint_indices]  # (N, 1)
        return_dict['pos'] = torch.cat(positions, dim=0)  # (1, N, 3)
        return_dict['cov3D'] = torch.cat(covariances, dim=0) # (1, N, 6)
        return_dict['rot'] = torch.cat(rotations, dim=0)  # (1, N, 3, 3)
        return_dict['opacity'] = init_opacity[keypoint_indices]
        return_dict['shs'] = init_shs[keypoint_indices]  # (N, 9)
        # visualize_object_seg(init_pos[keypoint_indices], obj_ids[keypoint_indices], ply_dir, 0)
        self.get_object_info(return_dict)
        return return_dict
    
    def _process_seq_dir(self, seq_dir):
        # Load global attributes (opacity, shs)
        with h5py.File(os.path.join(seq_dir, "opacity.h5"), 'r') as f:
            opacity = torch.tensor(f['opacity'][()], dtype=self.dtype)

        with h5py.File(os.path.join(seq_dir, "shs.h5"), 'r') as f:
            shs = torch.tensor(f['shs'][()], dtype=self.dtype)

        positions = []
        covariances = []
        rotations = []
        keypoint_indices = None
        return_dict = {}

        for frame_idx in range(self.num_frames):
            frame_path = os.path.join(seq_dir, f"{frame_idx:04d}.h5")
            with h5py.File(frame_path, 'r') as f:
                pos = torch.tensor(f['pos'][()], dtype=self.dtype)
                cov3D = torch.tensor(f['cov3D'][()].reshape(-1, 6), dtype=self.dtype)
                rot = torch.tensor(f['rot'][()].reshape(-1, 3, 3), dtype=self.dtype)

            if frame_idx == 0:
                obj_ids = self.group_objects(pos, None, seq_dir)
                if self.num_keypoints is not None and pos.shape[0] >= self.num_keypoints:
                    keypoint_indices = farthest_point_sampling_indices(pos, obj_ids, self.num_keypoints)
                elif self.num_keypoints is None:
                    keypoint_indices = torch.arange(0, pos.shape[0], dtype=torch.long)
                else:
                    raise ValueError(f"Not enough points in frame {frame_idx} of sequence {seq_dir}")

                sampled_opacity = opacity[keypoint_indices]
                sampled_shs = shs[keypoint_indices]
                return_dict['obj_ids'] = obj_ids[keypoint_indices]
                # visualize_object_seg(pos, obj_ids, seq_dir, frame_idx)

            positions.append(pos[keypoint_indices].unsqueeze(0))
            covariances.append(cov3D[keypoint_indices].unsqueeze(0))
            rotations.append(rot[keypoint_indices].unsqueeze(0))

        return_dict['pos'] = torch.cat(positions, dim=0) # (T, N, 3)
        return_dict['cov3D'] = torch.cat(covariances, dim=0)
        return_dict['rot'] = torch.cat(rotations, dim=0)
        return_dict['opacity'] = sampled_opacity
        return_dict['shs'] = sampled_shs
        return_dict['keypoint_indices'] = keypoint_indices
        self.get_object_info(return_dict)
        return return_dict

    def get_object_info(self, return_dict):
        # object centric pos
        num_objs = len(torch.unique(return_dict['obj_ids']))
        return_dict['points'] = torch.zeros((return_dict['pos'].shape[0], num_objs, self.num_keypoints, 3), dtype=self.dtype) # (T, num_objs, N, 3)
        return_dict['com'] = torch.zeros((return_dict['pos'].shape[0], num_objs, 3), dtype=self.dtype) # (T, num_objs, 3)
        return_dict['angle'] = torch.zeros_like(return_dict['com'])  # (T, N, 3)
        return_dict['padding'] = torch.ones((num_objs, self.num_keypoints), dtype=self.dtype) # (num_objs, N)
        return_dict['knn'] = torch.zeros((num_objs, self.num_keypoints, self.k), dtype=torch.long)
        for j in range(num_objs):
            # extract object points
            index = return_dict['obj_ids'] == torch.unique(return_dict['obj_ids'])[j]
            p = return_dict['pos'][:, index.squeeze(-1), :]  # (T, N, 3)
            return_dict['com'][:, j] = p.mean(dim=1) # (T, 3)
            
            num_points = p.shape[1]
            # padding to N points
            if num_points < self.num_keypoints:
                p = F.pad(p, (0, 0, 0, self.num_keypoints - num_points), mode='constant', value=0)
                # knn = F.pad(knn, (0, 0, 0, self.num_keypoints - num_points), mode='constant', value=0)
                return_dict['padding'][j][num_points:] = 0
            return_dict['points'][:, j] = p
            return_dict['knn'][j] = torch.zeros((self.num_keypoints, self.k), dtype=torch.long)

        return_dict['com_vel'] = return_dict['com'][1:] - return_dict['com'][:-1]
        return_dict['com_vel'] = torch.cat((torch.zeros_like(return_dict['com'][0:1]), return_dict['com_vel']), dim=0)
        return_dict['angle_vel'] = torch.zeros_like(return_dict['com_vel'])  # (T, N, 3)

    def group_objects(self, positions, opacity_mask, seq_dir):
        """
        Group point cloud into objects using file name.

        Args:
            positions (torch.Tensor): [N, 3] tensor of point positions.
            opacity_mask (torch.Tensor): [N, 1] tensor indicating which points are visible.

        Returns:
            obj_ids (torch.Tensor): [N, 1] tensor with object IDs assigned to each point.
        """
        if 'segmentation.npy' in os.listdir(seq_dir):
            # For testing, use segmentation file
            seg_path = os.path.join(seq_dir, 'segmentation.npy')
            obj_ids = torch.tensor(np.load(seg_path), dtype=torch.long).unsqueeze(-1)
            # the loaded object ids includes low opacity kernels, so we need to filter them out
            obj_ids = obj_ids[opacity_mask, :]
        else:
            # For training, use predefined object IDs based on sequence directory
            obj_ids = torch.zeros(positions.shape[0], dtype=torch.long).unsqueeze(-1)  # (N, 1)
            seq_dir = seq_dir.replace('mpm', 'scenes').replace('table_', '')
            scene_object_idxs = find_indexes(seq_dir)
            # filter objects to only those in the predefined list and keep the order
            objects = [INDEX_OBJ[i] for i in scene_object_idxs if i in INDEX_OBJ.keys()]
            objects = [obj for obj in objects if obj in OBJPART.keys()]
            current_point_num = 0
            print(len(objects))
            for i in range(0, len(objects)):
                obj_ids[current_point_num : current_point_num + OBJPART[objects[i]]] = i
                current_point_num = current_point_num + OBJPART[objects[i]]
        
        return obj_ids