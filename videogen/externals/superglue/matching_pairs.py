from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import sys
import os
import cv2
import torch.nn.functional as F
from tqdm import tqdm
from .matching import Matching
from .utils import (make_matching_plot, read_image,)

def prepare_superglue(max_keypoints=1024,keypoint_threshold=0.005,nms_radius=4,sinkhorn_iterations=20,match_threshold=0.2,superglue='indoor',device='cuda'):
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    return matching

@torch.no_grad()
def match_image_lists(matching, image_list_a, image_list_b, output_dir, resize=[-1],
                      resize_float=False, viz=True, show_keypoints=False,
                      viz_extension='png'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device "{}"'.format(device))

    if viz:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        print('Will write visualization images to directory "{}"'.format(output_dir))

    def _to_gray_uint8(img):
        try:
            from PIL import Image
            if isinstance(img, Image.Image):
                return np.array(img.convert('L'))
        except Exception:
            pass
        if isinstance(img, np.ndarray):
            arr = img
            if arr.ndim == 3:
                # assume RGB/RGBA
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.float32)
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
                    else:
                        arr = arr.round().clip(0, 255).astype(np.uint8)
                if arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                elif arr.shape[2] == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
                else:
                    arr = arr[..., 0]
                return arr
            elif arr.ndim == 2:
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
                    else:
                        arr = arr.round().clip(0, 255).astype(np.uint8)
                return arr
        raise TypeError('Unsupported image type for match_image_lists')

    # Ensure list types
    list_a = list(image_list_a)
    list_b = list(image_list_b)

    match_nums = []
    match_results = []

    # Always use product mode: all combinations
    iterable = []
    for ia, a in enumerate(list_a):
        for ib, b in enumerate(list_b):
            iterable.append((ia, ib, _to_gray_uint8(a), _to_gray_uint8(b), f'pair_{ia}_{ib}'))

    # Track best per A (product mode semantics)
    best_per_a = [
            {
                'count': -1,
                'out_matches': None,
                'image0': None,
                'image1': None,
                'mkpts0': None,
                'mkpts1': None,
                'mconf': None,
                'name': None,
                'ia': None,
                'ib': None,
            }
            for _ in range(len(list_a))
        ]

    for ia, ib, image0, image1, name in iterable:

        rot0, rot1 = 0, 0
        image0_gray, inp0, scales0 = read_image(image0, device, resize, rot0, resize_float)
        image1_gray, inp1, scales1 = read_image(image1, device, resize, rot1, resize_float)

        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                       'matches': matches, 'match_confidence': conf}

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        match_count = len(mkpts0)

        # product mode: only keep best per image A
        if match_count > best_per_a[ia]['count']:
            best_per_a[ia] = {
                'count': match_count,
                'out_matches': out_matches,
                'image0': image0_gray,
                'image1': image1_gray,
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
                'mconf': mconf,
                'name': name,
                'ia': ia,
                'ib': ib,
            }

    # Finalize for product mode: save only best per A
    best_pairs = []
    for best in best_per_a:
        if best['out_matches'] is None:
            continue
        best_pairs.append(best)
        match_results.append(best['out_matches'])
        match_nums.append(best['count'])
        name = best['name']

        if viz:
            viz_path = output_dir / f'{name}.{viz_extension}'
            color = cm.jet(best['mconf'])
            text = [
                'SuperGlue',
                'Matches: {}'.format(best['count']),
            ]
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(matching.superpoint.config['keypoint_threshold']),
                'Match Threshold: {:.2f}'.format(matching.superglue.config['match_threshold']),
                'Pair: {}'.format(name),
            ]
            make_matching_plot(
                best['image0'], best['image1'],
                best['out_matches']['keypoints0'], best['out_matches']['keypoints1'],
                best['mkpts0'], best['mkpts1'], color,
                text, viz_path, show_keypoints, small_text)

    best_pose = int(np.argmax(match_nums)) if len(match_nums) > 0 else -1
    best_result = match_results[best_pose] if best_pose >= 0 else None
    return best_pose, best_result, best_pairs


@torch.no_grad()
def match_image_lists_gpu(matching, image_list_a, image_list_b, output_dir, resize=[-1],
                          resize_float=False, viz=True, show_keypoints=False,
                          viz_extension='png', batch_size=50):

    device = 'cuda'

    if viz:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        print('Will write visualization images to directory "{}"'.format(output_dir))

    def _to_tensor_on_gpu(img):
        try:
            from PIL import Image
            if isinstance(img, Image.Image):
                img = np.array(img)
        except Exception:
            pass
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img)
        elif torch.is_tensor(img):
            t = img
        else:
            raise TypeError('Unsupported image type for match_image_lists_gpu')

        # Ensure H x W x C or H x W
        if t.ndim == 3:
            # If channels-first, move to channels-last
            if t.shape[0] in (1, 3, 4) and (t.shape[2] != 1 and t.shape[2] != 3 and t.shape[2] != 4):
                t = t.permute(1, 2, 0)
        return t.to(device)

    def _rgb_to_gray01_gpu(t_hw_or_hw3):
        # t on device, dtype could be uint8/float/others
        if t_hw_or_hw3.dtype.is_floating_point:
            x = t_hw_or_hw3.float()
            maxv = float(torch.amax(x)) if x.numel() > 0 else 1.0
            if maxv > 1.0 + 1e-6:
                x = x / 255.0
        else:
            x = t_hw_or_hw3.float() / 255.0

        if x.ndim == 2:
            # already grayscale HxW in [0,1]
            g = x
        elif x.ndim == 3:
            # assume last dim channels
            if x.shape[-1] == 1:
                g = x[..., 0]
            elif x.shape[-1] >= 3:
                # RGB to Gray
                r = x[..., 0]
                g_ = x[..., 1]
                b = x[..., 2]
                g = 0.2989 * r + 0.5870 * g_ + 0.1140 * b
            else:
                # Fallback: take first channel
                g = x[..., 0]
        else:
            raise ValueError('Unexpected tensor shape for image: {}'.format(tuple(x.shape)))
        return g.clamp(0.0, 1.0)

    def _process_resize_dims(w, h, resize_arg):
        assert(len(resize_arg) > 0 and len(resize_arg) <= 2)
        if len(resize_arg) == 1 and resize_arg[0] > -1:
            scale = resize_arg[0] / max(h, w)
            w_new, h_new = int(round(w * scale)), int(round(h * scale))
        elif len(resize_arg) == 1 and resize_arg[0] == -1:
            w_new, h_new = w, h
        else:
            w_new, h_new = resize_arg[0], resize_arg[1]
        return w_new, h_new

    def _resize_gray01_gpu(gray01, w_new, h_new):
        # gray01: HxW in [0,1] float32/float64
        inp = gray01[None, None, :, :]
        out = F.interpolate(inp, size=(h_new, w_new), mode='bilinear', align_corners=False)
        return out[0, 0]

    # Ensure list types
    list_a = list(image_list_a)
    list_b = list(image_list_b)

    match_nums = []
    match_results = []
    best_pairs = []

    # Always use product mode: all combinations
    iterable = []
    for ia, a in enumerate(list_a):
        for ib, b in enumerate(list_b):
            iterable.append((ia, ib, a, b, f'pair_{ia}_{ib}'))

    # Preprocess all images to GPU tensors, and compute resized inputs and viz images
    def preprocess_list(img_list):
        proc = {
            'gray_uint8': [],   # for viz (numpy uint8)
            'inp': [],          # torch [1,1,H,W] on device in [0,1]
            'shape': [],        # (H, W)
            'scales': [],       # (sx, sy)
        }
        for img in img_list:
            t = _to_tensor_on_gpu(img)
            g01 = _rgb_to_gray01_gpu(t)
            h, w = int(g01.shape[0]), int(g01.shape[1])
            w_new, h_new = _process_resize_dims(w, h, resize)
            scales = (float(w) / float(w_new), float(h) / float(h_new))
            g01r = _resize_gray01_gpu(g01, w_new, h_new)
            inp = g01r[None, None, :, :].contiguous()
            if viz:
                gray_u8 = (g01r * 255.0).round().clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
                proc['gray_uint8'].append(gray_u8)
            else:
                proc['gray_uint8'].append(None)
            proc['inp'].append(inp)
            proc['shape'].append((int(g01r.shape[0]), int(g01r.shape[1])))
            proc['scales'].append(scales)
        return proc
    

    proc_a = preprocess_list(list_a)
    proc_b = preprocess_list(list_b)

    def run_superpoint_batched(proc):
        # Build buckets by (H, W)
        buckets = {}
        for idx, (h, w) in enumerate(proc['shape']):
            buckets.setdefault((h, w), []).append(idx)
        kpts_list, scores_list, desc_list = [None] * len(proc['inp']), [None] * len(proc['inp']), [None] * len(proc['inp'])
        for (h, w), indices in buckets.items():
            for start in range(0, len(indices), max(1, int(batch_size))):
                chunk_idx = indices[start:start + int(batch_size)]
                batch = torch.cat([proc['inp'][i] for i in chunk_idx], dim=0)  # [B,1,H,W]
                sp_pred = matching.superpoint({'image': batch})
                for local_i, global_i in enumerate(chunk_idx):
                    kpts_list[global_i] = sp_pred['keypoints'][local_i]
                    scores_list[global_i] = sp_pred['scores'][local_i]
                    desc_list[global_i] = sp_pred['descriptors'][local_i]
        return kpts_list, scores_list, desc_list

    kpts_a, scores_a, desc_a = run_superpoint_batched(proc_a)
    kpts_b, scores_b, desc_b = run_superpoint_batched(proc_b)

    # Track best per A (product semantics)
    best_per_a = [
        {
            'count': -1,
            'out_matches': None,
            'image0': None,
            'image1': None,
            'mkpts0': None,
            'mkpts1': None,
            'mconf': None,
            'name': None,
            'ia': None,
            'ib': None,
        }
        for _ in range(len(list_a))
    ]

    for ia, a_img in enumerate(list_a):
        img0 = proc_a['inp'][ia]
        k0 = kpts_a[ia]
        s0 = scores_a[ia]
        d0 = desc_a[ia]

        # Build batch from all images in B
        B = len(list_b)
        k1_list = [kpts_b[ib] for ib in range(B)]
        s1_list = [scores_b[ib] for ib in range(B)]
        d1_list = [desc_b[ib] for ib in range(B)]

        # Determine max number of keypoints in this batch for padding
        n0 = k0.shape[0]
        n1_max = max([k.shape[0] for k in k1_list]) if B > 0 else 0

        # Prepare padded tensors
        C = d0.shape[0]

        # pad keypoints: k0 -> [B, n0, 2], k1_padded -> [B, n1_max, 2]
        kpts0_b = k0[None].repeat(B, 1, 1)
        kpts1_padded = torch.zeros((B, n1_max, 2), device=k0.device, dtype=k0.dtype)
        scores1_padded = torch.zeros((B, n1_max), device=s1_list[0].device, dtype=s1_list[0].dtype)
        desc1_padded = torch.zeros((B, C, n1_max), device=d1_list[0].device, dtype=d1_list[0].dtype)
        for ib in range(B):
            kn = k1_list[ib].shape[0]
            if kn > 0:
                kpts1_padded[ib, :kn] = k1_list[ib]
                scores1_padded[ib, :kn] = s1_list[ib]
                desc1_padded[ib, :, :kn] = d1_list[ib]

        # pad descriptors/keypoints for image0 to match batch dims (insert batch dim for desc0)
        # repeat descriptors/scores of image0 to match batch dim B
        desc0_padded = d0[None].repeat(B, 1, 1)
        scores0_padded = s0[None].repeat(B, 1)
        # image0 needs to be repeated B times for shape info
        img0_batch = torch.cat([img0 for _ in range(B)], dim=0)

        data = {
            'image0': img0_batch,
            'image1': torch.cat([proc_b['inp'][ib] for ib in range(B)], dim=0),
            'keypoints0': kpts0_b,  # [1, n0, 2]
            'keypoints1': kpts1_padded,  # [B, n1_max, 2]
            'scores0': scores0_padded,  # [1, n0]
            'scores1': scores1_padded,  # [B, n1_max]
            'descriptors0': desc0_padded,  # [1, C, n0]
            'descriptors1': desc1_padded,  # [B, C, n1_max]
        }

        for img_key in ('image0', 'image1'):
            img_t = data[img_key]
            if img_t.dim() > 4:
                leading = int(np.prod(img_t.shape[:-3]))
                c, h, w = img_t.shape[-3], img_t.shape[-2], img_t.shape[-1]
                data[img_key] = img_t.reshape(leading, c, h, w)

        sg_pred = matching.superglue(data)

        matches0_batch = sg_pred['matches0']
        scores0_batch = sg_pred['matching_scores0']

        for ib in range(B):
            # detach and copy to numpy for safe in-place edits
            matches0 = matches0_batch[ib].detach().cpu().numpy().copy()
            mscore0 = scores0_batch[ib].detach().cpu().numpy().copy()

            kpts0_np = k0.detach().cpu().numpy()
            kpts1_np = k1_list[ib].detach().cpu().numpy()

            # In padded-batch mode SuperGlue may return indices referring to
            # padded slots (>= len(kpts1_np)). Treat those as invalid (-1).
            if matches0.size > 0:
                valid_index_mask = (matches0 >= 0) & (matches0 < kpts1_np.shape[0])
                if not valid_index_mask.all():
                    matches0[~valid_index_mask] = -1
                    mscore0[~valid_index_mask] = 0.0

            out_matches = {
                'keypoints0': kpts0_np,
                'keypoints1': kpts1_np,
                'matches': matches0,
                'match_confidence': mscore0,
            }

            # valid matches are those >=0 after clamping
            valid = matches0 > -1
            mkpts0 = kpts0_np[valid]
            mkpts1 = kpts1_np[matches0[valid]]
            mconf = mscore0[valid]
            match_count = len(mkpts0)

            name = f'pair_{ia}_{ib}'
            # product mode: only keep best per image A
            if match_count > best_per_a[ia]['count']:
                best_per_a[ia] = {
                    'count': match_count,
                    'out_matches': out_matches,
                    'image0': proc_a['gray_uint8'][ia],
                    'image1': proc_b['gray_uint8'][ib],
                    'mkpts0': mkpts0,
                    'mkpts1': mkpts1,
                    'mconf': mconf,
                    'name': name,
                    'ia': ia,
                    'ib': ib,
                }

    # Finalize: save only best per A
    for best in best_per_a:
        if best['out_matches'] is None:
            continue
        best_pairs.append(best)
        match_results.append(best['out_matches'])
        match_nums.append(best['count'])
        name = best['name']
        if viz:
            viz_path = output_dir / f'{name}.{viz_extension}'
            color = cm.jet(best['mconf'])
            text = [
                'SuperGlue',
                'Matches: {}'.format(best['count']),
            ]
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(matching.superpoint.config['keypoint_threshold']),
                'Match Threshold: {:.2f}'.format(matching.superglue.config['match_threshold']),
                'Pair: {}'.format(name),
            ]
            make_matching_plot(
                best['image0'], best['image1'],
                best['out_matches']['keypoints0'], best['out_matches']['keypoints1'],
                best['mkpts0'], best['mkpts1'], color,
                text, viz_path, show_keypoints, small_text)

    best_pose = int(np.argmax(match_nums)) if len(match_nums) > 0 else -1
    best_result = match_results[best_pose] if best_pose >= 0 else None
    return best_pose, best_result, best_pairs