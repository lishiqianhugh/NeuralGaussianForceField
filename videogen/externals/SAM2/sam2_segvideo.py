import os
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import torch

def segment_image_sequence(
    images: List[np.ndarray],
    sam2_video_predictor,
    point_prompts: Optional[np.ndarray] = None,
    bbox_prompts: Optional[np.ndarray] = None,
    detection_frame_idx: int = 0,
) -> List[np.ndarray]:
    """
    使用 SAM2 的视频预测器和给定的正点提示对图片序列进行分割。

    Args:
        images: 列表，每个元素为 BGR 格式的 numpy 图像
        point_prompts: 可选，形状为 (N, 2) 的像素坐标数组，(x, y)，位于 detection_frame_idx 帧；每一行对应一个对象
        bbox_prompts: 可选，形状为 (M, 4) 的像素坐标数组，(xmin, ymin, xmax, ymax)，位于 detection_frame_idx 帧；每一行对应一个对象
        checkpoint_path: SAM2 模型权重路径
        model_cfg: SAM2 模型配置文件路径
        detection_frame_idx: 注入提示的帧索引（默认第 0 帧）

    Returns:
        List[np.ndarray]: mask 序列，每个 mask 是单通道图像，像素值为对象 ID (1..K)，K 为注入对象数量
    """
    assert len(images) > 0

    first_shape = images[0].shape[:2]

    assert torch.cuda.is_available(), "CUDA is not available"

    # 选择用于注入提示的帧
    detection_frame_idx = max(0, min(detection_frame_idx, len(images) - 1))

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
            cv2.imwrite(frame_path, img)

        # 构建 SAM2 视频预测器
        inference_state = sam2_video_predictor.init_state(video_path=temp_dir)

        # 规范化与校验提示（点与框任给其一，若两者皆为空则返回空掩码）
        num_objects = 0
        points = None
        boxes = None

        # 支持多种提示同时存在：points、boxes 检测框会共同注入

        if point_prompts is not None:
            points = np.asarray(point_prompts, dtype=np.float32)
            if points.ndim == 1 and points.shape[0] == 2:
                points = points[None, :]
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("point_prompts must have shape (N, 2) as (x, y) pixel coords.")
            if points.shape[0] == 0:
                points = None

        if bbox_prompts is not None:
            boxes = np.asarray(bbox_prompts, dtype=np.float32)
            if boxes.ndim == 1 and boxes.shape[0] == 4:
                boxes = boxes[None, :]
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("bbox_prompts must have shape (M, 4) as (xmin, ymin, xmax, ymax).")
            if boxes.shape[0] == 0:
                boxes = None

        # 混合精度/TF32（可选），按对象注入点或框
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            obj_id_counter = 1
            if points is not None:
                for idx, (x, y) in enumerate(points):
                    obj_id = int(obj_id_counter)
                    obj_id_counter += 1
                    pt = np.array([[float(x), float(y)]], dtype=np.float32)  # (1,2)
                    labels = np.ones((1,), dtype=np.int32)
                    _ = sam2_video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=detection_frame_idx,
                        obj_id=obj_id,
                        points=pt,
                        labels=labels,
                    )

            if boxes is not None:
                for idx in range(boxes.shape[0]):
                    obj_id = int(obj_id_counter)
                    obj_id_counter += 1
                    box = boxes[idx].astype(np.float32)
                    _ = sam2_video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=detection_frame_idx,
                        obj_id=obj_id,
                        box=box,
                    )

        # 在整个序列中传播 mask
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 生成最终的 mask 序列（像素值为 object ID）
        result_masks: List[np.ndarray] = []
        for frame_idx in range(len(images)):
            mask_result = np.zeros(first_shape, dtype=np.uint8)

            if frame_idx in video_segments:
                for obj_id, mask in video_segments[frame_idx].items():
                    mask_squeezed = mask.squeeze()

                    if mask_squeezed.shape != first_shape:
                        mask_squeezed = cv2.resize(
                            mask_squeezed.astype(np.uint8),
                            (first_shape[1], first_shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                    mask_result[mask_squeezed] = obj_id

            result_masks.append(mask_result)

        return result_masks
