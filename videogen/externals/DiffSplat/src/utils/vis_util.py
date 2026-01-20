from typing import *
from PIL.Image import Image as PILImage
from numpy import ndarray
from torch import Tensor
from wandb import Image as WandbImage

from PIL import Image
import numpy as np
import torch
from einops import rearrange
import wandb


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def wandb_mvimage_log(outputs: Dict[str, Tensor], max_num: int = 4, max_view: int = 8) -> List[WandbImage]:
    """Organize multi-view images in Dict `outputs` for wandb logging.

    Only process values in Dict `outputs` that have keys containing the word "images",
    which should be in the shape of (B, V, 3, H, W).
    """
    formatted_images = []
    for k in outputs.keys():
        if "images" in k and outputs[k] is not None:  # (B, V, 3, H, W)
            assert outputs[k].ndim == 5
            num, view = outputs[k].shape[:2]
            num, view = min(num, max_num), min(view, max_view)
            mvimages = rearrange(outputs[k][:num, :view], "b v c h w -> c (b h) (v w)")
            formatted_images.append(
                wandb.Image(
                    tensor_to_image(mvimages.detach()),
                    caption=k
                )
            )

    return formatted_images


def tensor_to_image(tensor: Tensor, return_pil: bool = False) -> Union[ndarray, PILImage]:
    if tensor.ndim == 4:  # (B, C, H, W)
        tensor = rearrange(tensor, "b c h w -> c h (b w)")
    assert tensor.ndim == 3  # (C, H, W)

    assert tensor.shape[0] in [1, 3]  # grayscale, RGB (not consider RGBA here)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    image = (tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)  # (H, W, C)
    if return_pil:
        image = Image.fromarray(image)
    return image


def load_image(image_path: str, rgba: bool = False, imagenet_norm: bool = False) -> Tensor:
    image = Image.open(image_path)
    tensor_image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.  # (C, H, W) in [0, 1]

    if not rgba and tensor_image.shape[0] == 4:
        mask = tensor_image[3:4]
        tensor_image = tensor_image[:3] * mask + (1. - mask)  # white background

    if imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor_image.dtype, device=tensor_image.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor_image.dtype, device=tensor_image.device).view(3, 1, 1)
        tensor_image = (tensor_image - mean) / std

    return tensor_image  # (C, H, W)
