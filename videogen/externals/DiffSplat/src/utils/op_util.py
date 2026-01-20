from typing import *
from torch import Tensor

import os
from einops import rearrange


def rembg_and_center_wrapper(
    image_path: str, image_size: int,
    border_ratio: float, center: bool = True,
    model_name: str = "u2net",  # see https://github.com/danielgatis/rembg#models
) -> str:
    """Run `extensions/rembg_and_center.py` to remove background and center the image, and return the path to the new image."""
    os.system(
        f"python3 extensions/rembg_and_center.py {image_path}" +
        f" --size {image_size} --border_ratio {border_ratio} --model {model_name}" +
        f" --center" if center else ""
    )
    directory, _ = os.path.split(image_path)
    file_base = os.path.basename(image_path).split(".")[0]
    new_filename = f"{file_base}_rgba.png"
    new_image_path = os.path.join(directory, new_filename)
    return new_image_path


def patchify(x: Tensor, patch_size: Union[int, Tuple[int, int]], tokenize: bool = True):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    p1, p2 = patch_size
    if tokenize:
        return rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p1, p2=p2)
    else:
        return rearrange(x, "b c (h p1) (w p2) -> b (p1 p2 c) h w", p1=p1, p2=p2)


def unpatchify(x: Tensor, patch_size: Union[int, Tuple[int, int]], input_size: Union[int, Tuple[int, int]], tokenize: bool = True):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    (p1, p2), (h, w) = patch_size, input_size
    if tokenize:
        return rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=p1, p2=p2)
    else:
        return rearrange(x, "b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=p1, p2=p2)
