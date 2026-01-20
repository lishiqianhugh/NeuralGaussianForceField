from typing import *
from PIL.Image import Image as PILImage
from torch import Tensor

import numpy as np
from skimage.metrics import structural_similarity as calculate_ssim
import torch
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor, CLIPVisionModelWithProjection,
    CLIPTokenizer, CLIPTextModelWithProjection,
)
import ImageReward as RM
from kiui.lpips import LPIPS


class TextConditionMetrics:
    def __init__(self,
        clip_name: str = "openai/clip-vit-base-patch32",
        rm_name: str = "ImageReward-v1.0",
        device_idx: int = 0,
    ):
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_name)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_name).to(f"cuda:{device_idx}").eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_name).to(f"cuda:{device_idx}").eval()

        self.rm_model = RM.load(rm_name)

        self.device = f"cuda:{device_idx}"

    @torch.no_grad()
    def evaluate(self,
        image: Union[PILImage, List[PILImage]],
        text: Union[str, List[str]],
    ) -> Tuple[float, float, float]:
        if isinstance(image, PILImage):
            image = [image]
        if isinstance(text, str):
            text = [text]

        assert len(image) == len(text)

        image_inputs = self.image_processor(image, return_tensors="pt").pixel_values.to(self.device)
        image_embeds = self.image_encoder(image_inputs).image_embeds.float()  # (N, D)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        text_embeds = self.text_encoder(text_input_ids).text_embeds.float()  # (N, D)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        assert image_embeds.shape == text_embeds.shape

        clip_scores = image_embeds @ text_embeds.T  # (N, N)

        # 1. CLIP similarity
        clip_sim = clip_scores.diag().mean().item()

        # 2. CLIP R-Precision
        clip_rprec = (clip_scores.argmax(dim=1) == torch.arange(len(text)).to(self.device)).float().mean().item()

        # 3. ImageReward
        rm_scores = []
        for img, txt in zip(image, text):
            rm_scores.append(self.rm_model.score(txt, img))
        rm_scores = torch.tensor(rm_scores, device=self.device)
        rm_score = rm_scores.mean().item()

        return clip_sim, clip_rprec, rm_score


class ImageConditionMetrics:
    def __init__(self,
        lpips_net: str = "vgg",
        lpips_res: int = 256,
        device_idx: int = 0,
    ):
        self.lpips_loss = LPIPS(net=lpips_net).to(f"cuda:{device_idx}").eval()

        self.lpips_res = lpips_res
        self.device = f"cuda:{device_idx}"

    @torch.no_grad()
    def evaluate(self,
        image: Union[Tensor, PILImage, List[PILImage]],
        gt: Union[Tensor, PILImage, List[PILImage]],
        chunk_size: Optional[int] = None,
        input_tensor: bool = False,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        if not input_tensor:
            if isinstance(image, PILImage):
                image = [image]
            if isinstance(gt, PILImage):
                gt = [gt]

            assert len(image) == len(gt)

            if chunk_size is None:
                chunk_size = len(image)

            def image_to_tensor(img: PILImage):
                return torch.tensor(np.array(img).transpose(2, 0, 1) / 255., device=self.device).unsqueeze(0).float()  # (1, 3, H, W)
            image_pt = torch.cat([image_to_tensor(img) for img in image], dim=0)
            gt_pt = torch.cat([image_to_tensor(img) for img in gt], dim=0)
        else:
            image_pt = image.to(device=self.device)
            gt_pt = gt.to(device=self.device)

        # 1. LPIPS
        lpips = []
        for i in range(0, len(image), chunk_size):
            _lpips = self.lpips_loss(
                F.interpolate(
                    image_pt[i:min(len(image), i+chunk_size)] * 2. - 1.,
                    (self.lpips_res, self.lpips_res), mode="bilinear", align_corners=False
                ),
                F.interpolate(
                    gt_pt[i:min(len(image), i+chunk_size)] * 2. - 1.,
                    (self.lpips_res, self.lpips_res), mode="bilinear", align_corners=False
                )
            )
            lpips.append(_lpips)
        lpips = torch.cat(lpips)
        lpips_mean, lpips_std = lpips.mean().item(), lpips.std().item()

        # 2. PSNR
        psnr = -10. * torch.log10((gt_pt - image_pt).pow(2).mean(dim=[1, 2, 3]))
        psnr_mean, psnr_std = psnr.mean().item(), psnr.std().item()

        # 3. SSIM
        ssim = []
        for i in range(len(image)):
            _ssim = calculate_ssim(
                (image_pt[i].cpu().float().numpy() * 255.).astype(np.uint8),
                (gt_pt[i].cpu().float().numpy() * 255.).astype(np.uint8),
                channel_axis=0,
            )
            ssim.append(_ssim)
        ssim = np.array(ssim)
        ssim_mean, ssim_std = ssim.mean(), ssim.std()

        return (psnr_mean, psnr_std), (ssim_mean, ssim_std), (lpips_mean, lpips_std)
