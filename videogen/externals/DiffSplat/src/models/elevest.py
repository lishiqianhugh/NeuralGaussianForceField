from typing import *
from torch import Tensor

import torch
from torch import nn
import torch.nn.functional as tF
from einops import rearrange

from src.options import Options
from src.utils import IMAGENET_MEAN, IMAGENET_STD


class ElevEst(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        # self.backbone: nn.Module = torch.hub.load("facebookresearch/dinov2", opt.elevest_backbone_name)
        # self.backbone: nn.Module = torch.hub.load("/", "dinov2_vitb14_reg")
        self.backbone: nn.Module = torch.hub.load('/mnt/nfs_project_a/shared/models/facebook/dinov2',opt.elevest_backbone_name,source='local',pretrained=False)
        self.backbone.load_state_dict(torch.load('/mnt/nfs_project_a/shared/models/facebook/dinov2_vitb14_reg4_pretrain.pth', map_location='cpu'))

        if opt.freeze_backbone:
            self.backbone.requires_grad_(False)
        else:
            self.backbone.mask_token.requires_grad_(False)  # not used

        self.dim = dim = {
            "dinov2_vits14_reg": 384,
            "dinov2_vitb14_reg": 768,
            "dinov2_vitl14_reg": 1024,
        }[opt.elevest_backbone_name]

        self.cls_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, opt.elevest_num_classes),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        self.interval = (opt.ele_max - opt.ele_min) / opt.elevest_num_classes
        self.register_buffer("lower_bounds", torch.linspace(opt.ele_min, opt.ele_max, opt.elevest_num_classes+1)[:-1])

    def state_dict(self, **kwargs):
        # Remove frozen parameters without gradients
        state_dict = super().state_dict(**kwargs)
        if self.opt.freeze_backbone:
            for k in list(state_dict.keys()):
                if "backbone" in k:
                    del state_dict[k]
        return state_dict

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss(self, data: Dict[str, Tensor], dtype: torch.dtype = torch.float32):
        outputs = {}

        input_images = data["image"].to(dtype)  # (B, V, 3, H, W)
        gt_elev = data["cam_pose"].to(dtype)[:, :, 0].rad2deg()  # (B, V)

        input_images = rearrange(input_images, "b v c h w -> (b v) c h w")
        gt_elev = rearrange(gt_elev, "b v -> (b v)")  # (B*V,)

        gt_class = torch.floor((gt_elev - self.opt.ele_min) / self.interval).long()
        gt_offset = gt_elev - self.lower_bounds[gt_class]
        assert torch.all((gt_class >= 0) & (gt_class < self.opt.elevest_num_classes))
        assert torch.all((gt_offset + 1e-8 >= 0) & (gt_offset - 1e-8 < self.interval))

        # ImageNet normalization
        mean = torch.tensor(IMAGENET_MEAN, device=input_images.device, dtype=dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=input_images.device, dtype=dtype).view(3, 1, 1)
        input_images = (input_images - mean) / std

        # Predict
        features = self.backbone(input_images.to(dtype=dtype), is_training=True)
        cls_token = features["x_norm_clstoken"]  # (B*V, D)
        logits = self.cls_head(cls_token)  # (B*V, C)
        pred_offset = self.offset_head(cls_token).squeeze(-1).clamp(0., self.interval)  # (B*V,)

        # Loss
        outputs["loss_cls"] = tF.cross_entropy(logits, gt_class)
        outputs["loss_offset"] = tF.mse_loss(pred_offset, gt_offset)
        outputs["loss"] = outputs["loss_cls"] + self.opt.elevest_reg_weight * outputs["loss_offset"]

        with torch.no_grad():
            pred_elev = self.lower_bounds[torch.argmax(logits, dim=-1)] + pred_offset  # (B*V,)
            outputs["err_degree"] = torch.mean(torch.abs(pred_elev - gt_elev))

        return outputs

    @torch.no_grad()
    def predict_elev(self, input_images: Tensor, dtype: torch.dtype = torch.float32):
        # Input image preprocessing
        input_images = tF.interpolate(input_images, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        input_images = input_images.to(device=self.lower_bounds.device, dtype=dtype)
        mean = torch.tensor(IMAGENET_MEAN, device=input_images.device, dtype=dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=input_images.device, dtype=dtype).view(3, 1, 1)
        input_images = (input_images - mean) / std

        features = self.backbone(input_images, is_training=True)
        cls_token = features["x_norm_clstoken"]  # (B, D)
        logits = self.cls_head(cls_token)  # (B, C)
        pred_offset = self.offset_head(cls_token).squeeze(-1).clamp(0., self.interval)  # (B,)

        pred_elev = self.lower_bounds[torch.argmax(logits, dim=-1)] + pred_offset  # (B,)
        return pred_elev
