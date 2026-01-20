from typing import *
from torch import Tensor
from lpips import LPIPS

from skimage.metrics import structural_similarity as calculate_ssim
import numpy as np
from torch import nn
import torch.nn.functional as tF
from einops import rearrange

from src.models.networks.attention import *
from src.models.gs_render import GaussianRenderer
from src.options import Options
from src.utils import plucker_ray, patchify, unpatchify


class GSRecon(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        # Image tokenizer
        in_channels = 3 + 6  # RGB + plucker
        if opt.input_normal:
            in_channels += 3
        if opt.input_coord:
            in_channels += 3
        if opt.input_mr:
            in_channels += 2
        self.x_embedder = nn.Linear(in_channels * (opt.patch_size**2), opt.dim)

        # Transformer backbone
        self.transformer = Transformer(opt.num_blocks, opt.dim, opt.num_heads, llama_style=opt.llama_style)
        self.ln_out = nn.LayerNorm(opt.dim)
        if opt.grad_checkpoint:
            self.transformer.set_grad_checkpointing()

        # Output heads
        self.inter_res = opt.input_res // opt.patch_size
        self.out_depth = nn.Linear(opt.dim, 1 * (opt.patch_size**2), bias=False)
        self.out_rgb = nn.Linear(opt.dim, 3 * (opt.patch_size**2), bias=False)
        self.out_scale = nn.Linear(opt.dim, 3 * (opt.patch_size**2), bias=False)
        self.out_rotation = nn.Linear(opt.dim, 4 * (opt.patch_size**2), bias=False)
        self.out_opacity = nn.Linear(opt.dim, 1 * (opt.patch_size**2), bias=False)

        # Rendering
        self.gs_renderer = GaussianRenderer(opt)

        # Initialize weights
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.zeros_(self.out_depth.weight)  # zero init.
        nn.init.xavier_uniform_(self.out_rgb.weight)
        nn.init.zeros_(self.out_scale.weight)  # zero init.
        nn.init.xavier_uniform_(self.out_rotation.weight)
        nn.init.zeros_(self.out_opacity.weight)  # zero init.

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss(self, data: Dict[str, Tensor], lpips_loss: LPIPS, step: int, dtype: torch.dtype = torch.float32):
        outputs = {}

        color_name = "albedo" if self.opt.input_albedo else "image"

        images = data[color_name].to(dtype)  # (B, V, 3, H, W)
        masks = data["mask"].to(dtype)  # (B, V, 1, H, W)
        C2W = data["C2W"].to(dtype)  # (B, V, 4, 4)
        fxfycxcy = data["fxfycxcy"].to(dtype)  # (B, V, 4)

        # Input views
        V_in = self.opt.num_input_views
        input_images = images[:, :V_in, ...]
        input_C2W = C2W[:, :V_in, ...]
        input_fxfycxcy = fxfycxcy[:, :V_in, ...]

        if self.opt.input_normal:
            input_images = torch.cat([input_images, data["normal"][:, :V_in, ...]], dim=2)
        if self.opt.input_coord:
            input_images = torch.cat([input_images, data["coord"][:, :V_in, ...]], dim=2)
        if self.opt.input_mr:
            input_images = torch.cat([input_images, data["mr"][:, :V_in, :2]], dim=2)

        model_outputs = self.forward_gaussians(input_images, input_C2W, input_fxfycxcy)
        render_outputs = self.gs_renderer.render(model_outputs, input_C2W, input_fxfycxcy, C2W, fxfycxcy)
        for k in render_outputs.keys():
            if isinstance(render_outputs[k], Tensor):
                render_outputs[k] = render_outputs[k].to(dtype)
        render_images = render_outputs["image"]  # (B, V, 3, H, W)
        render_masks = render_outputs["alpha"]  # (B, V, 1, H, W)
        render_coords = render_outputs["coord"]  # (B, V, 3, H, W)
        render_normals = render_outputs["normal"]  # (B, V, 3, H, W)

        # For visualization
        outputs["images_render"] = render_images
        outputs["images_gt"] = images
        if self.opt.vis_coords:
            outputs["images_coord"] = render_coords
            if self.opt.load_coord:
                outputs["images_gt_coord"] = data["coord"]
        if self.opt.vis_normals:
            outputs["images_normal"] = render_normals
            if self.opt.load_normal:
                outputs["images_gt_normal"] = data["normal"]
        # if self.opt.input_mr:
        #     outputs["images_mr"] = data["mr"]

        ################################ Compute reconstruction losses/metrics ################################

        outputs["image_mse"] = image_mse = tF.mse_loss(images, render_images)
        outputs["mask_mse"] = mask_mse = tF.mse_loss(masks, render_masks)
        loss = image_mse + mask_mse

        # Coord & Normal
        if self.opt.coord_weight > 0:
            assert self.opt.load_coord
            outputs["coord_mse"] = coord_mse = tF.mse_loss(data["coord"], render_coords)
            loss += self.opt.coord_weight * coord_mse
        if self.opt.normal_weight > 0:
            assert self.opt.load_normal
            outputs["normal_cosim"] = normal_cosim = tF.cosine_similarity(data["normal"], render_normals, dim=2).mean()
            loss += self.opt.normal_weight * (1. - normal_cosim)

        # LPIPS
        if step < self.opt.lpips_warmup_start:
            lpips_weight = 0.
        elif step > self.opt.lpips_warmup_end:
            lpips_weight = self.opt.lpips_weight
        else:
            lpips_weight = self.opt.lpips_weight * (step - self.opt.lpips_warmup_start) / (
                self.opt.lpips_warmup_end - self.opt.lpips_warmup_start)
        if lpips_weight > 0.:
            outputs["lpips"] = lpips = lpips_loss(
                # Downsampled to at most 256 to reduce memory cost
                tF.interpolate(
                    rearrange(images, "b v c h w -> (b v) c h w") * 2. - 1.,
                    (self.opt.lpips_resize, self.opt.lpips_resize), mode="bilinear", align_corners=False
                ) if self.opt.lpips_resize > 0 else rearrange(images, "b v c h w -> (b v) c h w") * 2. - 1.,
                tF.interpolate(
                    rearrange(render_images, "b v c h w -> (b v) c h w") * 2. - 1.,
                    (self.opt.lpips_resize, self.opt.lpips_resize), mode="bilinear", align_corners=False
                ) if self.opt.lpips_resize > 0 else rearrange(render_images, "b v c h w -> (b v) c h w") * 2. - 1.,
            ).mean()
            loss += lpips_weight * lpips

        outputs["loss"] = loss

        # Metric: PSNR, SSIM and LPIPS
        with torch.no_grad():
            outputs["psnr"] = -10 * torch.log10(torch.mean((images - render_images.detach()) ** 2))
            outputs["ssim"] = torch.tensor(calculate_ssim(
                (rearrange(images, "b v c h w -> (b v c) h w")
                    .cpu().float().numpy() * 255.).astype(np.uint8),
                (rearrange(render_images.detach(), "b v c h w -> (b v c) h w")
                    .cpu().float().numpy() * 255.).astype(np.uint8),
                channel_axis=0,
            ), device=images.device)
            if lpips_weight <= 0.:
                outputs["lpips"] = lpips = lpips_loss(
                    # Downsampled to at most 256 to reduce memory cost
                    tF.interpolate(
                        rearrange(images, "b v c h w -> (b v) c h w") * 2. - 1.,
                        (self.opt.lpips_resize, self.opt.lpips_resize), mode="bilinear", align_corners=False
                    ) if self.opt.lpips_resize > 0 else rearrange(images, "b v c h w -> (b v) c h w") * 2. - 1.,
                    tF.interpolate(
                        rearrange(render_images.detach(), "b v c h w -> (b v) c h w") * 2. - 1.,
                        (256, 256), mode="bilinear", align_corners=False
                    ) if self.opt.lpips_resize > 0 else rearrange(render_images.detach(), "b v c h w -> (b v) c h w") * 2. - 1.,
                ).mean()

        return outputs

    def forward_gaussians(self, input_images: Tensor, input_C2W: Tensor, input_fxfycxcy: Tensor):
        """
        Inputs:
            - `input_images`: (B, V_in, C, H, W)
            - `input_C2W`: (B, V_in, 4, 4)
            - `input_fxycxcy`: (B, V_in, 4)
        """
        _, V_in, _, H, W = input_images.shape
        plucker, _ = plucker_ray(H, W, input_C2W, input_fxfycxcy)  # (B, V_in, 6, H, W)
        images_plucker = torch.cat([input_images * 2. - 1., plucker], dim=2)
        images_plucker = rearrange(images_plucker, "b v c h w -> (b v) c h w")
        x = patchify(images_plucker, self.opt.patch_size)  # (B*V_in, N, C)
        x = rearrange(x, "(b v) n c -> b v n c", v=V_in)
        x = self.x_embedder(x)  # (B, V_in, N, D)

        x = rearrange(x, "b v n d -> b (v n) d")
        x = self.transformer(x)
        x = self.ln_out(x)

        def _reshape_feature(features: Tensor):
            features = rearrange(features, "b (v h w) d -> (b v) (h w) d", v=V_in, h=self.inter_res)
            features = unpatchify(features, self.opt.patch_size, int(features.shape[1]**0.5))
            features = rearrange(features, "(b v) c h w -> b v c h w", v=V_in)  # (B, V_in, `dim`, H, W)
            return features

        depth = _reshape_feature(self.out_depth(x))
        rgb = _reshape_feature(self.out_rgb(x))
        scale = _reshape_feature(self.out_scale(x))
        rotation = _reshape_feature(self.out_rotation(x))
        opacity = _reshape_feature(self.out_opacity(x))

        depth = torch.sigmoid(depth) * 2. - 1.  # [0, 1] -> [-1, 1]
        rgb = torch.sigmoid(rgb) * 2. - 1.  # [0, 1] -> [-1, 1]
        scale = torch.sigmoid(scale) * 2. - 1.  # [0, 1] -> [-1, 1]
        rotation = tF.normalize(rotation, p=2, dim=2)  # L2 normalize [-1, 1]
        opacity = torch.sigmoid(opacity - 2.) * 2. - 1.  # [0, 1] -> [-1, 1]; `-2.` cf. GS-LRM Appendix A.4

        return {
            "depth": depth,
            "rgb": rgb,
            "scale": scale,
            "rotation": rotation,
            "opacity": opacity,
        }
