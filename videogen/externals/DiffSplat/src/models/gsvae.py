from typing import *
from torch import Tensor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from lpips import LPIPS
from src.models.gsrecon import GSRecon

from skimage.metrics import structural_similarity as calculate_ssim
import numpy as np
import torch
from torch import nn
import torch.nn.functional as tF
from einops import rearrange
from diffusers import AutoencoderKL, AutoencoderTiny
from diffusers.models.autoencoders.autoencoder_kl import Decoder
from diffusers.models.autoencoders.autoencoder_tiny import DecoderTiny

from src.options import Options


TAE_DICT = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "madebyollin/taesd",
    "stabilityai/stable-diffusion-2-1": "madebyollin/taesd",
    "PixArt-alpha/PixArt-XL-2-512x512": "madebyollin/taesd",
    "stabilityai/stable-diffusion-xl-base-1.0": "madebyollin/taesdxl",
    "madebyollin/sdxl-vae-fp16-fix": "madebyollin/taesdxl",
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": "madebyollin/taesdxl",
    "stabilityai/stable-diffusion-3-medium-diffusers": "madebyollin/taesd3",
    "stabilityai/stable-diffusion-3.5-medium": "madebyollin/taesd3",
    "stabilityai/stable-diffusion-3.5-large": "madebyollin/taesd3",
    "black-forest-labs/FLUX.1-dev": "madebyollin/taef1",
}


class GSAutoencoderKL(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        AutoencoderKL_from = AutoencoderKL.from_config if opt.vae_from_scratch else AutoencoderKL.from_pretrained
        AutoencoderTiny_from = AutoencoderTiny.from_config if opt.vae_from_scratch else AutoencoderTiny.from_pretrained

        if not opt.use_tinyae:
            if "fp16" not in opt.pretrained_model_name_or_path:
                if "Sigma" in opt.pretrained_model_name_or_path:  # PixArt-Sigma
                    self.vae = AutoencoderKL_from("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="vae")
                else:
                    self.vae = AutoencoderKL_from(opt.pretrained_model_name_or_path, subfolder="vae")
            else:  # fixed fp16 VAE for SDXL
                self.vae = AutoencoderKL_from(opt.pretrained_model_name_or_path)
            self.vae.enable_slicing()  # to save memory
        else:
            self.vae = AutoencoderTiny_from(TAE_DICT[opt.pretrained_model_name_or_path])

        # Encode input Conv
        new_conv_in = nn.Conv2d(
            12,  # number of GS properties
            self.vae.config.block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if not opt.use_tinyae:
            init_conv_in_weight = torch.cat([self.vae.encoder.conv_in.weight.data]*4, dim=1)
        else:
            init_conv_in_weight = torch.cat([self.vae.encoder.layers[0].weight.data]*4, dim=1)
        # init_conv_in_weight /= 4  # rescale input conv weight parameters
        new_conv_in.weight.data.copy_(init_conv_in_weight)
        if not opt.use_tinyae:
            new_conv_in.bias.data.copy_(self.vae.encoder.conv_in.bias.data)
            self.vae.encoder.conv_in = new_conv_in
        else:
            new_conv_in.bias.data.copy_(self.vae.encoder.layers[0].bias.data)
            self.vae.encoder.layers[0] = new_conv_in

        # Decoder output Conv
        new_conv_out = nn.Conv2d(
            self.vae.config.block_out_channels[0],
            12,  # number of GS properties
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if not opt.use_tinyae:
            init_conv_out_weight = torch.cat([self.vae.decoder.conv_out.weight.data]*4, dim=0)
        else:
            init_conv_out_weight = torch.cat([self.vae.decoder.layers[-1].weight.data]*4, dim=0)
        new_conv_out.weight.data.copy_(init_conv_out_weight)
        if not opt.use_tinyae:
            init_conv_out_bias = torch.cat([self.vae.decoder.conv_out.bias.data]*4, dim=0)
        else:
            init_conv_out_bias = torch.cat([self.vae.decoder.layers[-1].bias.data]*4, dim=0)
        new_conv_out.bias.data.copy_(init_conv_out_bias)
        if not opt.use_tinyae:
            self.vae.decoder.conv_out = new_conv_out
        else:
            self.vae.decoder.layers[-1] = new_conv_out

        if opt.freeze_encoder:
            self.vae.encoder.requires_grad_(False)
            self.vae.quant_conv.requires_grad_(False)

        self.scaling_factor = opt.scaling_factor if opt.scaling_factor is not None else self.vae.config.scaling_factor
        self.scaling_factor = self.scaling_factor if self.scaling_factor is not None else 1.
        self.shift_factor = opt.shift_factor if opt.shift_factor is not None else self.vae.config.shift_factor
        self.shift_factor = self.shift_factor if self.shift_factor is not None else 0.

        # TinyAE
        tae = AutoencoderTiny_from(TAE_DICT[opt.pretrained_model_name_or_path])
        # Tiny decoder output Conv
        new_conv_out = nn.Conv2d(
            tae.config.block_out_channels[0],  # the same as `self.vae.config.block_out_channels[0]`
            12,  # number of GS properties
            kernel_size=3,
            stride=1,
            padding=1,
        )
        init_conv_out_weight = torch.cat([tae.decoder.layers[-1].weight.data]*4, dim=0)
        new_conv_out.weight.data.copy_(init_conv_out_weight)
        init_conv_out_bias = torch.cat([tae.decoder.layers[-1].bias.data]*4, dim=0)
        new_conv_out.bias.data.copy_(init_conv_out_bias)
        tae.decoder.layers[-1] = new_conv_out
        self.tiny_decoder = tae.decoder

        if opt.use_tiny_decoder:
            assert not opt.use_tinyae  # so 2 decoders in this model

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss(self,
        data: Optional[Dict[str, Tensor]],
        lpips_loss: LPIPS,
        gsrecon: GSRecon,
        step: int,
        latents: Optional[Tensor] = None,
        kl: Optional[float] = None,
        gs: Optional[Tensor] = None,
        use_tiny_decoder: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
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

        # Get GS latents, KL divergence and ground-truth GS
        if latents is None or kl is None or gs is None:
            context = torch.no_grad() if use_tiny_decoder else torch.enable_grad()
            # Reconstruct & Encode
            with context:
                latents, kl, gs = self.get_gslatents(gsrecon, input_images, input_C2W, input_fxfycxcy, return_kl=True, return_gs=True)

        outputs["kl"] = kl / (sum(latents.shape[1:]))

        # Decode
        recon_gs = self.decode(latents, use_tiny_decoder)
        recon_gs = rearrange(recon_gs, "(b v) c h w -> b v c h w", v=V_in)
        gs = rearrange(gs, "(b v) c h w -> b v c h w", v=V_in)
        recon_model_outputs = {
            "rgb": recon_gs[:, :, :3, ...],
            "scale": recon_gs[:, :, 3:6, ...],
            "rotation": recon_gs[:, :, 6:10, ...],
            "opacity": recon_gs[:, :, 10:11, ...],
            "depth": recon_gs[:, :, 11:12, ...],
        }

        render_outputs = gsrecon.gs_renderer.render(recon_model_outputs, input_C2W, input_fxfycxcy, C2W, fxfycxcy)
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

        outputs["latent_mse"] = latent_mse = tF.mse_loss(gs, recon_gs)

        outputs["image_mse"] = image_mse = tF.mse_loss(images, render_images)
        outputs["mask_mse"] = mask_mse = tF.mse_loss(masks, render_masks)
        loss = image_mse + mask_mse

        # Depth & Normal
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

        outputs["loss"] = self.opt.recon_weight * latent_mse + self.opt.render_weight * loss

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
                        (self.opt.lpips_resize, self.opt.lpips_resize), mode="bilinear", align_corners=False
                    ) if self.opt.lpips_resize > 0 else rearrange(render_images.detach(), "b v c h w -> (b v) c h w") * 2. - 1.,
                ).mean()

        return outputs

    def get_gslatents(self,
        gsrecon: GSRecon,
        input_images: Tensor,
        input_C2W: Tensor,
        input_fxfycxcy: Tensor,
        return_kl: bool = False,
        return_gs: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        (B, V_in), chunk = input_images.shape[:2], self.opt.chunk_size

        # Reconstruction
        gs = []
        for i in range(0, B, chunk):
            gsrecon_outputs = gsrecon.forward_gaussians(
                input_images[i:min(B, i+chunk)],
                input_C2W[i:min(B, i+chunk)],
                input_fxfycxcy[i:min(B, i+chunk)],
            )
            _gs = torch.cat([
                gsrecon_outputs["rgb"],
                gsrecon_outputs["scale"],
                gsrecon_outputs["rotation"],
                gsrecon_outputs["opacity"],
                gsrecon_outputs["depth"],
            ], dim=2)  # (`chunk`, V_in, C=12, H, W)
            gs.append(_gs)
        gs = torch.cat(gs, dim=0)  # (B, V_in, C=12, H, W)
        gs = rearrange(gs, "b v c h w -> (b v) c h w")

        # GSVAE encoding
        latents, kl = [], 0.
        for i in range(0, B*V_in, chunk):
            _latents, _kl = self.encode(gs[i:min(B*V_in, i+chunk)], deterministic=(not self.training))  # (`chunk`, D=4, H', W')
            latents.append(_latents)
            kl += (_latents.shape[0] * _kl)
        latents = torch.cat(latents, dim=0)  # (B*V_in, D=4, H', W')
        kl /= latents.shape[0]

        results = [latents]
        if return_kl:
            results.append(kl)
        if return_gs:
            results.append(gs)
        if len(results) == 1:  # only return latents
            return results[0]
        else:
            return tuple(results)

    def decode_gslatents(self, latents: Tensor, use_tiny_decoder: bool = False) -> Dict[str, Tensor]:
        V_in = self.opt.num_input_views
        B, chunk = latents.shape[0] // self.opt.num_input_views, self.opt.chunk_size

        # GSVAE decoding
        recon_gs = []
        for i in range(0, B*V_in, chunk):
            _recon_gs = self.decode(latents[i:min(B*V_in, i+chunk)], use_tiny_decoder)  # (`chunk`, C=12, H, W)
            recon_gs.append(_recon_gs)
        recon_gs = torch.cat(recon_gs, dim=0)  # (B*V_in, C=12, H, W)
        recon_gs = rearrange(recon_gs, "(b v) c h w -> b v c h w", v=V_in)

        recon_gsrecon_outputs = {
            "rgb": recon_gs[:, :, :3, ...],
            "scale": recon_gs[:, :, 3:6, ...],
            "rotation": recon_gs[:, :, 6:10, ...],
            "opacity": recon_gs[:, :, 10:11, ...],
            "depth": recon_gs[:, :, 11:12, ...],
        }

        return recon_gsrecon_outputs

    def decode_and_render_gslatents(self,
        gsrecon: GSRecon,
        latents: Tensor,
        input_C2W: Tensor,
        input_fxfycxcy: Tensor,
        C2W: Optional[Tensor] = None,
        fxfycxcy: Optional[Tensor] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        scaling_modifier: int = 1,
        opacity_threshold: float = 0.,
        use_tiny_decoder: bool = False,
    ) -> Dict[str, Tensor]:
        C2W = C2W if C2W is not None else input_C2W
        fxfycxcy = fxfycxcy if fxfycxcy is not None else input_fxfycxcy

        recon_gsrecon_outputs = self.decode_gslatents(latents, use_tiny_decoder)
        render_outputs = gsrecon.gs_renderer.render(
            recon_gsrecon_outputs,
            input_C2W, input_fxfycxcy, C2W, fxfycxcy,
            height=height, width=width,
            scaling_modifier=scaling_modifier,
            opacity_threshold=opacity_threshold,
        )
        return render_outputs  # (B, V, 3 or 1, H, W)

    def encode(self, gs: Tensor, deterministic=False) -> Tuple[Tensor, Tensor]:
        if self.opt.freeze_encoder or self.opt.use_tinyae:
            self.vae.encoder.eval()
            self.vae.quant_conv.eval()

        assert gs.ndim == 4  # (B*V, C=12, H, W)

        if not self.opt.use_tinyae:
            latent_dist: DiagonalGaussianDistribution = self.vae.encode(gs).latent_dist
            latents = latent_dist.sample() if not deterministic else latent_dist.mode()  # (B*V, D=4, H, W)
            kl = latent_dist.kl().mean()
        else:
            latents = self.vae.encode(gs).latents  # (B*V, D=4, H, W)
            kl = torch.zeros(1, dtype=latents.dtype, device=latents.device)  # dummy

        return latents, kl

    def decode(self, z: Tensor, use_tiny_decoder: bool = False) -> Tensor:
        if not hasattr(self, "tiny_decoder"):
            use_tiny_decoder = False

        if use_tiny_decoder:
            original_decoder = self.vae.decoder
            self.vae.decoder = self.tiny_decoder
            assert isinstance(self.vae.decoder, DecoderTiny)

            # NOTE: NOT exclude the origin `self.vae.post_quant_conv` for tiny decoder here
            # But we conduct full fine-tuning for VAE and tiny decoder, so it should be fine

            z = self.scaling_factor * (z - self.shift_factor)  # `AutoencoderTiny` uses scaled (and shifted) latents

        recon_gs = self.vae.decode(z).sample.clamp(-1., 1.)  # (B*V, C=12, H, W)

        # Change back to the original decoder
        if use_tiny_decoder:
            self.vae.decoder = original_decoder
            assert isinstance(self.vae.decoder, Decoder)

        return recon_gs
