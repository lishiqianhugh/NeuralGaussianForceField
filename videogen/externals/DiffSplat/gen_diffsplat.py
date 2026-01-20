import os
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as tF
from PIL import Image
import rembg

from src.options import opt_dict
from src.models import GSAutoencoderKL, GSRecon, ElevEst
from src.utils import util as _util
from src.utils import geo_util as _geo_util

from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection,T5EncoderModel, T5TokenizerFast
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
)

from extensions.diffusers_diffsplat import (
    UNetMV2DConditionModel,
    SD3TransformerMV2DModel,
    StableMVDiffusionPipeline,
    StableMVDiffusion3Pipeline,
    FlowDPMSolverMultistepScheduler
)


class DiffSplat:
    def __init__(self,seed: int = 0, device: Optional[str] = None,checkpoints_dir: str = 'checkpoints/DiffSplat',sd_version: str = 'sd15'):
        warnings.filterwarnings("ignore")

        self.sd_version = sd_version

        # Static config (mirrors infer.sh/infer_gsdiff_sd defaults)
        if sd_version == 'sd15':
            self.config_file = os.path.join(os.path.dirname(__file__), "configs/gsdiff_sd15.yaml")
        elif sd_version == 'sd35':
            self.config_file = os.path.join(os.path.dirname(__file__), "configs/gsdiff_sd35m_80g.yaml")
        else:
            raise ValueError(f"Invalid SD version: {sd_version}")
        
        if sd_version == 'sd15':
            self.tag = "gsdiff_gobj83k_sd15_image__render"
        elif sd_version == 'sd35':
            self.tag = "gsdiff_gobj83k_sd35m_image__render"
        else:
            raise ValueError(f"Invalid SD version: {sd_version}")
        
        self.checkpoints_dir = checkpoints_dir
        self.hdfs_dir = None
        self.seed = seed
        self.gpu_id = 0
        self.rembg_model_name = "u2net"
        self.border_ratio = 0.2
        # Defaults that differ between SD1.5 and SD3.5
        if self.sd_version == 'sd35':
            self.scheduler_type = "flow"
            self.num_inference_steps = 28
            self.guidance_scale = 2.0
            self.triangle_cfg_scaling = True
            self.min_guidance_scale = 1.0
        else:  # sd15
            self.scheduler_type = "sde-dpmsolver++"
            self.num_inference_steps = 20
            self.guidance_scale = 2.0
            self.triangle_cfg_scaling = True
            self.min_guidance_scale = 1.0

        self.eta = 1.0
        self.not_use_t5 = False
        self.init_std = 0.0
        self.init_noise_strength = 0.98
        self.init_bg = 0.0
        self.guess_mode = False
        self.controlnet_scale = 1.0
        self.distance = 1.4
        self.render_res: Optional[int] = None  # fallback to opt.input_res
        self.opacity_threshold = 0.0
        self.load_pretrained_gsrecon = "gsrecon_gobj265k_cnp_even4"
        self.load_pretrained_gsrecon_ckpt = -1

        if self.sd_version == 'sd15':
            self.load_pretrained_gsvae = "gsvae_gobj265k_sd"
        elif self.sd_version == 'sd35':
            self.load_pretrained_gsvae = "gsvae_gobj265k_sd3"
        else:
            raise ValueError(f"Invalid SD version: {self.sd_version}")

        self.load_pretrained_gsvae_ckpt = -1
        self.load_pretrained_elevest = "elevest_gobj265k_b_C25"
        self.load_pretrained_elevest_ckpt = -1
        self.infer_from_iter = -1
        self.negative_prompt="worst quality, normal quality, low quality, low res, blurry, ugly, disgusting"

        # Device
        if device is None:
            device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        
        self.device = device

        # Load configs & options
        configs = _util.get_configs(self.config_file, [])
        self.opt = opt_dict[configs["opt_type"]]

        # Image-conditioned settings
        self.opt.prediction_type = "v_prediction"
        self.opt.view_concat_condition = True
        self.opt.input_concat_binary_mask = True

        # Build models
        # Base channels differ between SD1.5 (4 latent channels) and SD3 (16 channels)
        in_channels = 16 if self.sd_version == 'sd35' else 4
        if self.opt.input_concat_plucker:
            in_channels += 6
        if self.opt.input_concat_binary_mask:
            in_channels += 1

        # Transformer kwargs for SD3 (use same computed in_channels)
        if self.sd_version == 'sd15':
            unet_from_pretrained_kwargs = {
                "sample_size": self.opt.input_res // 8,
                "in_channels": in_channels,
                "zero_init_conv_in": self.opt.zero_init_conv_in,
                "view_concat_condition": self.opt.view_concat_condition,
                "input_concat_plucker": self.opt.input_concat_plucker,
                "input_concat_binary_mask": self.opt.input_concat_binary_mask,
            }
        elif self.sd_version == 'sd35':
            transformer_from_pretrained_kwargs = {
                "sample_size": self.opt.input_res // 8,
                "in_channels": in_channels,
                "zero_init_conv_in": self.opt.zero_init_conv_in,
                "view_concat_condition": self.opt.view_concat_condition,
                "input_concat_plucker": self.opt.input_concat_plucker,
                "input_concat_binary_mask": self.opt.input_concat_binary_mask,
            }

        # Tokenizer / Text encoder / VAE selection depending on SD version
        if self.sd_version == 'sd35':
            # SD3: use CLIPTextModelWithProjection and flow-matching scheduler
            self.tokenizer = CLIPTokenizer.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16")
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="tokenizer_2")
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="text_encoder_2", variant="fp16")
            if not self.not_use_t5:
                self.tokenizer_3 = T5TokenizerFast.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="tokenizer_3")
                self.text_encoder_3 = T5EncoderModel.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="text_encoder_3", variant="fp16")
            else:
                self.tokenizer_3 = None
                self.text_encoder_3 = None
            self.vae = AutoencoderKL.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="vae")
        else:
            # SD1.5: use CLIPTextModel
            self.tokenizer = CLIPTokenizer.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16")
            self.vae = AutoencoderKL.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="vae")

        self.gsvae = GSAutoencoderKL(self.opt)
        self.gsrecon = GSRecon(self.opt)

        # Scheduler
        if self.sd_version == 'sd15':
            # SD15 default scheduler
            if self.scheduler_type == "ddim":
                self.noise_scheduler = DDIMScheduler.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="scheduler")
            elif "dpmsolver" in self.scheduler_type:
                self.noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="scheduler")
                self.noise_scheduler.config.algorithm_type = self.scheduler_type
            elif self.scheduler_type == "edm":
                self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="scheduler")
            else:
                raise NotImplementedError(f"Scheduler [{self.scheduler_type}] is not supported by now")
            if self.opt.common_tricks:
                self.noise_scheduler.config.timestep_spacing = "trailing"
                self.noise_scheduler.config.rescale_betas_zero_snr = True
            if self.opt.prediction_type is not None:
                self.noise_scheduler.config.prediction_type = self.opt.prediction_type
            if self.opt.beta_schedule is not None:
                self.noise_scheduler.config.beta_schedule = self.opt.beta_schedule
        else:
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="scheduler")
            if "dpmsolver" in self.scheduler_type:
                self.new_noise_scheduler = FlowDPMSolverMultistepScheduler.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="scheduler")
                self.new_noise_scheduler.config.algorithm_type = self.scheduler_type
                self.new_noise_scheduler.config.flow_shift = self.noise_scheduler.config.shift
                self.noise_scheduler = self.new_noise_scheduler


        if self.opt.common_tricks:
            self.noise_scheduler.config.timestep_spacing = "trailing"
            self.noise_scheduler.config.rescale_betas_zero_snr = True
        if self.opt.prediction_type is not None:
            self.noise_scheduler.config.prediction_type = self.opt.prediction_type
        if self.opt.beta_schedule is not None:
            self.noise_scheduler.config.beta_schedule = self.opt.beta_schedule

        # Load GSDiff checkpoint (UNet)
        ckpt_dir = os.path.join(self.checkpoints_dir, self.tag, "checkpoints")
        if not os.path.exists(os.path.join(ckpt_dir, f"{self.infer_from_iter:06d}")):
            self.infer_from_iter = _util.load_ckpt(ckpt_dir, self.infer_from_iter, self.hdfs_dir, None)
        path = os.path.join(ckpt_dir, f"{self.infer_from_iter:06d}")

        if self.sd_version == 'sd15':
            os.system(f"python3 {os.path.join(os.path.dirname(__file__), 'extensions/merge_safetensors.py')} {os.path.join(path, 'unet_ema')}")
            unet, loading_info = UNetMV2DConditionModel.from_pretrained_new(
                path,
                subfolder="unet_ema",
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
                output_loading_info=True,
                **unet_from_pretrained_kwargs,
            )
            for key in loading_info.keys():
                assert len(loading_info[key]) == 0
            self.unet = unet
        elif self.sd_version == 'sd35':
            os.system(f"python3 {os.path.join(os.path.dirname(__file__), 'extensions/merge_safetensors.py')} {os.path.join(path, 'transformer_ema')}")
            transformer, loading_info = SD3TransformerMV2DModel.from_pretrained_new(
                path,
                subfolder="transformer_ema",
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
                output_loading_info=True,
                **transformer_from_pretrained_kwargs,
            )
            for key in loading_info.keys():
                assert len(loading_info[key]) == 0  # no missing_keys, unexpected_keys, mismatched_keys, error_msgs
            self.transformer = transformer

        # Freeze models and move to device
        if self.sd_version == 'sd15':
            for m in [self.text_encoder, self.vae, self.gsvae, self.gsrecon, self.unet]:
                m.requires_grad_(False)
                m.eval()
            self.text_encoder = self.text_encoder.to(self.device)
            self.vae = self.vae.to(self.device)
            self.gsvae = self.gsvae.to(self.device)
            self.gsrecon = self.gsrecon.to(self.device)
            self.unet = self.unet.to(self.device)
        else:  # sd35: transformer
            for m in [self.text_encoder, self.text_encoder_2, self.vae, self.gsvae, self.gsrecon, self.transformer]:
                m.requires_grad_(False)
                m.eval()
            if not self.not_use_t5:
                self.text_encoder_3.requires_grad_(False)
                self.text_encoder_3.eval()
            
            self.text_encoder = self.text_encoder.to(self.device)
            self.text_encoder_2 = self.text_encoder_2.to(self.device)
            if not self.not_use_t5:
                self.text_encoder_3 = self.text_encoder_3.to(self.device)
            self.vae = self.vae.to(self.device)
            self.gsvae = self.gsvae.to(self.device)
            self.gsrecon = self.gsrecon.to(self.device)
            self.transformer = self.transformer.to(self.device)

        # Load pretrained GS models
        self.gsvae = _util.load_ckpt(
            os.path.join(self.checkpoints_dir, self.load_pretrained_gsvae, "checkpoints"),
            self.load_pretrained_gsvae_ckpt,
            None if self.hdfs_dir is None else os.path.join(self.hdfs_dir, self.load_pretrained_gsvae),
            self.gsvae,
        )
        self.gsrecon = _util.load_ckpt(
            os.path.join(self.checkpoints_dir, self.load_pretrained_gsrecon, "checkpoints"),
            self.load_pretrained_gsrecon_ckpt,
            None if self.hdfs_dir is None else os.path.join(self.hdfs_dir, self.load_pretrained_gsrecon),
            self.gsrecon,
        )

        # Elevation model
        self.use_elevest = True
        if self.use_elevest:
            self.elevest = ElevEst(self.opt).to(self.device)
            self.elevest.requires_grad_(False)
            self.elevest.eval()
            self.elevest = _util.load_ckpt(
                os.path.join(self.checkpoints_dir, self.load_pretrained_elevest, "checkpoints"),
                self.load_pretrained_elevest_ckpt,
                None if self.hdfs_dir is None else os.path.join(self.hdfs_dir, self.load_pretrained_elevest),
                self.elevest,
            )
        else:
            self.elevest = None

        # Pipeline
        self.V_in = self.opt.num_input_views
        # Pipeline selection: unet for sd15, transformer pipeline for sd35
        if self.sd_version == 'sd35':
            # Use the SD3 pipeline which accepts a transformer instead of unet
            self.pipeline = StableMVDiffusion3Pipeline(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                text_encoder_2=self.text_encoder_2,tokenizer_2=self.tokenizer_2,
                text_encoder_3=self.text_encoder_3,tokenizer_3=self.tokenizer_3,
                vae=self.vae,
                transformer=self.transformer,
                scheduler=self.noise_scheduler,
            )
        else:
            self.pipeline = StableMVDiffusionPipeline(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                vae=self.vae,
                unet=self.unet,
                scheduler=self.noise_scheduler,
            )
        self.pipeline.set_progress_bar_config(disable=True)

        # Generator
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

    def infer(self, image: Image.Image, text_prompt: str):
        # Preprocess image (rembg + center), then to tensor
        rgb = _rembg_and_center_pil(image, size=self.opt.input_res, border_ratio=self.border_ratio, model_name=self.rembg_model_name)
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W), float32 in [0,1]
        image_tensor = tF.interpolate(image_tensor, size=(self.opt.input_res, self.opt.input_res), mode="bilinear", align_corners=False, antialias=True)
        image_tensor = image_tensor.unsqueeze(1).to(device=self.device)  # (B=1, V_cond=1, 3, H, W)

        # Elevation estimation
        with torch.no_grad():
            elevation_val = -self.elevest.predict_elev(image_tensor.squeeze(1)).detach().cpu().item() if self.elevest is not None else 10.0
        
        # elevation_val = 20.0

        # Camera setup
        fxfycxcy = torch.tensor([self.opt.fxfy, self.opt.fxfy, 0.5, 0.5], device=self.device).float()
        elevations = torch.tensor([-elevation_val] * 4, device=self.device).deg2rad().float()
        azimuths = torch.tensor([0.0, 90.0, 180.0, 270.0], device=self.device).deg2rad().float()
        radius = torch.tensor([self.distance] * 4, device=self.device).float()
        input_C2W = _geo_util.orbit_camera(elevations, azimuths, radius, is_degree=False)
        input_C2W[:, :3, 1:3] *= -1  # OpenGL -> OpenCV
        input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(input_C2W.shape[0], 1)

        if self.opt.input_concat_plucker:
            H = W = self.opt.input_res
            plucker, _ = _geo_util.plucker_ray(H, W, input_C2W.unsqueeze(0), input_fxfycxcy.unsqueeze(0))
            plucker = plucker.squeeze(0)  # (V_in, 6, H, W)
            if self.opt.view_concat_condition:
                plucker = torch.cat([plucker[0:1, ...], plucker], dim=0)
        else:
            plucker = None

        # Inference
        if self.render_res is None:
            self.render_res = self.opt.input_res

        device_type = "cuda" if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                if self.sd_version == 'sd15':
                    out = self.pipeline(
                        image_tensor,
                        prompt=text_prompt,
                        negative_prompt=self.negative_prompt,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        triangle_cfg_scaling=self.triangle_cfg_scaling,
                        min_guidance_scale=self.min_guidance_scale,
                        max_guidance_scale=self.guidance_scale,
                        output_type="latent",
                        eta=self.eta,
                        generator=self.generator,
                        plucker=plucker,
                        num_views=self.V_in,
                        init_std=self.init_std,
                        init_noise_strength=self.init_noise_strength,
                        init_bg=self.init_bg,
                        guess_mode=self.guess_mode,
                        controlnet_conditioning_scale=float(self.controlnet_scale),
                    ).images
                else:
                    out = self.pipeline(
                        image_tensor,
                        prompt=text_prompt,
                        prompt_2=text_prompt,
                        prompt_3=text_prompt,
                        negative_prompt=self.negative_prompt,
                        negative_prompt_2=self.negative_prompt,
                        negative_prompt_3=self.negative_prompt,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        triangle_cfg_scaling=self.triangle_cfg_scaling,
                        min_guidance_scale=self.min_guidance_scale,
                        max_guidance_scale=self.guidance_scale,
                        output_type="latent",
                        generator=self.generator,
                        plucker=plucker,
                        num_views=self.V_in,
                        init_std=self.init_std,
                        init_noise_strength=self.init_noise_strength,
                        init_bg=self.init_bg,
                    ).images

        # Decode to Gaussian and return the 3D model
        out = out / self.gsvae.scaling_factor + self.gsvae.shift_factor
        render_outputs = self.gsvae.decode_and_render_gslatents(
            self.gsrecon,
            out,
            input_C2W.unsqueeze(0),
            input_fxfycxcy.unsqueeze(0),
            height=self.render_res,
            width=self.render_res,
            opacity_threshold=self.opacity_threshold,
        )
        gaussian_model = render_outputs["pc"][0]
        return gaussian_model

def _rembg_and_center_pil(image: Image.Image, size: int = 256, border_ratio: float = 0.2, model_name: str = "u2net") -> np.ndarray:
    """Remove background and center object on a square canvas. Returns RGB float array in [0,1], shape (size, size, 3)."""
    image_np = np.array(image.convert("RGBA"))
    h, w = image_np.shape[:2]
    scale = size / max(h, w)
    h2, w2 = int(h * scale), int(w * scale)
    image_np = np.array(Image.fromarray(image_np).resize((w2, h2), Image.BILINEAR))

    session = rembg.new_session(model_name=model_name)
    carved_rgba = rembg.remove(image_np, session=session)  # (H, W, 4)
    if carved_rgba.ndim != 3 or carved_rgba.shape[-1] != 4:
        if carved_rgba.ndim == 3 and carved_rgba.shape[-1] == 3:
            alpha = np.full((*carved_rgba.shape[:2], 1), 255, dtype=np.uint8)
            carved_rgba = np.concatenate([carved_rgba, alpha], axis=-1)
        else:
            carved_rgba = np.dstack([carved_rgba] * 3 + [np.full_like(carved_rgba, 255)])

    mask = carved_rgba[..., 3] > 0
    if not np.any(mask):
        final_rgba = np.zeros((size, size, 4), dtype=np.uint8)
        y2_min = (size - h2) // 2
        x2_min = (size - w2) // 2
        final_rgba[y2_min:y2_min + h2, x2_min:x2_min + w2] = carved_rgba
    else:
        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        obj_h = max(1, x_max - x_min)
        obj_w = max(1, y_max - y_min)
        desired_size = int(size * (1 - border_ratio))
        scale2 = desired_size / max(obj_h, obj_w)
        obj_h2 = max(1, int(obj_h * scale2))
        obj_w2 = max(1, int(obj_w * scale2))
        y2_min = (size - obj_h2) // 2
        x2_min = (size - obj_w2) // 2
        final_rgba = np.zeros((size, size, 4), dtype=np.uint8)
        crop = carved_rgba[x_min:x_max, y_min:y_max]
        crop_resized = np.array(Image.fromarray(crop).resize((obj_w2, obj_h2), Image.BILINEAR))
        final_rgba[y2_min:y2_min + obj_h2, x2_min:x2_min + obj_w2] = crop_resized

    rgb = final_rgba[..., :3].astype(np.float32) / 255.0
    a = (final_rgba[..., 3:4].astype(np.float32) / 255.0)
    rgb = rgb * a + (1.0 - a)
    return rgb
