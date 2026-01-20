import warnings
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

from typing import *
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from accelerate.data_loader import DataLoaderShard

import os
import argparse
import logging
import math
from collections import defaultdict
from packaging import version
import gc

from tqdm import tqdm
import wandb
import numpy as np
from skimage.metrics import structural_similarity as calculate_ssim
from lpips import LPIPS

import torch
import torch.nn.functional as tF
from einops import rearrange, repeat
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin
from diffusers import DDPMScheduler, DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, AutoencoderKL
from diffusers.training_utils import compute_snr

from src.options import opt_dict, Options
from src.data import GObjaverseParquetDataset, ParquetChunkDataSource, MultiEpochsChunkedDataLoader, yield_forever
from src.models import GSAutoencoderKL, GSRecon, get_optimizer, get_lr_scheduler
import src.utils.util as util
import src.utils.geo_util as geo_util
import src.utils.vis_util as vis_util

from extensions.diffusers_diffsplat import MyEMAModel, PixArtTransformerMV2DModel, PixArtSigmaMVPipeline


@torch.no_grad()
def log_validation(
    dataloader, negative_prompt_embed, negative_prompt_attention_mask, lpips_loss, gsrecon, gsvae, vae, transformer,
    global_step, accelerator, args, opt: Options,
):
    if not opt.edm_style_training:
        if opt.noise_scheduler_type == "ddim":
            noise_scheduler = DDIMScheduler.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="scheduler")
        elif "dpmsolver" in opt.noise_scheduler_type:
            noise_scheduler = DPMSolverMultistepScheduler.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="scheduler")
            noise_scheduler.config.algorithm_type = opt.noise_scheduler_type
        else:
            raise NotImplementedError  # TODO: support more noise schedulers
    else:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="scheduler")
    if opt.common_tricks:
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config.rescale_betas_zero_snr = True
    if opt.prediction_type is not None:
        noise_scheduler.config.prediction_type = opt.prediction_type
    if opt.beta_schedule is not None:
        noise_scheduler.config.beta_schedule = opt.beta_schedule

    pipeline = PixArtSigmaMVPipeline(
        text_encoder=None, tokenizer=None,
        vae=vae, transformer=accelerator.unwrap_model(transformer),
        scheduler=noise_scheduler,
    )

    pipeline.set_progress_bar_config(disable=True)
    # pipeline.enable_xformers_memory_efficient_attention()

    if args.seed >= 0:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images_dictlist, metrics_dictlist = defaultdict(list), defaultdict(list)

    val_progress_bar = tqdm(
        range(len(dataloader)) if args.max_val_steps is None else range(args.max_val_steps),
        desc=f"Validation",
        ncols=125,
        disable=not accelerator.is_main_process
    )

    for i, batch in enumerate(dataloader):
        V_in, V_cond, V = opt.num_input_views, opt.num_cond_views, opt.num_views  # TODO: not support V_cond > V_in by now
        cond_idx = [0]  # the first view must be in inputs
        if V_cond > 1:
            cond_idx += np.random.choice(range(1, V), V_cond-1, replace=False).tolist()
        imgs_cond = batch["image"][:, cond_idx, ...]  # (B, V_cond, 3, H, W)
        B = imgs_cond.shape[0]

        imgs_out = batch["image"]  # (B, V, 3, H, W); for visualization and evaluation
        imgs_out = rearrange(imgs_out, "b v c h w -> (b v) c h w")
        prompt_embeds = batch["prompt_embed"]  # (B, N, D)
        prompt_attention_masks = batch["prompt_attention_mask"]  # (B, N)
        negative_prompt_embeds = repeat(negative_prompt_embed.to(accelerator.device), "n d -> b n d", b=B)
        negative_prompt_attention_masks = repeat(negative_prompt_attention_mask.to(accelerator.device), "n -> b n", b=B)

        C2W = batch["C2W"]
        fxfycxcy = batch["fxfycxcy"]
        input_C2W = C2W[:, :V_in, ...]
        input_fxfycxcy = fxfycxcy[:, :V_in, ...]
        cond_C2W = C2W[:, cond_idx,...]
        cond_fxfycxcy = fxfycxcy[:, cond_idx,...]

        # Plucker embeddings
        if opt.input_concat_plucker:
            H = W = opt.input_res
            plucker, _ = geo_util.plucker_ray(H, W, input_C2W, input_fxfycxcy)  # (B, V_in, 6, H, W)
            if opt.view_concat_condition:
                cond_plucker, _ = geo_util.plucker_ray(H, W, cond_C2W, cond_fxfycxcy)  # (B, V_cond, 6, H, W)
                plucker = torch.cat([cond_plucker, plucker], dim=1)  # (B, V_cond+V_in, 6, H, W)
            plucker = rearrange(plucker, "b v c h w -> (b v) c h w")
        else:
            plucker = None

        images_dictlist["gt"].append(imgs_out)  # (B*V, C=3, H, W)
        if opt.vis_coords and opt.load_coord:
            coords_out = rearrange(batch["coord"], "b v c h w -> (b v) c h w")  # (B*V, C=3, H, W)
            images_dictlist["gt_coord"].append(coords_out)
        if opt.vis_normals and opt.load_normal:
            normals_out = rearrange(batch["normal"], "b v c h w -> (b v) c h w")  # (B*V, C=3, H, W)
            images_dictlist["gt_normal"].append(normals_out)

        with torch.autocast("cuda", torch.bfloat16):
            for guidance_scale in sorted(args.val_guidance_scales):
                out = pipeline(
                    imgs_cond, num_inference_steps=opt.num_inference_steps, guidance_scale=guidance_scale,
                    output_type="latent", eta=1., generator=generator,
                    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                    prompt_attention_mask=prompt_attention_masks, negative_prompt_attention_mask=negative_prompt_attention_masks,
                    plucker=plucker, num_views=V_in,
                    init_std=opt.init_std, init_noise_strength=opt.init_noise_strength, init_bg=opt.init_bg,
                ).images

                # Rendering GS latents
                out = out / gsvae.scaling_factor + gsvae.shift_factor
                render_outputs = gsvae.decode_and_render_gslatents(gsrecon, out, input_C2W, input_fxfycxcy, C2W, fxfycxcy)
                render_images = rearrange(render_outputs["image"], "b v c h w -> (b v) c h w")  # (B*V, C=3, H, W)
                images_dictlist[f"pred_cfg{guidance_scale:.1f}"].append(render_images)
                if opt.vis_coords:
                    render_coords = rearrange(render_outputs["coord"], "b v c h w -> (b v) c h w")  # (B*V, 3, H, W)
                    images_dictlist[f"pred_coord_cfg{guidance_scale:.1f}"].append(render_coords)
                if opt.vis_normals:
                    render_normals = rearrange(render_outputs["normal"], "b v c h w -> (b v) c h w")  # (B*V, 3, H, W)
                    images_dictlist[f"pred_normal_cfg{guidance_scale:.1f}"].append(render_normals)

                # Decode to pseudo images
                if opt.vis_pseudo_images:
                    out = (out - gsvae.shift_factor) * gsvae.scaling_factor / vae.config.scaling_factor
                    images = vae.decode(out).sample.clamp(-1., 1.) * 0.5 + 0.5
                    images_dictlist[f"pred_image_cfg{guidance_scale:.1f}"].append(images)  # (B*V_in, 3, H, W)

                ################################ Compute generation metrics ################################

                lpips = lpips_loss(
                    # Downsampled to at most 256 to reduce memory cost
                    tF.interpolate(imgs_out * 2. - 1., (256, 256), mode="bilinear", align_corners=False),
                    tF.interpolate(render_images * 2. - 1., (256, 256), mode="bilinear", align_corners=False)
                ).mean()

                psnr = -10. * torch.log10(tF.mse_loss(imgs_out, render_images))

                ssim = torch.tensor(calculate_ssim(
                    (rearrange(imgs_out, "bv c h w -> (bv c) h w").cpu().float().numpy() * 255.).astype(np.uint8),
                    (rearrange(render_images, "bv c h w -> (bv c) h w").cpu().float().numpy() * 255.).astype(np.uint8),
                    channel_axis=0,
                ), device=render_images.device)

                lpips = accelerator.gather_for_metrics(lpips.repeat(B)).mean()
                psnr = accelerator.gather_for_metrics(psnr.repeat(B)).mean()
                ssim = accelerator.gather_for_metrics(ssim.repeat(B)).mean()

                metrics_dictlist[f"lpips_cfg{guidance_scale:.1f}"].append(lpips)
                metrics_dictlist[f"psnr_cfg{guidance_scale:.1f}"].append(psnr)
                metrics_dictlist[f"ssim_cfg{guidance_scale:.1f}"].append(ssim)

                if opt.coord_weight > 0.:
                    assert opt.load_coord
                    coord_mse = tF.mse_loss(coords_out, render_coords)
                    coord_mse = accelerator.gather_for_metrics(coord_mse.repeat(B)).mean()
                    metrics_dictlist[f"coord_mse_cfg{guidance_scale:.1f}"].append(coord_mse)
                if opt.normal_weight > 0.:
                    assert opt.load_normal
                    normal_cosim = tF.cosine_similarity(normals_out, render_normals, dim=2).mean()
                    normal_cosim = accelerator.gather_for_metrics(normal_cosim.repeat(B)).mean()
                    metrics_dictlist[f"normal_cosim_cfg{guidance_scale:.1f}"].append(normal_cosim)

            # Only log the last (biggest) cfg metrics in the progress bar
            val_logs = {
                "lpips": lpips.item(),
                "psnr": psnr.item(),
                "ssim": ssim.item(),
            }
            val_progress_bar.set_postfix(**val_logs)
            val_progress_bar.update(1)

            if args.max_val_steps is not None and i == (args.max_val_steps - 1):
                break

        val_progress_bar.close()

    if accelerator.is_main_process:
        formatted_images = []
        for k, v in images_dictlist.items():  # "gs_gt", "pred_cfg1.0", "pred_cfg3.0", ...
            mvimages = torch.cat(v, dim=0)  # (N*B*V, C, H, W)
            mvimages = rearrange(mvimages, "(nb v) c h w -> nb v c h w", v=V if "image" not in k else V_in)
            mvimages = mvimages[:min(mvimages.shape[0], 4), ...]  # max show `4` samples; TODO: make it configurable
            mvimages = rearrange(mvimages, "nb v c h w -> c (nb h) (v w)")
            mvimages = vis_util.tensor_to_image(mvimages.detach())
            formatted_images.append(wandb.Image(mvimages, caption=k))
        wandb.log({"images/validation": formatted_images}, step=global_step)

        for k, v in metrics_dictlist.items():  # "lpips_cfg1.0", "psnr_cfg3.0", ...
            if "cfg1.0" in k:
                wandb.log({f"validation_cfg1.0/{k}": torch.tensor(v).mean().item()}, step=global_step)
            else:
                wandb.log({f"validation/{k}": torch.tensor(v).mean().item()}, step=global_step)


def main():
    PROJECT_NAME = "DiffSplat"

    parser = argparse.ArgumentParser(
        description="Train a diffusion model for 3D object generation",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Path to the HDFS directory to save checkpoints"
    )
    parser.add_argument(
        "--wandb_token_path",
        type=str,
        default="wandb/token",
        help="Path to the WandB login token"
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=None,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=1,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for training"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--val_guidance_scales",
        type=list,
        nargs="+",
        default=[1., 3., 7.5],
        help="CFG scale used for validation"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )

    parser.add_argument(
        "--load_pretrained_gsrecon",
        type=str,
        default="gsrecon_gobj265k_cnp_even4",
        help="Tag of a pretrained GSRecon in this project"
    )
    parser.add_argument(
        "--load_pretrained_gsrecon_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSRecon checkpoint"
    )
    parser.add_argument(
        "--load_pretrained_gsvae",
        type=str,
        default="gsvae_gobj265k_sdxl_fp16",
        help="Tag of a pretrained GSVAE in this project"
    )
    parser.add_argument(
        "--load_pretrained_gsvae_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSVAE checkpoint"
    )
    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Tag of a pretrained MVTransformer in this project"
    )
    parser.add_argument(
        "--load_pretrained_model_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained MVTransformer checkpoint"
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()
    args.val_guidance_scales = [float(x[0]) if isinstance(x, list) else float(x) for x in args.val_guidance_scales]

    # Parse the config file
    configs = util.get_configs(args.config_file, extras)  # change yaml configs by `extras`

    # Parse the option dict
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)
    opt.__post_init__()

    # Create an experiment directory using the `tag`
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.hdfs_dir is not None:
        args.project_hdfs_dir = args.hdfs_dir
        args.hdfs_dir = os.path.join(args.hdfs_dir, args.tag)
        os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir}")

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    # Initialize the accelerator
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
    )
    logger.info(f"Accelerator state:\n{accelerator.state}\n")

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Prepare dataset
    if accelerator.is_local_main_process:
        if not os.path.exists("/tmp/test_dataset"):
            os.system(opt.dataset_setup_script)
    accelerator.wait_for_everyone()  # other processes wait for the main process

    # Load the training and validation dataset
    assert opt.file_dir_train is not None and opt.file_name_train is not None and \
        opt.file_dir_test is not None and opt.file_name_test is not None

    train_dataset = GObjaverseParquetDataset(
        data_source=ParquetChunkDataSource(opt.file_dir_train, opt.file_name_train),
        shuffle=True,
        shuffle_buffer_size=-1,  # `-1`: not shuffle actually
        chunks_queue_max_size=1,  # number of preloading chunks
        # GObjaverse
        opt=opt,
        training=True,
    )
    val_dataset = GObjaverseParquetDataset(
        data_source=ParquetChunkDataSource(opt.file_dir_test, opt.file_name_test),
        shuffle=True,  # shuffle for various visualization
        shuffle_buffer_size=-1,  # `-1`: not shuffle actually
        chunks_queue_max_size=1,  # number of preloading chunks
        # GObjaverse
        opt=opt,
        training=False,
    )
    train_loader = MultiEpochsChunkedDataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    val_loader = MultiEpochsChunkedDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )

    logger.info(f"Load [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n")
    negative_prompt_embed = train_dataset.negative_prompt_embed
    negative_prompt_attention_mask = train_dataset.negative_prompt_attention_mask.squeeze(0)

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * \
        accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]

    # LPIPS loss
    if accelerator.is_main_process:
        _ = LPIPS(net="vgg")
        del _
    accelerator.wait_for_everyone()  # wait for pretrained backbone weights to be downloaded
    lpips_loss = LPIPS(net="vgg").to(accelerator.device)
    lpips_loss = lpips_loss.requires_grad_(False)
    lpips_loss.eval()

    # GSRecon
    gsrecon = GSRecon(opt)
    gsrecon = gsrecon.requires_grad_(False)
    gsrecon = gsrecon.eval()

    # Initialize the model, optimizer and lr scheduler
    in_channels = 4  # hard-coded for SD 1.5/2.1
    if opt.input_concat_plucker:
        in_channels += 6
    if opt.input_concat_binary_mask:
        in_channels += 1
    transformer_from_pretrained_kwargs = {
        "sample_size": opt.input_res // 8,  # `8` hard-coded for SD 1.5/2.1
        "in_channels": in_channels,
        "out_channels": 8,  # hard-coded for PixArt-alpha
        "zero_init_conv_in": opt.zero_init_conv_in,
        "view_concat_condition": opt.view_concat_condition,
        "input_concat_plucker": opt.input_concat_plucker,
        "input_concat_binary_mask": opt.input_concat_binary_mask,
    }
    if opt.load_fp16vae_for_sdxl and args.mixed_precision == "fp16":  # fixed fp16 VAE for SDXL
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    else:
        vae = AutoencoderKL.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="vae")
    if args.load_pretrained_model is None:
        transformer, loading_info = PixArtTransformerMV2DModel.from_pretrained_new(opt.pretrained_model_name_or_path, subfolder="transformer",
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True, output_loading_info=True, **transformer_from_pretrained_kwargs)
        logger.info(f"Loading info: {loading_info}\n")
    else:
        logger.info(f"Load MVTransformer EMA checkpoint from [{args.load_pretrained_model}] iteration [{args.load_pretrained_model_ckpt:06d}]\n")
        args.load_pretrained_model_ckpt = util.load_ckpt(
            os.path.join(args.output_dir, args.load_pretrained_model, "checkpoints"),
            args.load_pretrained_model_ckpt,
            None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_model),
            None,  # `None`: not load model ckpt here
            accelerator,  # manage the process states
        )
        path = f"out/{args.load_pretrained_model}/checkpoints/{args.load_pretrained_model_ckpt:06d}"
        os.system(f"python3 extensions/merge_safetensors.py {path}/transformer_ema")  # merge safetensors for loading
        transformer, loading_info = PixArtTransformerMV2DModel.from_pretrained_new(path, subfolder="transformer_ema",
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True, output_loading_info=True, **transformer_from_pretrained_kwargs)
        logger.info(f"Loading info: {loading_info}\n")

    gsvae = GSAutoencoderKL(opt)

    if not opt.edm_style_training:
        noise_scheduler = DDPMScheduler.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="scheduler")
    else:
        logger.info("Performing EDM-style training")
        noise_scheduler = EulerDiscreteScheduler.from_pretrained("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", subfolder="scheduler")
    if opt.common_tricks:
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config.rescale_betas_zero_snr = True
    if opt.prediction_type is not None:
        noise_scheduler.config.prediction_type = opt.prediction_type
    if opt.beta_schedule is not None:
        noise_scheduler.config.beta_schedule = opt.beta_schedule

    if args.use_ema:
        ema_transformer = MyEMAModel(
            transformer.parameters(),
            model_cls=PixArtTransformerMV2DModel,
            model_config=transformer.config,
            **configs["train"]["ema_kwargs"]
        )

    # Freeze VAE and GSVAE
    vae.requires_grad_(False)
    gsvae.requires_grad_(False)
    vae.eval()
    gsvae.eval()

    trainable_module_names = []
    if opt.trainable_modules is None:
        transformer.requires_grad_(True)
    else:
        transformer.requires_grad_(False)
        for name, module in transformer.named_modules():
            for module_name in tuple(opt.trainable_modules.split(",")):
                if module_name in name:
                    for params in module.parameters():
                        params.requires_grad = True
                    trainable_module_names.append(name)
    logger.info(f"Trainable parameter names: {trainable_module_names}\n")

    # transformer.enable_xformers_memory_efficient_attention()  # use `tF.scaled_dot_product_attention` instead

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "transformer"))

                    # Make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = MyEMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), PixArtTransformerMV2DModel)
                ema_transformer.load_state_dict(load_model.state_dict())
                ema_transformer.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # Pop models so that they are not loaded again
                model = models.pop()

                # Load diffusers style into model
                load_model = PixArtTransformerMV2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if opt.grad_checkpoint:
        transformer.enable_gradient_checkpointing()

    params, params_lr_mult, names_lr_mult = [], [], []
    for name, param in transformer.named_parameters():
        if opt.name_lr_mult is not None:
            for k in opt.name_lr_mult.split(","):
                if k in name:
                    params_lr_mult.append(param)
                    names_lr_mult.append(name)
            if name not in names_lr_mult:
                params.append(param)
        else:
            params.append(param)
    optimizer = get_optimizer(
        params=[
            {"params": params, "lr": configs["optimizer"]["lr"]},
            {"params": params_lr_mult, "lr": configs["optimizer"]["lr"] * opt.lr_mult}
        ],
        **configs["optimizer"]
    )
    logger.info(f"Learning rate x [{opt.lr_mult}] parameter names: {names_lr_mult}\n")

    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader) // accelerator.num_processes / args.gradient_accumulation_steps)  # only account updated steps
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes  # for lr scheduler setting
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes  # reset for multi-gpu
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes  # reset for multi-gpu

    # Load pretrained reconstruction and gsvae models
    logger.info(f"Load GSVAE checkpoint from [{args.load_pretrained_gsvae}] iteration [{args.load_pretrained_gsvae_ckpt:06d}]\n")
    gsvae = util.load_ckpt(
        os.path.join(args.output_dir, args.load_pretrained_gsvae, "checkpoints"),
        args.load_pretrained_gsvae_ckpt,
        None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_gsvae),
        gsvae, accelerator
    )
    logger.info(f"Load GSRecon checkpoint from [{args.load_pretrained_gsrecon}] iteration [{args.load_pretrained_gsrecon_ckpt:06d}]\n")
    gsrecon = util.load_ckpt(
        os.path.join(args.output_dir, args.load_pretrained_gsrecon, "checkpoints"),
        args.load_pretrained_gsrecon_ckpt,
        None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_gsrecon),
        gsrecon, accelerator
    )

    # Prepare everything with `accelerator`
    transformer, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        transformer, optimizer, lr_scheduler, train_loader, val_loader
    )
    # Set classes explicitly for everything
    transformer: DistributedDataParallel
    optimizer: AcceleratedOptimizer
    lr_scheduler: AcceleratedScheduler
    train_loader: DataLoaderShard
    val_loader: DataLoaderShard

    if args.use_ema:
        ema_transformer.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move `gsrecon`, `vae` and `gsvae` to gpu and cast to `weight_dtype`
    gsrecon.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    gsvae.to(accelerator.device, dtype=weight_dtype)

    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    assert configs["train"]["epochs"] * updated_steps_per_epoch == total_updated_steps
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: [{len(val_loader)}]\n")

    # (Optional) Load checkpoint
    global_update_step = 0
    if args.resume_from_iter is not None:
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        # Download from HDFS
        if not os.path.exists(os.path.join(ckpt_dir, f'{args.resume_from_iter:06d}')):
            args.resume_from_iter = util.load_ckpt(
                ckpt_dir,
                args.resume_from_iter,
                args.hdfs_dir,
                None,  # `None`: not load model ckpt here
                accelerator,  # manage the process states
            )
        # Load everything
        accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"))  # torch < 2.4.0 here for `weights_only=False`
        global_update_step = int(args.resume_from_iter)

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = util.save_experiment_params(args, configs, opt, exp_dir)
        util.save_model_architecture(accelerator.unwrap_model(transformer), exp_dir)

    # WandB logger
    if accelerator.is_main_process:
        if args.offline_wandb:
            os.environ["WANDB_MODE"] = "offline"
        with open(args.wandb_token_path, "r") as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()
        wandb.init(
            project=PROJECT_NAME, name=args.tag,
            config=exp_params, dir=exp_dir,
            resume=True
        )
        # Wandb artifact for logging experiment information
        arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
        arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
        arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
        arti_exp_info.add_file(os.path.join(exp_dir, "log.txt"))  # only save the log before training
        wandb.log_artifact(arti_exp_info)

    def get_sigmas(timesteps: Tensor, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(dtype=dtype, device=accelerator.device)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero()[0].item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Start training
    logger.logger.propagate = False  # not propagate to the root logger (console)
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        ncols=125,
        disable=not accelerator.is_main_process
    )
    for batch in yield_forever(train_loader):

        if global_update_step == args.max_train_steps:
            progress_bar.close()
            logger.logger.propagate = True  # propagate to the root logger (console)
            if accelerator.is_main_process:
                wandb.finish()
            logger.info("Training finished!\n")
            return

        transformer.train()

        with accelerator.accumulate(transformer):
            V_in, V_cond, V = opt.num_input_views, opt.num_cond_views, opt.num_views  # TODO: not support V_cond > V_in by now
            cond_idx = [0]  # the first view must be in inputs
            if V_cond > 1:
                cond_idx += np.random.choice(range(1, V), V_cond-1, replace=False).tolist()
            imgs_cond = batch["image"][:, cond_idx, ...]  # (B, V_cond, 3, H, W)
            B = imgs_cond.shape[0]

            prompt_embeds = batch["prompt_embed"]  # (B, N, D)
            prompt_attention_masks = batch["prompt_attention_mask"]  # (B, N)
            negative_prompt_embeds = repeat(negative_prompt_embed.to(accelerator.device), "n d -> b n d", b=B)
            negative_prompt_attention_masks = repeat(negative_prompt_attention_mask.to(accelerator.device), "n -> b n", b=B)

            imgs_out = batch["image"][:, :V_in, ...]
            C2W = batch["C2W"]
            fxfycxcy = batch["fxfycxcy"]

            imgs_cond, prompt_embeds, negative_prompt_embeds, imgs_out, C2W, fxfycxcy = (
                imgs_cond.to(weight_dtype),
                prompt_embeds.to(weight_dtype),
                negative_prompt_embeds.to(weight_dtype),
                imgs_out.to(weight_dtype),
                C2W.to(weight_dtype),
                fxfycxcy.to(weight_dtype),
            )

            input_C2W = C2W[:, :V_in, ...]
            input_fxfycxcy = fxfycxcy[:, :V_in, ...]
            cond_C2W = C2W[:, cond_idx, ...]
            cond_fxfycxcy = fxfycxcy[:, cond_idx,...]

            # (Optional) Plucker embeddings
            if opt.input_concat_plucker:
                H = W = opt.input_res
                plucker, _ = geo_util.plucker_ray(H, W, input_C2W, input_fxfycxcy)  # (B, V_in, 6, H, W)
                if opt.view_concat_condition:
                    cond_plucker, _ = geo_util.plucker_ray(H, W, cond_C2W, cond_fxfycxcy)  # (B, V_cond, 6, H, W)
                    plucker = torch.cat([cond_plucker, plucker], dim=1)  # (B, V_cond+V_in, 6, H, W)
                plucker = rearrange(plucker, "b v c h w -> (b v) c h w")
            else:
                plucker = None

            # VAE input image condition
            if opt.view_concat_condition:
                with torch.no_grad():
                    imgs_cond = rearrange(imgs_cond, "b v c h w -> (b v) c h w")
                    image_latents = vae.config.scaling_factor * vae.encode(imgs_cond * 2. - 1.).latent_dist.sample()  # (B*V_cond, 4, H', W')
                    image_latents = rearrange(image_latents, "(b v) c h w -> b v c h w", v=V_cond)  # (B, V_cond, 4, H', W')

            # Get GS latents
            if opt.input_normal:
                imgs_out = torch.cat([imgs_out, batch["normal"][:, :V_in, ...].to(weight_dtype)], dim=2)
            if opt.input_coord:
                imgs_out = torch.cat([imgs_out, batch["coord"][:, :V_in, ...].to(weight_dtype)], dim=2)
            with torch.no_grad():
                latents = gsvae.scaling_factor * (gsvae.get_gslatents(gsrecon, imgs_out, input_C2W, input_fxfycxcy) - gsvae.shift_factor)  # (B*V_in, 4, H', W')

            noise = torch.randn_like(latents)
            if not opt.edm_style_training:
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=latents.device)
                timesteps = timesteps.long()
            else:
                # In EDM formulation, the model is conditioned on the pre-conditioned noise levels
                # instead of discrete timesteps, so here we sample indices to get the noise levels
                # from `scheduler.timesteps`
                indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,))
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)
            timesteps = repeat(timesteps, "b -> (b v)", v=V_in)  # same noise scale for different views of the same object

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            if not opt.edm_style_training:
                latent_model_input = noisy_latents
            else:
                # For EDM-style training, we first obtain the sigmas based on the continuous timesteps
                # Then precondition the final model inputs based on these sigmas instead of the timesteps
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364
                sigmas = get_sigmas(timesteps, len(noisy_latents.shape), weight_dtype)
                latent_model_input = noisy_latents / ((sigmas**2 + 1)**0.5)

            if opt.cfg_dropout_prob > 0.:
                # Drop a group of multi-view images as a whole
                random_p = torch.rand(B, device=latents.device)

                # Sample masks for the conditioning VAE images
                if opt.view_concat_condition:
                    image_mask_dtype = image_latents.dtype
                    image_mask = 1 - (
                        (random_p >= opt.cfg_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * opt.cfg_dropout_prob).to(image_mask_dtype)
                    )  # actual dropout rate is 2 * `cfg.condition_drop_rate`
                    image_mask = image_mask.reshape(B, 1, 1, 1, 1)
                    # Final VAE image conditioning
                    image_latents = image_mask * image_latents

                # Sample masks for the conditioning text prompts
                text_mask_dtype = prompt_embeds.dtype
                text_mask = 1 - (
                    (random_p < 2 * opt.cfg_dropout_prob).to(text_mask_dtype)
                )  # actual dropout rate is 2 * `cfg.condition_drop_rate`
                text_mask = text_mask.reshape(B, 1, 1)
                # Final text conditioning
                prompt_embeds = text_mask * prompt_embeds + (1 - text_mask) * negative_prompt_embeds
                prompt_attention_masks = text_mask.squeeze(-1) * prompt_attention_masks + (1 - text_mask.squeeze(-1)) * negative_prompt_attention_masks

            prompt_embeds = repeat(prompt_embeds, "b n d -> (b v) n d", v=V_in + (V_cond if opt.view_concat_condition else 0))
            prompt_attention_masks = repeat(prompt_attention_masks, "b n -> (b v) n", v=V_in + (V_cond if opt.view_concat_condition else 0))

            # Concatenate input latents with others
            latent_model_input = rearrange(latent_model_input, "(b v) c h w -> b v c h w", v=V_in)
            if opt.view_concat_condition:
                latent_model_input = torch.cat([image_latents, latent_model_input], dim=1)  # (B, V_in+V_cond, 4, H', W')
            if opt.input_concat_plucker:
                plucker = tF.interpolate(plucker, size=latent_model_input.shape[-2:], mode="bilinear", align_corners=False)
                plucker = rearrange(plucker, "(b v) c h w -> b v c h w", v=V_in + (V_cond if opt.view_concat_condition else 0))
                latent_model_input = torch.cat([latent_model_input, plucker], dim=2)  # (B, V_in(+V_cond), 4+6, H', W')
                plucker = rearrange(plucker, "b v c h w -> (b v) c h w")
            if opt.input_concat_binary_mask:
                if opt.view_concat_condition:
                    latent_model_input = torch.cat([
                        torch.cat([latent_model_input[:, :V_cond, ...], torch.zeros_like(latent_model_input[:, :V_cond, 0:1, ...])], dim=2),
                        torch.cat([latent_model_input[:, V_cond:, ...], torch.ones_like(latent_model_input[:, V_cond:, 0:1, ...])], dim=2),
                    ], dim=1)  # (B, V_in+V_cond, 4+6+1, H', W')
                else:
                    latent_model_input = torch.cat([
                        torch.cat([latent_model_input, torch.ones_like(latent_model_input[:, :, 0:1, ...])], dim=2),
                    ], dim=1)  # (B, V_in, 4+6+1, H', W')
            latent_model_input = rearrange(latent_model_input, "b v c h w -> (b v) c h w")

            # Concatenate input timesteps along the view dimension
            timesteps_input = rearrange(timesteps, "(b v) -> b v", v=V_in)
            if opt.view_concat_condition:
                timesteps_input = torch.cat([timesteps_input[:, :V_cond], timesteps_input], dim=1)  # (B, V_in+V_cond)
            timesteps_input = rearrange(timesteps_input, "b v -> (b v)")

            # Micro-conditions
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            if transformer_from_pretrained_kwargs["sample_size"] == 128:
                added_cond_kwargs = {
                    "resolution": torch.tensor([opt.input_res, opt.input_res],
                        dtype=weight_dtype, device=latents.device).repeat(B, 1),
                    "aspect_ratio": torch.tensor([1.],
                        dtype=weight_dtype, device=latents.device).repeat(B, 1),
                }

            model_pred = transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_masks,
                timestep=timesteps_input,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=dict(
                    num_views=opt.num_input_views + (V_cond if opt.view_concat_condition else 0),
                ),
            ).sample
            model_pred = model_pred.chunk(2, dim=1)[0]  # hard-coded for PixArt-Sigma

            # Only keep the noise prediction for the latents
            if opt.view_concat_condition:
                model_pred = rearrange(model_pred, "(b v) c h w -> b v c h w", v=V_in+V_cond)
                model_pred = rearrange(model_pred[:, V_cond:, ...], "b v c h w -> (b v) c h w")

            if not opt.edm_style_training:
                weighting = 1.
            else:
                # Similar to the input preconditioning, the model predictions are also preconditioned
                # on noised model inputs (before preconditioning) and the sigmas
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364
                if noise_scheduler.config.prediction_type in ["original_sample", "sample"]:
                    model_pred = model_pred
                elif noise_scheduler.config.prediction_type == "epsilon":
                    model_pred = model_pred * (-sigmas) + noisy_latents
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (noisy_latents / (sigmas**2 + 1))
                else:
                    raise ValueError(f"Unknown prediction type [{noise_scheduler.config.prediction_type}]")
                weighting = (sigmas**-2.).float()

            # Get the target for loss depending on the prediction type
            if opt.edm_style_training or noise_scheduler.config.prediction_type in ["original_sample", "sample"]:
                target = latents
            elif noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type [{noise_scheduler.config.prediction_type}]")

            if opt.snr_gamma <= 0.:
                loss = weighting * tF.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = rearrange(loss, "(b v) c h w -> b v c h w", v=V_in)
                loss = loss.mean(dim=list(range(1, len(loss.shape))))
            else:
                assert not opt.edm_style_training, "Min-SNR formulation is not supported when conducting EDM-style training"
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise/v instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, opt.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights /= snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights /= (1. + snr)
                else:
                    raise ValueError(f"Unknown prediction type [{noise_scheduler.config.prediction_type}]")
                # We first calculate the original loss. Then we mean over the non-batch dimensions and
                # rebalance the sample-wise losses with their respective loss weights.
                # Finally, we take the mean of the rebalanced loss.
                loss = tF.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = rearrange(loss, "(b v) c h w -> b v c h w", v=V_in)
                loss = mse_loss_weights * loss.mean(dim=list(range(1, len(loss.shape))))

            # Rendering loss
            use_rendering_loss = np.random.rand() < opt.rendering_loss_prob
            if use_rendering_loss:
                # Get predicted x_0
                if isinstance(noise_scheduler, DDPMScheduler) or isinstance(noise_scheduler, DDIMScheduler):
                    alpha_prod_t = noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps]  # (B*V_in,)
                    beta_prod_t = 1. - alpha_prod_t  # (B*V_in,)
                    while alpha_prod_t.ndim < latents.ndim:
                        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
                        beta_prod_t = beta_prod_t.unsqueeze(-1)
                    if noise_scheduler.config.prediction_type in ["original_sample", "sample"]:
                        pred_original_latents = model_pred
                    elif noise_scheduler.config.prediction_type == "epsilon":
                        pred_original_latents = (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt()
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        pred_original_latents = alpha_prod_t.sqrt() * noisy_latents - beta_prod_t.sqrt() * model_pred
                    else:
                        raise ValueError(f"Unknown prediction type [{noise_scheduler.config.prediction_type}]")
                elif isinstance(noise_scheduler, EulerDiscreteScheduler):
                    if noise_scheduler.config.prediction_type in ["original_sample", "sample"]:
                        pred_original_latents = model_pred
                    elif noise_scheduler.config.prediction_type == "epsilon":
                        pred_original_latents = model_pred * (-sigmas) + noisy_latents
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        pred_original_latents = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (noisy_latents / (sigmas**2 + 1))
                    else:
                        raise ValueError(f"Unknown prediction type [{noise_scheduler.config.prediction_type}]")
                else:
                    raise NotImplementedError  # TODO: support more noise schedulers

                # Render the predicted latents
                pred_original_latents = pred_original_latents.to(weight_dtype)
                pred_original_latents = pred_original_latents / gsvae.scaling_factor + gsvae.shift_factor
                pred_render_outputs = gsvae.decode_and_render_gslatents(
                    gsrecon, pred_original_latents, input_C2W, input_fxfycxcy, C2W, fxfycxcy,
                    use_tiny_decoder=opt.use_tiny_decoder,
                )  # (B, V, 3 or 1, H, W)

                image_mse = tF.mse_loss(batch["image"], pred_render_outputs["image"], reduction="none")
                mask_mse = tF.mse_loss(batch["mask"], pred_render_outputs["alpha"], reduction="none")
                render_loss = image_mse + mask_mse  # (B, V, C, H, W)

                # Depth & Normal
                if opt.coord_weight > 0:
                    assert opt.load_coord
                    coord_mse = tF.mse_loss(batch["coord"], pred_render_outputs["coord"], reduction="none")
                    render_loss += opt.coord_weight * coord_mse  # (B, V, C, H, W)
                else:
                    coord_mse = None
                if opt.normal_weight > 0:
                    assert opt.load_normal
                    normal_cosim = tF.cosine_similarity(batch["normal"], pred_render_outputs["normal"], dim=2).unsqueeze(2)
                    render_loss += opt.normal_weight * (1. - normal_cosim)  # (B, V, C, H, W)
                else:
                    normal_cosim = None

                # LPIPS
                if opt.lpips_weight > 0.:
                    lpips, chunk = [], opt.chunk_size
                    for i in range(B*V):
                        _lpips = lpips_loss(
                            # Downsampled to at most 256 to reduce memory cost
                            tF.interpolate(
                                rearrange(batch["image"], "b v c h w -> (b v) c h w")[i:min(B*V, i+chunk), ...] * 2. - 1.,
                                (256, 256), mode="bilinear", align_corners=False
                            ),
                            tF.interpolate(
                                rearrange(pred_render_outputs["image"], "b v c h w -> (b v) c h w")[i:min(B*V, i+chunk), ...] * 2. - 1.,
                                (256, 256), mode="bilinear", align_corners=False
                            )
                        )  # (`chunk`, 1, 1, 1)
                        lpips.append(_lpips)
                    lpips = torch.cat(lpips, dim=0)  # (B*V, 1, 1, 1)
                    lpips = rearrange(lpips, "(b v) c h w -> b v c h w", v=V)
                    render_loss += opt.lpips_weight * lpips  # (B, V, C, H, W)

                render_loss = render_loss.mean(dim=list(range(1, len(render_loss.shape))))  # (B,)

                if opt.snr_gamma_rendering > 0.:
                    timesteps = rearrange(timesteps, "(b v) -> b v", v=V_in)[:, 0]  # (B,)
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, opt.snr_gamma_rendering * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    render_loss = mse_loss_weights * render_loss

                loss = opt.diffusion_weight * loss + opt.render_weight * render_loss  # (B,)

                # Metric: PNSR, SSIM and LPIPS
                with torch.no_grad():
                    psnr = -10 * torch.log10(torch.mean((batch["image"] - pred_render_outputs["image"].detach()) ** 2))
                    ssim = torch.tensor(calculate_ssim(
                        (rearrange(batch["image"], "b v c h w -> (b v c) h w")
                            .cpu().float().numpy() * 255.).astype(np.uint8),
                        (rearrange(pred_render_outputs["image"].detach(), "b v c h w -> (b v c) h w")
                            .cpu().float().numpy() * 255.).astype(np.uint8),
                        channel_axis=0,
                    ), device=batch["image"].device)
                    if opt.lpips_weight <= 0.:
                        lpips = lpips_loss(
                            # Downsampled to at most 256 to reduce memory cost
                            tF.interpolate(
                                rearrange(batch["image"], "b v c h w -> (b v) c h w") * 2. - 1.,
                                (256, 256), mode="bilinear", align_corners=False
                            ),
                            tF.interpolate(
                                rearrange(pred_render_outputs["image"].detach(), "b v c h w -> (b v) c h w") * 2. - 1.,
                                (256, 256), mode="bilinear", align_corners=False
                            )
                        )

            # Backpropagate
            accelerator.backward(loss.mean())
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            # Gather the losses across all processes for logging (if we use distributed training)
            loss = accelerator.gather(loss.detach()).mean()
            if use_rendering_loss:
                psnr = accelerator.gather(psnr.detach()).mean()
                ssim = accelerator.gather(ssim.detach()).mean()
                lpips = accelerator.gather(lpips.detach()).mean()
                render_loss = accelerator.gather(render_loss.detach()).mean()
                if coord_mse is not None:
                    coord_mse = accelerator.gather(coord_mse.detach()).mean()
                if normal_cosim is not None:
                    normal_cosim = accelerator.gather(normal_cosim.detach()).mean()

            logs = {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            if args.use_ema:
                ema_transformer.step(transformer.parameters())
                logs.update({"ema": ema_transformer.cur_decay_value})
            if use_rendering_loss:
                logs.update({"render_loss": render_loss.item()})

            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            global_update_step += 1

            logger.info(
                f"[{global_update_step:06d} / {total_updated_steps:06d}] " +
                f"loss: {logs['loss']:.4f}, lr: {logs['lr']:.2e}" +
                f", ema: {logs['ema']:.4f}" if args.use_ema else "" +
                f", render: {logs['render_loss']:.4f}" if use_rendering_loss else ""
            )

            # Log the training progress
            if global_update_step % configs["train"]["log_freq"] == 0 or global_update_step == 1 \
                or global_update_step % updated_steps_per_epoch == 0:  # last step of an epoch
                if accelerator.is_main_process:
                    wandb.log({
                        "training/loss": logs["loss"],
                        "training/lr": logs["lr"],
                    }, step=global_update_step)
                    if args.use_ema:
                        wandb.log({
                            "training/ema": logs["ema"]
                        }, step=global_update_step)
                    if use_rendering_loss:
                        wandb.log({
                            "training/psnr": psnr.item(),
                            "training/ssim": ssim.item(),
                            "training/lpips": lpips.item(),
                            "training/render_loss": logs["render_loss"]
                        }, step=global_update_step)
                        if coord_mse is not None:
                            wandb.log({
                                "training/coord_mse": coord_mse.item()
                            }, step=global_update_step)
                        if normal_cosim is not None:
                            wandb.log({
                                "training/normal_cosim": normal_cosim.item()
                            }, step=global_update_step)

            # Save checkpoint
            if (global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                or global_update_step == total_updated_steps):  # 3. last step of an epoch

                gc.collect()
                if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                elif accelerator.is_main_process:
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                accelerator.wait_for_everyone()  # ensure all processes have finished saving
                if accelerator.is_main_process:
                    if args.hdfs_dir is not None:
                        util.save_ckpt(ckpt_dir, global_update_step, args.hdfs_dir)
                gc.collect()

            # Evaluate on the validation set
            if (global_update_step == 1
                or (global_update_step % configs["train"]["early_eval_freq"] == 0 and
                global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                or global_update_step == total_updated_steps):  # 4. last step of an epoch

                # Visualize images for rendering loss
                if accelerator.is_main_process and use_rendering_loss:
                    train_vis_dict = {
                        "images_render": pred_render_outputs["image"],  # (B, V, 3, H, W)
                        "images_gt": batch["image"],  # (B, V, 3, H, W)
                    }
                    if opt.vis_coords:
                        train_vis_dict.update({
                            "images_coord": pred_render_outputs["coord"],  # (B, V, 3, H, W)
                        })
                        if opt.load_coord:
                            train_vis_dict.update({
                                "images_gt_coord": batch["coord"]  # (B, V, 3, H, W)
                            })
                    if opt.vis_normals:
                        train_vis_dict.update({
                            "images_normal": pred_render_outputs["normal"],  # (B, V, 3, H, W)
                        })
                        if opt.load_normal:
                            train_vis_dict.update({
                                "images_gt_normal": batch["normal"]  # (B, V, 3, H, W)
                            })
                    wandb.log({
                        "images/training": vis_util.wandb_mvimage_log(train_vis_dict)
                    }, step=global_update_step)

                torch.cuda.empty_cache()
                gc.collect()

                # Use EMA parameters for evaluation
                if args.use_ema:
                    # Store the Transformer parameters temporarily and load the EMA parameters to perform inference
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())

                transformer.eval()

                log_validation(
                    val_loader,
                    negative_prompt_embed,
                    negative_prompt_attention_mask,
                    lpips_loss,
                    gsrecon,
                    gsvae,
                    vae,
                    transformer,
                    global_update_step,
                    accelerator,
                    args,
                    opt,
                )

                if args.use_ema:
                    # Switch back to the original Transformer parameters
                    ema_transformer.restore(transformer.parameters())

                torch.cuda.empty_cache()
                gc.collect()


if __name__ == "__main__":
    main()
