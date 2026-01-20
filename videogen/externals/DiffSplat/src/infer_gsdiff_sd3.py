import warnings
warnings.filterwarnings("ignore")  # ignore all warnings

from typing import *

import os
import argparse
import logging
import time

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn.functional as tF
from einops import rearrange
import accelerate
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from kiui.cam import orbit_camera

from src.options import opt_dict
from src.models import GSAutoencoderKL, GSRecon, ElevEst
import src.utils.util as util
import src.utils.op_util as op_util
import src.utils.geo_util as geo_util
import src.utils.vis_util as vis_util
from src.utils.metrics import TextConditionMetrics

from extensions.diffusers_diffsplat import SD3TransformerMV2DModel, StableMVDiffusion3Pipeline, FlowDPMSolverMultistepScheduler


def main():
    parser = argparse.ArgumentParser(
        description="Infer a diffusion model for 3D object generation"
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
        default=None,
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
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use"
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Use half precision for inference"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )
    parser.add_argument(
        "--not_use_t5",
        action="store_true",
        help="Not use T5 for text embedding"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the image for reconstruction"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to the directory of images for reconstruction"
    )
    parser.add_argument(
        "--infer_from_iter",
        type=int,
        default=-1,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--rembg_and_center",
        action="store_true",
        help="Whether or not to remove background and center the image"
    )
    parser.add_argument(
        "--rembg_model_name",
        default="u2net",
        type=str,
        help="Rembg model, see https://github.com/danielgatis/rembg#models"
    )
    parser.add_argument(
        "--border_ratio",
        default=0.2,
        type=float,
        help="Rembg output border ratio"
    )

    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="flow",  # "sde-dpmsolver++", "dpmsolver++", ...
        help="Type of diffusion scheduler"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Diffusion steps for inference"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.,
        help="Classifier-free guidance scale for inference"
    )
    parser.add_argument(
        "--triangle_cfg_scaling",
        action="store_true",
        help="Whether or not to use triangle classifier-free guidance scaling"
    )
    parser.add_argument(
        "--min_guidance_scale",
        type=float,
        default=1.,
        help="Minimum of triangle cfg scaling"
    )

    parser.add_argument(
        "--init_std",
        type=float,
        default=0.,
        help="Standard deviation of Gaussian grids (cf. Instant3D) for initialization"
    )
    parser.add_argument(
        "--init_noise_strength",
        type=float,
        default=0.98,
        help="Noise strength of Gaussian grids (cf. Instant3D) for initialization"
    )
    parser.add_argument(
        "--init_bg",
        type=float,
        default=0.,
        help="Gray background of Gaussian grids for initialization"
    )

    parser.add_argument(
        "--elevation",
        type=float,
        default=None,
        help="The elevation of rendering"
    )
    parser.add_argument(
        "--use_elevest",
        action="store_true",
        help="Whether or not to use an elevation estimation model"
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=1.4,
        help="The distance of rendering"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Caption prompt for generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        # default="worst quality, normal quality, low quality, low res, blurry, ugly, disgusting",
        default="",
        help="Negative prompt for better classifier-free guidance"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to the file of text prompts for generation"
    )

    parser.add_argument(
        "--render_res",
        type=int,
        default=None,
        help="Resolution of GS rendering"
    )
    parser.add_argument(
        "--opacity_threshold",
        type=float,
        default=0.,
        help="The min opacity value for filtering floater Gaussians"
    )
    parser.add_argument(
        "--opacity_threshold_ply",
        type=float,
        default=0.,
        help="The min opacity value for filtering floater Gaussians in ply file"
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Whether or not to save the generated Gaussian ply file"
    )
    parser.add_argument(
        "--output_video_type",
        type=str,
        default=None,
        help="Type of the output video"
    )

    parser.add_argument(
        "--name_by_id",
        action="store_true",
        help="Whether or not to name the output by the prompt/image ID"
    )
    parser.add_argument(
        "--eval_text_cond",
        action="store_true",
        help="Whether or not to evaluate text-conditioned generation"
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
        default="gsvae_gobj265k_sd3",
        help="Tag of a pretrained GSVAE in this project"
    )
    parser.add_argument(
        "--load_pretrained_gsvae_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSVAE checkpoint"
    )
    parser.add_argument(
        "--load_pretrained_elevest",
        type=str,
        default="elevest_gobj265k_b_C25",
        help="Tag of a pretrained GSRecon in this project"
    )
    parser.add_argument(
        "--load_pretrained_elevest_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSRecon checkpoint"
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()

    # Parse the config file
    configs = util.get_configs(args.config_file, extras)  # change yaml configs by `extras`

    # Parse the option dict
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = time.strftime("%Y-%m-%d_%H:%M") + "_" + \
            os.path.split(args.config_file)[-1].split()[0]  # config file name

    # Create the experiment directory
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    infer_dir = os.path.join(exp_dir, "inference")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(infer_dir, exist_ok=True)
    if args.hdfs_dir is not None:
        args.project_hdfs_dir = args.hdfs_dir
        args.hdfs_dir = os.path.join(args.hdfs_dir, args.tag)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, args.tag, "log_infer.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)
    logger.propagate = True  # propagate to the root logger (console)

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Set options for image-conditioned models
    if args.image_path is not None or args.image_dir is not None:
        opt.prediction_type = "v_prediction"
        opt.view_concat_condition = True
        opt.input_concat_binary_mask = True
        if args.guidance_scale > 3.:
            logger.info(
                f"WARNING: guidance scale ({args.guidance_scale}) is too large for image-conditioned models. " +
                "Please set it to a smaller value (e.g., 2.0) for better results.\n"
            )

    # Load the image for reconstruction
    if args.image_dir is not None:
        logger.info(f"Load images from [{args.image_dir}]\n")
        image_paths = [
            os.path.join(args.image_dir, filename)
            for filename in os.listdir(args.image_dir)
            if filename.endswith(".png") or filename.endswith(".jpg") or \
                filename.endswith(".jpeg") or filename.endswith(".webp")
        ]
        image_paths = sorted(image_paths)
    elif args.image_path is not None:
        logger.info(f"Load image from [{args.image_path}]\n")
        image_paths = [args.image_path]
    else:
        logger.info(f"No image condition\n")
        image_paths = [None]

    # Load text prompts for generation
    if args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            prompts = prompts_2 = prompts_3 = [line.strip() for line in f.readlines() if line.strip() != ""]
        negative_prompt = negative_prompt_2 = negative_prompt_3 = args.negative_prompt.replace("_", " ")
        negative_promts, negative_promts_2, negative_prompts_3 = \
            [negative_prompt] * len(prompts), [negative_prompt_2] * len(prompts_2), [negative_prompt_3] * len(prompts_3)
    else:
        prompt = prompt_2 = prompt_3 = args.prompt.replace("_", " ")
        negative_prompt = negative_prompt_2 = negative_prompt_3 = args.negative_prompt.replace("_", " ")
        prompts, prompts_2, prompts_3, negative_promts, negative_promts_2, negative_prompts_3 = \
            [prompt], [prompt_2], [prompt_3], [negative_prompt], [negative_prompt_2], [negative_prompt_3]

    # Initialize the model, optimizer and lr scheduler
    in_channels = 16  # hard-coded for SD3
    if opt.input_concat_plucker:
        in_channels += 6
    if opt.input_concat_binary_mask:
        in_channels += 1
    transformer_from_pretrained_kwargs = {
        "sample_size": opt.input_res // 8,  # `8` hard-coded for SD3
        "in_channels": in_channels,
        "zero_init_conv_in": opt.zero_init_conv_in,
        "view_concat_condition": opt.view_concat_condition,
        "input_concat_plucker": opt.input_concat_plucker,
        "input_concat_binary_mask": opt.input_concat_binary_mask,
    }
    tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(opt.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16")
    tokenizer_2 = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(opt.pretrained_model_name_or_path, subfolder="text_encoder_2", variant="fp16")
    if not args.not_use_t5:
        tokenizer_3 = T5TokenizerFast.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer_3")
        text_encoder_3 = T5EncoderModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder="text_encoder_3", variant="fp16")
    else:
        tokenizer_3 = None
        text_encoder_3 = None
    vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")

    gsvae = GSAutoencoderKL(opt)
    gsrecon = GSRecon(opt)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
    if "dpmsolver" in args.scheduler_type:
        new_noise_scheduler = FlowDPMSolverMultistepScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
        new_noise_scheduler.config.algorithm_type = args.scheduler_type
        new_noise_scheduler.config.flow_shift = noise_scheduler.config.shift
        noise_scheduler = new_noise_scheduler

    # Load checkpoint
    logger.info(f"Load checkpoint from iteration [{args.infer_from_iter}]\n")
    if not os.path.exists(os.path.join(ckpt_dir, f"{args.infer_from_iter:06d}")):
        args.infer_from_iter = util.load_ckpt(
            ckpt_dir,
            args.infer_from_iter,
            args.hdfs_dir,
            None,  # `None`: not load model ckpt here
        )
    path = os.path.join(ckpt_dir, f"{args.infer_from_iter:06d}")
    os.system(f"python3 extensions/merge_safetensors.py {path}/transformer_ema")  # merge safetensors for loading
    transformer, loading_info = SD3TransformerMV2DModel.from_pretrained_new(path, subfolder="transformer_ema",
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True, output_loading_info=True, **transformer_from_pretrained_kwargs)
    for key in loading_info.keys():
        assert len(loading_info[key]) == 0  # no missing_keys, unexpected_keys, mismatched_keys, error_msgs

    # Freeze all models
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    gsvae.requires_grad_(False)
    gsrecon.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder.eval()
    text_encoder_2.eval()
    vae.eval()
    gsvae.eval()
    gsrecon.eval()
    transformer.eval()
    if not args.not_use_t5:
        text_encoder_3.requires_grad_(False)
        text_encoder_3.eval()

    # Load pretrained reconstruction and gsvae models
    logger.info(f"Load GSVAE checkpoint from [{args.load_pretrained_gsvae}] iteration [{args.load_pretrained_gsvae_ckpt:06d}]\n")
    gsvae = util.load_ckpt(
        os.path.join(args.output_dir, args.load_pretrained_gsvae, "checkpoints"),
        args.load_pretrained_gsvae_ckpt,
        None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_gsvae),
        gsvae,
    )
    logger.info(f"Load GSRecon checkpoint from [{args.load_pretrained_gsrecon}] iteration [{args.load_pretrained_gsrecon_ckpt:06d}]\n")
    gsrecon = util.load_ckpt(
        os.path.join(args.output_dir, args.load_pretrained_gsrecon, "checkpoints"),
        args.load_pretrained_gsrecon_ckpt,
        None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_gsrecon),
        gsrecon,
    )

    text_encoder = text_encoder.to(f"cuda:{args.gpu_id}")
    text_encoder_2 = text_encoder_2.to(f"cuda:{args.gpu_id}")
    vae = vae.to(f"cuda:{args.gpu_id}")
    gsvae = gsvae.to(f"cuda:{args.gpu_id}")
    gsrecon = gsrecon.to(f"cuda:{args.gpu_id}")
    transformer = transformer.to(f"cuda:{args.gpu_id}")
    if not args.not_use_t5:
        text_encoder_3 = text_encoder_3.to(f"cuda:{args.gpu_id}")

    # Set diffusion pipeline
    V_in = opt.num_input_views
    pipeline = StableMVDiffusion3Pipeline(
        text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
        text_encoder_3=text_encoder_3, tokenizer_3=tokenizer_3,
        vae=vae, transformer=transformer,
        scheduler=noise_scheduler,
    )
    pipeline.set_progress_bar_config(disable=False)
    # pipeline.enable_xformers_memory_efficient_attention()

    if args.seed >= 0:
        generator = torch.Generator(device=f"cuda:{args.gpu_id}").manual_seed(args.seed)
    else:
        generator = None

    # Set rendering resolution
    if args.render_res is None:
        args.render_res = opt.input_res

    # Load elevation estimation model
    if args.use_elevest:
        elevest = ElevEst(opt)
        elevest.requires_grad_(False)
        elevest.eval()

        logger.info(f"Load ElevEst checkpoint from [{args.load_pretrained_elevest}] iteration [{args.load_pretrained_elevest_ckpt:06d}]\n")
        elevest = util.load_ckpt(
            os.path.join(args.output_dir, args.load_pretrained_elevest, "checkpoints"),
            args.load_pretrained_elevest_ckpt,
            None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_elevest),
            elevest,
        )
        elevest = elevest.to(f"cuda:{args.gpu_id}")

    # Save all experimental parameters of this run to a file (args and configs)
    _ = util.save_experiment_params(args, configs, opt, infer_dir)

    # Evaluation for text-conditioned generation
    text_condition_metrics = TextConditionMetrics(device_idx=args.gpu_id) if args.eval_text_cond else None

    # Inference
    CLIPSIM, CLIPRPREC, IMAGEREWARD = [], [], []
    for i in range(len(image_paths)):  # to save outputs with the same name as the input image
        image_path = image_paths[i]
        if image_path is not None:
            # (Optional) Remove background and center the image
            if args.rembg_and_center:
                image_path = op_util.rembg_and_center_wrapper(image_path, opt.input_res, args.border_ratio, model_name=args.rembg_model_name)

            image_name = image_path.split('/')[-1].split('.')[0]

            image = plt.imread(image_path)
            if image.shape[-1] == 4:
                image = image[..., :3] * image[..., 3:4] + (1. - image[..., 3:4])  # RGBA to RGB white background
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            image = tF.interpolate(
                image, size=(opt.input_res, opt.input_res),
                mode="bilinear", align_corners=False, antialias=True
            )
            image = image.unsqueeze(1).to(device=f"cuda:{args.gpu_id}")  # (B=1, V_cond=1, 3, H, W)
        else:
            image_name = ""
            image = None

        # Elevation estimation
        if image is not None:
            if args.elevation is None:
                assert args.use_elevest, "Elevation estimation is required for image-conditioned generation if `args.elevation` is not provided"
                with torch.no_grad():
                    elevation = -elevest.predict_elev(image.squeeze(1)).cpu().item()
                logger.info(f"Elevation estimation: [{elevation}] deg\n")
            else:
                elevation = args.elevation
        else:
            elevation = args.elevation if args.elevation is not None else 10.

        # Get plucker embeddings
        fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5], device=f"cuda:{args.gpu_id}").float()
        elevations = torch.tensor([-elevation] * 4, device=f"cuda:{args.gpu_id}").deg2rad().float()
        azimuths = torch.tensor([0., 90., 180., 270.], device=f"cuda:{args.gpu_id}").deg2rad().float()  # hard-coded
        radius = torch.tensor([args.distance] * 4, device=f"cuda:{args.gpu_id}").float()
        input_C2W = geo_util.orbit_camera(elevations, azimuths, radius, is_degree=False)  # (V_in, 4, 4)
        input_C2W[:, :3, 1:3] *= -1  # OpenGL -> OpenCV
        input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(input_C2W.shape[0], 1)  # (V_in, 4)
        if opt.input_concat_plucker:
            H = W = opt.input_res
            plucker, _ = geo_util.plucker_ray(H, W, input_C2W.unsqueeze(0), input_fxfycxcy.unsqueeze(0))
            plucker = plucker.squeeze(0)  # (V_in, 6, H, W)
            if opt.view_concat_condition:
                plucker = torch.cat([plucker[0:1, ...], plucker], dim=0)  # (V_in+1, 6, H, W)
        else:
            plucker = None

        IMAGES = []
        for j in range(len(prompts)):
            prompt, prompt_2, prompt_3, negative_prompt, negative_prompt_2, negative_prompt_3 = \
                prompts[j], prompts_2[j], prompts_3[j], negative_promts[j], negative_promts_2[j], negative_prompts_3[j]

            MAX_NAME_LEN = 20  # TODO: make `20` configurable
            prompt_name = prompt[:MAX_NAME_LEN] + "..." if prompt[:MAX_NAME_LEN] != "" else prompt
            if not args.name_by_id:
                name = f"[{image_name}]_[{prompt_name}]_{args.infer_from_iter:06d}"
            else:
                name = f"{i:03d}_{j:03d}_{args.infer_from_iter:06d}"

            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16 if args.half_precision else torch.float32):
                    out = pipeline(
                        image, prompt=prompt, negative_prompt=negative_prompt,
                        prompt_2=prompt_2, negative_prompt_2=negative_prompt_2,
                        prompt_3=prompt_3, negative_prompt_3=negative_prompt_3,
                        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                        triangle_cfg_scaling=args.triangle_cfg_scaling,
                        min_guidance_scale=args.min_guidance_scale, max_guidance_scale=args.guidance_scale,
                        output_type="latent", generator=generator,
                        plucker=plucker, num_views=V_in,
                        init_std=args.init_std, init_noise_strength=args.init_noise_strength, init_bg=args.init_bg,
                    ).images

                    out = out / gsvae.scaling_factor + gsvae.shift_factor
                    render_outputs = gsvae.decode_and_render_gslatents(
                        gsrecon,
                        out, input_C2W.unsqueeze(0), input_fxfycxcy.unsqueeze(0),
                        height=args.render_res, width=args.render_res,
                        opacity_threshold=args.opacity_threshold,
                    )
                    images = render_outputs["image"].squeeze(0)  # (V_in, 3, H, W)
                    IMAGES.append(images)
                    images = vis_util.tensor_to_image(rearrange(images, "v c h w -> c h (v w)"))  # (H, V*W, 3)
                    imageio.imwrite(os.path.join(infer_dir, f"{name}_gs.png"), images)

                    # Save Gaussian ply file
                    if args.save_ply:
                        ply_path = os.path.join(infer_dir, f"{name}.ply")
                        render_outputs["pc"][0].save_ply(ply_path, args.opacity_threshold_ply)

                    # Render video
                    if args.output_video_type is not None:
                        fancy_video = "fancy" in args.output_video_type
                        save_gif = "gif" in args.output_video_type

                        if fancy_video:
                            render_azimuths = np.arange(0., 720., 4)
                        else:
                            render_azimuths = np.arange(0., 360., 2)

                        C2W = []
                        for i in range(len(render_azimuths)):
                            c2w = torch.from_numpy(
                                orbit_camera(-elevation, render_azimuths[i], radius=args.distance, opengl=True)
                            ).to(f"cuda:{args.gpu_id}")
                            c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV
                            C2W.append(c2w)
                        C2W = torch.stack(C2W, dim=0)  # (V, 4, 4)
                        fxfycxcy_V = fxfycxcy.unsqueeze(0).repeat(C2W.shape[0], 1)

                        images = []
                        for v in tqdm(range(C2W.shape[0]), desc="Rendering", ncols=125):
                            render_outputs = gsvae.decode_and_render_gslatents(
                                gsrecon,
                                out,  # (V_in, 4, H', W')
                                input_C2W.unsqueeze(0),  # (1, V_in, 4, 4)
                                input_fxfycxcy.unsqueeze(0),  # (1, V_in, 4)
                                C2W[v].unsqueeze(0).unsqueeze(0),  # (B=1, V=1, 4, 4)
                                fxfycxcy_V[v].unsqueeze(0).unsqueeze(0),  # (B=1, V=1, 4)
                                height=args.render_res, width=args.render_res,
                                scaling_modifier=min(render_azimuths[v] / 360, 1) if fancy_video else 1.,
                                opacity_threshold=args.opacity_threshold,
                            )
                            image = render_outputs["image"].squeeze(0).squeeze(0)  # (3, H, W)
                            images.append(vis_util.tensor_to_image(image, return_pil=save_gif))

                        if save_gif:
                            images[0].save(
                                os.path.join(infer_dir, f"{name}.gif"),
                                save_all=True,
                                append_images=images[1:],
                                optimize=False,
                                duration=1000 // 30,
                                loop=0,
                            )
                        else:  # save mp4
                            images = np.stack(images, axis=0)  # (V, H, W, 3)
                            imageio.mimwrite(os.path.join(infer_dir, f"{name}.mp4"), images, fps=30)

        # Evaluate text-conditioned generation across views
        if text_condition_metrics is not None:
            IMAGES = torch.stack(IMAGES, dim=0)  # (N_prompt, V_in, 3, H, W)
            for v in range(V_in):
                clipsim, cliprprec, imagereward = text_condition_metrics.evaluate(
                    [vis_util.tensor_to_image(IMAGES[i, v, ...], return_pil=True) for i in range(len(IMAGES))],
                    prompts,
                )
                CLIPSIM.append(clipsim)
                CLIPRPREC.append(cliprprec)
                IMAGEREWARD.append(imagereward)

        if image_path is not None and args.rembg_and_center:
            os.system(f"rm {image_path}")

    logger.info(f"Mean\t CosSim: {np.mean(CLIPSIM):.6f}\t R-Prec: {np.mean(CLIPRPREC):.6f}\t ImageReward: {np.mean(IMAGEREWARD):.6f}")
    logger.info(f"Std\t CosSim: {np.std(CLIPSIM):.6f}\t R-Prec: {np.std(CLIPRPREC):.6f}\t ImageReward: {np.std(IMAGEREWARD):.6f}")
    logger.info("Inference finished!\n")


if __name__ == "__main__":
    main()
