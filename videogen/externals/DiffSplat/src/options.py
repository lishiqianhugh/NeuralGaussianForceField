from typing import *

from dataclasses import dataclass
from copy import deepcopy


HDFS_DIR = "<HDFS_DIR>"  # data is stored in an internal HDFS in this project


@dataclass
class Options:
    # Dataset
    input_res: int = 256
        ## Camera
    num_input_views: int = 4
    num_views: int = 8
    load_even_views: bool = True
    exclude_topdown_views: bool = False
    norm_camera: bool = True
    norm_radius: float = 1.4  # the min distance in GObjaverse (cf. `RichDreamer` Sec. 3.1); only used when `norm_camera` is True
    fxfy: float = 1422.222 / 1024  # for GObjaverse only (https://github.com/modelscope/richdreamer/issues/10#issuecomment-1890870640)
        ## Content
    load_albedo: bool = False
    load_normal: bool = True
    load_coord: bool = True
    load_mr: bool = False
    load_canny: bool = False
    load_depth: bool = False
    normalize_depth: bool = True  # to [0, 1]
    dataset_name: Literal[
        "gobj83k",
        "gobj265k",
    ] = "gobj83k"
    dataset_size: int = None  # set later
    prompt_embed_dir: Optional[str] = None  # set later
        ## ParquetDataset
    file_dir_train: str = f"{HDFS_DIR}/GObjaverse_parquet"
    file_name_train: str = None  # set later
    file_dir_test: str = "/tmp/test_dataset"
    file_name_test: str = "GObjaverse-val"
    dataset_setup_script: str = f"mkdir -p /tmp/test_dataset && hdfs dfs -ls {HDFS_DIR}/GObjaverse_parquet/GObjaverse-val-* | grep '^-' | " + "awk '{print $8}' | xargs -n 1 -P 5 -I {} hdfs dfs -get {} /tmp/test_dataset"

    # GSRecon
    input_albedo: bool = False
    input_normal: bool = True
    input_coord: bool = True
    input_mr: bool = False
        ## Transformer
    llama_style: bool = True
    patch_size: int = 8
    dim: int = 512
    num_blocks: int = 12
    num_heads: int = 8
    grad_checkpoint: bool = True
        ## Rendering
    render_type: Literal[
        "default",
        "deferred",
    ] = "default"
    deferred_bp_patch_size: int = 64
    znear: float = 0.01
    zfar: float = 100.
    scale_min: float = 0.0005
    scale_max: float = 0.02

    # Elevation estimation
    elevest_backbone_name: Literal[
        "dinov2_vits14_reg",
        "dinov2_vitb14_reg",
        "dinov2_vitl14_reg",
    ] = "dinov2_vitb14_reg"
    freeze_backbone: bool = False
    ele_min: float = -40.  # actual min: -30.
    ele_max: float = 10.  # actual max: 5.
    elevest_num_classes: int = 25
    elevest_reg_weight: float = 1.

    # GSVAE
    vae_from_scratch: bool = False
    use_tinyae: bool = False
    freeze_encoder: bool = False
    use_tiny_decoder: bool = False
    scaling_factor: Optional[float] = None
    shift_factor: Optional[float] = None

    # GSDiff
    pretrained_model_name_or_path: Literal[
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "PixArt-alpha/PixArt-XL-2-256x256",
        "PixArt-alpha/PixArt-XL-2-512x512",
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        "PixArt-alpha/PixArt-Sigma-XL-2-256x256",
        "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-medium",
        "stabilityai/stable-diffusion-3.5-large",
        "black-forest-labs/FLUX.1-dev",
        "madebyollin/sdxl-vae-fp16-fix",
        "lambdalabs/sd-image-variations-diffusers",
        "stabilityai/stable-diffusion-2-1-unclip",
        "chenguolin/sv3d-diffusers",
    ] = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    load_fp16vae_for_sdxl: bool = True
        ## Config
    from_scratch: bool = False
    cfg_dropout_prob: float = 0.05  # actual prob is x2; see the training code
    snr_gamma: float = 0.  # Min-SNR trick; `0.` menas not used
    num_inference_steps: int = 20
    noise_scheduler_type: Literal[
        "ddim",
        "dpmsolver++",
        "sde-dpmsolver++",
    ] = "dpmsolver++"
    prediction_type: Optional[str] = None  # `None` means using default prediction type
    beta_schedule: Optional[str] = None  # `None` means using the default beta schedule
    edm_style_training: bool = False  # EDM scheduling; cf. https://arxiv.org/pdf/2206.00364
    common_tricks: bool = True  # cf. https://arxiv.org/pdf/2305.08891 (including: 1. trailing timestep spacing, 2. rescaling to zero snr)
            ### SD3; cf. https://arxiv.org/pdf/2403.03206
    weighting_scheme: Literal[
        "sigma_sqrt",
        "logit_normal",
        "mode",
        "cosmap",
    ] = "logit_normal"
    logit_mean: float = 0.
    logit_std: float = 1.
    mode_scale: float = 1.29
    precondition_outputs: bool = False  # whether prediction x_0
        ## Model
    trainable_modules: Optional[str] = None  # train all parameters if None
    name_lr_mult: Optional[str] = None
    lr_mult: float = 1.
            ### Conditioning
    zero_init_conv_in: bool = True  # whether zero_init new conv_in params
    view_concat_condition: bool = False  # `True` for image-cond
    input_concat_plucker: bool = True
    input_concat_binary_mask: bool = False
    num_cond_views: int = 1
            ### Inference
    init_std: float = 0.  # cf. Instant3D inference trick, `0.` means not used
    init_noise_strength: float = 0.98  # used with `init_std`; cf. Instant3D inference trick, `1.` means not used
    init_bg: float = 0.  # used with `init_std` and `init_noise_strength`; gray background for the initialization
            ### ControlNet
    controlnet_type: Literal[
        "normal",
        "depth",
        "canny",
    ] = "normal"
    controlnet_input_channels: int = 3
    guess_mode: bool = False
    controlnet_scale: float = 1.
        ## Rendering loss
    rendering_loss_prob: float = 0.
    snr_gamma_rendering: float = 0.  # Min-SNR trick for rendering loss; `0.` menas not used

    # Training
    chunk_size: int = 1  # chunk size for GSRecon and GSVAE inference to save memory
    coord_weight: float = 0.  # render coords for supervision
    normal_weight: float = 0.  # render normals for supervision
    recon_weight: float = 1.  # GSVAE reconstruction weight
    render_weight: float = 1.  # GSVAE rendering weight
    diffusion_weight: float = 1.  # GSDiff diffusion weight
        ## LPIPS
    lpips_resize: int = 256  # `0` means no resizing
    lpips_weight: float = 1.  # lpips weight in GSRecon, GSVAE, GSDiff rendering
    lpips_warmup_start: int = 0
    lpips_warmup_end: int = 0

    # Visualization
    vis_pseudo_images: bool = False  # decode Gaussian latents by the image decoder
    vis_coords: bool = False
    vis_normals: bool = False

    def __post_init__(self):
        if self.dataset_name == "gobj83k":
            self.dataset_size = 83296
            self.file_name_train = "GObjaverse-train-280k-83k"
        elif self.dataset_name == "gobj265k":
            self.dataset_size = 265232
            self.file_name_train = "GObjaverse-train-280k"
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")


def _update_opt(opt: Options, **kwargs) -> Options:
    new_opt = deepcopy(opt)
    for k, v in kwargs.items():
        setattr(new_opt, k, v)
    return new_opt


# Set all options for different tasks and models
opt_dict: Dict[str, Options] = {}


# GRM
opt_dict["gsrecon"] = Options(dataset_name="gobj265k")


# Elevation estimation
opt_dict["elevest"] = Options(
    input_res=224,
    load_even_views=False,
    exclude_topdown_views=True,
    load_normal=False,
    load_coord=False,
    dataset_name="gobj265k",
    name_lr_mult="backbone",
    lr_mult=0.1,
)


# GSVAE
    ## SD-based
opt_dict["gsvae"] = Options(dataset_name="gobj265k")
    ## SDXL-based
opt_dict["gsvae_sdxl"] = _update_opt(
    opt_dict["gsvae"],
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
)
opt_dict["gsvae_sdxl_fp16"] = _update_opt(
    opt_dict["gsvae"],
    pretrained_model_name_or_path="madebyollin/sdxl-vae-fp16-fix",
)
    ## SD3-based
opt_dict["gsvae_sd3m"] = _update_opt(
    opt_dict["gsvae"],
    pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers",
)
opt_dict["gsvae_sd35m"] = _update_opt(
    opt_dict["gsvae"],
    pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium",
)


# GSDiff
    ## SD15-based
opt_dict["gsdiff_sd15"] = Options(
    prompt_embed_dir="/tmp/GObjaverse_sd15_prompt_embeds",
    pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
)
    ## SDXL-based
opt_dict["gsdiff_sdxl"] = Options(
    prompt_embed_dir="/tmp/GObjaverse_sdxl_prompt_embeds",
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
)
    ## PAA-based
opt_dict["gsdiff_paa"] = Options(
    prompt_embed_dir="/tmp/GObjaverse_paa_prompt_embeds",
    pretrained_model_name_or_path="PixArt-alpha/PixArt-XL-2-512x512",
)
    ## PAS-based
opt_dict["gsdiff_pas"] = Options(
    prompt_embed_dir="/tmp/GObjaverse_pas_prompt_embeds",
    pretrained_model_name_or_path="PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
)
    ## SD3-based
opt_dict["gsdiff_sd3m"] = Options(
    prompt_embed_dir="/tmp/GObjaverse_sd3m_prompt_embeds",
    pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers",
)
opt_dict["gsdiff_sd35m"] = Options(
    prompt_embed_dir="/tmp/GObjaverse_sd35m_prompt_embeds",
    pretrained_model_name_or_path="stabilityai/stable-diffusion-3.5-medium",
)
