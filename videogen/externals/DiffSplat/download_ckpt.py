import os
import argparse
from huggingface_hub import snapshot_download


def download_ckpt():
    parser = argparse.ArgumentParser(description="Download checkpoints from HuggingFace Hub")
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./out",
        help="Local directory to save the checkpoints"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="sd15",
        choices=["sd15", "pas", "sd35m", "depth", "normal", "canny", "elevest"],
        help="Model type to download"
    )
    parser.add_argument(
        "--image_cond",
        action="store_true",
        help="Whether to download image-conditioned models"
    )

    args = parser.parse_args()

    repo_id, local_dir = "chenguolin/DiffSplat", args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    model_type, image_cond = args.model_type, args.image_cond
    suffix = "_image" if image_cond else ""

    # DiffSplat (SD1.5)
    if model_type == "sd15":
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=[
                "gsrecon_gobj265k_cnp_even4/*",  # `GSRecon`
                "gsvae_gobj265k_sd/*",  # `GSVAE (SD)`
                f"gsdiff_gobj83k_sd15{suffix}__render/*",  # `DiffSplat (SD)`
            ]
        )
    # DiffSplat (PixArt-Sigma)
    elif model_type == "pas":
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=[
                "gsrecon_gobj265k_cnp_even4/*",  # `GSRecon`
                "gsvae_gobj265k_sdxl_fp16/*",  # `GSVAE (SDXL)`
                f"gsdiff_gobj83k_pas_fp16{suffix}__render/*",  # `DiffSplat (PixArt-Sigma)`
            ]
        )
    # DiffSplat (SD3.5m)
    elif model_type == "sd35m":
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=[
                "gsrecon_gobj265k_cnp_even4/*",  # `GSRecon`
                "gsvae_gobj265k_sd3/*",  # `GSVAE (SD3)`
                f"gsdiff_gobj83k_sd35m{suffix}__render/*",  # `DiffSplat (SD3.5m)`
            ]
        )
    # DiffSplat ControlNet (SD1.5)
    elif model_type in ["depth", "normal", "canny"]:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=[
                f"gsdiff_gobj83k_sd15__render__{model_type}/*",  # `DiffSplat ControlNet (SD1.5)`
            ]
        )
    # Elevation Estimation
    elif model_type == "elevest":
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=[
                "elevest_gobj265k_b_C25/*",
            ]
        )
    else:
        raise ValueError(f"Choose from ['sd15', 'pas', 'sd35m', 'depth', 'normal', 'canny', 'elevest'], but got [{model_type}]")


if __name__ == "__main__":
    download_ckpt()
