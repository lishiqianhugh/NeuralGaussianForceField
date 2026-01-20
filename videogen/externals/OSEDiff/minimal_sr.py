import os
import sys
import torch
from dataclasses import dataclass
import math
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler

try:
    from .models.autoencoder_kl import AutoencoderKL
except:
    from models.autoencoder_kl import AutoencoderKL

try:
    from .models.unet_2d_condition import UNet2DConditionModel
except:
    from models.unet_2d_condition import UNet2DConditionModel

from peft import LoraConfig

try:
    from .ram.models.ram_lora import ram
except:
    from ram.models.ram_lora import ram

try:
    from .ram import inference_ram as ram_infer
except:
    from ram import inference_ram as ram_infer

def load_models(sd_base_dir: str, lora_path: str, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(sd_base_dir, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_base_dir, subfolder="text_encoder").to(device, dtype=dtype)
    # scheduler = DDPMScheduler.from_pretrained(sd_base_dir, subfolder="scheduler")
    # scheduler.set_timesteps(1, device=device)
    scheduler = DDPMScheduler.from_pretrained(sd_base_dir, subfolder="scheduler", timestep_spacing="trailing")
    scheduler.set_timesteps(1, device=device)


    vae = AutoencoderKL.from_pretrained(sd_base_dir, subfolder="vae").to(device, dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(sd_base_dir, subfolder="unet").to(device, dtype=dtype)

    ckpt = torch.load(lora_path, map_location="cpu")

    # UNet LoRA adapters
    lora_conf_encoder = LoraConfig(r=ckpt["rank_unet"], init_lora_weights="gaussian", target_modules=ckpt["unet_lora_encoder_modules"])
    lora_conf_decoder = LoraConfig(r=ckpt["rank_unet"], init_lora_weights="gaussian", target_modules=ckpt["unet_lora_decoder_modules"])
    lora_conf_others = LoraConfig(r=ckpt["rank_unet"], init_lora_weights="gaussian", target_modules=ckpt["unet_lora_others_modules"])
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    for name, param in unet.named_parameters():
        if ("lora" in name or "conv_in" in name) and name in ckpt["state_dict_unet"]:
            param.data.copy_(ckpt["state_dict_unet"][name])
    unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

    # VAE LoRA adapters
    vae_lora_conf_encoder = LoraConfig(r=ckpt["rank_vae"], init_lora_weights="gaussian", target_modules=ckpt["vae_lora_encoder_modules"])
    vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
    for name, param in vae.named_parameters():
        if "lora" in name and name in ckpt["state_dict_vae"]:
            param.data.copy_(ckpt["state_dict_vae"][name])
    vae.set_adapter(["default_encoder"])

    return tokenizer, text_encoder, scheduler, vae, unet


@torch.no_grad()
def encode_prompt(tokenizer: AutoTokenizer, text_encoder: CLIPTextModel, prompt: str) -> torch.Tensor:
    input_ids = tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(text_encoder.device)
    return text_encoder(input_ids)[0]


@dataclass
class SRModels:
    tokenizer: AutoTokenizer
    text_encoder: CLIPTextModel
    scheduler: DDPMScheduler
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    ram_model: torch.nn.Module
    device: str
    dtype: torch.dtype


class OSEDiff:
    def __init__(
        self,
        checkpoints_dir: str = "checkpoints/OSEDiff",
        device: str = "cuda",
        dtype: str | torch.dtype = "bfloat16",
    ) -> None:
        if isinstance(dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            dtype = dtype_map.get(dtype.lower(), torch.float16)

        sd_base_dir = os.path.join(checkpoints_dir, "stable-diffusion-2-1-base")
        lora_path = os.path.join(checkpoints_dir, "osediff.pkl")
        ram_path = os.path.join(checkpoints_dir, "ram_swin_large_14m.pth")
        ram_ft_path = os.path.join(checkpoints_dir, "DAPE.pth")

        self.device = device
        self.dtype = dtype

        tokenizer, text_encoder, scheduler, vae, unet = load_models(sd_base_dir, lora_path, device, dtype)
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

        dape = ram(pretrained=ram_path, pretrained_condition=ram_ft_path, image_size=384, vit='swin_l')
        dape.eval()
        dape.to(device)
        dape = dape.to(dtype=dtype)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.vae = vae
        self.unet = unet
        self.ram_model = dape

    def __call__(self, img: Image.Image | list[Image.Image]) -> Image.Image | list[Image.Image]:
        device = self.device
        dtype = self.dtype

        # Accept single image or a list of images
        is_list_input = isinstance(img, (list, tuple))
        imgs: list[Image.Image] = list(img) if is_list_input else [img]
        imgs = [im.convert("RGB") for im in imgs]

        # GPU preprocessing: resize each image so longest side == 768, keep aspect ratio
        # Then center-pad to common canvas (ceil to multiples of 8) with white background
        to_tensor = T.ToTensor()
        img_tensors = [to_tensor(im).to(device) for im in imgs]  # list of (3,H,W) on GPU

        resized_tensors: list[torch.Tensor] = []  # each (1,3,h,w)
        sizes_hw: list[tuple[int, int]] = []
        target_long = 768
        for t in img_tensors:
            _, h, w = t.shape
            longest = max(h, w)
            if longest == 0:
                # safeguard, skip invalid image
                resized_tensors.append(t.unsqueeze(0))
                sizes_hw.append((h, w))
                continue
            scale = float(target_long) / float(longest)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            resized = F.interpolate(t.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            resized_tensors.append(resized)
            sizes_hw.append((new_h, new_w))

        max_h = max(h for h, _ in sizes_hw)
        max_w = max(w for _, w in sizes_hw)
        # make canvas dims multiples of 8
        max_h8 = int(math.ceil(max_h / 8.0) * 8)
        max_w8 = int(math.ceil(max_w / 8.0) * 8)

        B = len(resized_tensors)
        lq = torch.ones((B, 3, max_h8, max_w8), device=device)
        for i, resized in enumerate(resized_tensors):
            th, tw = sizes_hw[i]
            top = (max_h8 - th) // 2
            left = (max_w8 - tw) // 2
            lq[i, :, top:top+th, left:left+tw] = resized[0]

        # RAM on first image only
        ram_transforms = T.Compose([
            T.Resize((384, 384)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        lq_ram = ram_transforms(lq[0:1]).to(dtype=dtype)
        tags = ram_infer(lq_ram, self.ram_model)
        validation_prompt = f"{tags[0]}, "

        # Scale to [-1, 1]
        lq = lq * 2 - 1

        # Encode prompt once; replicate per minibatch
        prompt_embeds_single = encode_prompt(self.tokenizer, self.text_encoder, validation_prompt)

        # Minibatch forward (chunk size 5)
        outs: list[Image.Image] = []
        bs_total = lq.shape[0]
        chunk_size = 15
        with torch.no_grad():
            for start in range(0, bs_total, chunk_size):
                end = min(start + chunk_size, bs_total)
                lq_chunk = lq[start:end]
                bs_chunk = lq_chunk.shape[0]

                prompt_embeds = prompt_embeds_single.repeat(bs_chunk, 1, 1)
                # Use scalar timestep for compatibility with installed diffusers scheduler broadcasting
                timestep = torch.tensor(999, device=device).long()

                latents = self.vae.encode(lq_chunk.to(dtype)).latent_dist.sample() * self.vae.config.scaling_factor
                model_pred = self.unet(latents, timestep, encoder_hidden_states=prompt_embeds.to(dtype)).sample
                x_denoised = self.scheduler.step(model_pred, timestep, latents, return_dict=True).prev_sample
                decoded = self.vae.decode(x_denoised.to(dtype) / self.vae.config.scaling_factor).sample
                decoded = decoded.clamp(-1, 1)

                with torch.amp.autocast('cuda', enabled=False):
                    decoded01 = (decoded.cpu() * 0.5 + 0.5).clamp(0, 1).to(torch.float32)
                    for i in range(decoded01.shape[0]):
                        img_i = Image.fromarray((decoded01[i].permute(1, 2, 0).numpy() * 255).astype("uint8"))
                        outs.append(img_i)

        return outs if is_list_input else outs[0]


def main():
    # use class API
    client = OSEDiff()

    inp_path = "0000.png"
    img = Image.open(inp_path).convert("RGB")
    out_img = client(img)
    out_img.save("0000_sr.png")


if __name__ == "__main__":
    main()

