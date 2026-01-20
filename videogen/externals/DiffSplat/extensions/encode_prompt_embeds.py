from typing import *

import os
import argparse
import json
from multiprocessing import Process

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer,
    T5EncoderModel, T5Tokenizer, T5TokenizerFast,
)


@torch.no_grad()
def text_encode(
    text_encoder: Union[CLIPTextModel, CLIPTextModelWithProjection], tokenizer: CLIPTokenizer,
    oids: List[str], gpu_id: int,
    text_encoder_2: Optional[Union[CLIPTextModelWithProjection, T5EncoderModel]] = None,
    tokenizer_2: Optional[Union[CLIPTokenizer, T5Tokenizer]] = None,
    text_encoder_3: Optional[T5EncoderModel] = None,
    tokenizer_3: Optional[T5TokenizerFast] = None,
):
    global caption_dict, MODEL_NAME, BATCH, dataset_name
    device = f"cuda:{gpu_id}"

    text_encoder = text_encoder.to(device)

    if MODEL_NAME in ["sdxl", "sd3m", "sd35m", "sd35l"]:
        assert text_encoder_2 is not None and tokenizer_2 is not None
        text_encoder_2 = text_encoder_2.to(device)
    if MODEL_NAME in ["sd3m", "sd35m", "sd35l"]:
        assert text_encoder_3 is not None and tokenizer_3 is not None
        text_encoder_3 = text_encoder_3.to(device)

    for i in tqdm(range(0, len(oids), BATCH), desc=pretrained_model_name_or_path, ncols=125):
        batch_oids = oids[i:min(i+BATCH, len(oids))]
        batch_captions = [caption_dict[oid] for oid in batch_oids]

        if MODEL_NAME in ["sd15", "sd21"]:
            batch_text_inputs = tokenizer(
                batch_captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            batch_text_input_ids = batch_text_inputs.input_ids.to(device)  # (B, N)
            batch_prompt_embeds = text_encoder(batch_text_input_ids)
            batch_prompt_embeds = batch_prompt_embeds[0]  # (B, N, D)
        elif MODEL_NAME in ["sdxl"]:
            batch_text_inputs = tokenizer(
                batch_captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            batch_text_input_ids = batch_text_inputs.input_ids.to(device)  # (B, N)
            batch_prompt_embeds = text_encoder(batch_text_input_ids, output_hidden_states=True)
            batch_prompt_embeds_1 = batch_prompt_embeds.hidden_states[-2]  # (B, N, D1); `-2` because SDXL always indexes from the penultimate layer
            # Text encoder 2
            batch_text_inputs = tokenizer_2(
                batch_captions,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            batch_text_input_ids = batch_text_inputs.input_ids.to(device)  # (B, N)
            batch_prompt_embeds = text_encoder_2(batch_text_input_ids, output_hidden_states=True)
            batch_pooled_prompt_embeds = batch_prompt_embeds.text_embeds  # (B, D2)
            batch_prompt_embeds_2 = batch_prompt_embeds.hidden_states[-2]  # (B, N, D); `-2` because SDXL always indexes from the penultimate layer
            batch_prompt_embeds = torch.cat([batch_prompt_embeds_1, batch_prompt_embeds_2], dim=-1)  # (B, N, D1+D2)
        elif MODEL_NAME in ["paa", "pas"]:
            max_length = {"paa": 120, "pas": 300}  # hard-coded for PAA and PAS
            batch_captions = [t.lower().strip() for t in batch_captions]
            batch_text_inputs = tokenizer(
                batch_captions,
                padding="max_length",
                max_length=max_length[MODEL_NAME],
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            batch_text_input_ids = batch_text_inputs.input_ids.to(device)  # (B, N)
            batch_prompt_attention_mask = batch_text_inputs.attention_mask.to(device)  # (B, N)
            batch_prompt_embeds = text_encoder(batch_text_input_ids, attention_mask=batch_prompt_attention_mask)
            batch_prompt_embeds = batch_prompt_embeds[0]  # (B, N, D)
        elif MODEL_NAME in ["sd3m", "sd35m", "sd35l"]:
            batch_text_inputs = tokenizer(
                batch_captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            batch_text_input_ids = batch_text_inputs.input_ids.to(device)  # (B, N)
            batch_prompt_embeds = text_encoder(batch_text_input_ids, output_hidden_states=True)
            batch_pooled_prompt_embeds_1 = batch_prompt_embeds.text_embeds  # (B, D)
            batch_prompt_embeds_1 = batch_prompt_embeds.hidden_states[-2]  # (B, N, D); `-2` because SD3(.5) always indexes from the penultimate layer
            # Text encoder 2
            batch_text_inputs = tokenizer_2(
                batch_captions,
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            batch_text_input_ids = batch_text_inputs.input_ids.to(device)  # (B, N)
            batch_prompt_embeds = text_encoder_2(batch_text_input_ids, output_hidden_states=True)
            batch_pooled_prompt_embeds_2 = batch_prompt_embeds.text_embeds  # (B, D)
            batch_pooled_prompt_embeds = torch.cat([batch_pooled_prompt_embeds_1, batch_pooled_prompt_embeds_2], dim=-1)  # (B, D1+D2)
            batch_prompt_embeds_2 = batch_prompt_embeds.hidden_states[-2]  # (B, N, D); `-2` because SD3(.5) always indexes from the penultimate layer
            batch_clip_prompt_embeds = torch.cat([batch_prompt_embeds_1, batch_prompt_embeds_2], dim=-1)  # (B, N, D1+D2)
            # Text encoder 3
            batch_text_inputs = tokenizer_3(
                batch_captions,
                padding="max_length",
                max_length=256,  # hard-coded for SD3(.5)
                truncation=True,
                return_tensors="pt",
            )
            batch_text_input_ids = batch_text_inputs.input_ids.to(device)  # (B, N3)
            batch_prompt_embeds = text_encoder_3(batch_text_input_ids)
            batch_prompt_embeds_3 = batch_prompt_embeds[0]  # (B, N3, D3)
            batch_clip_prompt_embeds = F.pad(
                batch_clip_prompt_embeds,
                (0, batch_prompt_embeds_3.shape[-1] - batch_clip_prompt_embeds.shape[-1]),
            )  # (B, N, D3)
            batch_prompt_embeds = torch.cat([batch_clip_prompt_embeds, batch_prompt_embeds_3], dim=-2)  # (B, N+N3, D3)

        DATASET_NAME = {
            "gobj265k": "GObjaverse",
            "gobj83k": "GObjaverse",
        }[dataset_name]
        dir = f"/tmp/{DATASET_NAME}_{MODEL_NAME}_prompt_embeds"
        os.makedirs(dir, exist_ok=True)
        for j, oid in enumerate(batch_oids):
            np.save(f"{dir}/{oid}.npy", batch_prompt_embeds[j].float().cpu().numpy())
            if MODEL_NAME in ["sdxl", "sd3m", "sd35m", "sd35l"]:
                np.save(f"{dir}/{oid}_pooled.npy", batch_pooled_prompt_embeds[j].float().cpu().numpy())
            if MODEL_NAME in ["paa", "pas"]:
                np.save(f"{dir}/{oid}_attention_mask.npy", batch_prompt_attention_mask[j].float().cpu().numpy())


if __name__ == "__main__":
    args = argparse.ArgumentParser("Encode prompt embeddings")
    args.add_argument("model_name", type=str, choices=["sd15", "sd21", "sdxl", "paa", "pas", "sd3m", "sd35m", "sd35l"])
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--dataset_name", default="gobj83k", choices=["gobj265k", "gobj83k"])
    args = args.parse_args()

    MODEL_NAME = args.model_name
    pretrained_model_name_or_path = {
        "sd15": "chenguolin/stable-diffusion-v1-5",
        "sd21": "stabilityai/stable-diffusion-2-1",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "paa": "PixArt-alpha/PixArt-XL-2-512x512",  # "PixArt-alpha/PixArt-XL-2-1024-MS"
        "pas": "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        "sd3m": "stabilityai/stable-diffusion-3-medium-diffusers",
        "sd35m": "stabilityai/stable-diffusion-3.5-medium",
        "sd35l": "stabilityai/stable-diffusion-3.5-large",
    }[MODEL_NAME]
    NUM_GPU = torch.cuda.device_count()
    BATCH = args.batch_size
    dataset_name = args.dataset_name

    variant = "fp16" if MODEL_NAME not in ["pas", "sd3m", "sd35m", "sd35l"] else None  # text encoders of PAS and SD3(.5) are already in fp16

    if MODEL_NAME in ["sd15", "sdxl"]:
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", variant=variant)
    elif MODEL_NAME in ["paa", "pas"]:
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", variant=variant)
    elif MODEL_NAME in ["sd3m", "sd35m", "sd35l"]:
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", variant=variant)

    if MODEL_NAME in ["sdxl", "sd3m", "sd35m", "sd35l"]:
        tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2", variant=variant)
    else:
        tokenizer_2 = None
        text_encoder_2 = None

    if MODEL_NAME in ["sd3m", "sd35m", "sd35l"]:
        tokenizer_3 = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_3")
        text_encoder_3 = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_3", variant=variant)
    else:
        tokenizer_3 = None
        text_encoder_3 = None

    # GObjaverse Cap3D
    if "gobj" in dataset_name:
        if not os.path.exists("extensions/assets/Cap3D_automated_Objaverse_full.csv"):
            os.system("wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/Cap3D_automated_Objaverse_full.csv -P extensions/assets/")
        captions = pd.read_csv("extensions/assets/Cap3D_automated_Objaverse_full.csv", header=None)
        caption_dict = {}
        for i in tqdm(range(len(captions)), desc="Preparing caption dict", ncols=125):
            caption_dict[captions.iloc[i][0]] = captions.iloc[i][1]

        if not os.path.exists("extensions/assets/gobjaverse_280k_index_to_objaverse.json"):
            os.system("wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/gobjaverse_280k_index_to_objaverse.json -P extensions/assets/")
        gids_to_oids = json.load(open("extensions/assets/gobjaverse_280k_index_to_objaverse.json", "r"))

        if dataset_name == "gobj83k":  # 83k subset
            if not os.path.exists("extensions/assets/gobj_merged.json"):
                os.system("wget https://raw.githubusercontent.com/ashawkey/objaverse_filter/main/gobj_merged.json -P extensions/assets/")
            gids = json.load(open("extensions/assets/gobj_merged.json", "r"))
            all_oids = [gids_to_oids[gid].split("/")[1].split(".")[0] for gid in gids]
        elif dataset_name == "gobj265k":  # GObjaverse all 265k
            all_oids = [oid.split("/")[1].split(".")[0] for oid in gids_to_oids.values()]

    assert all(oid in caption_dict.keys() for oid in all_oids)

    oids_split = np.array_split(all_oids, NUM_GPU)
    processes = [
        Process(
            target=text_encode,
            args=(text_encoder, tokenizer, oids_split[i], i, text_encoder_2, tokenizer_2, text_encoder_3, tokenizer_3),
        )
        for i in range(NUM_GPU)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    with torch.no_grad():
        if MODEL_NAME in ["sd15", "sd21"]:
            null_text_inputs = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            null_text_input_ids = null_text_inputs.input_ids  # (1, N)
            null_prompt_embed = text_encoder(null_text_input_ids)
            null_prompt_embed = null_prompt_embed[0].squeeze(0)  # (N, D)
        elif MODEL_NAME in ["sdxl"]:
            null_text_inputs = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            null_text_input_ids = null_text_inputs.input_ids  # (1, N)
            null_prompt_embed = text_encoder(null_text_input_ids, output_hidden_states=True)
            null_prompt_embed_1 = null_prompt_embed.hidden_states[-2].squeeze(0)  # (N, D1); `-2` because SDXL always indexes from the penultimate layer
            # Text encoder 2
            null_text_inputs = tokenizer_2(
                "",
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            null_text_input_ids = null_text_inputs.input_ids  # (1, N)
            null_prompt_embed = text_encoder_2(null_text_input_ids, output_hidden_states=True)
            null_pooled_prompt_embed = null_prompt_embed.text_embeds.squeeze(0)  # (D2)
            null_prompt_embed_2 = null_prompt_embed.hidden_states[-2].squeeze(0)  # (N, D2); `-2` because SDXL always indexes from the penultimate layer
            null_prompt_embed = torch.cat([null_prompt_embed_1, null_prompt_embed_2], dim=1)  # (N, D1+D2)
        elif MODEL_NAME in ["paa", "pas"]:
            max_length = {"paa": 120, "pas": 300}  # hard-coded for PAA and PAS
            null_text_inputs = tokenizer(
                "",
                padding="max_length",
                max_length=max_length[MODEL_NAME],
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            null_text_input_ids = null_text_inputs.input_ids  # (1, N)
            null_attention_mask = null_text_inputs.attention_mask  # (1, N)
            null_prompt_embed = text_encoder(null_text_input_ids, attention_mask=null_attention_mask)
            null_prompt_embed = null_prompt_embed[0].squeeze(0)  # (N, D)
            null_attention_mask = null_attention_mask.squeeze(0)  # (N)
        elif MODEL_NAME in ["sd3m", "sd35m", "sd35l"]:
            null_text_inputs = tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            null_text_input_ids = null_text_inputs.input_ids  # (1, N)
            null_prompt_embed = text_encoder(null_text_input_ids, output_hidden_states=True)
            null_pooled_prompt_embed_1 = null_prompt_embed.text_embeds.squeeze(0)  # (D1)
            null_prompt_embed_1 = null_prompt_embed.hidden_states[-2].squeeze(0)  # (N, D1); `-2` because SD3(.5) always indexes from the penultimate layer            
            # Text encoder 2
            null_text_inputs = tokenizer_2(
                "",
                padding="max_length",
                max_length=tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            null_text_input_ids = null_text_inputs.input_ids  # (1, N)
            null_prompt_embed = text_encoder_2(null_text_input_ids, output_hidden_states=True)
            null_pooled_prompt_embed_2 = null_prompt_embed.text_embeds.squeeze(0)  # (D2)
            null_pooled_prompt_embed = torch.cat([null_pooled_prompt_embed_1, null_pooled_prompt_embed_2], dim=-1)  # (D1+D2)
            null_prompt_embed_2 = null_prompt_embed.hidden_states[-2].squeeze(0)  # (N, D2); `-2` because SD3(.5) always indexes from the penultimate layer
            null_clip_prompt_embed = torch.cat([null_prompt_embed_1, null_prompt_embed_2], dim=1)  # (N, D1+D2)
            # Text encoder 3
            null_text_inputs = tokenizer_3(
                "",
                padding="max_length",
                max_length=256,  # hard-coded for SD3(.5)
                truncation=True,
                return_tensors="pt",
            )
            null_text_input_ids = null_text_inputs.input_ids  # (1, N3)
            null_prompt_embed = text_encoder_3(null_text_input_ids)
            null_prompt_embed_3 = null_prompt_embed[0].squeeze(0)  # (N3, D3)
            null_clip_prompt_embed = F.pad(
                null_clip_prompt_embed,
                (0, null_prompt_embed_3.shape[-1] - null_clip_prompt_embed.shape[-1]),
            )  # (N, D3)
            null_prompt_embed = torch.cat([null_clip_prompt_embed, null_prompt_embed_3], dim=-2)  # (N+N3, D3)

        DATASET_NAME = {
            "gobj265k": "GObjaverse",
            "gobj83k": "GObjaverse",
        }[dataset_name]
        dir = f"/tmp/{DATASET_NAME}_{MODEL_NAME}_prompt_embeds"
        os.makedirs(dir, exist_ok=True)
        np.save(f"{dir}/null.npy", null_prompt_embed.float().cpu().numpy())
        if MODEL_NAME in ["sdxl", "sd3m", "sd35m", "sd35l"]:
            np.save(f"{dir}/null_pooled.npy", null_pooled_prompt_embed.float().cpu().numpy())
        if MODEL_NAME in ["paa", "pas"]:
            np.save(f"{dir}/null_attention_mask.npy", null_attention_mask.float().cpu().numpy())
