from PIL.Image import Image as PILImage
from torch import Tensor

import PIL.Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import *
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import *


# Copied from https://github.com/camenduru/GRM/blob/master/third_party/generative_models/instant3d.py
def build_gaussians(H: int, W: int, std: float, bg: float = 0.) -> Tensor:
    assert H == W  # TODO: support non-square latents

    x_vals = torch.arange(W)
    y_vals = torch.arange(H)
    x_vals, y_vals = torch.meshgrid(x_vals, y_vals, indexing="ij")
    x_vals = x_vals.unsqueeze(0).unsqueeze(0)
    y_vals = y_vals.unsqueeze(0).unsqueeze(0)
    center_x, center_y = W//2., H//2.

    gaussian = torch.exp(-((x_vals - center_x) ** 2 + (y_vals - center_y) ** 2) / (2 * (std * H) ** 2))  # cf. Instant3D A.5
    gaussian = gaussian / gaussian.max()
    gaussian = (gaussian + bg).clamp(0., 1.)  # gray background for `bg` > 0.
    gaussian = gaussian.repeat(1, 3, 1, 1)
    gaussian = 1. - gaussian    # (1, 3, H, W) in [0, 1]

    gaussian = torch.cat([gaussian, gaussian], dim=-1)
    gaussian = torch.cat([gaussian, gaussian], dim=-2)  # (1, 3, 2H, 2W)
    gaussians = F.interpolate(gaussian, (H, W), mode="bilinear", align_corners=False)
    gaussians = gaussians * 2. - 1.  # (1, 3, H, W) in [-1, 1]
    return gaussians


# Copied from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion
def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


class StableMVDiffusion3Pipeline(StableDiffusion3Pipeline):
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps_img2img(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.prepare_latents
    def prepare_latents_img2img(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if image.shape[1] == self.vae.config.latent_channels:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.scale_noise(init_latents, timestep, noise)
        latents = init_latents.to(device=device, dtype=dtype)

        return latents

    def prepare_image_latents(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
        dtype = next(self.vae.parameters()).dtype

        assert isinstance(image, Tensor)
        assert image.ndim == 5 and image.shape[2] == 3

        V_cond = image.shape[1]
        image = rearrange(image, "b v c h w -> (b v) c h w")

        # VAE latent
        image = image.to(device).to(dtype)  # not resize like CLIP preprocessing
        image = image * 2. - 1.
        image_latents = self.vae.encode(image).latent_dist.mode() * self.vae.config.scaling_factor

        image_latents = rearrange(image_latents, "(b v) c h w -> b v c h w", v=V_cond)

        # duplicate image latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.unsqueeze(1)
        bs_latent, _, v, c, h, w = image_latents.shape
        image_latents = image_latents.repeat(1, num_images_per_prompt, 1, 1, 1, 1)
        image_latents = image_latents.view(bs_latent * num_images_per_prompt, v, c, h, w)

        if do_classifier_free_guidance:
            negative_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_latents, image_latents])

        return image_latents

    def prepare_plucker(self, plucker, num_images_per_prompt, do_classifier_free_guidance):
        plucker = plucker.to(dtype=self.transformer.dtype, device=self.transformer.device)

        # duplicate plucker embeddings for each generation per prompt, using mps friendly method
        plucker = plucker.unsqueeze(1)
        bs, _, c, h, w = plucker.shape
        plucker = plucker.repeat(1, num_images_per_prompt, 1, 1, 1)
        plucker = plucker.view(bs * num_images_per_prompt, c, h, w)

        if do_classifier_free_guidance:
            plucker = torch.cat([plucker]*2, dim=0)

        return plucker

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    # Refine for triangle cfg scaling
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,

        num_views: int = 4,
        plucker: Optional[torch.FloatTensor] = None,
        triangle_cfg_scaling: bool = False,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        init_std: Optional[float] = 0.,
        init_noise_strength: Optional[float] = 1.,
        init_bg: Optional[float] = 0.,

        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale if not triangle_cfg_scaling else max_guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}
        self._interrupt = False

        V_cond = 0
        if image is not None:
            assert image.ndim == 5  # (B, V_cond, 3, H, W)
            V_cond = image.shape[1]
        self.joint_attention_kwargs.update(num_views=num_views + (V_cond if self.transformer.config.view_concat_condition else 0))

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3.1 Prepare input image latents
        if self.transformer.config.view_concat_condition:
            if image is not None:
                image_latents = self.prepare_image_latents(image, device, num_images_per_prompt, self.do_classifier_free_guidance)
            else:
                image_latents = torch.zeros(
                    (
                        batch_size * num_images_per_prompt,
                        self.transformer.config.out_channels,  # `num_channels_latents`; self.transformer.config.in_channels
                        int(height) // self.vae_scale_factor,
                        int(width) // self.vae_scale_factor,
                    ),
                    dtype=prompt_embeds.dtype,
                    device=device,
                )
                if V_cond > 0:
                    image_latents = image_latents.unsqueeze(1).repeat(1, V_cond, 1, 1, 1)
                if self.do_classifier_free_guidance:
                    image_latents = torch.cat([image_latents] * 2, dim=0)

        # 3.2 Prepare Plucker embeddings
        if plucker is not None:
            assert plucker.shape[0] == batch_size * (num_views + (V_cond if self.transformer.config.view_concat_condition else 0))
            plucker = self.prepare_plucker(plucker, num_images_per_prompt, self.do_classifier_free_guidance)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.out_channels  # self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt * num_views,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.base_image_seq_len,
                self.scheduler.config.max_image_seq_len,
                self.scheduler.config.base_shift,
                self.scheduler.config.max_shift,
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5.1 Gaussian blobs initialization; cf. Instant3D
        if init_std > 0. and init_noise_strength < 1.:
            row = int(num_views**0.5)
            col = num_views - row
            init_image = build_gaussians(row * height, col * width, init_std, init_bg).to(device=device, dtype=latents.dtype)
            init_image = rearrange(init_image, "b d (r h) (c w) -> (b r c) d h w", r=row, c=col)
            timesteps, num_inference_steps = self.get_timesteps_img2img(num_inference_steps, init_noise_strength, device)
            self._num_timesteps = len(timesteps)
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            latents = self.prepare_latents_img2img(
                init_image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        # 5.2 Prepare guidance scale
        if triangle_cfg_scaling:
            # Triangle CFG scaling; the first view is input condition
            guidance_scale = torch.cat([
                torch.linspace(min_guidance_scale, max_guidance_scale, num_views//2 + 1).unsqueeze(0),
                torch.linspace(max_guidance_scale, min_guidance_scale, num_views - (num_views//2 + 1) + 2)[1:-1].unsqueeze(0)
            ], dim=-1)
            guidance_scale = guidance_scale.to(device, latents.dtype)
            guidance_scale = guidance_scale.repeat(batch_size * num_images_per_prompt, 1)
            guidance_scale = _append_dims(guidance_scale, latents.unsqueeze(1).ndim)  # (B, V, 1, 1, 1)
            guidance_scale = rearrange(guidance_scale, "b v c h w -> (b v) c h w")

            self._guidance_scale = guidance_scale

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0] // num_views)

                # Concatenate input latents with others
                latent_model_input = rearrange(latent_model_input, "(b v) c h w -> b v c h w", v=num_views)
                if self.transformer.config.view_concat_condition:
                    latent_model_input = torch.cat([image_latents, latent_model_input], dim=1)  # (B, V_in+V_cond, 4, H', W')
                if self.transformer.config.input_concat_plucker:
                        plucker = F.interpolate(plucker, size=latent_model_input.shape[-2:], mode="bilinear", align_corners=False)
                        plucker = rearrange(plucker, "(b v) c h w -> b v c h w", v=num_views + (V_cond if self.transformer.config.view_concat_condition else 0))
                        latent_model_input = torch.cat([latent_model_input, plucker], dim=2)  # (B, V_in(+V_cond), 4+6, H', W')
                        plucker = rearrange(plucker, "b v c h w -> (b v) c h w")
                if self.transformer.config.input_concat_binary_mask:
                    if self.transformer.config.view_concat_condition:
                        latent_model_input = torch.cat([
                            torch.cat([latent_model_input[:, :V_cond, ...], torch.zeros_like(latent_model_input[:, :V_cond, 0:1, ...])], dim=2),
                            torch.cat([latent_model_input[:, V_cond:, ...], torch.ones_like(latent_model_input[:, V_cond:, 0:1, ...])], dim=2),
                        ], dim=1)  # (B, V_in+V_cond, 4+6+1, H', W')
                    else:
                        latent_model_input = torch.cat([
                            torch.cat([latent_model_input, torch.ones_like(latent_model_input[:, :, 0:1, ...])], dim=2),
                        ], dim=1)  # (B, V_in, 4+6+1, H', W')
                latent_model_input = rearrange(latent_model_input, "b v c h w -> (b v) c h w")

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Only keep the noise prediction for the latents
                if self.transformer.config.view_concat_condition:
                    noise_pred = rearrange(noise_pred, "(b v) c h w -> b v c h w", v=num_views+V_cond)
                    noise_pred = rearrange(noise_pred[:, V_cond:, ...], "b v c h w -> (b v) c h w")

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    should_skip_layers = (
                        True
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep = t.expand(latents.shape[0])
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
