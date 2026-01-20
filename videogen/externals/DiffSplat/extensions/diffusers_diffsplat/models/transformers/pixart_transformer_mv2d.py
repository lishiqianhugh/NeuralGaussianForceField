import copy

from diffusers import __version__
from diffusers.models.modeling_utils import (
    _add_variant, _get_checkpoint_shard_files, _get_model_file,  # diffusers.utils
    _determine_device_map, _fetch_index_file,  # diffusers.models.model_loading_utils
)
from diffusers.models.modeling_utils import *
from diffusers.models.transformers.pixart_transformer_2d import *

from extensions.diffusers_diffsplat.models.mv_attention import BasicMVTransformerBlock


if is_torch_version(">=", "1.9.0"):
    _LOW_CPU_MEM_USAGE_DEFAULT = True
else:
    _LOW_CPU_MEM_USAGE_DEFAULT = False


# Copied from diffusers.models.transformers.pixart_transformer_2d.PixArtTransformer2DModel
# The only modifications: `BasicTransformerBlock` -> `BasicMVTransformerBlock`
class PixArtTransformerMV2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D Transformer model as introduced in PixArt family of models (https://arxiv.org/abs/2310.00426,
    https://arxiv.org/abs/2403.04692).

    Parameters:
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        in_channels (int, defaults to 4): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        cross_attention_dim (int, optional):
            The dimensionality for cross-attention layers, typically matching the encoder's hidden dimension.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 128):
            The width of the latent images. This parameter is fixed during training.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-6):
            A small constant added to the denominator in normalization layers to prevent division by zero.
        interpolation_scale (int, optional): Scale factor to use during interpolating the position embeddings.
        use_additional_conditions (bool, optional): If we're using additional conditions as inputs.
        attention_type (str, optional, defaults to "default"): Kind of attention mechanism to be used.
        caption_channels (int, optional, defaults to None):
            Number of channels to use for projecting the caption embeddings.
        use_linear_projection (bool, optional, defaults to False):
            Deprecated argument. Will be removed in a future version.
        num_vector_embeds (bool, optional, defaults to False):
            Deprecated argument. Will be removed in a future version.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicMVTransformerBlock", "PatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = 8,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        sample_size: int = 128,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        use_additional_conditions: Optional[bool] = None,
        caption_channels: Optional[int] = None,
        attention_type: Optional[str] = "default",
    ):
        super().__init__()

        # Validate inputs.
        if norm_type != "ada_norm_single":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_single" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        if use_additional_conditions is None:
            if sample_size == 128:
                use_additional_conditions = True
            else:
                use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = self.config.sample_size
        self.width = self.config.sample_size

        interpolation_scale = (
            self.config.interpolation_scale
            if self.config.interpolation_scale is not None
            else max(self.config.sample_size // 64, 1)
        )
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=interpolation_scale,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicMVTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels)

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=self.use_additional_conditions
        )
        self.caption_projection = None
        if self.config.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.config.caption_channels, hidden_size=self.inner_dim
            )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.

        Safe to just use `AttnProcessor()` as PixArt doesn't have any exotic attention processors in default model.
        """
        self.set_attn_processor(AttnProcessor())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`PixArtTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep (`torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            added_cond_kwargs: (`Dict[str, Any]`, *optional*): Additional conditions to be used as inputs.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        # 2. Blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    None,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                )

        # 3. Output
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    # Copied from diffusers.models.modeling_utils.ModelingMixin.from_pretrained
    @classmethod
    @validate_hf_hub_args
    def from_pretrained_new(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],

        sample_size: int = 32,  # `input_res` / 8
        in_channels: int = 4,
        out_channels: int = 8,
        zero_init_conv_in: bool = True,
        view_concat_condition: bool = False,
        input_concat_plucker: bool = False,
        input_concat_binary_mask: bool = False,
        from_scratch: bool = False,  # do not load pretrained parameters

        **kwargs
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Loading and dispatching requires `accelerate`. Please make sure to install accelerate or set"
                " `device_map=None`. You can install accelerate with `pip install accelerate`."
            )

        # Check if we can handle device_map and dispatching the weights
        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to `False` while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        # change device_map into a map if we passed an int, a str or a torch.device
        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        if device_map is not None:
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

        if low_cpu_mem_usage:
            if device_map is not None and not is_torch_version(">=", "1.10"):
                # The max memory utils require PyTorch >= 1.10 to have torch.cuda.mem_get_info.
                raise ValueError("`low_cpu_mem_usage` and `device_map` require PyTorch >= 1.10.")

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )

        # Modify configs for the multi-view cross-domain diffusion model
        config["_class_name"] = cls.__name__
        config["sample_size"] = sample_size  # training resolution
        config["in_channels"] = in_channels
        config["out_channels"] = out_channels

        config["view_concat_condition"] = view_concat_condition
        config["input_concat_plucker"] = input_concat_plucker
        config["input_concat_binary_mask"] = input_concat_binary_mask

        # Determine if we're loading from a directory of sharded checkpoints.
        is_sharded = False
        index_file = None
        is_local = os.path.isdir(pretrained_model_name_or_path)
        index_file = _fetch_index_file(
            is_local=is_local,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder or "",
            use_safetensors=use_safetensors,
            cache_dir=cache_dir,
            variant=variant,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            user_agent=user_agent,
            commit_hash=commit_hash,
        )
        if index_file is not None and index_file.is_file():
            is_sharded = True

        if is_sharded and from_flax:
            raise ValueError("Loading of sharded checkpoints is not supported when `from_flax=True`.")

        # load model
        model_file = None
        if from_flax:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=FLAX_WEIGHTS_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                commit_hash=commit_hash,
            )
            model = cls.from_config(config, **unused_kwargs)

            # Convert the weights
            from diffusers.models.modeling_pytorch_flax_utils import load_flax_checkpoint_in_pytorch_model

            if not from_scratch:
                model = load_flax_checkpoint_in_pytorch_model(model, model_file)
        else:
            if is_sharded:
                sharded_ckpt_cached_folder, sharded_metadata = _get_checkpoint_shard_files(
                    pretrained_model_name_or_path,
                    index_file,
                    cache_dir=cache_dir,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder or "",
                )

            elif use_safetensors and not is_sharded:
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                    )

                except IOError as e:
                    logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                    if not allow_pickle:
                        raise
                    logger.warning(
                        "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                    )

            if model_file is None and not is_sharded:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                )

            if low_cpu_mem_usage:
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **unused_kwargs)

                if not from_scratch:
                    # if device_map is None, load the state dict and move the params from meta device to the cpu
                    if device_map is None and not is_sharded:
                        param_device = "cpu"
                        state_dict = load_state_dict(model_file, variant=variant)
                        model._convert_deprecated_attention_blocks(state_dict)
                        # move the params from meta device to cpu
                        missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                        if len(missing_keys) > 0:
                            raise ValueError(
                                f"Cannot load {cls} from {pretrained_model_name_or_path} because the following keys are"
                                f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                                " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                                " those weights or else make sure your checkpoint file is correct."
                            )

                        unexpected_keys = load_model_dict_into_meta(
                            model,
                            state_dict,
                            device=param_device,
                            dtype=torch_dtype,
                            model_name_or_path=pretrained_model_name_or_path,
                        )

                        if cls._keys_to_ignore_on_load_unexpected is not None:
                            for pat in cls._keys_to_ignore_on_load_unexpected:
                                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                        if len(unexpected_keys) > 0:
                            logger.warning(
                                f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                            )

                    else:  # else let accelerate handle loading and dispatching.
                        # Load weights and dispatch according to the device_map
                        # by default the device_map is None and the weights are loaded on the CPU
                        force_hook = True
                        device_map = _determine_device_map(model, device_map, max_memory, torch_dtype)
                        if device_map is None and is_sharded:
                            # we load the parameters on the cpu
                            device_map = {"": "cpu"}
                            force_hook = False
                        try:
                            accelerate.load_checkpoint_and_dispatch(
                                model,
                                model_file if not is_sharded else index_file,
                                device_map,
                                max_memory=max_memory,
                                offload_folder=offload_folder,
                                offload_state_dict=offload_state_dict,
                                dtype=torch_dtype,
                                force_hooks=force_hook,
                                strict=True,
                            )
                        except AttributeError as e:
                            # When using accelerate loading, we do not have the ability to load the state
                            # dict and rename the weight names manually. Additionally, accelerate skips
                            # torch loading conventions and directly writes into `module.{_buffers, _parameters}`
                            # (which look like they should be private variables?), so we can't use the standard hooks
                            # to rename parameters on load. We need to mimic the original weight names so the correct
                            # attributes are available. After we have loaded the weights, we convert the deprecated
                            # names to the new non-deprecated names. Then we _greatly encourage_ the user to convert
                            # the weights so we don't have to do this again.

                            if "'Attention' object has no attribute" in str(e):
                                logger.warning(
                                    f"Taking `{str(e)}` while using `accelerate.load_checkpoint_and_dispatch` to mean {pretrained_model_name_or_path}"
                                    " was saved with deprecated attention block weight names. We will load it with the deprecated attention block"
                                    " names and convert them on the fly to the new attention block format. Please re-save the model after this conversion,"
                                    " so we don't have to do the on the fly renaming in the future. If the model is from a hub checkpoint,"
                                    " please also re-upload it or open a PR on the original repository."
                                )
                                model._temp_convert_self_to_deprecated_attention_blocks()
                                accelerate.load_checkpoint_and_dispatch(
                                    model,
                                    model_file if not is_sharded else index_file,
                                    device_map,
                                    max_memory=max_memory,
                                    offload_folder=offload_folder,
                                    offload_state_dict=offload_state_dict,
                                    dtype=torch_dtype,
                                    force_hooks=force_hook,
                                    strict=True,
                                )
                                model._undo_temp_convert_self_to_deprecated_attention_blocks()
                            else:
                                raise e

                loading_info = {
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }
            else:
                model = cls.from_config(config, **unused_kwargs)

                if not from_scratch:
                    state_dict = load_state_dict(model_file, variant=variant)
                    model._convert_deprecated_attention_blocks(state_dict)
                    state_dict_original = copy.deepcopy(state_dict)

                    model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                        model,
                        state_dict,
                        model_file,
                        pretrained_model_name_or_path,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                    )

                    loading_info = {
                        "missing_keys": missing_keys,
                        "unexpected_keys": unexpected_keys,
                        "mismatched_keys": mismatched_keys,
                        "error_msgs": error_msgs,
                    }
                else:
                    loading_info = {
                        "missing_keys": [],
                        "unexpected_keys": [],
                        "mismatched_keys": [],
                        "error_msgs": [],
                    }

        if not from_scratch:
            # Handle initilizations for some layers
            ## Patch embedding conv
            pos_embed_proj_weight = state_dict_original["pos_embed.proj.weight"]
            latent_channels = pos_embed_proj_weight.shape[1]
            if model.pos_embed.proj.weight.data.shape[1] != latent_channels:
                # Initialize from the original weights
                model.pos_embed.proj.weight.data[:, :latent_channels] = pos_embed_proj_weight
                # Whether to place all zero to new layers ?
                if zero_init_conv_in:
                    model.pos_embed.proj.weight.data[:, latent_channels:] = 0

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            model = model.to(torch_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        if output_loading_info:
            return model, loading_info

        return model
