import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, solve_camera_intrinsics_batch,se3_inverse, depth_edge
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, FeatureHead
from .layers.camera_head import CameraHead
from .gaussians.gaussian_adapter import GaussianAdapter
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from .pi3_gs import Pi3_gs
from ..utils.types import Gaussians
from .cuda_splatter import DecoderSplattingCUDA
from ..utils.debug import memory_monitor,memoryit

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

class Pi3_splat_Config(PretrainedConfig):
    def __init__(
            self,
            sh_degree=2,
            pos_type='rope100',
            decoder_size='large',
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.sh_degree = sh_degree
        self.pos_type = pos_type
        self.decoder_size = decoder_size

class Pi3_splat(PreTrainedModel):
    config_class = Pi3_splat_Config
    def __init__(
            self,
            config: Pi3_splat_Config,
        ):
        super().__init__(config)
        self.build_encoder(pos_type=config.pos_type, decoder_size=config.decoder_size, sh_degree=config.sh_degree)
        self.build_decoder()
    
    def build_encoder(self,pos_type='rope100',decoder_size='large',sh_degree=2):
        self.encoder = Pi3_gs(pos_type=pos_type,decoder_size=decoder_size,sh_degree=sh_degree)

    def build_decoder(self):
        self.decoder = DecoderSplattingCUDA()

    def forward(self,input_images: torch.Tensor,use_masks=False):

        b,v,c,h,w = input_images.shape

        encoder_output = self.encoder(input_images)

        camera_poses = encoder_output['camera_poses'] # [B,V,4,4]
        gaussians = encoder_output['gaussians']
        local_points = encoder_output['local_points'] # [B,V,H,W,3]
        conf = encoder_output['conf'] # [B,V,H,W,1]

        masks = torch.sigmoid(conf[..., 0]) > 0.1
        non_edge = ~depth_edge(local_points[..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge).squeeze(-1) # [B,N,H,W]

        if use_masks:
            decoder_output = self.decoder.forward(
                gaussians=gaussians,
                extrinsics=encoder_output['extrinsics'],
                intrinsics=encoder_output['intrinsics'],
                masks=masks,
                image_shape=(h, w),
            )
        else:
            decoder_output = self.decoder.forward(
                gaussians=gaussians,
                extrinsics=encoder_output['extrinsics'],
                intrinsics=encoder_output['intrinsics'],
                masks=torch.ones_like(masks),
                image_shape=(h, w),
            )

        return encoder_output, decoder_output

    @torch.no_grad()
    def inference(self,input_images,require_decoder=False,use_masks=False):
        b,v,c,h,w = input_images.shape
        encoder_output = self.encoder(input_images)

        encoder_output['gaussians'].calc_world_scale_and_rot() # As downstream tasks require world space scales and rot

        if not require_decoder:
            return encoder_output
        
        masks = torch.sigmoid(encoder_output['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(encoder_output['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge).squeeze(-1) # [B,N,H,W]

        if use_masks:
            decoder_output = self.decoder.forward(
                gaussians=encoder_output['gaussians'],
                extrinsics=encoder_output['extrinsics'],
                intrinsics=encoder_output['intrinsics'],
                masks=masks,
                image_shape=(h, w),
            )
        else:
            decoder_output = self.decoder.forward(
                gaussians=encoder_output['gaussians'],
                extrinsics=encoder_output['extrinsics'],
                intrinsics=encoder_output['intrinsics'],
                masks=torch.ones_like(masks),
                image_shape=(h, w),
            )

        return encoder_output,decoder_output
    
