from .attention import FlashAttentionRope
from .residual_cnn import ResidualCNN, get_norm, get_activation
from .upsample import UpsampleProj
from .block import BlockRope
from ..dinov2.layers import Mlp
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
from ...utils.debug import memory_monitor
import torch.nn.functional as F
import torch
   
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        out = self.linear_out(hidden)
        return out

class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat.permute(0, 2, 3, 1)


class FeatureHead (nn.Module):
    """ 
    Feature Head use for feature aggregation and gaussian output
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, feature_dim=128,output_dim=83,
                 cnn_depth=3,cnn_norm="bn", cnn_activation="relu",
                 up_hidden=(512,512,256), up_res_blocks=2, up_refine_blocks=2,use_checkpoint=True):
        super().__init__()
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        self.upsample_proj = UpsampleProj(
            in_channels=dec_embed_dim,
            out_channels=feature_dim,
            patch_size=patch_size,
            hidden_channels=up_hidden,
            num_res_blocks=up_res_blocks,
            refine_blocks=up_refine_blocks,
            norm=cnn_norm,
            activation=cnn_activation,
        )

        self.conv_net = ResidualCNN(
            in_channels=3,
            out_channels=feature_dim,
            depth=cnn_depth,
            hidden_channels=feature_dim,
            norm=cnn_norm,
            activation=cnn_activation,
            first_conv=True,
            use_checkpoint=self.use_checkpoint,
        )
        self.down_proj = nn.Linear(feature_dim * 2, output_dim)

    def forward(self, decout, imgs, img_shape):
        H, W = img_shape
        tokens = decout[-1]  # (B, S, D)
        
        # 1. Reshape tokens to match convolution layer input format (B, D, H/patch, W/patch)
        tokens_reshaped = tokens.transpose(1, 2).reshape(
            tokens.shape[0], -1, H // self.patch_size, W // self.patch_size
        )
        
        # 2. Use transpose convolution for upsampling
        if self.training and self.use_checkpoint:
            feat_conv = checkpoint(self.upsample_proj, tokens_reshaped, use_reentrant=False)
        else:
            feat_conv = self.upsample_proj(tokens_reshaped)

        feat = feat_conv.permute(0, 2, 3, 1) # (B, H, W, feature_dim)

        # imgs: (B,C,H,W) -> conv_feat: (B,feature_dim,H,W)

        conv_feat = self.conv_net(imgs)

        cnn_feat = conv_feat.permute(0, 2, 3, 1).contiguous() # (B,H,W,feature_dim)

        if self.training and self.use_checkpoint:
            gaussian_params = checkpoint(self.down_proj, torch.cat([feat, cnn_feat], dim=-1), use_reentrant=False)
        else:
            gaussian_params = self.down_proj(torch.cat([feat, cnn_feat], dim=-1))  # [B,H,W,output_dim]

        return dict(gaussian_params=gaussian_params)