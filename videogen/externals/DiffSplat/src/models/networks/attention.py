from typing import *
from torch import Tensor

import torch
from torch import nn
import torch.nn.functional as tF
from torch.utils.checkpoint import checkpoint
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        context_dim: Optional[int] = None,
    ):
        super().__init__()

        if context_dim is None:
            context_dim = dim

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.wq = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(context_dim, num_heads * head_dim, bias=False)
        self.wv = nn.Linear(context_dim, num_heads * head_dim, bias=False)
        self.wo = nn.Linear(num_heads * head_dim, dim, bias=False)

        if qk_norm:
            self.q_norm = nn.LayerNorm(num_heads * head_dim)
            self.k_norm = nn.LayerNorm(num_heads * head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Initialize weights
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        if context is None:
            context = x

        q, k, v = self.wq(x), self.wk(context), self.wv(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        output = rearrange(tF.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0., is_causal=False,
        ), "b h n d -> b n (h d)")
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,  # ensure `hidden_dim` is a multiple of this value
        ffn_dim_multiplier: Optional[float] = None,  # custom mulitplier for `hidden_dim`
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        # Custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        # Initialize weights
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def _forward_silu_gating(self, x1: Tensor, x3: Tensor):
        return tF.silu(x1) * x3

    def forward(self, x: Tensor):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class LLaMaTransformerBlock(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int,
        use_cross_attention: bool = False,
        context_dim: Optional[int] = None,
        qk_norm: bool = True,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, norm_eps)
        self.attn = Attention(dim, num_heads, qk_norm)
        self.norm2 = RMSNorm(dim, norm_eps)
        self.mlp = FeedForward(dim, dim * 4, multiple_of, ffn_dim_multiplier)

        if use_cross_attention:
            self.norm3 = RMSNorm(dim, norm_eps)
            self.cross_attn = Attention(dim, num_heads, qk_norm, context_dim)

        self.use_cross_attention = use_cross_attention

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        x = x + self.attn(self.norm1(x))
        if context is not None:
            x = x + self.cross_attn(self.norm3(x), context)
        else:
            assert not self.use_cross_attention
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int,
        use_cross_attention: bool = False,
        context_dim: Optional[int] = None,
        **kwargs,  # for compatibility with `LLaMaTransformerBlock`
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qk_norm=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        if use_cross_attention:
            self.norm3 = nn.LayerNorm(dim)
            self.cross_attn = Attention(dim, num_heads, qk_norm=False, context_dim=context_dim)

        self.use_cross_attention = use_cross_attention

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        x = x + self.attn(self.norm1(x))
        if context is not None:
            x = x + self.cross_attn(self.norm3(x), context)
        else:
            assert not self.use_cross_attention
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
        num_blocks: int = 12,
        dim: int = 512,
        num_heads: int = 8,
        llama_style: bool = True,
        use_cross_attention: bool = False,
        context_dim: Optional[int] = None,
    ):
        super().__init__()

        Block = LLaMaTransformerBlock if llama_style else TransformerBlock
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, use_cross_attention, context_dim)
            for _ in range(num_blocks)
        ])

        self.grad_checkpointing = False

    def set_grad_checkpointing(self, flag=True):
        self.grad_checkpointing = flag

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        for block in self.blocks:
            if self.grad_checkpointing:
                x = checkpoint(block, x, context, use_reentrant=False)
            else:
                x = block(x, context)

        return x
