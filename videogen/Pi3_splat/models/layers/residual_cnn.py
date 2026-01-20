"""
Residual Convolutional Network
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

__all__ = ["ResidualBlock", "ResidualCNN"]


def get_activation(name: str) -> nn.Module:
	name = name.lower()
	if name == "relu":
		return nn.ReLU(inplace=True)
	if name == "gelu":
		return nn.GELU()
	if name == "silu" or name == "swish":
		return nn.SiLU(inplace=True)
	raise ValueError(f"Unsupported activation: {name}")


def get_norm(norm: str, num_channels: int) -> nn.Module:
	norm = norm.lower()
	if norm == "bn" or norm == "batchnorm":
		return nn.BatchNorm2d(num_channels)
	if norm == "gn" or norm == "groupnorm":
		# 32 or fallback to num_channels
		groups = 32 if num_channels % 32 == 0 else min(8, num_channels)
		return nn.GroupNorm(groups, num_channels)
	if norm == "ln" or norm == "layernorm":
		# 以 channel-last 方案包装; 便于与 torchvision 的 conv 输出 (N,C,H,W) 兼容
		return ChannelLayerNorm(num_channels)
	if norm in ("id", "identity", "none"):
		return nn.Identity()
	raise ValueError(f"Unsupported norm: {norm}")


class ChannelLayerNorm(nn.Module):
	"""LayerNorm over channel 维 (保持 (N,C,H,W) 接口).
	计算时将张量变 reshape 为 (N,H,W,C) 做 LayerNorm, 再换回。
	"""

	def __init__(self, num_channels: int, eps: float = 1e-6):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(num_channels))
		self.bias = nn.Parameter(torch.zeros(num_channels))
		self.eps = eps

	def forward(self, x: Tensor) -> Tensor:
		# x: (N, C, H, W)
		mean = x.mean(dim=1, keepdim=True)
		var = (x - mean).pow(2).mean(dim=1, keepdim=True)
		x_hat = (x - mean) / (var + self.eps).sqrt()
		return x_hat * self.weight[:, None, None] + self.bias[:, None, None]


class ResidualBlock(nn.Module):
	"""
	Residual block: Conv->Norm->Act->Conv->Norm + shortcut.
	If in_channels != out_channels, use 1x1 convolution to match dimensions.
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		norm: str = "bn",
		activation: str = "relu",
		dropout: float = 0.0,
		use_checkpoint: bool = True,
	) -> None:
		super().__init__()
		
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.norm1 = get_norm(norm, out_channels)
		self.act = get_activation(activation)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.norm2 = get_norm(norm, out_channels)
		self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

		self.use_checkpoint = use_checkpoint

		if in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
				get_norm(norm, out_channels),
			)
		else:
			self.shortcut = nn.Identity()

		self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1.0)
				nn.init.constant_(m.bias, 0.0)
		if isinstance(self.shortcut, nn.Sequential):  # ensure last norm init done
			pass

	def _forward_impl(self, x: Tensor) -> Tensor:
		identity = self.shortcut(x)
		out = self.conv1(x)
		out = self.norm1(out)
		out = self.act(out)
		out = self.conv2(out)
		out = self.norm2(out)
		out = self.dropout(out)
		out = out + identity
		out = self.act(out)
		return out

	def forward(self, x: Tensor) -> Tensor:
		if self.training and self.use_checkpoint:
			x = checkpoint(self._forward_impl, x, use_reentrant=False)
		else:
			x = self._forward_impl(x)
		return x


class ResidualCNN(nn.Module):
	"""
	Multi-layer Residual CNN Feature Extractor.
	Args:
		in_channels: Number of input channels (C)
		out_channels: Number of output channels (C')
		depth: Number of residual blocks (>=1)
		hidden_channels: Number of hidden channels, if None uses out_channels
		norm: Normalization type (bn|gn|ln|id)
		activation: Activation function (relu|gelu|silu)
		dropout: Dropout2d rate applied after the second conv in each res block
		first_conv: Whether to add a head conv to project in_channels to hidden_channels
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		depth: int = 4,
		hidden_channels: Optional[int] = None,
		norm: str = "bn",
		activation: str = "relu",
		dropout: float = 0.0,
		first_conv: bool = True,
		use_checkpoint: bool = True,
	) -> None:
		super().__init__()
		assert depth >= 1, "depth must be >= 1"

		self.use_checkpoint = use_checkpoint
		
		hidden_channels = hidden_channels or out_channels

		self.layers = nn.ModuleList()
		
		if first_conv:
			first_layer = nn.Sequential(
				nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
				get_norm(norm, hidden_channels),
				get_activation(activation),
			)
			self.layers.append(first_layer)
			in_c = hidden_channels
		else:
			in_c = in_channels

		# residual blocks (depth - 1 if first_conv projects & out_channels mapping at end)
		for i in range(depth):
			out_c_block = hidden_channels if i < depth - 1 else out_channels
			block = ResidualBlock(
				in_channels=in_c,
				out_channels=out_c_block,
				norm=norm,
				activation=activation,
				dropout=dropout,
				use_checkpoint=self.use_checkpoint,
			)
			self.layers.append(block)
			in_c = out_c_block

		self.out_channels = out_channels

	def forward(self, x: Tensor) -> Tensor:
		for layer in self.layers:
			x = layer(x)
		return x


if __name__ == "__main__":
	model = ResidualCNN(3, 64, depth=3, hidden_channels=32)
	x = torch.randn(2, 3, 128, 128)
	y = model(x)
	print("output shape:", y.shape)
	assert y.shape == (2, 64, 128, 128)
