import torch.nn as nn
from .residual_cnn import get_norm, get_activation


class ResidualBlock2d(nn.Module):
    """

    Args:
        channels: Number of input/output channels.
        norm: Normalization type, passed to get_norm.
        activation: Activation function type, passed to get_activation.
    """

    def __init__(self, channels: int, norm: str = "bn", activation: str = "relu") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = get_norm(norm, channels)
        self.act = get_activation(activation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = get_norm(norm, channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        out = self.act(out)
        return out


class UpsampleProj(nn.Module):
    """
    input:  (B, C_in, H/ps, W/ps)
    output:  (B, C_out, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        hidden_channels=(512, 512, 256),
        num_res_blocks: int = 1,
        refine_blocks: int = 1,
        norm: str = "bn",
        activation: str = "relu",
    ) -> None:
        super().__init__()

        layers = []
        prev_ch = in_channels
        for h in hidden_channels:
            layers.append(nn.Conv2d(prev_ch, h, kernel_size=3, padding=1, bias=False))
            layers.append(get_norm(norm, h))
            layers.append(get_activation(activation))
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock2d(h, norm=norm, activation=activation))
            prev_ch = h

        # Upsample to full resolution
        layers.append(
            nn.ConvTranspose2d(
                in_channels=prev_ch,
                out_channels=out_channels,
                kernel_size=patch_size,
                stride=patch_size,
            )
        )

        # Refine at full resolution
        for _ in range(refine_blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(get_norm(norm, out_channels))
            layers.append(get_activation(activation))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
