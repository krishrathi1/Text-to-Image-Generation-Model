import torch
import torch.nn as nn

from .cross_attention import CrossAttention
from .resnet_block import ResnetBlock
from .self_attention import SelfAttention
from .sinusoidal_time_embedder import SinusoidalTimeEmbedder


class SDUNetV1(nn.Module):

    class _DownBlock(nn.Module):

        _res1: ResnetBlock
        _res2: ResnetBlock
        _attn: SelfAttention
        _downsample: nn.Conv2d

        def __init__(
            self, in_channels: int, out_channels: int, time_channels: int
        ) -> None:
            super().__init__()
            self._res1 = ResnetBlock(in_channels, out_channels, time_channels)
            self._res2 = ResnetBlock(out_channels, out_channels, time_channels)
            self._attn = SelfAttention(out_channels, num_heads=1)
            self._downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

        def forward(
            self, x: torch.Tensor, t: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = self._res1(x, t)
            out = self._res2(out, t)
            out = self._attn(out)
            skip = out
            out = self._downsample(out)
            return out, skip

    class _MidBlock(nn.Module):

        _res1: ResnetBlock
        _attn1: SelfAttention
        _attn2: CrossAttention
        _res2: ResnetBlock

        def __init__(self, channels: int, time_dim: int) -> None:
            super().__init__()
            self._res1 = ResnetBlock(channels, channels, time_dim)
            self._attn1 = SelfAttention(channels, num_heads=1)
            self._attn2 = CrossAttention(channels)
            self._res2 = ResnetBlock(channels, channels, time_dim)

        def forward(
            self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
        ) -> torch.Tensor:
            out = self._res1(x, t)
            out = self._attn1(out)
            out = self._attn2(out, context)
            out = self._res2(out, t)
            return out

    class _UpBlock(nn.Module):

        _upsample: nn.ConvTranspose2d
        _res1: ResnetBlock
        _res2: ResnetBlock
        _attn: SelfAttention

        def __init__(
            self, in_channels: int, out_channels: int, time_channels: int
        ) -> None:
            super().__init__()
            self._upsample = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=4, stride=2, padding=1
            )
            self._res1 = ResnetBlock(in_channels * 2, out_channels, time_channels)
            self._res2 = ResnetBlock(out_channels, out_channels, time_channels)
            self._attn = SelfAttention(out_channels, num_heads=1)

        def forward(
            self,
            x: torch.Tensor,
            skip: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            out = self._upsample(x)
            out = torch.cat([out, skip], dim=1)
            out = self._res1(out, t)
            out = self._res2(out, t)
            out = self._attn(out)
            return out

    _time_emb: SinusoidalTimeEmbedder
    _conv_in: nn.Conv2d
    _down1: _DownBlock
    _down2: _DownBlock
    _down3: _DownBlock
    _mid: _MidBlock
    _up3: _UpBlock
    _up2: _UpBlock
    _up1: _UpBlock
    _conv_out: nn.Sequential

    def __init__(
        self, time_channels=256, latent_channels=4, breadth=320, num_groups=32
    ) -> None:
        super().__init__()
        self._time_emb = SinusoidalTimeEmbedder(time_channels)
        self._conv_in = nn.Conv2d(
            latent_channels, breadth, kernel_size=3, stride=1, padding=1
        )
        self._down1 = self._DownBlock(breadth, breadth, time_channels * 4)
        self._down2 = self._DownBlock(breadth, breadth * 2, time_channels * 4)
        self._down3 = self._DownBlock(breadth * 2, breadth * 4, time_channels * 4)
        self._mid = self._MidBlock(breadth * 4, time_channels * 4)
        self._up1 = self._UpBlock(breadth * 4, breadth * 2, time_channels * 4)
        self._up2 = self._UpBlock(breadth * 2, breadth, time_channels * 4)
        self._up3 = self._UpBlock(breadth, breadth, time_channels * 4)
        self._conv_out = nn.Sequential(
            nn.GroupNorm(num_groups, breadth),
            nn.SiLU(),
            nn.Conv2d(breadth, latent_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        x (pixel space) --VAE--> z (latent space)
        z: latent feature (B,4,64,64)
        t: timestep batch (B,)
        context: CLIP text embedding (B,T,context_dim)
        T: token length (77)
        """
        t_emb = self._time_emb(t)
        z = self._conv_in(z)
        z, skip1 = self._down1(z, t_emb)
        z, skip2 = self._down2(z, t_emb)
        z, skip3 = self._down3(z, t_emb)
        z = self._mid(z, t_emb, context)
        z = self._up1(z, skip3, t_emb)
        z = self._up2(z, skip2, t_emb)
        z = self._up3(z, skip1, t_emb)
        z = self._conv_out(z)
        return z
