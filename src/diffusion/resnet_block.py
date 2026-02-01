import torch
import torch.nn as nn


class ResnetBlock(nn.Module):

    _norm_silu_conv1: nn.Sequential
    _time_emb: nn.Sequential
    _norm_silu_conv2: nn.Sequential
    _shortcut: nn.Conv2d | nn.Identity

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels=256,
        num_groups=32,
    ) -> None:
        super().__init__()
        self._norm_silu_conv1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self._time_emb = nn.Sequential(
            nn.SiLU(), nn.Linear(time_channels, out_channels)
        )
        self._norm_silu_conv2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self._shortcut = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if (in_channels != out_channels)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_vec: torch.Tensor) -> torch.Tensor:
        out = self._norm_silu_conv1(x)
        out += self._time_emb(time_vec)[:, :, None, None]
        out = self._norm_silu_conv2(out)
        out += self._shortcut(x)
        return out
