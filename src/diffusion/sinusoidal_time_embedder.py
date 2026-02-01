import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedder(nn.Module):

    _time_dim: int
    _linear1: nn.Linear
    _linear2: nn.Linear

    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self._time_dim = time_dim
        self._linear1 = nn.Linear(time_dim, time_dim * 4)
        self._linear2 = nn.Linear(time_dim * 4, time_dim * 4)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self._time_dim // 2
        """
        0 <= k=torch.arange(half_dim, device=t.device) <= half_dim-1
        """
        time_vec = torch.exp(
            torch.arange(half_dim, device=t.device) * -math.log(10000) / (half_dim - 1)
        )
        """
        time_vec[2k] = sin(t*time_vec)
        time_vec[2k+1]= cos(t*time_vec)
        """
        time_vec = t[:, None] * time_vec[None, :]
        time_vec = torch.cat([torch.sin(time_vec), torch.cos(time_vec)], dim=-1)
        time_vec = self._linear1(time_vec)
        time_vec = F.silu(time_vec)
        time_vec = self._linear2(time_vec)
        return time_vec
