import torch.nn as nn


class SelfAttention(nn.Module):

    _layer_norm: nn.LayerNorm
    _mha: nn.MultiheadAttention
    _ff_self: nn.Sequential

    def __init__(self, channels: int, num_heads=4):
        super().__init__()
        self._layer_norm = nn.LayerNorm([channels])
        self._mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self._ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).swapaxes(1, 2)  # (B, H*W, C)
        q, k, v = self._layer_norm(x), self._layer_norm(x), self._layer_norm(x)
        attention_value, _ = self._mha(q, k, v)
        attention_value = attention_value + x
        attention_value = attention_value + self._ff_self(attention_value)
        return attention_value.swapaxes(2, 1).view(B, C, H, W)
