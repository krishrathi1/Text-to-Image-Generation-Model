import torch
import torch.nn as nn


class CrossAttention(nn.Module):

    _query: nn.Linear
    _key: nn.Linear
    _value: nn.Linear
    _n_heads: int
    _scale: float
    _out: nn.Linear

    def __init__(self, channel_dim: int, context_dim=768, n_heads=8) -> None:
        super().__init__()
        self._query = nn.Linear(channel_dim, channel_dim)
        self._key = nn.Linear(context_dim, channel_dim)
        self._value = nn.Linear(context_dim, channel_dim)
        self._n_heads = n_heads
        self._scale = (channel_dim // n_heads) ** -0.5
        self._out = nn.Linear(channel_dim, channel_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        """
        C: channel_dim
        X: image latent feature (B, C, H, W) -> (B, H*W, C)
        Context: text tokens (B, T, context_dim)
        T: token length (77)
        h: number of heads
        d = C // h
        Q = Wq @ X (B, H*W, C) -> (B, H*W, h, d) -> (B, h, H*W, d)
        K = Wk @ C (B, T, C) -> (B, T, h, d) -> (B, h, T, d)
        V = Wv @ C (B, T, C) -> (B, T, h, d) -> (B, h, T, d)
        """
        q = self._query(x)
        k = self._key(context)
        v = self._value(context)

        def multi_head_reshape(t):
            return t.view(B, -1, self._n_heads, C // self._n_heads).transpose(1, 2)

        q, k, v = map(multi_head_reshape, (q, k, v))
        """
        Attn(X, Context) = softmax(Q @ K.T / sqrt(d)) @ V
        (B, h, H*W, T) -> (B, h, H*W, d) -> (B, H*W, C) -> (B, C, H, W)
        """
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self._scale, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self._out(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out
