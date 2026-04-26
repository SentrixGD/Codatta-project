from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        heads: int,
        dim: int,
        dropout: float,
        window_size: int,
        device: str = "cuda",
    ) -> None:

        super().__init__()

        assert dim % heads == 0, "dim must be divisible by heads"

        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        coords = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(window_size, device=device),
                    torch.arange(window_size, device=device),
                    indexing="ij",
                )
            )
            .reshape(2, -1)
            .T
        )  # (L, 2)
        rel = coords[:, None, :] - coords[None, :, :]
        dx = rel[..., 0]
        dy = rel[..., 1]

        self.register_buffer("dx", dx)
        self.register_buffer("dy", dy)
        self.register_buffer("coords", coords)
        self.device = device
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, attn_mask: torch.Tensor):

        assert x_q.shape == x_kv.shape, "This implementation assumes self-attention"

        B, W, L, _ = x_q.shape

        assert (
            attn_mask.shape[0] == B and attn_mask.shape[1] == W
        ), "attn_mask must be of shape (batch_size, num_windows, seq_len, seq_len)"

        q = self.proj_q(x_q)
        k = self.proj_k(x_kv)
        v = self.proj_v(x_kv)

        dtype = q.dtype

        # heads (NOT windows)
        slopes = torch.logspace(0, -1, self.heads, device=self.device)

        mask = attn_mask.reshape(B * W, L, L).bool()
        mask_heads = mask[:, None, :, :]
        bias = -slopes[:, None, None] * (self.dx.abs() + self.dy.abs())  # (heads, L, L)
        bias = bias.unsqueeze(0).expand(B * W, -1, -1, -1)  # (B*W, heads, L, L)

        attn_mask_float = torch.zeros_like(mask_heads, dtype=dtype)
        attn_mask_float = attn_mask_float.masked_fill(~mask_heads, float("-inf"))
        print(attn_mask_float.shape, bias.shape)

        attn_bias = attn_mask_float + bias.to(dtype)

        print(attn_bias)

        q = q.view(B * W, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B * W, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B * W, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        print(q.shape)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=(self.dropout.p if self.training else 0.0),
        )
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B, W, L, self.dim)

        return self.proj_out(attn)


class Merge(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convolutional = nn.Conv2d(
            channels, channels * 2, kernel_size=2, stride=2, bias=False
        )
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        padded = F.pad(x, pad=(0, W % 2, 0, H % 2), mode="constant", value=0)
        merged = self.convolutional(padded)
        return merged


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def windowing(
    x: torch.Tensor, window_size: int, shift_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Turns a tensor of shape (batch_size, channels, height, width) into a tensor of shape (batch_size, num_windows, window_size * window_size, channels) for MHA and creates a window boolean mask composed of shift mask and validity mask

    args:
        x (torch.Tensor): input tensor of shape (batch_size, channels, height, width)
        window_size (int): size of the window
        shift_size (int): size of the shift of the windows
    returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of the windowed tensor and the window mask
    """
    B, C, H, W = x.shape
    height_padding = (window_size - H % window_size) % window_size
    width_padding = (window_size - W % window_size) % window_size
    padded = F.pad(
        x, pad=(0, width_padding, 0, height_padding), mode="constant", value=0
    )
    H_pad = H + height_padding
    W_pad = W + width_padding
    mask = torch.zeros(B, H_pad, W_pad, dtype=torch.bool)
    mask[:, :H, :W] = True

    # create window mask
    window_mask = torch.zeros(B, H_pad, W_pad, dtype=torch.int8)
    window_mask[:, :, W_pad - shift_size :] = 1
    window_mask[:, H_pad - shift_size :, :] = 2
    window_mask[:, H_pad - shift_size :, W_pad - shift_size :] = 3
    window_mask = window_mask.roll(shifts=(shift_size, shift_size), dims=(1, 2))
    window_mask = window_mask.view(
        B,
        H_pad // window_size,
        window_size,
        W_pad // window_size,
        window_size,
    )
    window_mask = window_mask.permute(0, 1, 3, 2, 4)
    windows = window_mask.reshape(B, -1, window_size * window_size)
    attn_mask = windows[:, :, None, :] == windows[:, :, :, None]

    padded = padded.roll(shifts=(shift_size, shift_size), dims=(2, 3))
    mask = mask.roll(shifts=(shift_size, shift_size), dims=(1, 2))
    padded = padded.view(
        B,
        C,
        H_pad // window_size,
        window_size,
        W_pad // window_size,
        window_size,
    )

    padded = padded.permute(0, 2, 4, 1, 3, 5)  # (B, nH, nW, C, ws, ws)

    padded = padded.reshape(
        B,
        (H_pad // window_size) * (W_pad // window_size),
        C,
        window_size,
        window_size,
    )
    mask = mask.view(
        B,
        H_pad // window_size,
        window_size,
        W_pad // window_size,
        window_size,
    )

    mask = mask.permute(0, 1, 3, 2, 4)

    mask = mask.reshape(
        B,
        (H_pad // window_size) * (W_pad // window_size),
        window_size,
        window_size,
    )
    valid = mask.reshape(
        B, (H_pad // window_size) * (W_pad // window_size) * window_size * window_size
    )
    valid_windows = valid.view(B, -1, window_size * window_size)
    valid_mask = valid_windows[:, :, :, None] & valid_windows[:, :, None, :]
    final_mask = attn_mask & valid_mask
    padded = padded.permute(0, 1, 3, 4, 2).contiguous()
    padded = padded.view(B, -1, window_size * window_size, C)
    return padded, final_mask


def unwindowing(
    x: torch.Tensor,
    shift_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    The system is FRAGILE! Works under assumption that the windows are not shuffled. If the windows are shuffled, the output will be wrong
    """
    # x is of shape (B, num_windows, ws**2, C)
    # x should be in the end of size (B, C, H, W)
    B, num_windows, ws2, C = x.shape
    ws = int(ws2**0.5)

    nH = H // ws
    nW = W // ws

    # (B, nH*nW, ws, ws, C)
    x = x.view(B, nH, nW, ws, ws, C)

    # bring channels forward
    x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, nH, ws, nW, ws)

    # reconstruct spatial image
    x = x.reshape(B, C, H, W)

    # undo shift
    x = x.roll((-shift_size, -shift_size), dims=(2, 3))
    return x


# k, mask = windowing(torch.randn(1, 48, 167, 167), 7, 3)
k, mask = windowing(torch.randn(1, 48, 167, 167, dtype=torch.float), 7, 3)
MHA = MultiHeadAttention(6, 48, 0.0, 7, "cpu")
k = MHA(k, k, mask)
b = unwindowing(k, 3, 168, 168)
b = b.permute(0, 2, 3, 1).contiguous()
print(b.shape)
merge = Merge(48)
# c = merge.forward(b)
