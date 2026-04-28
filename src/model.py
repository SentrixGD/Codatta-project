import math
from typing import List, Tuple

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
    ) -> None:

        super().__init__()

        assert dim % heads == 0, "dim must be divisible by heads"

        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        coords = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(window_size),
                    torch.arange(window_size),
                    indexing="ij",
                )
            )
            .reshape(2, -1)
            .T
        )

        coords = coords.to(torch.long)

        rel = coords[:, None, :] - coords[None, :, :]
        dx = rel[..., 0].contiguous()
        dy = rel[..., 1].contiguous()

        self.register_buffer("dx", dx)
        self.register_buffer("dy", dy)
        self.register_buffer("coords", coords)
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
        slopes = torch.logspace(0, -1, self.heads, device=q.device)

        mask = attn_mask.reshape(B * W, L, L).bool()
        mask_heads = mask[:, None, :, :]
        bias = -slopes[:, None, None] * (self.dx.abs() + self.dy.abs())  # (heads, L, L)
        bias = bias.unsqueeze(0).expand(B * W, -1, -1, -1)  # (B*W, heads, L, L)

        attn_mask_float = torch.zeros_like(mask_heads, dtype=dtype)
        attn_mask_float = attn_mask_float.masked_fill(~mask_heads, float("-inf")).cuda()
        attn_bias = attn_mask_float + bias.to(dtype)

        q = q.view(B * W, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B * W, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B * W, L, self.heads, self.head_dim).permute(0, 2, 1, 3)

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
    def __init__(self, dim: int, dropout: float, hidden_extension: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * hidden_extension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_extension, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DropPath(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        keep_prob = 1 - self.p

        # one mask per sample
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep_prob

        return x * mask / keep_prob


class Patching(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=4, stride=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def windowing(
    x: torch.Tensor,
    window_size: int,
    shift_size: int,
    height_padding: int,
    width_padding: int,
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
    B, C, H_pad, W_pad = x.shape
    device = x.device
    H = H_pad - height_padding
    W = W_pad - width_padding
    mask = torch.zeros(B, H_pad, W_pad, dtype=torch.bool, device=device)
    mask[:, :H, :W] = True

    # create window mask
    window_mask = torch.zeros(B, H_pad, W_pad, dtype=torch.int8, device=device)
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

    x = x.roll(shifts=(shift_size, shift_size), dims=(2, 3))
    mask = mask.roll(shifts=(shift_size, shift_size), dims=(1, 2))
    x = x.view(
        B,
        C,
        H_pad // window_size,
        window_size,
        W_pad // window_size,
        window_size,
    )

    x = x.permute(0, 2, 4, 1, 3, 5)  # (B, nH, nW, C, ws, ws)

    x = x.reshape(
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
    x = x.permute(0, 1, 3, 4, 2).contiguous()
    x = x.view(B, -1, window_size * window_size, C)
    return x, final_mask


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


class SwinBlock(nn.Module):
    def __init__(
        self,
        heads: int,
        dim: int,
        dropout_mha: float,
        dropout_mlp: float,
        dropout_outer: float,
        droppath: float,
        window_size: int,
    ):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.dropout_mha = dropout_mha
        self.dropout_mlp = dropout_mlp
        self.window_size = window_size

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_outer)
        self.MHA = MultiHeadAttention(heads, dim, dropout_mha, window_size)
        self.MLP = MLP(dim, dropout_mlp, 4)
        self.droppath = DropPath(droppath)

    def forward(self, x: torch.Tensor, shift_size: int) -> torch.Tensor:
        B, C, H, W = x.shape

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        H = H + pad_h
        W = W + pad_w
        x = F.pad(x, pad=(0, pad_w, 0, pad_h), mode="constant", value=0)
        attn = x.permute(0, 2, 3, 1).contiguous()
        attn = self.ln1(attn)
        attn = attn.permute(0, 3, 1, 2).contiguous()
        attn, mask = windowing(attn, self.window_size, shift_size, pad_h, pad_w)
        attn = self.MHA(attn, attn, mask)
        attn = unwindowing(attn, shift_size, H, W)
        attn = self.dropout(attn)
        x = x + self.droppath(attn)
        mlp = x.permute(0, 2, 3, 1).contiguous()
        mlp = mlp.view(B, H * W, C)
        mlp = self.ln2(mlp)
        mlp = self.MLP(mlp)
        mlp = mlp.view(B, H, W, C)
        mlp = mlp.permute(0, 3, 1, 2).contiguous()
        return x + self.droppath(mlp)


class SwinModel(nn.Module):
    def __init__(
        self,
        heads_ratio: int,
        dim: int,
        dropout_mha: float,
        dropout_mlp: float,
        dropout_outer: float,
        droppath: float,
        window_size: int,
        input_channels: int,
        depths: List[int],
        stage_num: int,
    ):
        super().__init__()

        self.heads_ratio = heads_ratio
        self.dim = dim
        self.dropout_mha = dropout_mha
        self.dropout_mlp = dropout_mlp
        self.dropout_outer = dropout_outer
        self.droppath = droppath
        self.window_size = window_size
        self.input_channels = input_channels
        self.depths = depths
        self.stage_num = stage_num

        self.patch_embed = Patching(self.input_channels, self.input_channels * 16)

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()

        self.dims = list(dim * 2**i for i in range(stage_num))
        self.droppath_values = torch.linspace(0, droppath, sum(depths))

        for i in range(stage_num):
            blocks = nn.ModuleList(
                [
                    SwinBlock(
                        heads=self.dims[i] // heads_ratio,
                        dim=self.dims[i],
                        dropout_mha=dropout_mha,
                        dropout_mlp=dropout_mlp,
                        dropout_outer=dropout_outer,
                        droppath=self.droppath_values[sum(depths[:i]) + j],
                        window_size=window_size,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(blocks)

            self.merges.append(Merge(self.dims[i]))

        self.shared_mlp = MLP(self.dims[-1], dropout_mlp, 2)

    def forward(self, x):
        x = self.patch_embed(x)

        for stage_idx, blocks in enumerate(self.stages):

            for block_idx, block in enumerate(blocks):
                shift = 0 if block_idx % 2 == 0 else self.window_size // 2
                x = block(x, shift)

            if stage_idx < len(self.merges):
                x = self.merges[stage_idx](x)

        return x


device = torch.device("cuda")

x = torch.randn(1, 3, 448, 672, device=device)

model = SwinModel(
    heads_ratio=6,
    dim=48,
    dropout_mha=0,
    dropout_mlp=0,
    dropout_outer=0,
    droppath=0.1,
    window_size=7,
    input_channels=3,
    depths=[2, 2, 8, 2],
    stage_num=4,
).to(device)
out = model(x)
print(out.shape)

# make increasing droppath probability with depth
# make final mlp with multiple heads
# make
