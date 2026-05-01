"""
File: model.py

Description:
    Modular file containing the model, all its modules and initial weight initialization. The file is built to be imported and used as a module.

Purpose:
    To initialize and train the model, this file contains all the necessary components and the model class itself.

Inputs:
    - No external inputs are required.

Outputs:
    - No outputs are returned. The file is used as an importable module.

Dependencies:
    - torch

Usage:
    python -m src.model
"""

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
        """
        Multi-head attention module. Relative coordinates are stored as buffers.

        Args:
            heads (int): Number of attention heads.
            dim (int): Dimensionality of the input and output.
            dropout (float): Dropout probability.
            window_size (int): Size of the attention window.
        """

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

        # store relative coordinates as buffers to avoid reusing
        self.register_buffer("dx", dx)
        self.register_buffer("dy", dy)
        self.register_buffer("coords", coords)

        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x_q: torch.Tensor, x_kv: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the multi-head attention module. Assumes self-attention.

        Args:
            x_q (torch.Tensor): Query tensor of shape (batch_size, num_windows, seq_len, dim).
            x_kv (torch.Tensor): Key-value tensor of shape (batch_size, num_windows, seq_len, dim).
            attn_mask (torch.Tensor): Attention mask of shape (batch_size, num_windows, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_windows, seq_len, dim).
        """

        assert x_q.shape == x_kv.shape, "This implementation assumes self-attention"

        B, W, L, _ = x_q.shape

        assert (
            attn_mask.shape[0] == B and attn_mask.shape[1] == W
        ), "attn_mask must be of shape (batch_size, num_windows, seq_len, seq_len)"

        q = self.proj_q(x_q)
        k = self.proj_k(x_kv)
        v = self.proj_v(x_kv)

        dtype = q.dtype

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
    def __init__(self, channels: int, squeeze: bool, inflation: float) -> None:
        """
        Merging layers, halving the spatial resolution by a factor of 2 and doubling the number of channels.

        Args:
            channels (int): Number of channels in the input tensor.
            squeeze (bool): Whether to squeeze the spatial resolution by a factor of 2.
            inflation (float): Inflation factor for the convolutional layer.
        """
        super().__init__()
        self.convolutional = nn.Conv2d(
            channels,
            int(channels * inflation),
            kernel_size=2 if squeeze else 1,
            stride=2 if squeeze else 1,
            bias=False,
        )
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the merging layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels * 2, height // 2, width // 2).
        """
        _, _, H, W = x.shape
        padded = F.pad(x, pad=(0, W % 2, 0, H % 2), mode="constant", value=0)
        merged = self.convolutional(padded)
        return merged


class MLP(nn.Module):
    def __init__(self, dim: int, dropout: float, hidden_extension: float) -> None:
        """
        Multi-layer perceptron module.

        Args:
            dim (int): Input dimension.
            dropout (float): Dropout probability.
            hidden_extension (float): Extension factor for the hidden layer.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * hidden_extension)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * hidden_extension), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
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
        width_padding (int): width padding
        height_padding (int): height padding
    returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of the windowed tensor and the window mask
    """
    B, C, H_pad, W_pad = x.shape
    device = x.device
    H = H_pad - height_padding
    W = W_pad - width_padding
    mask = torch.zeros(B, H_pad, W_pad, dtype=torch.bool, device=device)
    mask[:, :H, :W] = True

    # create window mask that enables masked shifting, so that the opposite sides in a window do not see each other
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

    # create windowed shifted tensors of samples and masks
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

    # create validity mask that enables only valid patches, disabling padding from attending to other tokens and vice versa
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
    Reconstructs a tensor of shape (batch_size, num_windows, window_size * window_size, channels) to a tensor of shape (batch_size, channels, height, width)
    The system is FRAGILE! Works under assumption that the windows are not shuffled. If the windows are shuffled, the output will be wrong.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_windows, window_size * window_size, channels)
        shift_size (int): Size of the shift of the windows
        H (int): Height of the pre-windowed post-padding image
        W (int): Width of the pre-windowed post-padding image

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, channels, height, width)
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
        """
        Swin Transformer block consisting of MHA and MLP.

        Args:
            heads (int): Number of attention heads.
            dim (int): Input dimension (number of channels).
            dropout_mha (float): Dropout probability inside the MHA module (inside attention).
            dropout_mlp (float): Dropout probability for the MLP module.
            dropout_outer (float): Dropout probability for the MHA module (outside the sublayer).
            droppath (float): Dropout probability for the drop path.
            window_size (int): Window size for the MHA module.
        """
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
        """
        Forward pass of the Swin Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            shift_size (int): Size of the shift of the windows.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        B, C, H, W = x.shape

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        H = H + pad_h
        W = W + pad_w
        x = F.pad(x, pad=(0, pad_w, 0, pad_h), mode="constant", value=0)

        attn = x.permute(0, 2, 3, 1).contiguous()
        attn = self.ln1(attn)
        attn = attn.permute(0, 3, 1, 2).contiguous()

        # split the image into windows for attention module
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
        dropout_swin_mlp: float,
        dropout_outer: float,
        dropout_shared_mlp: float,
        dropout_pre_output: float,
        droppath: float,
        window_size: int,
        input_channels: int,
        depths: List[int],
        stage_num: int,
        shared_mlp_size: int,
        ingredients_mlp_size: int,
        portions_mlp_size: int,
        dish_names_mlp_size: int,
        food_type_classes: int,
        ingredients_classes: int,
        portion_size_classes: int,
        dish_names_classes: int,
        cooking_method_classes: int,
        binary_classes: int,
    ):
        """
        Swin Transformer model.

        Args:
            heads_ratio (int): Ratio of attention heads. Defines the number of channels per head.
            dim (int): Input dimension (number of channels).
            dropout_mha (float): Dropout probability inside the MHA module (inside attention).
            dropout_swin_mlp (float): Dropout probability for the Swin MLP module.
            dropout_outer (float): Dropout probability for the MHA module (outside the sublayer).
            dropout_shared_mlp (float): Dropout probability for the shared MLP module.
            dropout_pre_output (float): Dropout probability for the heads after the shared MLP.
            droppath (float): Dropout probability for the drop path.
            window_size (int): Window size for the MHA module.
            input_channels (int): Number of input channels.
            depths (List[int]): List of depths for each stage.
            stage_num (int): Number of stages.
            shared_mlp_size (int): Size of the shared MLP module.
            ingredients_mlp_size (int): Size of the ingredients MLP module.
            portions_mlp_size (int): Size of the portions MLP module.
            dish_names_mlp_size (int): Size of the dish names MLP module.
            food_type_classes (int): Number of food type classes.
            ingredients_classes (int): Number of ingredients classes.
            portion_size_classes (int): Number of portion size classes.
            dish_names_classes (int): Number of dish names classes.
            cooking_method_classes (int): Number of cooking method classes.
            binary_classes (int): Number of binary outputs.
        """
        super().__init__()

        self.heads_ratio = heads_ratio
        self.dim = dim
        self.dropout_mha = dropout_mha
        self.dropout_swin_mlp = dropout_swin_mlp
        self.dropout_outer = dropout_outer
        self.dropout_shared_mlp = dropout_shared_mlp
        self.dropout_pre_output = dropout_pre_output
        self.droppath = droppath
        self.window_size = window_size
        self.input_channels = input_channels
        self.depths = depths
        self.stage_num = stage_num
        self.shared_mlp_size = shared_mlp_size
        self.ingredients_mlp_size = ingredients_mlp_size
        self.portions_mlp_size = portions_mlp_size
        self.dish_names_mlp_size = dish_names_mlp_size
        self.food_type_classes = food_type_classes
        self.ingredients_classes = ingredients_classes
        self.portion_size_classes = portion_size_classes
        self.dish_names_classes = dish_names_classes
        self.cooking_method_classes = cooking_method_classes
        self.binary_classes = binary_classes

        self.patch_embed = Patching(self.input_channels, self.input_channels * 16)

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()

        self.dims = [48] + list(dim * 2**i for i in range(1, stage_num + 1))

        assert all(
            i % self.heads_ratio == 0 for i in self.dims
        ), "dim must be divisible by heads_ratio"

        self.droppath_values = torch.linspace(0, droppath, sum(depths))

        for i in range(stage_num):
            blocks = nn.ModuleList(
                [
                    SwinBlock(
                        heads=self.dims[i] // heads_ratio,
                        dim=self.dims[i],
                        dropout_mha=dropout_mha,
                        dropout_mlp=dropout_swin_mlp,
                        dropout_outer=dropout_outer,
                        droppath=self.droppath_values[sum(depths[:i]) + j],
                        window_size=window_size,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(blocks)
            print(self.dims[i], self.dims[i + 1] / self.dims[i])
            self.merges.append(
                Merge(
                    self.dims[i],
                    squeeze=True if i > 0 else False,
                    inflation=self.dims[i + 1] / self.dims[i],
                )
            )
        self.final_dim = self.dims[-1]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(self.final_dim),
            nn.Linear(self.final_dim, self.shared_mlp_size),
            nn.GELU(),
            nn.Dropout(dropout_shared_mlp),
            nn.Linear(self.shared_mlp_size, self.shared_mlp_size // 2),
        )
        self.food_type_head = nn.Linear(self.shared_mlp_size // 2, food_type_classes)
        self.ingredients_head = nn.Sequential(
            nn.Dropout(dropout_pre_output),
            nn.Linear(self.shared_mlp_size // 2, ingredients_mlp_size),
            nn.GELU(),
            nn.Dropout(dropout_pre_output),
            nn.Linear(ingredients_mlp_size, ingredients_classes),
        )
        self.portion_shared_head = nn.Sequential(
            nn.Dropout(dropout_pre_output),
            nn.Linear(self.shared_mlp_size // 2, portions_mlp_size),
            nn.GELU(),
        )
        self.dish_names_head = nn.Sequential(
            nn.Dropout(dropout_pre_output),
            nn.Linear(self.shared_mlp_size // 2, dish_names_mlp_size),
            nn.GELU(),
            nn.Dropout(dropout_pre_output),
            nn.Linear(dish_names_mlp_size, dish_names_classes),
        )
        self.cooking_method_head = nn.Linear(
            self.shared_mlp_size // 2, cooking_method_classes
        )
        self.calories_head = nn.Linear(self.shared_mlp_size // 2, 1)
        self.fats_head = nn.Linear(self.shared_mlp_size // 2, 1)
        self.carbohydrates_head = nn.Linear(self.shared_mlp_size // 2, 1)
        self.proteins_head = nn.Linear(self.shared_mlp_size // 2, 1)
        self.binary_head = nn.Linear(self.shared_mlp_size // 2, binary_classes)
        self.portion_weight_output = nn.Linear(portions_mlp_size, portion_size_classes)
        self.portion_presence_output = nn.Linear(
            portions_mlp_size, portion_size_classes
        )

    def forward(self, x):
        x = self.patch_embed(x)

        for stage_idx, blocks in enumerate(self.stages):

            for block_idx, block in enumerate(blocks):
                shift = 0 if block_idx % 2 == 0 else self.window_size // 2
                x = block(x, shift)

            if stage_idx < len(self.merges):
                x = self.merges[stage_idx](x)

        x = self.gap(x)
        x = self.flatten(x)
        x = self.shared_mlp(x)

        food_type_logits = self.food_type_head(x)
        ingredients_logits = self.ingredients_head(x)
        portion_embedding = self.portion_shared_head(x)
        dish_name_logits = self.dish_names_head(x)
        cooking_method_logits = self.cooking_method_head(x)
        self.calories_logits = self.calories_head(x)
        self.fats_logits = self.fats_head(x)
        self.carbohydrates_logits = self.carbohydrates_head(x)
        self.proteins_logits = self.proteins_head(x)
        binary_logits = self.binary_head(x)

        portion_weight_logits = self.portion_weight_output(portion_embedding)
        portion_presence_logits = self.portion_presence_output(portion_embedding)
        return {
            "food_type_logits": food_type_logits,
            "ingredients_logits": ingredients_logits,
            "portion_weight_logits": portion_weight_logits,
            "portion_presence_logits": portion_presence_logits,
            "dish_name_logits": dish_name_logits,
            "cooking_method_logits": cooking_method_logits,
            "calories_logits": self.calories_logits,
            "fats_logits": self.fats_logits,
            "carbohydrates_logits": self.carbohydrates_logits,
            "proteins_logits": self.proteins_logits,
            "binary_logits": binary_logits,
        }


def init_weights(m):
    """
    Initialize weights for the model.

    Args:
        m (torch.nn.Module): The module to initialize weights for.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


device = torch.device("cuda")

x = torch.randn(2, 3, 448, 672, device=device)

model = SwinModel(
    heads_ratio=16,
    dim=64,
    dropout_mha=0,
    dropout_swin_mlp=0,
    dropout_outer=0,
    dropout_shared_mlp=0.25,
    dropout_pre_output=0.1,
    droppath=0.1,
    window_size=7,
    input_channels=3,
    depths=[2, 2, 18, 2],
    stage_num=4,
    shared_mlp_size=1024,
    ingredients_mlp_size=768,
    portions_mlp_size=768,
    dish_names_mlp_size=512,
    food_type_classes=5,
    ingredients_classes=589,
    portion_size_classes=437,
    dish_names_classes=191,
    cooking_method_classes=15,
    binary_classes=2,
).to(device)
model.apply(init_weights)
out = model(x)
for i in out:
    print(i, out[i].shape)

from torchinfo import summary

summary(model, input_size=x.shape)
