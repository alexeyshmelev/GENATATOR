from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DownSample1D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, num_layers: int = 2):
        super().__init__()
        layers = [nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)]
        layers += [nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1) for _ in range(num_layers - 1)]
        self.conv_layers = nn.ModuleList(layers)
        self.activation_fn = nn.SiLU()
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor):
        for conv_layer in self.conv_layers:
            x = self.activation_fn(conv_layer(x))
        hidden = x
        x = self.avg_pool(hidden)
        return x, hidden


class UpSample1D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, num_layers: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose1d(input_channels, output_channels, kernel_size=2, stride=2)
        layers = [nn.Conv1d(output_channels * 2, output_channels, kernel_size=3, padding=1)]
        layers += [nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1) for _ in range(num_layers - 1)]
        self.conv_layers = nn.ModuleList(layers)
        self.activation_fn = nn.SiLU()

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor):
        x = self.up(x)
        diff = skip_connection.size(2) - x.size(2)
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, : skip_connection.size(2)]
        x = torch.cat([skip_connection, x], dim=1)
        for conv_layer in self.conv_layers:
            x = self.activation_fn(conv_layer(x))
        return x


class FinalConv1D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, num_layers: int = 2):
        super().__init__()
        layers = [nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)]
        layers += [nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1) for _ in range(num_layers - 1)]
        self.conv_layers = nn.ModuleList(layers)
        self.activation_fn = nn.SiLU()

    def forward(self, x: torch.Tensor):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i < len(self.conv_layers) - 1:
                x = self.activation_fn(x)
        return x


class UNET1DSegmentationHead(nn.Module):
    """Same UNET topology as the supplied RMT code, but input dimensions are explicit."""

    def __init__(self, embed_dim: int, num_classes: int, output_channels_list: Sequence[int] | None = None, num_conv_layers_per_block: int = 2):
        super().__init__()
        if output_channels_list is None:
            # For hidden=768 this reproduces [192, 384, 768]; for hidden=1024 -> [256, 512, 1024].
            base = max(32, embed_dim // 8)
            output_channels_list = [base, base * 2, base * 4]
        output_channels_list = [int(x) for x in output_channels_list]
        logger.info("[UNET] embed_dim=%d num_classes=%d channels=%s conv_layers=%d", embed_dim, num_classes, output_channels_list, num_conv_layers_per_block)
        downsample_input_channels_list = [embed_dim] + output_channels_list[:-1]
        self.downsample_blocks = nn.ModuleList([
            DownSample1D(in_ch, out_ch, num_conv_layers_per_block)
            for in_ch, out_ch in zip(downsample_input_channels_list, output_channels_list)
        ])
        reversed_output_channels_list = output_channels_list[::-1]
        upsample_input_channels_list = [output_channels_list[-1]] + reversed_output_channels_list[:-1]
        self.upsample_blocks = nn.ModuleList([
            UpSample1D(in_ch, out_ch, num_conv_layers_per_block)
            for in_ch, out_ch in zip(upsample_input_channels_list, reversed_output_channels_list)
        ])
        self.final_block = FinalConv1D(output_channels_list[0], num_classes, num_conv_layers_per_block)

    def forward(self, x: torch.Tensor):
        original_len = x.shape[-1]
        hiddens = []
        for downsample_block in self.downsample_blocks:
            x, hidden = downsample_block(x)
            hiddens.append(hidden)
        for i, upsample_block in enumerate(self.upsample_blocks):
            x = upsample_block(x, hiddens[-(i + 1)])
        x = self.final_block(x)
        if x.shape[-1] != original_len:
            x = x[:, :, :original_len]
        return x
