from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


DEFAULT_UNET_CHUNK_SIZE = 8192


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
        if x.ndim != 3 or int(x.shape[0]) != 1:
            raise RuntimeError(
                "UNET1DSegmentationHead must be called with exactly one unpadded sample; "
                f"got shape={tuple(x.shape)}"
            )
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


def run_unet_in_chunks_single_sample(
    x: torch.Tensor,
    *,
    unet: nn.Module,
    activation_fn: nn.Module,
    chunk_size: int,
) -> torch.Tensor:
    """Run a nucleotide tensor through UNET without ever batching samples.

    ``x`` contains only real nucleotide positions and has shape ``[1, N, C]``.
    Chunks are not padded: the final call receives exactly the remaining number
    of nucleotides.  Padding inside the convolution layers is part of the UNET
    topology and is unrelated to Transformer PAD tokens.
    """

    if x.ndim != 3 or x.shape[0] != 1:
        raise RuntimeError(
            "UNET calls must receive exactly one sample with shape [1, nucleotides, channels]; "
            f"got {tuple(x.shape)}"
        )
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise RuntimeError(f"unet_chunk_size must be positive, got {chunk_size}")
    if x.shape[1] == 0:
        raise RuntimeError("UNET received an empty nucleotide sequence")

    chunks = []
    for start in range(0, int(x.shape[1]), chunk_size):
        chunk = x[:, start : start + chunk_size, :]
        z = activation_fn(unet(chunk.transpose(1, 2))).transpose(1, 2)
        if z.shape != chunk.shape:
            raise RuntimeError(
                "UNET output must preserve its input shape for residual refinement: "
                f"input={tuple(chunk.shape)} output={tuple(z.shape)}"
            )
        chunks.append(z)
    return torch.cat(chunks, dim=1)


def _sample_pos_weight(pos_weight: torch.Tensor | None, sample_index: int, device: torch.device):
    if pos_weight is None:
        return None
    weight = pos_weight
    if weight.ndim == 3:
        weight = weight[sample_index, 0, :]
    elif weight.ndim == 2:
        # Batched [B, C] is preferred.  A legacy unbatched [T, C] tensor has
        # identical rows in the supplied datasets, so row zero is sufficient.
        row = sample_index if weight.shape[0] > sample_index else 0
        weight = weight[row, :]
    elif weight.ndim != 1:
        raise RuntimeError(f"Unsupported pos_weight shape: {tuple(weight.shape)}")
    return weight.to(device).float()


def run_samplewise_chunked_unet(
    *,
    token_hidden: torch.Tensor,
    token_content_mask: torch.Tensor,
    embedding_repeater: torch.Tensor,
    letter_level_tokens: torch.Tensor,
    letter_level_attention_mask: torch.Tensor | None,
    letter_level_labels: torch.Tensor | None,
    letter_level_labels_mask: torch.Tensor | None,
    pos_weight: torch.Tensor | None,
    nucleotide_embedding: nn.Module,
    unet: nn.Module,
    activation_fn: nn.Module,
    classifier: nn.Module,
    cycles: int,
    chunk_size: int,
    context: str,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Expand batched BPE states, then run UNET one real sample at a time.

    The Transformer/AMT/RMT output may have any batch size.  Transformer PAD
    and nucleotide PAD positions are removed before this function makes each
    UNET call.  Loss is a global masked BCE over all valid label elements, so a
    short sample is not weighted the same as a much longer sample.
    """

    if token_hidden.ndim != 3:
        raise RuntimeError(f"{context}: token_hidden must be [B, T, H], got {tuple(token_hidden.shape)}")
    if token_content_mask is None:
        raise RuntimeError(f"{context}: token_content_mask is required for BPE-to-nucleotide alignment")
    if embedding_repeater is None or letter_level_tokens is None:
        raise RuntimeError(f"{context}: embedding_repeater and letter_level_tokens are required")
    if letter_level_attention_mask is None:
        # Backward-compatible fallback for older materialized batches.  New
        # batches provide the attention mask explicitly so labels never define
        # which nucleotide inputs enter UNET.
        if letter_level_labels_mask is None:
            raise RuntimeError(f"{context}: letter_level_attention_mask is required")
        letter_level_attention_mask = letter_level_labels_mask

    batch_size = int(token_hidden.shape[0])
    if any(int(x.shape[0]) != batch_size for x in (token_content_mask, embedding_repeater, letter_level_tokens, letter_level_attention_mask)):
        raise RuntimeError(f"{context}: inconsistent batch dimensions in UNET inputs")
    cycles = int(cycles)
    chunk_size = int(chunk_size)
    if cycles < 1:
        raise RuntimeError(f"{context}: UNET cycles must be >= 1, got {cycles}")
    if chunk_size <= 0:
        raise RuntimeError(f"{context}: unet_chunk_size must be positive, got {chunk_size}")

    output_length = int(letter_level_tokens.shape[1])
    num_labels = int(getattr(classifier, "out_features"))
    full_logits = token_hidden.new_zeros((batch_size, output_length, num_labels))
    loss_sum = None
    loss_element_count = 0

    for sample_index in range(batch_size):
        content_mask = token_content_mask[sample_index].bool()
        sample_token_hidden = token_hidden[sample_index, content_mask, :].unsqueeze(0)
        if sample_token_hidden.shape[1] == 0:
            raise RuntimeError(f"{context}: sample {sample_index} has no retained BPE content tokens")

        repeater_full = embedding_repeater[sample_index].long()
        nucleotide_input_mask = letter_level_attention_mask[sample_index].bool()
        unet_mask = nucleotide_input_mask & (repeater_full >= 0)
        # Positions with repeater < 0 were truncated away by BPE tokenization.
        # Exclude them silently from the U-Net input, labels, loss, and output
        # assembly; per-sample logging here would flood the training progress bar.
        repeater = repeater_full[unet_mask]
        if repeater.numel() == 0:
            raise RuntimeError(f"{context}: sample {sample_index} has no nucleotide positions covered by BPE tokens")
        if int(repeater.min().item()) < 0 or int(repeater.max().item()) >= int(sample_token_hidden.shape[1]):
            raise RuntimeError(
                f"{context}: sample {sample_index} repeater range "
                f"[{int(repeater.min().item())}, {int(repeater.max().item())}] is incompatible with "
                f"{sample_token_hidden.shape[1]} retained BPE tokens"
            )

        # Indexing by unet_mask removes nucleotide PAD tokens before embedding
        # and before every UNET call.
        nucleotide_hidden = nucleotide_embedding(
            letter_level_tokens[sample_index, unet_mask].unsqueeze(0)
        )
        x = torch.cat((nucleotide_hidden, sample_token_hidden[:, repeater, :]), dim=-1)

        sample_target = None
        target_mask = None
        weight = None
        if letter_level_labels is not None:
            sample_target = letter_level_labels[sample_index, unet_mask, :].unsqueeze(0)
            if letter_level_labels_mask is None:
                target_mask = torch.ones(sample_target.shape[:2], dtype=torch.bool, device=sample_target.device)
            else:
                target_mask = letter_level_labels_mask[sample_index, unet_mask].bool().unsqueeze(0)
            weight = _sample_pos_weight(pos_weight, sample_index, x.device)

        sample_logits = None
        for _ in range(cycles):
            z = run_unet_in_chunks_single_sample(
                x,
                unet=unet,
                activation_fn=activation_fn,
                chunk_size=chunk_size,
            )
            sample_logits = classifier(z)
            if sample_target is not None:
                if not bool(target_mask.any()):
                    raise RuntimeError(f"{context}: sample {sample_index} has an empty nucleotide label mask")
                cycle_loss = F.binary_cross_entropy_with_logits(
                    sample_logits[target_mask].float(),
                    sample_target[target_mask].float(),
                    pos_weight=weight,
                    reduction="sum",
                )
                loss_sum = cycle_loss if loss_sum is None else loss_sum + cycle_loss
                loss_element_count += int(target_mask.sum().item()) * num_labels
            x = x + z

        if sample_logits is None:
            raise RuntimeError(f"{context}: UNET produced no logits")
        # Accelerate may convert the backbone output to FP32 while autocast keeps
        # the U-Net/classifier output in BF16/FP16. Advanced-index assignment
        # requires identical dtypes, so cast only for the assembled output tensor.
        # The loss above still uses the original mixed-precision logits.
        full_logits[sample_index, unet_mask, :] = sample_logits[0].to(dtype=full_logits.dtype)

    loss = None
    if letter_level_labels is not None:
        if loss_sum is None or loss_element_count <= 0:
            raise RuntimeError(f"{context}: no valid elements were available for nucleotide BCE loss")
        loss = loss_sum / float(loss_element_count)
    return loss, full_logits
