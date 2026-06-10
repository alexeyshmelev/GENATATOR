from typing import Optional

import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = mask.bool()
    logits = logits[mask]
    targets = targets[mask].float()
    keep = targets != -100
    logits = logits[keep]
    targets = targets[keep]
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device, logits.dtype)
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float(), pos_weight=pos_weight)


def binary_sequence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().view(-1))
