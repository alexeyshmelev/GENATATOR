from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss


class CADUSEUS_for_token_classification_middle_loss(nn.Module):
    """Caduceus segmentation model selected from the provided working code.

    Original class name in the old pipeline:
    `CADUSEUS_for_token_classification_middle_loss`.

    Changes are deliberately minimal and marked here:
    - `nn.LazyLinear(5)` replaces fixed `nn.Linear(256, 5)` so both PH/PS
      checkpoints work when the HF remote code changes the output width.
    - batch size > 1 is supported by keeping the original per-sample loop.
    """
    def __init__(self, caduseus_model):
        super().__init__()
        self.caduseus_model = caduseus_model
        self.fc = nn.LazyLinear(5)
        self.fc2 = nn.LazyLinear(5)

    def forward(self, input_ids=None, letter_level_labels=None, letter_level_labels_mask=None, **kwargs):
        bs = input_ids.shape[0]
        batched_collected_logits, batched_losses = [], []
        for b in range(bs):
            valid_input = input_ids[b, input_ids[b] != 4].unsqueeze(0)
            res = self.caduseus_model(valid_input, output_hidden_states=True)
            all_hidden_states = res.hidden_states
            hidden_states = res.last_hidden_state
            middle_hidden_states = all_hidden_states[len(all_hidden_states) // 2]
            curr_logits = self.fc(hidden_states[:, :-1, :])
            middle_curr_logits = self.fc2(middle_hidden_states[:, :-1, :])
            if letter_level_labels is not None:
                y = letter_level_labels[b, letter_level_labels_mask[b], :].unsqueeze(0)
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(curr_logits, y)
                middle_loss = loss_fct(middle_curr_logits, y)
                batched_losses.append((loss + middle_loss) / 2)
            target_len = letter_level_labels.shape[1] if letter_level_labels is not None else input_ids.shape[1]
            collected = F.pad(curr_logits, (0, 0, target_len - curr_logits.shape[1] - 1, 1))
            batched_collected_logits.append(collected)
        out = {"logits": torch.cat(batched_collected_logits, dim=0)}
        if batched_losses:
            out["loss"] = torch.stack(batched_losses).mean()
        return out


class CADUSEUS_for_token_classification(nn.Module):
    """Original Caduceus segmentation head with only fixed-width Linear changed to LazyLinear."""
    def __init__(self, caduseus_model):
        super().__init__()
        self.caduseus_model = caduseus_model
        self.fc = nn.LazyLinear(5)

    def forward(self, input_ids=None, letter_level_labels=None, letter_level_labels_mask=None, **kwargs):
        hidden_states = self.caduseus_model(input_ids).last_hidden_state
        bs = hidden_states.shape[0]
        batched_collected_logits, batched_losses = [], []
        for b in range(bs):
            curr_logits = self.fc(hidden_states[b, letter_level_labels_mask[b], :].unsqueeze(0))
            if letter_level_labels is not None:
                loss = BCEWithLogitsLoss()(curr_logits, letter_level_labels[b, letter_level_labels_mask[b], :].unsqueeze(0))
                batched_losses.append(loss)
            target_len = letter_level_labels.shape[1] if letter_level_labels is not None else input_ids.shape[1]
            collected = F.pad(curr_logits, (0, 0, target_len - curr_logits.shape[1] - 1, 1))
            batched_collected_logits.append(collected)
        out = {"logits": torch.cat(batched_collected_logits, dim=0)}
        if batched_losses:
            out["loss"] = torch.stack(batched_losses).mean()
        return out


class CADUSEUS_for_token_classification_transcript_type(nn.Module):
    """Transcript type model selected from the provided working code.

    Original class name in the old pipeline:
    `CADUSEUS_for_token_classification_transcript_type`.

    Changes:
    - fixed `Linear(512, ...)` layers became `LazyLinear(...)` for PS/PH output-width compatibility.
    - `logits` returned to Trainer is the transcript-type logit, while the old nucleotide logits
      are preserved as `segmentation_logits` for debugging/inference inspection.
    """
    def __init__(self, caduseus_model):
        super().__init__()
        self.caduseus_model = caduseus_model
        self.fc = nn.LazyLinear(2)
        self.ttype = nn.LazyLinear(1)

    def forward(self, input_ids=None, letter_level_labels=None, letter_level_labels_mask=None, transcript_type=None, **kwargs):
        hidden_states = self.caduseus_model(input_ids).last_hidden_state
        bs = hidden_states.shape[0]
        batched_seg_logits, batched_type_logits, batched_losses = [], [], []
        for b in range(bs):
            curr_logits = self.fc(hidden_states[b, letter_level_labels_mask[b], :].unsqueeze(0))
            curr_logits_ttype = self.ttype(hidden_states[b, -1, :].unsqueeze(0)).squeeze(-1)
            if letter_level_labels is not None and transcript_type is not None:
                loss_fct = BCEWithLogitsLoss()
                loss_seg = loss_fct(curr_logits, letter_level_labels[b, letter_level_labels_mask[b], :].unsqueeze(0))
                loss_type = loss_fct(curr_logits_ttype, transcript_type[b])
                batched_losses.append((loss_seg + loss_type) / 2)
            target_len = letter_level_labels.shape[1] if letter_level_labels is not None else input_ids.shape[1]
            batched_seg_logits.append(F.pad(curr_logits, (0, 0, target_len - curr_logits.shape[1] - 1, 1)))
            batched_type_logits.append(curr_logits_ttype)
        out = {
            "logits": torch.cat(batched_type_logits, dim=0).view(bs, 1),
            "segmentation_logits": torch.cat(batched_seg_logits, dim=0),
        }
        if batched_losses:
            out["loss"] = torch.stack(batched_losses).mean()
        return out
