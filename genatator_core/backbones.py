from __future__ import annotations

import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

from .config import local_or_remote


class BackboneAsLetterLevelTokenClassification(nn.Module):
    """GENA backbone adapter for the provided RMT repeater classes.

    The provided RMT classes expect a model whose forward returns `TokenClassifierOutput`
    with `logits == sequence hidden states`, and whose `.base_model.embeddings.word_embeddings`
    can be extended with memory tokens. This adapter preserves that contract while loading
    only the pretrained backbone from HF/local path.
    """
    def __init__(self, backbone_path: str, trust_remote_code: bool = True):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(local_or_remote(backbone_path), trust_remote_code=trust_remote_code)
        self.config = self.base_model.config

    def resize_token_embeddings(self, *args, **kwargs):
        return self.base_model.resize_token_embeddings(*args, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        return TokenClassifierOutput(logits=out.last_hidden_state, hidden_states=out.hidden_states, attentions=out.attentions)
