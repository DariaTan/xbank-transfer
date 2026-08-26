"""NEP: from-scratch decoder-only causal transformer for next-event
prediction.

No existing repo to adapt -- this reuses HF
`transformers`' own `LlamaModel` for the actual attention/RoPE/GQA
implementation (Llama-style architecture per the NVIDIA transaction-FM
blueprint this design is modeled on) fed our own per-event feature
embeddings via `inputs_embeds`, rather than hand-writing attention.
`LlamaModel` always applies its own causal mask internally regardless of
the passed `attention_mask` (which only conveys padding) -- exactly the
behavior needed here.
"""
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel

from xbank.models.event_heads import EventPredictionHeads
from xbank.models.trx_embedding import TrxEmbedding


class NEP(nn.Module):
    def __init__(
        self,
        category_dictionary_sizes: Dict[str, int],
        numeric_cols: List[str],
        embed_dim: int = 32,
        d_model: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        intermediate_size: int = 384,
        max_position_embeddings: int = 512,
    ):
        super().__init__()
        self.trx_embedding = TrxEmbedding(
            category_dictionary_sizes, numeric_cols, embed_dim, d_model
        )
        config = LlamaConfig(
            vocab_size=1,  # unused: we always pass inputs_embeds, never input_ids
            hidden_size=d_model,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=None,
        )
        self.backbone = LlamaModel(config)
        self.heads = EventPredictionHeads(category_dictionary_sizes, numeric_cols, d_model)

    def encode(self, payload: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        """Full-sequence causal encoding -- used for embedding extraction
        (hidden state at the last real event, built from the entire causal
        history up to it), not for the training loss (which shifts by one
        step, see `loss`).
        """
        x = self.trx_embedding(payload)
        return self.backbone(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state

    def loss(self, payload: Dict[str, torch.Tensor], seq_len_mask: torch.Tensor) -> torch.Tensor:
        x = self.trx_embedding(payload)
        hidden = self.backbone(
            inputs_embeds=x[:, :-1], attention_mask=seq_len_mask[:, :-1]
        ).last_hidden_state

        targets = {col: payload[col][:, 1:] for col in self.heads.category_cols + self.heads.numeric_cols}
        supervise_mask = seq_len_mask[:, 1:].float()

        cat_logits, num_pred = self.heads(hidden)
        return self.heads.loss(cat_logits, num_pred, targets, supervise_mask)


def last_event_embedding(hidden: torch.Tensor, seq_len_mask: torch.Tensor) -> torch.Tensor:
    """hidden: (B, T, d_model) full-sequence encoding. Returns (B, d_model),
    the hidden state at each sequence's last real (non-pad) position.
    """
    lengths = seq_len_mask.sum(dim=1).long()
    idx = (lengths - 1).clamp(min=0)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
