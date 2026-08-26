"""MLM: from-scratch bidirectional (BERT-style) transformer, masked
next... rather masked EVENT prediction over the full xbank feature set.

Shares TrxEmbedding/EventPredictionHeads with NEP; differs in transformer
body (HF `BertModel`, bidirectional, no causal mask) and objective (mask
~15% of real events per sequence, predict their original feature values
from context on both sides, BERT-style).

Correctness note for later use as an embedding extractor: bidirectional
attention means a masked position's prediction can see events AFTER it --
that's fine for the pretraining objective, but it means "hidden state at
the last position" is only a safe, no-future-leakage embedding at
inference time if the INPUT sequence itself was already truncated to end
at the desired cutoff `t` (i.e. splits.py's history-up-to-t window, same
as every other model here). Never feed MLM a sequence extending past `t`
and try to read out an embedding at an earlier position -- it will have
looked ahead.
"""
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from xbank.models.event_heads import EventPredictionHeads
from xbank.models.trx_embedding import TrxEmbedding


class MLM(nn.Module):
    def __init__(
        self,
        category_dictionary_sizes: Dict[str, int],
        numeric_cols: List[str],
        embed_dim: int = 32,
        d_model: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        intermediate_size: int = 384,
        max_position_embeddings: int = 512,
        mask_prob: float = 0.15,
    ):
        super().__init__()
        self.trx_embedding = TrxEmbedding(
            category_dictionary_sizes, numeric_cols, embed_dim, d_model
        )
        self.mask_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        config = BertConfig(
            vocab_size=1,  # unused: we always pass inputs_embeds
            hidden_size=d_model,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
        )
        self.backbone = BertModel(config, add_pooling_layer=False)
        self.heads = EventPredictionHeads(category_dictionary_sizes, numeric_cols, d_model)
        self.mask_prob = mask_prob

    def encode(self, payload: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        """No masking applied -- full given sequence, bidirectionally
        encoded. See module docstring for the no-future-leakage caveat.
        """
        x = self.trx_embedding(payload)
        return self.backbone(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state

    def loss(self, payload: Dict[str, torch.Tensor], seq_len_mask: torch.Tensor) -> torch.Tensor:
        x = self.trx_embedding(payload)

        maskable = seq_len_mask.bool()
        rand_mask = (torch.rand_like(seq_len_mask, dtype=torch.float) < self.mask_prob) & maskable
        masked_x = torch.where(rand_mask.unsqueeze(-1), self.mask_embedding, x)

        hidden = self.backbone(
            inputs_embeds=masked_x, attention_mask=seq_len_mask
        ).last_hidden_state

        targets = {col: payload[col] for col in self.heads.category_cols + self.heads.numeric_cols}
        supervise_mask = rand_mask.float()

        cat_logits, num_pred = self.heads(hidden)
        return self.heads.loss(cat_logits, num_pred, targets, supervise_mask)


def last_event_embedding(hidden: torch.Tensor, seq_len_mask: torch.Tensor) -> torch.Tensor:
    lengths = seq_len_mask.sum(dim=1).long()
    idx = (lengths - 1).clamp(min=0)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
