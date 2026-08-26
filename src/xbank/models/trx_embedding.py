"""Shared per-event feature embedding for NEP and MLM.

Unlike COTIC/THP (vanilla marked TPPs, one categorical "event type" per
event), NEP and MLM are built to consume the full xbank feature set --
this is the actual point of including them alongside the single-mark TPP
models. Takes the same frequency-encoded ptls records `build_ptls_records`
already produces for CoLES (category columns 1-indexed, 0 = pad; numeric
columns as-is) and embeds every column, not just col_2.
"""
from typing import Dict, List

import torch
import torch.nn as nn


class TrxEmbedding(nn.Module):
    def __init__(
        self,
        category_dictionary_sizes: Dict[str, int],
        numeric_cols: List[str],
        embed_dim: int,
        d_model: int,
    ):
        super().__init__()
        self.category_cols = list(category_dictionary_sizes.keys())
        self.numeric_cols = list(numeric_cols)

        self.embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(size, embed_dim, padding_idx=0)
                for col, size in category_dictionary_sizes.items()
            }
        )
        self.numeric_proj = (
            nn.Linear(len(self.numeric_cols), embed_dim) if self.numeric_cols else None
        )
        self.numeric_norm = (
            nn.BatchNorm1d(len(self.numeric_cols)) if self.numeric_cols else None
        )

        concat_dim = embed_dim * len(self.category_cols) + (
            embed_dim if self.numeric_cols else 0
        )
        self.out_proj = nn.Linear(concat_dim, d_model)

    def forward(self, payload: Dict[str, torch.Tensor]) -> torch.Tensor:
        """`payload` is a dict of (B, T) tensors, e.g. `PaddedBatch.payload`
        from `ptls.data_load.utils.collate_feature_dict` -- long tensors
        for category columns, float for numeric ones.
        """
        parts = [self.embeddings[col](payload[col]) for col in self.category_cols]

        if self.numeric_proj is not None:
            # Padded positions are 0-filled by collate_feature_dict, so
            # they're included in the BatchNorm statistics here -- a minor
            # skew toward zero, acceptable for a smoke test but worth a
            # masked-mean/var version before any real training run.
            numeric = torch.stack([payload[col] for col in self.numeric_cols], dim=-1).float()
            b, t, f = numeric.shape
            numeric = self.numeric_norm(numeric.reshape(b * t, f)).reshape(b, t, f)
            parts.append(self.numeric_proj(numeric))

        return self.out_proj(torch.cat(parts, dim=-1))
