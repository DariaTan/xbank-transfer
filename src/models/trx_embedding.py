"""Shared per-event feature embedding for NEP and MLM.

Unlike COTIC/THP (vanilla marked TPPs, one categorical "event type" per
event), NEP and MLM are built to consume the full xbank feature set --
this is the actual point of including them alongside the single-mark TPP
models. Takes the same frequency-encoded ptls records `build_ptls_records`
already produces for CoLES (category columns 1-indexed, 0 = pad; numeric
columns as-is) and embeds every column, not just col_2.
"""
from typing import Dict, List, Optional

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

    def forward(
        self, payload: Dict[str, torch.Tensor], seq_len_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """`payload` is a dict of (B, T) tensors, e.g. `PaddedBatch.payload`
        from `ptls.data_load.utils.collate_feature_dict` -- long tensors
        for category columns, float for numeric ones.

        `seq_len_mask` (B, T), 1 at real positions / 0 at padding -- same
        tensor NEP/MLM already thread through as `seq_len_mask`/
        `attention_mask` for the backbone and loss. Pass it whenever the
        batch may contain padding (every real training/inference call);
        omitting it falls back to unmasked BatchNorm, only equivalent when
        the caller already knows every position is real (e.g. a batch
        with no padding at all).

        Category columns are cast to `.long()` before the embedding
        lookup -- not a no-op safety net, an actual fix: pytorch-
        lifestream's FrequencyEncoder.transform() does
        `pd_col.map(self.mapping).fillna(self.other_values_code)`, and if
        `x` contains any category value absent from `self.mapping` (a
        real possibility once train/valid use a shared train-fit
        vocabulary -- see build_ptls_records), the `.map()` step produces
        NaN for those entries, which upcasts that WHOLE column to
        float64; `.fillna()` fills the NaN values but never restores the
        int dtype. That float64 column survives all the way to
        `nn.Embedding`, which rejects non-integer index tensors.
        """
        parts = [self.embeddings[col](payload[col].long()) for col in self.category_cols]

        if self.numeric_proj is not None:
            numeric = torch.stack([payload[col] for col in self.numeric_cols], dim=-1).float()
            b, t, f = numeric.shape
            flat = numeric.reshape(b * t, f)

            if seq_len_mask is None:
                normed_flat = self.numeric_norm(flat)
            else:
                # BatchNorm1d normalizes (in train mode) using the CURRENT
                # batch's own mean/var, and (always) updates its running
                # mean/var from whatever it's given -- padded positions
                # are 0-filled by collate_feature_dict, so feeding the
                # full flat tensor lets padding skew both the running
                # stats and the normalization applied to the real
                # positions, depending on how much padding a given batch
                # happens to contain. Restrict BatchNorm entirely to real
                # positions instead; padded positions get exactly 0 here
                # -- an arbitrary but harmless placeholder, since they're
                # excluded from the loss/attention downstream by this
                # same mask, so their numeric value never affects
                # anything but has to exist and be finite for the shapes
                # to line up.
                flat_mask = seq_len_mask.reshape(b * t).bool()
                normed_flat = flat.new_zeros(flat.shape)
                normed_flat[flat_mask] = self.numeric_norm(flat[flat_mask])

            numeric = normed_flat.reshape(b, t, f)
            parts.append(self.numeric_proj(numeric))

        return self.out_proj(torch.cat(parts, dim=-1))
