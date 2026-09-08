"""Shared multi-task prediction heads for NEP and MLM: one classification
head per categorical column, one shared regression head for the numeric
columns. Both models predict the SAME thing (every feature of an event,
not just col_2 like COTIC/THP) -- they differ only in which positions get
supervised (next-position for NEP, randomly masked positions for MLM) and
whether the transformer body sees a causal or bidirectional mask.
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventPredictionHeads(nn.Module):
    def __init__(
        self,
        category_dictionary_sizes: Dict[str, int],
        numeric_cols: List[str],
        d_model: int,
    ):
        super().__init__()
        self.category_cols = list(category_dictionary_sizes.keys())
        self.numeric_cols = list(numeric_cols)

        self.category_heads = nn.ModuleDict(
            {col: nn.Linear(d_model, size) for col, size in category_dictionary_sizes.items()}
        )
        self.numeric_head = (
            nn.Linear(d_model, len(self.numeric_cols)) if self.numeric_cols else None
        )

    def forward(self, hidden: torch.Tensor):
        cat_logits = {col: head(hidden) for col, head in self.category_heads.items()}
        num_pred = self.numeric_head(hidden) if self.numeric_head is not None else None
        return cat_logits, num_pred

    def loss(
        self,
        cat_logits: Dict[str, torch.Tensor],
        num_pred: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        supervise_mask: torch.Tensor,
    ) -> torch.Tensor:
        """`targets` holds the same columns at the positions to be
        predicted (already aligned -- callers handle the NEP next-step
        shift or the MLM mask-position gather before calling this).
        `supervise_mask` (B, T) selects which positions actually count
        (non-pad AND, for NEP, not the last position with no "next").

        Category targets are cast to `.long()` before cross_entropy --
        same real fix as TrxEmbedding's embedding-lookup cast (see its
        docstring): pytorch-lifestream's FrequencyEncoder can hand back a
        float64 column when a category value isn't in the fitted
        vocabulary (fillna doesn't restore the int dtype after the
        map-to-NaN), and cross_entropy's target argument rejects
        non-integer dtypes the same way nn.Embedding's indices do.
        """
        denom = supervise_mask.sum().clamp(min=1)
        total = torch.zeros((), device=supervise_mask.device)

        for col, logits in cat_logits.items():
            ce = F.cross_entropy(
                logits.transpose(1, 2), targets[col].long(), reduction="none"
            )
            total = total + (ce * supervise_mask).sum() / denom

        if num_pred is not None:
            num_target = torch.stack([targets[c] for c in self.numeric_cols], dim=-1).float()
            se = (num_pred - num_target).pow(2).mean(dim=-1)
            total = total + (se * supervise_mask).sum() / denom

        return total
