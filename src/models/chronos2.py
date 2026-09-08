"""Chronos-2 adapter: zero-shot, no training. Unlike the other five models,
this wraps a real pretrained checkpoint downloaded from the Hugging Face
Hub (Apache-2.0, amazon/chronos-2) -- that's expected and fine here since
Chronos-2 itself is the thing being evaluated zero-shot, not something we
fit to xbank data. No proprietary data is ever sent anywhere; only the
public pretrained weights are fetched.

Chronos-2 also differs architecturally from the other five: it's a
generic time-series FM operating on a single regularly-spaced numeric
channel (see `data.loaders.build_chronos_series`), not our
multi-column per-event feature set.
"""
from typing import List

import numpy as np
import torch
from chronos import Chronos2Pipeline

DEFAULT_CHECKPOINT = "amazon/chronos-2"


def load_pipeline(checkpoint: str = DEFAULT_CHECKPOINT, device_map: str = "cuda") -> Chronos2Pipeline:
    return Chronos2Pipeline.from_pretrained(checkpoint, device_map=device_map)


def extract_reg_embeddings(
    pipeline: Chronos2Pipeline,
    series_list: List[np.ndarray],
    batch_size: int = 256,
) -> np.ndarray:
    """One embedding per client: the [REG] token from `Chronos2Pipeline.embed()`.

    `.embed()` returns, per series, a (n_variates, num_context_patches + 2,
    d_model) tensor: patch embeddings, then [REG], then a masked
    future-patch placeholder (last position) -- confirmed by reading
    `Chronos2Model.encode()` directly (REG is concatenated right after the
    context patches, before the future/output patch is appended). All
    series here are univariate (n_variates=1, from build_chronos_series's
    plain 1-D arrays), so index 0 on the first axis. The REG token (index
    -2 on the patch axis, NOT the last index) is the model's own
    whole-sequence summary token, closest in spirit to CoLES's pooled
    output / the other models' last-event hidden state -- taking the
    literal last token would grab the meaningless masked placeholder
    instead.
    """
    embeddings, _ = pipeline.embed(series_list, batch_size=batch_size)
    reg = torch.stack([e[0, -2] for e in embeddings])
    return reg.cpu().numpy()
