"""CoLES adapter: build the ptls seq_encoder/module/datamodule for xbank data."""
from functools import partial
from typing import Dict, List

import numpy as np
import torch
from ptls.data_load.utils import collate_feature_dict
from ptls.frames import PtlsDataModule
from ptls.frames.coles import ColesDataset, CoLESModule
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.nn import RnnSeqEncoder, TrxEncoder

from data.schema import CATEGORY_COLS, NUMERIC_COLS


def build_seq_encoder(
    category_dictionary_sizes: Dict[str, int],
    embedding_dim: int = 32,
    hidden_size: int = 128,
    rnn_type: str = "gru",
    num_layers: int = 2,
) -> RnnSeqEncoder:
    """`category_dictionary_sizes` comes from
    `PandasDataPreprocessor.get_category_dictionary_sizes()` (fitted on the
    actual sampled data), not from the raw distinct counts in schema.py --
    frequency encoding reserves index 0 for padding/rare, so table sizes
    are dictionary_size (already includes that +1), not distinct_count.
    """
    embeddings = {
        col: {"in": category_dictionary_sizes[col], "out": embedding_dim}
        for col in CATEGORY_COLS
    }
    numeric_values = {col: "identity" for col in NUMERIC_COLS}

    trx_encoder = TrxEncoder(embeddings=embeddings, numeric_values=numeric_values)

    return RnnSeqEncoder(
        trx_encoder=trx_encoder,
        hidden_size=hidden_size,
        type=rnn_type,
        num_layers=num_layers,
    )


def build_module(
    category_dictionary_sizes: Dict[str, int],
    embedding_dim: int = 32,
    hidden_size: int = 128,
    rnn_type: str = "gru",
    num_layers: int = 2,
    lr: float = 1e-3,
) -> CoLESModule:
    seq_encoder = build_seq_encoder(
        category_dictionary_sizes, embedding_dim, hidden_size, rnn_type, num_layers
    )
    return CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=lr),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9),
    )


def build_datamodule(
    train_records: List[dict],
    valid_records: List[dict] = None,
    split_count: int = 5,
    cnt_min: int = 15,
    cnt_max: int = 150,
    batch_size: int = 64,
    num_workers: int = 0,
) -> PtlsDataModule:
    splitter = SampleSlices(split_count=split_count, cnt_min=cnt_min, cnt_max=cnt_max)
    train_data = ColesDataset(data=train_records, splitter=splitter)
    valid_data = (
        ColesDataset(data=valid_records, splitter=splitter)
        if valid_records is not None
        else None
    )
    return PtlsDataModule(
        train_data=train_data,
        train_batch_size=batch_size,
        train_num_workers=num_workers,
        valid_data=valid_data,
        valid_batch_size=batch_size,
        valid_num_workers=num_workers,
    )


def extract_embeddings(
    module: CoLESModule,
    records: List[dict],
    batch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """One embedding per record: `seq_encoder`'s own pooled output over the
    FULL sequence (no splitter -- that's a training-time augmentation, not
    used at inference), same `is_reduce_sequence=True` default pooling
    CoLES trains against, so this is exactly the representation the
    contrastive loss was shaping.
    """
    module.eval()
    seq_encoder = module.seq_encoder.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = collate_feature_dict(records[i : i + batch_size]).to(device)
            out.append(seq_encoder(batch).cpu().numpy())
    return np.concatenate(out, axis=0)
