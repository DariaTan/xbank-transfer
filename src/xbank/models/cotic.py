"""COTIC adapter: build the upstream module/datamodule for xbank data.

Reuses third_party/COTIC's own classes directly (net, intensity/downstream
heads, EventDataset, Normalizer) rather than reimplementing the continuous-
time NLL -- that math is easy to get subtly wrong, and this is what "run
COTIC" should mean. We skip COTIC's own Hydra/on-disk-file plumbing
(`EventDataModule`/`load_time_series_data`), which is built for their
benchmark file format, and wire the same pieces up directly from
in-memory xbank sequences instead.

COTIC's own `src` package uses absolute imports (`from src.models...`),
so `third_party/COTIC` itself (not `third_party/`) must be on `sys.path`
before importing anything from it -- `_ensure_cotic_on_path()` does that.
"""
import os
import sys
from functools import partial
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


def _ensure_cotic_on_path() -> None:
    cotic_root = os.environ.get("COTIC_ROOT", "/app/third_party/COTIC")
    if cotic_root not in sys.path:
        sys.path.insert(0, cotic_root)


def _patch_pl_compat() -> None:
    """COTIC's src/utils/__init__.py (a Hydra-CLI helper module we never
    call, but which still runs on import since it's the package __init__)
    type-hints a function argument as `pl.loggers.LightningLoggerBase`.
    That class was renamed to `pl.loggers.Logger` well before our pinned
    pytorch-lightning==2.6.5 without this shim, importing anything from 
    `src.utils.*` raises `AttributeError: module 'pytorch_lightning.loggers'
    has no attribute 'LightningLoggerBase'` at COTIC's package-init time. 
    Patched here instead of editing third_party/COTIC directly.
    """
    if not hasattr(pl.loggers, "LightningLoggerBase"):
        pl.loggers.LightningLoggerBase = pl.loggers.Logger


_ensure_cotic_on_path()
_patch_pl_compat()

from src.datamodules.components.base_dset import EventDataset  # noqa: E402
from src.models.base_model import BaseEventModule  # noqa: E402
from src.models.components.cotic.cotic import COTIC  # noqa: E402
from src.models.components.cotic.head.downstream_head import DownstreamHeadLinear  # noqa: E402
from src.models.components.cotic.head.intensity_head import IntensityHead  # noqa: E402
from src.models.components.cotic.head.joined_head import JoinedHead  # noqa: E402
from src.utils.data_utils.normalizers import ExponentialNormalizerP99  # noqa: E402


def build_datasets(
    train_times: List[torch.Tensor],
    train_types: List[torch.Tensor],
    valid_times: List[torch.Tensor],
    valid_types: List[torch.Tensor],
    num_types: int,
) -> Tuple[EventDataset, EventDataset]:
    """Fit the inter-event-time normalizer on train, reuse it (not refit)
    on valid -- matches `EventDataModule.load_event_data`'s own pattern.
    """
    train_dataset = EventDataset(train_times, train_types, num_types)
    normalizer = train_dataset.normalize_data(ExponentialNormalizerP99)

    valid_dataset = EventDataset(valid_times, valid_types, num_types)
    valid_dataset.normalize_data(normalizer)

    return train_dataset, valid_dataset, normalizer


class InMemoryEventDataModule(LightningDataModule):
    """Minimal stand-in for COTIC's `EventDataModule`, sourcing from
    already-built `EventDataset`s instead of `load_time_series_data`'s
    on-disk format. `BaseEventModule.step()` reads
    `self.trainer.datamodule.normalizer`, so this needs to be a real
    LightningDataModule, not just a pair of DataLoaders.
    """

    def __init__(
        self,
        train_dataset: EventDataset,
        valid_dataset: EventDataset,
        normalizer,
        batch_size_train: int = 32,
        batch_size_valid: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.normalizer = normalizer
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size_valid,
            shuffle=False,
            num_workers=self.num_workers,
        )


def build_module(
    num_types: int,
    in_channels: int = 128,
    kernel_size: int = 3,
    nb_filters: int = 128,
    nb_layers: int = 6,
    mlp_layers: Optional[List[int]] = None,
    uniform_sample_size: int = 16,
    lr: float = 1e-3,
) -> BaseEventModule:
    mlp_layers = list(mlp_layers) if mlp_layers is not None else [32]

    net = COTIC(
        in_channels=in_channels,
        kernel_size=kernel_size,
        nb_filters=nb_filters,
        nb_layers=nb_layers,
        num_types=num_types,
    )
    intensity_head = IntensityHead(
        kernel_size=kernel_size,
        nb_filters=nb_filters,
        mlp_layers=mlp_layers,
        num_types=num_types,
    )
    # DownstreamHead (the MLP version) has a `forward()` that doesn't
    # accept a `stage` argument, but plain `JoinedHead.forward()` always
    # passes one -- an API mismatch in third_party/COTIC between the two.
    # DownstreamHeadLinear's signature does accept `stage` and is the one
    # actually wired to work with plain JoinedHead; using it here instead.
    downstream_head = DownstreamHeadLinear(
        nb_filters=nb_filters,
        num_types=num_types,
    )
    joined_head = JoinedHead(
        intensity_head=intensity_head,
        downstream_head=downstream_head,
        uniform_sample_size=uniform_sample_size,
    )

    return BaseEventModule(
        net=net,
        joined_head=joined_head,
        optimizer=partial(torch.optim.Adam, lr=lr),
        init_lr=lr,
        scheduler=None,
        scheduler_monitoring_params={},
    )
