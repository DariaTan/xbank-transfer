"""Shared full-scale-training utilities: index splitting, early stopping,
resumable checkpointing. Used by the three plain-PyTorch-loop training
scripts (train_thp.py, train_nep.py, train_mlm.py) -- CoLES/COTIC use
pytorch-lightning's own EarlyStopping/ModelCheckpoint callbacks instead,
since they already run through pl.Trainer.
"""
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from ptls.preprocessing import PandasDataPreprocessor
from ptls.preprocessing.multithread_dispatcher import DaskDispatcher


def split_indices(n: int, valid_frac: float = 0.05, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic train/valid split by permutation index, scaled up from
    the same pattern the smoke tests used inline (there just for a small
    sample, here for however many records the caller built).
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_valid = max(1, int(valid_frac * n))
    return idx[n_valid:], idx[:n_valid]


def split_df_by_client(
    df: pd.DataFrame, client_col: str, valid_frac: float = 0.05, seed: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a raw per-event dataframe into train/valid at CLIENT
    granularity, before any records/sequences are built from it.

    This must run before build_ptls_records/build_cotic_sequences/
    build_thp_sequences, not after -- those functions fit a vocabulary
    (frequency-encoding ranks, factorized event-type codes) on whatever
    dataframe they're given. Splitting only afterward (indexing into an
    already-built list, the pattern this replaces) still leaks: the
    vocabulary itself would already have been fit on the full population,
    valid clients included.
    """
    client_ids = df[client_col].unique()
    train_idx, valid_idx = split_indices(len(client_ids), valid_frac, seed)
    valid_ids = set(client_ids[valid_idx])
    valid_mask = df[client_col].isin(valid_ids)
    return df[~valid_mask].copy(), df[valid_mask].copy()


def check_or_save_run_config(checkpoint_dir: Path, args: Any, fields: List[str]) -> None:
    """Guards against blind resume: on a run's first epoch, saves the
    data-affecting CLI args (seed, valid_frac, n_clients, max_seq_len) to
    <checkpoint_dir>/run_config.json. On every later invocation -- which
    might be a genuine resume, or might be someone re-running the same
    command with a changed flag -- compares current args against the
    saved ones and raises if any differ.

    Without this, a resumed job silently trains/validates against
    whatever `df` the CURRENT args happen to produce -- a different seed
    or valid_frac gives a different client split than the checkpoint was
    fit on, and there's no natural error for that (unlike an architecture
    change like --hidden-size, which load_state_dict already rejects on
    its own via a shape mismatch -- no separate guard needed for those).

    Call this unconditionally, before checking whether a checkpoint
    exists to resume from -- it both writes the manifest on a fresh start
    and validates it on a resume.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "run_config.json"
    current: Dict[str, Any] = {f: getattr(args, f) for f in fields}

    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(current, indent=2))
        return

    saved: Dict[str, Any] = json.loads(manifest_path.read_text())
    mismatched = {f: (saved.get(f), current[f]) for f in fields if saved.get(f) != current[f]}
    if mismatched:
        details = "; ".join(f"{f}: checkpoint={old!r} vs now={new!r}" for f, (old, new) in mismatched.items())
        raise ValueError(
            f"Resume argument mismatch against {manifest_path} -- {details}. "
            "Resuming with different data-affecting args would silently train/"
            "validate against a different split than the checkpoint was "
            "produced with. Either match the original args, or delete "
            f"{checkpoint_dir} to start fresh."
        )


def save_preprocessor(path: Path, preprocessor: PandasDataPreprocessor) -> None:
    """Pickles a fitted `PandasDataPreprocessor` -- so a later process can
    `transform` new data against the SAME category->index mapping a
    checkpoint's embedding tables were trained against, instead of fitting
    a fresh (and on different, e.g. windowed, data, differently-skewed)
    vocabulary. `preprocessor.multithread_dispatcher` holds a live Dask
    client/task and is never picklable (`TypeError: cannot pickle
    '_asyncio.Task' object`); it's only needed while actively calling
    `fit_transform`/`transform`, not part of the fitted state itself, so
    it's dropped before pickling and restored on the live object
    afterward (`load_preprocessor` recreates an equivalent one on load).
    """
    dispatcher = preprocessor.multithread_dispatcher
    del preprocessor.multithread_dispatcher
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(preprocessor, f)
    finally:
        preprocessor.multithread_dispatcher = dispatcher


def load_preprocessor(path: Path) -> PandasDataPreprocessor:
    """Reverses `save_preprocessor`: unpickles the fitted preprocessor and
    reattaches a fresh `DaskDispatcher` (matching `n_jobs`, which does
    pickle) so `build_ptls_records(df, preprocessor=...)` can call
    `fit_transform`/`transform` on it as normal.
    """
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)
    preprocessor.multithread_dispatcher = DaskDispatcher(n_jobs=preprocessor.n_jobs)
    return preprocessor


class EarlyStopper:
    """Tracks a validation score across epochs; `patience` counts
    *consecutive* non-improving epochs, reset on any improvement -- not a
    fixed epoch budget, since there's no prior estimate of how many epochs
    full-scale training needs for any of these models (unlike the smoke
    tests, which just ran a fixed 1-2 epochs to prove the pipeline works).

    `mode="min"` for loss/NLL-style metrics, `"max"` for accuracy/recall/
    log-likelihood-style ones.
    """

    def __init__(self, patience: int = 5, mode: str = "min"):
        assert mode in ("min", "max")
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.best_epoch: int = -1
        self.bad_epochs = 0

    def step(self, score: float, epoch: int) -> bool:
        improved = self.best is None or (
            score < self.best if self.mode == "min" else score > self.best
        )
        if improved:
            self.best = score
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stopper: EarlyStopper,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": stopper.best,
            "best_epoch": stopper.best_epoch,
            "bad_epochs": stopper.bad_epochs,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stopper: EarlyStopper,
    device: torch.device,
) -> int:
    """Restores model/optimizer/stopper state in place; returns the epoch
    to resume from (one past the last epoch that was actually completed
    and saved).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    stopper.best = ckpt["best"]
    stopper.best_epoch = ckpt["best_epoch"]
    stopper.bad_epochs = ckpt["bad_epochs"]
    return ckpt["epoch"] + 1
