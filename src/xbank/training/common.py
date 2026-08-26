"""Shared full-scale-training utilities: index splitting, early stopping,
resumable checkpointing. Used by the three plain-PyTorch-loop training
scripts (train_thp.py, train_nep.py, train_mlm.py) -- CoLES/COTIC use
pytorch-lightning's own EarlyStopping/ModelCheckpoint callbacks instead,
since they already run through pl.Trainer.
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch


def split_indices(n: int, valid_frac: float = 0.05, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic train/valid split by permutation index, scaled up from
    the same pattern the smoke tests used inline (there just for a small
    sample, here for however many records the caller built).
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_valid = max(1, int(valid_frac * n))
    return idx[n_valid:], idx[:n_valid]


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
