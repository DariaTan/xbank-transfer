"""Read trusted local artifacts, including the original daily CLI pickles.

The 2026-10-02 CLI wrote two helper classes as ``__main__``. Resolve only
these exact references without rewriting artifacts or their manifests.
This is ordinary (unsafe for untrusted inputs) pickle, not a sandbox.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path

import torch


def legacy_class(name):
    if name == "CanonicalFrequencyEncoder":
        from training.pretrain_cache import CanonicalFrequencyEncoder
        return CanonicalFrequencyEncoder
    if name == "FlatCrossEntropy":
        from training.train_daily_encoders import FlatCrossEntropy
        return FlatCrossEntropy
    raise AttributeError(name)


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name in {"CanonicalFrequencyEncoder", "FlatCrossEntropy"}:
            return legacy_class(name)
        if module.startswith("src.models.components.") or module.startswith("src.utils.data_utils."):
            # COTIC embeds its upstream modules in Lightning hyperparameters.
            # Initialize the existing adapter/path shim before unpickling them.
            import models.cotic  # noqa: F401
        return super().find_class(module, name)


def load(stream, **kwargs):
    return Unpickler(stream, **kwargs).load()


def loads(value, **kwargs):
    return load(io.BytesIO(value), **kwargs)


def load_preprocessor(path: Path):
    from ptls.preprocessing.multithread_dispatcher import DaskDispatcher
    with Path(path).open("rb") as stream:
        preprocessor = load(stream)
    preprocessor.multithread_dispatcher = DaskDispatcher(n_jobs=preprocessor.n_jobs)
    return preprocessor


def load_torch_checkpoint(path, *, map_location="cpu"):
    import training.artifact_compat as compat_pickle
    return torch.load(path, map_location=map_location, weights_only=False, pickle_module=compat_pickle)
