"""Stable production entry point for daily preparation and training.

Invoke ``python -m training.daily_cli prepare ...`` or ``... train ...``.
Import implementations instead of executing their class definitions as
__main__. Keep the original training sources/identity guards unchanged so
already-running COTIC and completed THP remain compatible on resume.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys

from training.artifact_compat import legacy_class


@contextmanager
def legacy_main_aliases():
    """For old pickle references used by the unchanged Lightning loader."""
    main_module = sys.modules["__main__"]
    missing = object()
    previous = {}
    try:
        for name in ("CanonicalFrequencyEncoder", "FlatCrossEntropy"):
            previous[name] = getattr(main_module, name, missing)
            setattr(main_module, name, legacy_class(name))
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                delattr(main_module, name)
            else:
                setattr(main_module, name, value)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("stage", choices=("prepare", "train"))
    args, remaining = parser.parse_known_args()
    # --help and all stage-specific flags belong to the original parser.
    sys.argv = [sys.argv[0], *remaining]
    source = Path(__file__)
    compat_source = Path(sys.modules["training.artifact_compat"].__file__)
    print(f"DAILY_CLI serialization=v1 sha256={hashlib.sha256(source.read_bytes()).hexdigest()} "
          f"compat_sha256={hashlib.sha256(compat_source.read_bytes()).hexdigest()}", flush=True)
    with legacy_main_aliases():
        if args.stage == "prepare":
            from training.pretrain_cache import main as prepare
            prepare()
        else:
            from training.train_daily_encoders import main as train
            train()


if __name__ == "__main__":
    main()
