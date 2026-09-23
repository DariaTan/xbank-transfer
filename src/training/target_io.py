"""Validated target loading shared by downstream evaluations."""
from __future__ import annotations

from typing import List

import pandas as pd

from data.schema import TARGET_COLS, TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL


def load_targets(targets_path: str, target_cols: List[str] = TARGET_COLS) -> pd.DataFrame:
    targets = pd.read_parquet(targets_path)
    targets[TARGETS_DATE_COL] = targets[TARGETS_DATE_COL].astype(str)
    keys = [TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL]
    missing = set(keys + target_cols) - set(targets.columns)
    if missing:
        raise ValueError(f"target file is missing columns: {sorted(missing)}")
    if targets[keys + target_cols].isna().any().any():
        raise ValueError("target keys or selected labels contain nulls")
    if not targets[target_cols].isin([0, 1]).all().all():
        raise ValueError("selected target labels must be binary")
    duplicate_rows = targets.duplicated(keys, keep=False)
    if duplicate_rows.any():
        variants = targets.loc[duplicate_rows].groupby(keys, sort=False)[target_cols].nunique()
        conflicts = variants.gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError(f"{int(conflicts.sum())} client-date keys have conflicting selected labels")
        before = len(targets)
        targets = targets.drop_duplicates(keys).copy()
        print(f"Collapsed {before - len(targets)} identical selected-label target rows", flush=True)
    return targets[keys + target_cols]
