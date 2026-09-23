"""Small dependency-light helpers for safe embedding artifact I/O."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def embedding_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def validate_embedding_frame(df: pd.DataFrame, expected_date: Optional[str] = None) -> None:
    missing = {"inn", "date"} - set(df.columns)
    if missing:
        raise ValueError(f"embedding frame is missing columns: {sorted(missing)}")
    emb_cols = embedding_columns(df)
    if not emb_cols:
        raise ValueError("embedding frame has no emb_* columns")
    if df["inn"].duplicated().any():
        raise ValueError("embedding frame contains duplicate client ids")
    if expected_date is not None and set(df["date"].astype(str)) != {expected_date}:
        raise ValueError(f"embedding frame contains dates other than {expected_date}")
    if not np.isfinite(df[emb_cols].to_numpy(dtype=np.float32)).all():
        raise ValueError("embedding frame contains NaN or infinite values")


def validate_embedding_file(path: Path, expected_date: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    validate_embedding_frame(df, expected_date=expected_date)
    return df


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
