"""Data-quality checks on collected downstream embeddings."""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _psi(reference: np.ndarray, actual: np.ndarray, bin_edges: np.ndarray) -> float:
    """PSI for one 1-D distribution against fixed bin edges (Siddiqi 2006):
    sum over bins of (actual% - expected%) * ln(actual% / expected%). A
    small epsilon avoids log(0)/div-by-0 on empty bins.
    """
    eps = 1e-6
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    act_pct = act_counts / max(act_counts.sum(), 1) + eps
    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def _reference_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges fit on the reference distribution, with the
    outer edges opened to +/-inf so a later date's out-of-range values
    (a real possibility once the population has drifted) still land in
    the extreme bin instead of being silently dropped from the count.
    """
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    return edges


def check_embedding_psi(
    model: str,
    date_col: str = "date",
    threshold: float = 0.1,
    embeds_dir: str = "/app/data/embeds",
    n_bins: int = 10,
) -> None:
    """Check the Population Stability Index (PSI) of `model`'s collected
    embeddings across every date under `<embeds_dir>/<model>/*.parquet`.

    For each `emb_*` column, quantile bin edges (`n_bins`) are fit on the
    EARLIEST date's values (the reference/expected distribution); every
    later date's values (the actual distribution) are then compared
    against those SAME fixed edges -- moving the bins per-date would
    defeat the point, since PSI measures how much a fixed binning's
    population shares have shifted. Per-dimension PSI is averaged across
    every embedding dimension into one PSI-per-date number.

    PSI < 0.1 is the conventional "no significant shift" band, 0.1-0.25
    "moderate shift worth investigating," > 0.25 "major shift" -- but
    `threshold` (default 0.1) is what actually gates the printed pass/
    fail verdict here, not those bands.

    Prints one line per date (the reference date itself is skipped) with
    the mean PSI, a pass/fail verdict, and row count; a failing date also
    prints its 5 most-drifted individual embedding dimensions so the
    failure is actionable rather than just a single number.
    """
    paths = sorted(Path(embeds_dir, model).glob("*.parquet"))
    if not paths:
        print(f"No embedding files found under {embeds_dir}/{model}/")
        return

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        print(f"No emb_* columns found in {embeds_dir}/{model}/ files")
        return

    dates = sorted(df[date_col].unique())
    reference_date = dates[0]
    reference = df[df[date_col] == reference_date]

    bin_edges: Dict[str, np.ndarray] = {
        col: _reference_bin_edges(reference[col].to_numpy(), n_bins) for col in emb_cols
    }

    print(
        f"PSI check for model={model!r}, reference date={reference_date}, "
        f"{len(emb_cols)} embedding dims, threshold={threshold}"
    )
    print("-" * 70)

    for date in dates[1:]:
        actual = df[df[date_col] == date]
        per_dim_psi = {
            col: _psi(reference[col].to_numpy(), actual[col].to_numpy(), bin_edges[col])
            for col in emb_cols
        }
        mean_psi = float(np.mean(list(per_dim_psi.values())))
        verdict = "FAIL" if mean_psi > threshold else "OK"
        print(f"{date}: mean PSI={mean_psi:.4f}  [{verdict}]  (n={len(actual)})")
        if verdict == "FAIL":
            worst = sorted(per_dim_psi.items(), key=lambda kv: -kv[1])[:5]
            worst_str = ", ".join(f"{c}={v:.3f}" for c, v in worst)
            print(f"    most-drifted dims: {worst_str}")

    print("-" * 70)
