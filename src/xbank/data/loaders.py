"""Loading xbank transactions into ptls (pytorch-lifestream) records.

Uses duckdb to sample/filter directly on the parquet file (91.5M rows,
378K clients) rather than loading the whole table into pandas -- only the
sampled clients' rows ever become a pandas DataFrame.
"""
from typing import List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import torch
from ptls.preprocessing import PandasDataPreprocessor
from ptls.preprocessing.util import dt_to_timestamp

from xbank.data.schema import (
    CLIENT_ID_COL,
    EVENT_TIME_COL,
    CATEGORY_COLS,
    NUMERIC_COLS,
)


def sample_client_ids(parquet_path: str, n: int, seed: int = 0) -> List[str]:
    """Pick `n` distinct client ids at random, without reading the full table.

    Samples from the DISTINCT id set, not the raw rows -- sampling `n` raw
    rows directly would collapse to far fewer than `n` distinct clients
    (~242 rows/client on average), which isn't what callers want here.
    """
    con = duckdb.connect()
    df = con.execute(
        f"SELECT {CLIENT_ID_COL} FROM "
        f"(SELECT DISTINCT {CLIENT_ID_COL} FROM read_parquet(?)) "
        f"USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})",
        [parquet_path],
    ).df()
    return df[CLIENT_ID_COL].tolist()


def load_raw_for_clients(parquet_path: str, client_ids: List[str]) -> pd.DataFrame:
    """Pull all rows for the given client ids into a pandas DataFrame."""
    con = duckdb.connect()
    return con.execute(
        f"SELECT * FROM read_parquet(?) WHERE {CLIENT_ID_COL} IN "
        f"({', '.join('?' * len(client_ids))})",
        [parquet_path, *client_ids],
    ).df()


def load_all_raw(parquet_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load the full transactions table -- every client, not a sample --
    for full-scale training runs. `columns` projects to a subset (e.g. just
    id/event_time/event_type for COTIC/THP, which only use one mark column)
    to cut memory for models that don't need the other columns; omit for
    all columns.
    """
    con = duckdb.connect()
    col_list = ", ".join(columns) if columns else "*"
    return con.execute(f"SELECT {col_list} FROM read_parquet(?)", [parquet_path]).df()


def cap_rows_per_client(df: pd.DataFrame, max_seq_len: int) -> pd.DataFrame:
    """Keep at most the `max_seq_len` most recent rows per client.

    Guards against the small number of outlier clients seen in the data
    (e.g. 600+ rows in a single day for one client) blowing up sequence
    length / memory during preprocessing or training.
    """
    df = df.sort_values([CLIENT_ID_COL, EVENT_TIME_COL])
    return df.groupby(CLIENT_ID_COL, group_keys=False).tail(max_seq_len)


def build_ptls_records(
    df: pd.DataFrame,
) -> Tuple[List[dict], PandasDataPreprocessor]:
    """Convert the flat per-row transaction table into ptls format:
    one dict per client with feature arrays (event_time + categorical/
    numeric columns), category columns frequency-encoded to a contiguous
    1..N range with 0 reserved for padding.

    Returns the records plus the fitted preprocessor, so callers can read
    `preprocessor.get_category_dictionary_sizes()` to size embedding
    tables -- the raw distinct counts in schema.py are for documentation,
    the fitted sizes here are what the model must actually use.

    Note: `event_time_transformation="dt_to_timestamp"` is NOT used here.
    In pytorch-lifestream==0.7.0, `PandasDataPreprocessor`'s internal
    dispatcher wraps every unitary-transform column in `pd.DataFrame(...)`
    before calling the transformer (`multithread_dispatcher.evaluate_single`),
    but `DatetimeToTimestamp.transform` calls `pd.to_datetime()` directly on
    that DataFrame instead of re-extracting the Series first -- pandas then
    tries to assemble a date from year/month/day *columns* and raises
    `ValueError: to assemble mappings requires at least that [year, month,
    day] be specified`. `ColIdentityEncoder` (used by `"none"`) does
    re-extract the Series correctly, so we precompute the numeric
    timestamp ourselves with the same `dt_to_timestamp` utility and hand it
    to the preprocessor as an already-correct passthrough column.
    """
    df = df.copy()
    df["event_time"] = dt_to_timestamp(df[EVENT_TIME_COL])

    preprocessor = PandasDataPreprocessor(
        col_id=CLIENT_ID_COL,
        col_event_time="event_time",
        event_time_transformation="none",
        cols_category=CATEGORY_COLS,
        category_transformation="frequency",
        cols_numerical=NUMERIC_COLS,
        return_records=True,
    )
    records = preprocessor.fit_transform(df)
    return records, preprocessor


def build_cotic_sequences(
    df: pd.DataFrame,
    event_type_col: str = "col_2",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
    """Convert the flat per-row transaction table into the (times, types)
    per-client sequences COTIC's `EventDataset` expects.

    COTIC (and marked TPPs generally) model a single categorical "event
    type" per event, not our full 13-column feature set -- `event_type_col`
    defaults to col_2, the MCC-like category (52 values, confirmed against
    src/report.html), as the single most semantically meaningful mark.
    The other columns are not fed to COTIC at all; this is a real
    limitation of vanilla marked-TPP architectures, not an oversight, and
    is worth a line in the paper same as the day-granularity caveat in
    configs/data/xbank.yaml.

    `times` are days since each client's own first event in `df` (not a
    shared global origin) -- COTIC's normalizer expects a bounded,
    roughly-comparable time scale across clients, not raw epoch values.
    """
    df = df.sort_values([CLIENT_ID_COL, EVENT_TIME_COL])
    codes, uniques = pd.factorize(df[event_type_col], sort=True)
    df = df.assign(_event_type=codes)

    times_list, types_list = [], []
    for _, g in df.groupby(CLIENT_ID_COL, sort=False):
        t0 = g[EVENT_TIME_COL].iloc[0]
        days = (g[EVENT_TIME_COL] - t0).dt.days.astype(float).to_numpy()
        times_list.append(torch.tensor(days, dtype=torch.float32))
        types_list.append(torch.tensor(g["_event_type"].to_numpy(), dtype=torch.long))

    return times_list, types_list, len(uniques)


def build_thp_sequences(
    df: pd.DataFrame,
    event_type_col: str = "col_2",
) -> Tuple[List[List[float]], List[List[float]], List[List[int]], int]:
    """Convert the flat per-row transaction table into the plain-list
    (time_seqs, time_delta_seqs, type_seqs) format EasyTPP's `TPPDataset`/
    `EventTokenizer` expect -- unpadded Python lists per client, not
    tensors; the tokenizer does its own padding/masking from these.

    Same single-mark simplification as `build_cotic_sequences` (event_type
    = col_2), and same days-since-client's-own-first-event time origin.
    `time_delta_seqs[i][0] = 0.0` by EasyTPP convention (first event has no
    preceding gap); type ids are 0-indexed with no reserved pad slot here --
    EasyTPP's own pad_token_id (num_event_types, the next free index) is
    supplied separately when building the model/tokenizer configs.
    """
    df = df.sort_values([CLIENT_ID_COL, EVENT_TIME_COL])
    codes, uniques = pd.factorize(df[event_type_col], sort=True)
    df = df.assign(_event_type=codes)

    time_seqs, time_delta_seqs, type_seqs = [], [], []
    for _, g in df.groupby(CLIENT_ID_COL, sort=False):
        t0 = g[EVENT_TIME_COL].iloc[0]
        days = (g[EVENT_TIME_COL] - t0).dt.days.astype(float).to_numpy()
        deltas = [0.0] + list(days[1:] - days[:-1])
        time_seqs.append(days.tolist())
        time_delta_seqs.append(deltas)
        type_seqs.append(g["_event_type"].to_numpy().tolist())

    return time_seqs, time_delta_seqs, type_seqs, len(uniques)


def build_chronos_series(
    df: pd.DataFrame,
    value_col: str = "col_11",
    freq: str = "D",
) -> List[np.ndarray]:
    """Aggregate the flat per-row transaction table into one regularly
    spaced daily series per client -- Chronos-2 is a generic time-series
    FM, it expects a fixed-frequency grid, not raw irregular event
    timestamps like the other five models consume directly.

    Sums `value_col` per (client, day), then reindexes to every day in
    [client's first event, client's last event] so gap days become 0
    rather than being silently skipped -- a missing day is a real "no
    transactions" observation, not a hole in the series. This is the
    "generic time-series FM adapted to the amount channel only" model
    from the design discussion: only one numeric column is used, none of
    the categorical features other models see.
    """
    daily = (
        df.groupby([CLIENT_ID_COL, EVENT_TIME_COL])[value_col]
        .sum()
        .rename("value")
        .reset_index()
    )

    series_list = []
    for _, g in daily.groupby(CLIENT_ID_COL, sort=False):
        g = g.set_index(EVENT_TIME_COL).sort_index()
        full_index = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        series = g["value"].reindex(full_index, fill_value=0.0)
        series_list.append(series.to_numpy(dtype=np.float32))

    return series_list
