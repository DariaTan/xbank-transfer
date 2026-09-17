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

from data.schema import (
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


def get_all_client_ids(parquet_path: str) -> List[str]:
    """Every distinct client id in the table, without reading any raw rows
    -- cheap even at ~1.5M distinct ids (2026-09-14), since only the id
    column is ever scanned. Used by full-scale (`n_clients=None`) training
    to drive chunked/streaming loading (see train_nep.py's chunked path)
    instead of `load_all_raw`'s one-shot full materialization, which
    OOM-killed against MBD's ~550-950M-row tables.
    """
    con = duckdb.connect()
    df = con.execute(
        f"SELECT DISTINCT {CLIENT_ID_COL} FROM read_parquet(?)", [parquet_path]
    ).df()
    return df[CLIENT_ID_COL].tolist()


def load_raw_for_clients(
    parquet_path: str, client_ids: List[str], columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Pull all rows for the given client ids into a pandas DataFrame --
    the WHERE filter runs inside DuckDB before anything becomes a pandas
    object, so only the requested clients' rows are ever materialized
    (unlike `load_all_raw` followed by a pandas-side `.sample()`, which
    still pays for reading and materializing every row first). `columns`
    mirrors `load_all_raw`'s same-named parameter.
    """
    con = duckdb.connect()
    col_list = ", ".join(columns) if columns else "*"
    return con.execute(
        f"SELECT {col_list} FROM read_parquet(?) WHERE {CLIENT_ID_COL} IN "
        f"({', '.join('?' * len(client_ids))})",
        [parquet_path, *client_ids],
    ).df()


def load_all_raw(parquet_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load the full transactions table -- every client, not a sample --
    for full-scale training runs (`n_clients: null`). `columns` projects to
    a subset (e.g. just id/event_time/event_type for COTIC/THP, which only
    use one mark column) to cut memory for models that don't need the
    other columns; omit for all columns.

    When `n_clients` IS set (a bounded/debug run), callers should use
    `sample_client_ids` + `load_raw_for_clients` instead of this function
    -- filtering to the sampled clients inside DuckDB, before anything
    becomes a pandas object, rather than calling this function (which
    always reads and materializes every row regardless of any cap applied
    afterward) and then subsampling the resulting DataFrame. The latter
    pattern was the actual cause of two silent OOM-kills against MBD's
    ~550-950M-row tables (2026-09-14): `n_clients` was set, but the full
    table got loaded into pandas anyway before the cap was ever applied.

    `ORDER BY` on the client id pins row order across runs -- without it,
    DuckDB's parallel scan gives no ordering guarantee, which other
    downstream code (e.g. `cap_rows_per_client`'s groupby) relies on being
    stable given a fixed seed.
    """
    con = duckdb.connect()
    col_list = ", ".join(columns) if columns else "*"
    return con.execute(
        f"SELECT {col_list} FROM read_parquet(?) ORDER BY {CLIENT_ID_COL}", [parquet_path]
    ).df()


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
    preprocessor: Optional[PandasDataPreprocessor] = None,
) -> Tuple[List[dict], PandasDataPreprocessor]:
    """Convert the flat per-row transaction table into ptls format:
    one dict per client with feature arrays (event_time + categorical/
    numeric columns), category columns frequency-encoded to a contiguous
    1..N range with 0 reserved for padding.

    Returns the records plus the fitted preprocessor, so callers can read
    `preprocessor.get_category_dictionary_sizes()` to size embedding
    tables -- the raw distinct counts in schema.py are for documentation,
    the fitted sizes here are what the model must actually use.

    Pass `preprocessor=None` (the default) to fit a fresh one on `df` --
    do this for the TRAIN split only. For the VALID split, pass the
    preprocessor that was fit on train here, so validation category
    values get encoded against the train-only vocabulary instead of
    leaking into it. Calling `preprocessor.transform(df)` directly would
    NOT do this correctly: in pytorch-lifestream==0.7.0,
    `DataPreprocessor.transform` is `def transform(self, x): return
    self.fit_transform(x)` (ptls/preprocessing/base/data_preprocessor.py)
    -- i.e. it silently refits every column transformer (the
    FrequencyEncoder vocab, in particular) from scratch on whatever `df`
    it's given, so a "transform" call on valid data would produce a
    completely different vocabulary than train's, with the same integer
    index meaning a different category between the two splits. Worked
    around here by temporarily monkeypatching each already-fitted column
    transformer's own `fit_transform` to just `transform` (skipping the
    refit) for the duration of one `preprocessor.fit_transform(df)` call --
    confirmed correct by reading `multithread_dispatcher.evaluate_single`,
    which invokes `eval_func.fit_transform(data)` via attribute lookup at
    call time, so the instance-level override is picked up. None of these
    transformers (FrequencyEncoder, ColIdentityEncoder, ...) define their
    own `fit_transform` -- they rely on sklearn's TransformerMixin default
    -- so `del ct.fit_transform` afterward cleanly restores that.

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

    Category columns in the returned records are cast to `.long()` before
    returning, unconditionally. Real reason, not defensive boilerplate:
    `FrequencyEncoder.transform` does `pd_col.map(self.mapping).fillna(
    self.other_values_code)` -- if `df` contains any category value
    absent from the fitted vocabulary (only possible on the VALID path
    above, never on a fresh fit), `.map()` produces NaN for those
    entries, upcasting that whole column to float64; `.fillna()` fills
    the NaN values but never restores the int dtype. That float64 column
    otherwise survives all the way into whatever consumes these records
    (ptls's own `TrxEncoder` for CoLES, our `TrxEmbedding`/
    `EventPredictionHeads` for NEP/MLM) and crashes on a strict-dtype
    call (`nn.Embedding`, `F.cross_entropy`) -- fixed once here at the
    source rather than relying on every current and future consumer to
    defend against it individually.
    """
    df = df.copy()
    df["event_time"] = dt_to_timestamp(df[EVENT_TIME_COL])

    if preprocessor is None:
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
    else:
        fitted = preprocessor._all_col_transformers
        for ct in fitted:
            ct.fit_transform = ct.transform
        try:
            records = preprocessor.fit_transform(df)
        finally:
            for ct in fitted:
                del ct.fit_transform

    present_category_cols = set(CATEGORY_COLS) & set(records[0].keys()) if records else set()
    for record in records:
        for col in present_category_cols:
            record[col] = record[col].long()

    return records, preprocessor


def build_cotic_sequences(
    df: pd.DataFrame,
    event_type_col: str = "col_2",
    categories: Optional[np.ndarray] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int, np.ndarray, List]:
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

    Pass `categories=None` (the default) to factorize fresh on `df` -- do
    this for the TRAIN split only, and pass the returned `categories`
    array back in here for the VALID split, so event types get mapped
    against train's category set instead of `pd.factorize` fitting a
    second, disconnected vocabulary on valid alone (the same class of
    leakage `build_ptls_records` has, and the same fix: fit once, reuse).
    Any valid-only event value absent from `categories` gets dropped (with
    a printed count) rather than silently producing an invalid code --
    astronomically unlikely given ~50 categories across 90M+ rows, but
    worth failing loud/visibly rather than corrupting an index.

    Also returns `client_ids`, the id each returned sequence belongs to
    (same groupby order as `times_list`/`types_list`) -- needed by callers
    that must attribute a downstream embedding back to the client (or
    synthetic per-window id, see splits.py) it came from.
    """
    df = df.sort_values([CLIENT_ID_COL, EVENT_TIME_COL])

    if categories is None:
        codes, categories = pd.factorize(df[event_type_col], sort=True)
    else:
        codes = pd.Categorical(df[event_type_col], categories=categories).codes
        n_unseen = int((codes == -1).sum())
        if n_unseen:
            print(
                f"  WARNING: dropping {n_unseen} events with {event_type_col} "
                f"values unseen in train",
                flush=True,
            )
            keep = codes != -1
            df, codes = df[keep], codes[keep]

    df = df.assign(_event_type=codes)

    times_list, types_list, client_ids = [], [], []
    for client_id, g in df.groupby(CLIENT_ID_COL, sort=False):
        t0 = g[EVENT_TIME_COL].iloc[0]
        days = (g[EVENT_TIME_COL] - t0).dt.days.astype(float).to_numpy()
        times_list.append(torch.tensor(days, dtype=torch.float32))
        types_list.append(torch.tensor(g["_event_type"].to_numpy(), dtype=torch.long))
        client_ids.append(client_id)

    return times_list, types_list, len(categories), categories, client_ids


def build_thp_sequences(
    df: pd.DataFrame,
    event_type_col: str = "col_2",
    categories: Optional[np.ndarray] = None,
) -> Tuple[List[List[float]], List[List[float]], List[List[int]], int, np.ndarray, List]:
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

    Same train-fit/valid-reuse `categories` contract as
    `build_cotic_sequences` -- see its docstring for why and how. Also
    returns `client_ids` in the same order as the three sequence lists,
    same rationale as `build_cotic_sequences`.
    """
    df = df.sort_values([CLIENT_ID_COL, EVENT_TIME_COL])

    if categories is None:
        codes, categories = pd.factorize(df[event_type_col], sort=True)
    else:
        codes = pd.Categorical(df[event_type_col], categories=categories).codes
        n_unseen = int((codes == -1).sum())
        if n_unseen:
            print(
                f"  WARNING: dropping {n_unseen} events with {event_type_col} "
                f"values unseen in train",
                flush=True,
            )
            keep = codes != -1
            df, codes = df[keep], codes[keep]

    df = df.assign(_event_type=codes)

    time_seqs, time_delta_seqs, type_seqs, client_ids = [], [], [], []
    for client_id, g in df.groupby(CLIENT_ID_COL, sort=False):
        t0 = g[EVENT_TIME_COL].iloc[0]
        days = (g[EVENT_TIME_COL] - t0).dt.days.astype(float).to_numpy()
        deltas = [0.0] + list(days[1:] - days[:-1])
        time_seqs.append(days.tolist())
        time_delta_seqs.append(deltas)
        type_seqs.append(g["_event_type"].to_numpy().tolist())
        client_ids.append(client_id)

    return time_seqs, time_delta_seqs, type_seqs, len(categories), categories, client_ids


def build_chronos_series(
    df: pd.DataFrame,
    value_col: str = "col_11",
    freq: str = "D",
) -> Tuple[List[np.ndarray], List]:
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

    Returns `(series_list, client_ids)` -- `client_ids[i]` is the client
    `series_list[i]` belongs to (both follow the same groupby order), so
    callers can pair embeddings back to clients without relying on an
    implicit sort order.
    """
    daily = (
        df.groupby([CLIENT_ID_COL, EVENT_TIME_COL])[value_col]
        .sum()
        .rename("value")
        .reset_index()
    )

    series_list = []
    client_ids = []
    for client_id, g in daily.groupby(CLIENT_ID_COL, sort=False):
        g = g.set_index(EVENT_TIME_COL).sort_index()
        full_index = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        series = g["value"].reindex(full_index, fill_value=0.0)
        series_list.append(series.to_numpy(dtype=np.float32))
        client_ids.append(client_id)

    return series_list, client_ids

    return series_list
