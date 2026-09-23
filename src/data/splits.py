"""Downstream-embedding windowing: for each (client, target_date) pair on
a fixed calendar-date grid (e.g. "the 1st of each month"), slice that
client's transaction history down to exactly what an embedding at that
date should be built from -- strictly history up to and including
target_date, capped to the trailing `history_window_months` (never
crossing into the future), then further capped to `max_seq_len` most
recent events within that window -- same tail-keeping convention as
`cap_rows_per_client`.

Every one of the sequence builders in loaders.py (`build_ptls_records`,
`build_cotic_sequences`, `build_thp_sequences`, `build_chronos_series`)
groups by a single client-id column and produces exactly one sequence per
distinct id -- correct for pretraining (one sequence per real client), but
wrong here: the SAME client needs a DIFFERENT windowed sequence per
target_date. Rather than rewriting all four builders to understand
per-window grouping, this module stamps each windowed slice with a
synthetic composite id (f"{client_id}::{target_date}") in place of the
real client id column before handing the combined dataframe to the
existing builders unchanged -- each (client, date) window is then just
another "virtual client" as far as they're concerned. `unpack_window_id`
recovers the real (client_id, target_date) pair afterward.
"""
from typing import List, Optional, Tuple

import duckdb
import pandas as pd

from data.schema import CLIENT_ID_COL, EVENT_TIME_COL

WINDOW_ID_SEP = "::"


def window_client_history(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    history_window_months: int,
    max_seq_len: int,
) -> pd.DataFrame:
    """Slice one already-loaded client's raw events down to the window a
    downstream embedding at `cutoff` should be built from. `df` is assumed
    to already be for a single client -- no id filtering happens here.

    Single-client/single-cutoff convenience version of the batch path
    (`load_windowed_transactions_for_dates` below) -- useful for
    interactive debugging ("what does client X's window look like at date
    Y") without going through the duckdb join.
    """
    cutoff = pd.Timestamp(cutoff)
    window_start = cutoff - pd.DateOffset(months=history_window_months)
    windowed = df[(df[EVENT_TIME_COL] > window_start) & (df[EVENT_TIME_COL] <= cutoff)]
    return windowed.sort_values(EVENT_TIME_COL).tail(max_seq_len)


def unpack_window_id(window_id: str) -> Tuple[str, str]:
    """Reverses the `f"{client_id}::{target_date}"` stamp
    `load_windowed_transactions_for_dates` applies -- rsplit (not split)
    so a client id that happens to contain the separator still splits
    correctly, since the date suffix is always the last `::`-delimited
    piece.
    """
    client_id, date_str = window_id.rsplit(WINDOW_ID_SEP, 1)
    return client_id, date_str


def load_windowed_transactions_for_dates(
    transactions_path: str,
    target_dates: List[str],
    history_window_months: int,
    max_seq_len: int,
    client_ids: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
    include_cutoff: bool = True,
) -> pd.DataFrame:
    """Windowed/synthetic-id-stamped transactions for a fixed calendar-date
    grid applied to EVERY client (e.g. "the 1st of each month") -- the
    (client, target_date) pairs are a straight cross product between every
    distinct client id in the transactions table and `target_dates`,
    computed with a duckdb CROSS JOIN rather than materializing the cross
    product in Python first.

    `client_ids`, if given, restricts to a subset. `columns`, if given,
    projects inside DuckDB before materializing pandas data; it must include
    the client-id and event-time columns. This is especially important for
    THP/COTIC, which only need one event-type field from the wide table.
    `include_cutoff=False` excludes transactions on the target date. Xbank
    has transactions on its first-of-month target dates, so that setting
    avoids leaking same-day information into evaluation embeddings.
    """
    con = duckdb.connect()

    params: List = [transactions_path]
    client_filter = ""
    if client_ids is not None:
        placeholders = ", ".join("?" * len(client_ids))
        client_filter = f"AND tx.{CLIENT_ID_COL} IN ({placeholders})"
        params += list(client_ids)

    date_values = ", ".join(f"(DATE '{d}')" for d in target_dates)

    if columns is None:
        projected = f"tx.* EXCLUDE ({CLIENT_ID_COL})"
    else:
        required = {CLIENT_ID_COL, EVENT_TIME_COL}
        missing = required - set(columns)
        if missing:
            raise ValueError(f"columns must include {sorted(missing)}")
        projected_cols = [c for c in columns if c != CLIENT_ID_COL]
        projected = ", ".join(f"tx.{c}" for c in projected_cols)

    cutoff_operator = "<=" if include_cutoff else "<"
    windowed_query = f"""
        SELECT
            {projected},
            tx.{CLIENT_ID_COL} || '{WINDOW_ID_SEP}' || CAST(d.target_date AS VARCHAR)
                AS {CLIENT_ID_COL}
        FROM read_parquet(?) AS tx
        CROSS JOIN (VALUES {date_values}) AS d(target_date)
        WHERE tx.{EVENT_TIME_COL} {cutoff_operator} d.target_date
          AND tx.{EVENT_TIME_COL} > d.target_date - INTERVAL ({history_window_months}) MONTH
          {client_filter}
    """
    windowed = con.execute(windowed_query, params).df()

    windowed = (
        windowed.sort_values([CLIENT_ID_COL, EVENT_TIME_COL])
        .groupby(CLIENT_ID_COL, group_keys=False)
        .tail(max_seq_len)
    )

    return windowed
