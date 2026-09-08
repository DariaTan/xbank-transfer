"""Adapter mapping MBD (Sber AI Lab's Multimodal Banking Dataset) onto the
same (id, col_1, col_2..col_16) column shape xbank's pipeline
(data/schema.py, data/splits.py, data/loaders.py) already expects -- so
the existing windowing/record-building code can run over MBD data
completely unchanged, with an MBD-adapted parquet only ever entering as a
plain `transactions_path`/targets-path argument, same as xbank's own files.

This is a STRUCTURAL mapping only (which raw MBD field lands in which
xbank-shaped column slot). It does NOT align MBD's category vocabularies
with xbank's fitted PandasDataPreprocessor vocabulary -- those come from
genuinely different code systems, so a category value passed through
under (say) col_2 will mostly resolve to an out-of-vocabulary embedding
when scored by an xbank-pretrained model. That's a known, documented
limitation (see RESEARCH_PLAN.md, "MBD Integration", point 4), not
something this adapter tries to fix.

MBD ships as Hive-partitioned parquet under /app/data/mbd/ (verified
directly on 2026-09-08, not just from the paper/configs/data/mbd.yaml):
    detail/trx/fold={-1,0,1,2,3,4}/*.parquet   -- transactions
    targets/fold={0,1,2,3,4}/*.parquet         -- 4 binary monthly targets
    client_split/fold={0,1,2,3,4}/*.parquet    -- per-fold client id lists
fold=-1 is the large (562M-row) UNLABELED pool -- out of scope here, since
we only need embeddings for clients that have a target to evaluate
against. LABELED_FOLDS (0-4) are the ~1M labeled, client-disjoint folds.
"""
from pathlib import Path
from typing import Iterable, Optional

import duckdb

from data.schema import CLIENT_ID_COL, EVENT_TIME_COL, TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL

MBD_ROOT = "/app/data/mbd"
MBD_TRX_GLOB = f"{MBD_ROOT}/detail/trx/fold=*/*.parquet"
MBD_TARGETS_GLOB = f"{MBD_ROOT}/targets/fold=*/*.parquet"

LABELED_FOLDS = (0, 1, 2, 3, 4)

# MBD raw transaction column -> xbank column-naming slot. event_type is
# the closest analogue to xbank's MCC-like col_2 (per configs/data/mbd.yaml);
# the rest of the mapping just fills the remaining category slots in the
# order MBD exposes them -- there's no claim that col_N means the same
# thing across the two datasets, only that both are "some category field".
# src/dst_type* are stored as DOUBLE in MBD (verified via DuckDB DESCRIBE)
# despite being integer-like codes -- cast to BIGINT below so they land in
# these category slots as something index-like rather than a raw float.
TRX_CATEGORY_MAP = {
    "event_type": "col_2",
    "event_subtype": "col_3",
    "currency": "col_4",
    "src_type11": "col_5",
    "src_type12": "col_6",
    "dst_type11": "col_7",
    "dst_type12": "col_8",
    "src_type21": "col_9",
    "src_type22": "col_10",
    "src_type31": "col_13",
    "src_type32": "col_14",
}
TRX_DOUBLE_CATEGORY_COLS = ("src_type11", "src_type12", "dst_type11", "dst_type12", "src_type21", "src_type22", "src_type31", "src_type32")

# xbank's col_12 (a second normalized amount) and col_15/col_16 (two more
# low-cardinality category slots) have no MBD analogue -- filled with a
# constant placeholder rather than left absent, since code elsewhere
# (data/schema.py's ALL_FEATURE_COLS) assumes every one of these columns
# exists on any transactions table it's given.
PLACEHOLDER_FEATURE_COLS = ("col_12", "col_15", "col_16")

# MBD's 4 binary targets -> xbank's col_2..col_5 target slots (see
# data/schema.py's TARGET_COLS). Unrelated to, and not to be confused
# with, TRX_CATEGORY_MAP's col_2..col_5 above -- transactions and targets
# are different tables/files, each with their own independent col_2..col_5.
TARGET_COLUMN_MAP = {
    "bcard_target": "col_2",
    "cred_target": "col_3",
    "zp_target": "col_4",
    "acquiring_target": "col_5",
}


def _amount_bounds(con: duckdb.DuckDBPyConnection, folds: Iterable[int]) -> tuple:
    fold_list = ", ".join(str(f) for f in folds)
    lo, hi = con.execute(
        f"""
        SELECT MIN(amount), MAX(amount)
        FROM read_parquet('{MBD_TRX_GLOB}', hive_partitioning=true)
        WHERE fold IN ({fold_list})
        """
    ).fetchone()
    return lo, hi


def build_mbd_transactions(
    output_path: str,
    folds: Iterable[int] = LABELED_FOLDS,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> None:
    """Materializes MBD's labeled-fold transactions into a parquet file
    shaped like xbank's own transactions table (id, col_1, col_2..col_16),
    so it becomes a drop-in `transactions_path` for
    data.splits.load_windowed_transactions_for_dates and everything built
    on top of it.

    event_time is kept at MBD's native hour-of-day precision (NOT
    truncated to a bare date the way xbank's col_1 is) -- preserving that
    resolution gap is the whole point of the transfer experiment
    (RESEARCH_PLAN.md's temporal-resolution hypothesis); a THP/COTIC model
    pretrained on xbank's already-daily-aggregated rows should see MBD's
    raw multi-event days as genuinely new input, not have it silently
    coarsened back down to match.

    amount is min-max scaled to [0, 1] (bounds taken from `folds` only) to
    land in roughly the same range as xbank's already-normalized col_11 --
    a heuristic rescaling, not a calibrated cross-dataset one, since MBD's
    raw amount scale/currency mix isn't characterized here.
    """
    con = con or duckdb.connect()
    lo, hi = _amount_bounds(con, folds)
    span = hi - lo if hi > lo else 1.0

    category_select = ",\n                ".join(
        f"CAST({src} AS BIGINT) AS {dst}" if src in TRX_DOUBLE_CATEGORY_COLS else f"{src} AS {dst}"
        for src, dst in TRX_CATEGORY_MAP.items()
    )
    placeholder_select = ",\n                ".join(f"0 AS {col}" for col in PLACEHOLDER_FEATURE_COLS)
    fold_list = ", ".join(str(f) for f in folds)

    query = f"""
        COPY (
            SELECT
                client_id AS {CLIENT_ID_COL},
                event_time AS {EVENT_TIME_COL},
                (amount - {lo}) / {span} AS col_11,
                {category_select},
                {placeholder_select}
            FROM read_parquet('{MBD_TRX_GLOB}', hive_partitioning=true)
            WHERE fold IN ({fold_list})
        ) TO '{output_path}' (FORMAT PARQUET)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    con.execute(query)


def build_mbd_targets(
    output_path: str,
    folds: Iterable[int] = LABELED_FOLDS,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> None:
    """Materializes MBD's per-fold targets into xbank's targets shape
    (id, col_1, col_2..col_5), so it's a drop-in TARGETS_PATH for anything
    built against data.schema's TARGETS_CLIENT_ID_COL/TARGETS_DATE_COL/
    TARGET_COLS convention (e.g. train_downstream.py). Also keeps a
    `fold` column -- unlike xbank, MBD's benchmark protocol is a fixed
    5-fold client-disjoint split (client_split/fold=k), not a calendar
    train/test cut, so whatever consumes this needs `fold` to split
    train/test the way MBD's own benchmark does (RESEARCH_PLAN.md's ID/
    date splitting section) -- xbank's targets have no equivalent column.

    MBD's `mon` is a MONTH-END date string ("2022-02-28", ...); normalized
    here to the first of that month ("2022-02-01") to match xbank's
    month-start convention (data/splits.py's fixed calendar-date grid,
    and train_downstream.py's month_range()) -- a date-REPRESENTATION
    fix, not a data change; the target still describes that same
    calendar month.
    """
    con = con or duckdb.connect()
    target_select = ",\n                ".join(f"{src} AS {dst}" for src, dst in TARGET_COLUMN_MAP.items())
    fold_list = ", ".join(str(f) for f in folds)

    query = f"""
        COPY (
            SELECT
                client_id AS {TARGETS_CLIENT_ID_COL},
                DATE_TRUNC('month', CAST(mon AS DATE)) AS {TARGETS_DATE_COL},
                {target_select},
                fold
            FROM read_parquet('{MBD_TARGETS_GLOB}', hive_partitioning=true)
            WHERE fold IN ({fold_list})
        ) TO '{output_path}' (FORMAT PARQUET)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    con.execute(query)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Materialize MBD data adapted to xbank's column shape.")
    parser.add_argument("--transactions-out", default="/app/data/mbd_data/transactions_adapted.parquet")
    parser.add_argument("--targets-out", default="/app/data/mbd_data/targets_adapted.parquet")
    parser.add_argument("--folds", type=int, nargs="+", default=list(LABELED_FOLDS))
    args = parser.parse_args()

    print(f"Building adapted MBD transactions (folds={args.folds}) ...", flush=True)
    build_mbd_transactions(args.transactions_out, folds=args.folds)
    print(f"  wrote {args.transactions_out}", flush=True)

    print(f"Building adapted MBD targets (folds={args.folds}) ...", flush=True)
    build_mbd_targets(args.targets_out, folds=args.folds)
    print(f"  wrote {args.targets_out}", flush=True)
