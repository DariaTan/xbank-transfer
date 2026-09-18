"""Adapter mapping MBD (Sber AI Lab's Multimodal Banking Dataset) onto the
same (id, col_1, col_2..col_16) column shape xbank's pipeline
(data/schema.py, data/splits.py, data/loaders.py) already expects -- so
the existing windowing/record-building code can run over MBD data
completely unchanged, with an MBD-adapted parquet only ever entering as a
plain `transactions_path`/targets-path argument, same as xbank's own files.

This is a STRUCTURAL mapping only (which raw MBD field lands in which
xbank-shaped column slot). It does NOT align MBD's category vocabularies
with xbank's real category codes -- those come from genuinely different
code systems, so at xbank-transfer eval time (once a model is pretrained
on this adapted MBD data), xbank's real values passed through the same
named slot will mostly resolve to an out-of-vocabulary embedding. That's
a known, documented limitation (see RESEARCH_PLAN.md §4, point 5), not
something this adapter tries to fix.

MBD ships as Hive-partitioned parquet, now living at /app/data/mbd_data/raw/
(moved there 2026-09-14 -- previously /app/data/mbd/, still the layout
verified directly on 2026-09-08, not just from the paper/configs/data/
mbd.yaml):
    detail/trx/fold={-1,0,1,2,3,4}/*.parquet   -- transactions
    targets/fold={0,1,2,3,4}/*.parquet         -- 4 binary monthly targets
    client_split/fold={0,1,2,3,4}/*.parquet    -- per-fold client id lists
fold=-1 is the large (562M-row) UNLABELED pool -- no targets exist for it
(no `targets/fold=-1` partition), so it's excluded from build_mbd_targets
and from anything scored against MBD's own labels. It IS included in
build_mbd_transactions's default folds (ALL_FOLDS, added 2026-09-14): the
five FMs are self-supervised, so pretraining needs no labels at all, and
MBD's benchmark reserves fold=-1 specifically for exactly this use.
LABELED_FOLDS (0-4) are the ~1M labeled, client-disjoint folds used for
targets and for the in-domain MBD downstream probe's fold rotation.

IMPORTANT naming distinction (2026-09-14): `mbd_data/raw/` is this
SOURCE tree (untouched MBD, as shipped) -- NOT this adapter's output.
This adapter's own output (xbank-shaped, either aggregation level) goes
to a DIFFERENT pair of folders: `mbd_data/raw_adapted/` (freq=None) and
`mbd_data/daily_adapted/` (freq="D") -- see the CLI section below and
configs/data/mbd.yaml / mbd_daily.yaml, whose paths.transactions point at
the `_adapted` folders, never at `mbd_data/raw/` itself.

`build_mbd_transactions`'s `freq` controls the aggregation level: None
(default) keeps MBD's transactions completely UNTOUCHED, exactly as
shipped -- one row per raw transaction, at whatever native hour-of-day
timestamp precision MBD already has. This is NOT an "hourly aggregation"
-- nothing is grouped or binned to an hour; it's the un-aggregated
baseline, referred to as "raw" throughout (mbd_data/raw_adapted/,
configs/data/mbd.yaml), in contrast to "D" which DOES aggregate: to
xbank's own (client, day, category) row convention instead (see that
function's docstring). This lets a client's MBD history be pretrained on
at either granularity, to separate "changing institution" from "changing
aggregation level" as two independent ablation axes (RESEARCH_PLAN.md
§4) -- weekly was considered and dropped, only raw (None) and daily ("D")
are built.
"""
from pathlib import Path
from typing import Iterable, Optional

import duckdb

from data.schema import CLIENT_ID_COL, EVENT_TIME_COL, TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL

MBD_ROOT = "/app/data/mbd_data/raw"
MBD_TRX_GLOB = f"{MBD_ROOT}/detail/trx/fold=*/*.parquet"
MBD_TARGETS_GLOB = f"{MBD_ROOT}/targets/fold=*/*.parquet"

# DuckDB's default spill-to-disk temp directory is relative to the CWD
# (/app, the repo bind mount) -- which lives on the HOST's small root
# partition (~230GB, often near-full), not the big /mnt/storage mount
# /app/data is backed by. The daily aggregation's GROUP BY over ~950M rows
# needs real spill space and hit "No space left on device" against the
# root partition's leftover ~17GB (2026-09-14) even though /app/data had
# 4.8TB free. Every connection this module opens must set temp_directory
# here explicitly -- duckdb.connect() alone is not enough.
DUCKDB_TEMP_DIR = "/app/data/duckdb_tmp"


def _connect(temp_dir: str = DUCKDB_TEMP_DIR) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{temp_dir}'")
    # DuckDB's own default memory_limit (~80% of system RAM) leaves no
    # safety margin: it only starts spilling to temp_directory once
    # already close to that limit, by which point the OS's OOM killer can
    # win the race and silently kill the process first (no Python
    # traceback, no error in the log -- confirmed 2026-09-14, the daily
    # aggregation over all folds died this way twice). Capping it well
    # below actual available RAM forces DuckDB to spill early and
    # deliberately instead of gambling against the kernel.
    con.execute("SET memory_limit = '60GB'")
    return con

LABELED_FOLDS = (0, 1, 2, 3, 4)
# Labeled folds + the large unlabeled pool -- the default for
# build_mbd_transactions (pretraining needs no labels), NOT for
# build_mbd_targets (fold=-1 has no targets partition at all).
ALL_FOLDS = (-1,) + LABELED_FOLDS

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


def _amount_bounds(con: duckdb.DuckDBPyConnection, folds: Iterable[int], trx_glob: str = MBD_TRX_GLOB) -> tuple:
    fold_list = ", ".join(str(f) for f in folds)
    lo, hi = con.execute(
        f"""
        SELECT MIN(amount), MAX(amount)
        FROM read_parquet('{trx_glob}', hive_partitioning=true)
        WHERE fold IN ({fold_list})
        """
    ).fetchone()
    return lo, hi


def build_mbd_transactions(
    output_path: str,
    folds: Iterable[int] = ALL_FOLDS,
    freq: Optional[str] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    trx_glob: str = MBD_TRX_GLOB,
    temp_dir: str = DUCKDB_TEMP_DIR,
) -> None:
    """Materializes MBD transactions into a parquet file shaped like
    xbank's own transactions table (id, col_1, col_2..col_16), so it
    becomes a drop-in `transactions_path` for
    data.splits.load_windowed_transactions_for_dates and everything built
    on top of it. Defaults to ALL_FOLDS (labeled 0-4 + unlabeled -1) since
    this now primarily feeds PRETRAINING, which needs no labels --
    restrict to LABELED_FOLDS explicitly if the caller specifically wants
    only clients that also have targets (e.g. for embedding generation
    ahead of the in-domain MBD downstream probe).

    `freq`:
    - None (default): keeps MBD's transactions completely UNTOUCHED, one
      row per raw transaction, at whatever native hour-of-day precision
      MBD already has -- NOT an aggregation step, this is the un-
      aggregated baseline ("raw" throughout, mbd_data/raw/).
    - "D": aggregates to xbank's OWN convention instead -- one row per
      (client, day, full category-combination), `amount` SUMMED across
      whatever raw transactions collapse into it (see configs/data/
      xbank.yaml's "Row unit" note for the xbank-side definition this
      mirrors). Lets MBD's daily-aggregated history be compared against
      xbank pretraining/transfer as a controlled ablation of aggregation
      level ALONE, holding the institution fixed (RESEARCH_PLAN.md §4).

    amount is min-max scaled to [0, 1] AFTER aggregation (bounds computed
    on whatever the final row's amount actually is -- raw per-transaction
    for freq=None, the per-day sum for freq="D") to land in roughly the
    same range as xbank's already-normalized col_11 -- a heuristic
    rescaling, not a calibrated cross-dataset one, since MBD's raw amount
    scale/currency mix isn't characterized here. Scaling BEFORE summing
    would let a well-populated day exceed 1.0 and misrepresent what
    "already normalized" means at the row's actual grain.
    """
    if freq not in (None, "D"):
        raise ValueError(f"freq must be None or 'D', got {freq!r}")

    con = con or _connect(temp_dir)
    fold_list = ", ".join(str(f) for f in folds)
    placeholder_select = ",\n                ".join(f"0 AS {col}" for col in PLACEHOLDER_FEATURE_COLS)

    if freq is None:
        category_select = ",\n                ".join(
            f"CAST({src} AS BIGINT) AS {dst}" if src in TRX_DOUBLE_CATEGORY_COLS else f"{src} AS {dst}"
            for src, dst in TRX_CATEGORY_MAP.items()
        )
        lo, hi = _amount_bounds(con, folds, trx_glob)
        span = hi - lo if hi > lo else 1.0

        query = f"""
            COPY (
                SELECT
                    client_id AS {CLIENT_ID_COL},
                    event_time AS {EVENT_TIME_COL},
                    (amount - {lo}) / {span} AS col_11,
                    {category_select},
                    {placeholder_select}
                FROM read_parquet('{trx_glob}', hive_partitioning=true)
                WHERE fold IN ({fold_list})
            ) TO '{output_path}' (FORMAT PARQUET)
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        con.execute(query)
        return

    # freq == "D": aggregate to one row per (client, day, full category
    # combination). Split into up to TWO independent passes -- labeled
    # folds (0-4) and the unlabeled pool (fold=-1) -- rather than one
    # combined GROUP BY over all ~950M rows at once, which silently
    # OOM-killed the process twice (2026-09-14, confirmed by RSS climbing
    # toward available RAM then the process vanishing with no Python
    # traceback at all -- the signature of the kernel's OOM killer winning
    # a race DuckDB's own memory_limit didn't prevent in time). This is
    # CORRECT, not an approximation: folds are strictly client-disjoint
    # (verified 2026-09-14 -- 0 clients span more than one fold) and
    # client_id is part of the GROUP BY key, so no aggregation group can
    # ever span two fold-groups. Doing the GROUP BY in two smaller pieces,
    # each written to its own intermediate parquet file, then UNIONing
    # them for the final scale-and-write pass, is mathematically identical
    # to one big GROUP BY over everything -- just with a lower peak
    # memory footprint per pass (the final UNION+write pass is a plain
    # scan, not a hash aggregation, so it's cheap regardless of size --
    # the same shape of query that already succeeded in under 2 minutes
    # for the freq=None/raw case above).
    group_cols = list(TRX_CATEGORY_MAP.values())
    daily_category_select = ",\n                ".join(
        f"CAST({src} AS BIGINT) AS {dst}" if src in TRX_DOUBLE_CATEGORY_COLS else f"{src} AS {dst}"
        for src, dst in TRX_CATEGORY_MAP.items()
    )

    fold_groups = [g for g in ([f for f in folds if f != -1], [f for f in folds if f == -1]) if g]
    tmp_dir = Path(temp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    intermediate_paths = []
    try:
        for i, group_folds in enumerate(fold_groups):
            group_fold_list = ", ".join(str(f) for f in group_folds)
            intermediate_path = tmp_dir / f"_daily_agg_pass{i}.parquet"
            con.execute(f"""
                COPY (
                    SELECT
                        client_id AS {CLIENT_ID_COL},
                        DATE_TRUNC('day', event_time) AS {EVENT_TIME_COL},
                        {daily_category_select},
                        SUM(amount) AS amount_sum
                    FROM read_parquet('{trx_glob}', hive_partitioning=true)
                    WHERE fold IN ({group_fold_list})
                    GROUP BY client_id, DATE_TRUNC('day', event_time), {", ".join(group_cols)}
                ) TO '{intermediate_path}' (FORMAT PARQUET)
            """)
            intermediate_paths.append(str(intermediate_path))

        union_sql = " UNION ALL ".join(f"SELECT * FROM read_parquet('{p}')" for p in intermediate_paths)

        lo, hi = con.execute(f"SELECT MIN(amount_sum), MAX(amount_sum) FROM ({union_sql})").fetchone()
        span = hi - lo if hi > lo else 1.0

        query = f"""
            COPY (
                SELECT
                    {CLIENT_ID_COL},
                    {EVENT_TIME_COL},
                    (amount_sum - {lo}) / {span} AS col_11,
                    {", ".join(group_cols)},
                    {placeholder_select}
                FROM ({union_sql})
            ) TO '{output_path}' (FORMAT PARQUET)
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        con.execute(query)
    finally:
        for p in intermediate_paths:
            Path(p).unlink(missing_ok=True)


def build_mbd_targets(
    output_path: str,
    folds: Iterable[int] = LABELED_FOLDS,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    targets_glob: str = MBD_TARGETS_GLOB,
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
    con = con or _connect()
    target_select = ",\n                ".join(f"{src} AS {dst}" for src, dst in TARGET_COLUMN_MAP.items())
    fold_list = ", ".join(str(f) for f in folds)

    query = f"""
        COPY (
            SELECT
                client_id AS {TARGETS_CLIENT_ID_COL},
                DATE_TRUNC('month', CAST(mon AS DATE)) AS {TARGETS_DATE_COL},
                {target_select},
                fold
            FROM read_parquet('{targets_glob}', hive_partitioning=true)
            WHERE fold IN ({fold_list})
        ) TO '{output_path}' (FORMAT PARQUET)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    con.execute(query)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Materialize MBD data adapted to xbank's column shape.")
    parser.add_argument("--freq", choices=["raw", "daily"], default="raw", help="raw keeps MBD's transactions completely untouched (native per-transaction precision, no aggregation); daily aggregates to xbank's own (client, day, category) convention")
    parser.add_argument("--transactions-out", default=None, help="default /app/data/mbd_data/{raw_adapted,daily_adapted}/transactions.parquet depending on --freq")
    parser.add_argument("--targets-out", default=None, help="default /app/data/mbd_data/{raw_adapted,daily_adapted}/targets.parquet depending on --freq -- same content either way (targets don't depend on transaction aggregation), duplicated per-frequency so each configs/data/mbd*.yaml's paths.transactions/paths.targets pair is self-contained under one folder")
    parser.add_argument("--folds", type=int, nargs="+", default=list(ALL_FOLDS), help="folds for the TRANSACTIONS file only (default: ALL_FOLDS, labeled 0-4 + unlabeled -1 -- pretraining needs no labels). Targets always use only the labeled folds (0-4), independent of this flag, since fold=-1 has no targets partition.")
    parser.add_argument("--mbd-root", default=MBD_ROOT,
                        help="root of the MBD source tree (expects detail/trx/fold=*/ and targets/fold=*/ inside); default is the in-container path -- pass the host path when running outside the container")
    parser.add_argument("--temp-dir", default=DUCKDB_TEMP_DIR,
                        help="duckdb spill dir for the daily aggregation's intermediates (needs free disk roughly 2x the output size); default is the in-container path")
    args = parser.parse_args()

    freq = None if args.freq == "raw" else "D"
    # freq=None (raw, untouched) -> mbd_data/raw_adapted/, freq="D" (daily)
    # -> mbd_data/daily_adapted/ -- NOT mbd_data/raw/, which is the
    # untouched MBD SOURCE tree (see module docstring), not this adapter's
    # output. Matches configs/data/mbd.yaml (raw_adapted) and
    # configs/data/mbd_daily.yaml (daily_adapted)'s paths.transactions/
    # paths.targets. A future weekly variant would add "weekly_adapted"
    # alongside these two, same convention.
    freq_dir = "daily_adapted" if freq == "D" else "raw_adapted"
    transactions_out = args.transactions_out or f"/app/data/mbd_data/{freq_dir}/transactions.parquet"
    targets_out = args.targets_out or f"/app/data/mbd_data/{freq_dir}/targets.parquet"

    trx_glob = f"{args.mbd_root}/detail/trx/fold=*/*.parquet"
    targets_glob = f"{args.mbd_root}/targets/fold=*/*.parquet"
    # one shared connection with the CALLER's temp dir -- build_mbd_targets's
    # own _connect() default still points at the in-container path and would
    # crash on a host run
    con = _connect(args.temp_dir)

    print(f"Building adapted MBD transactions (folds={args.folds}, freq={args.freq}) ...", flush=True)
    build_mbd_transactions(transactions_out, folds=args.folds, freq=freq, con=con,
                           trx_glob=trx_glob, temp_dir=args.temp_dir)
    print(f"  wrote {transactions_out}", flush=True)

    print(f"Building adapted MBD targets (folds={list(LABELED_FOLDS)}) ...", flush=True)
    build_mbd_targets(targets_out, folds=LABELED_FOLDS, con=con, targets_glob=targets_glob)
    print(f"  wrote {targets_out}", flush=True)
