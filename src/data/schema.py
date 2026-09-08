"""Column semantics for trans_any_pos_anonym_encoded.parquet.

The anonymization pass stripped all column names down to `id`/`col_1..col_16`,
so nothing here is guaranteed by the schema itself -- it's what a duckdb
DESCRIBE + per-column distinct-count/min/max pass over the full 91.5M-row
table showed on 2026-08-25, cross-checked against src/report.html (which
independently flagged col_2 as an MCC-like category with 52 values). Treat
this as a working hypothesis to revisit if a model behaves unexpectedly on
a given column, not as ground truth from a data dictionary.

col_1  DATE     event date, range 2022-01-03 .. 2024-01-31
col_2  52 vals  category / MCC-like group code 
col_3  257 vals category, finer-grained than col_2
col_4  2 vals   binary flag
col_5  4 vals   category
col_6  21 vals  category
col_7  4 vals   category
col_8  4 vals   category
col_9  6 vals   category
col_10 27 vals  category
col_11 ~11.2M   continuous, range [0, 1] -- likely a normalized amount
col_12 ~9.7M    continuous, range [0, 1] -- likely a second normalized amount
col_13 6 vals   low-cardinality float in [0, 1] -- looks quantized/binned
col_14 6 vals   same pattern as col_13
col_15 6 vals   same pattern as col_13
col_16 5 vals   same pattern as col_13

`id` is the client identifier (hashed string, 378,305 distinct in this file
-- fewer than the 426,772 clients in targets_anonym_encoded.parquet, i.e.
some target clients have zero transactions in this window; not resolved
yet, doesn't block unsupervised pretraining).
"""

CLIENT_ID_COL = "id"
EVENT_TIME_COL = "col_1"

CATEGORY_COLS = [
    "col_2", "col_3", "col_4", "col_5", "col_6",
    "col_7", "col_8", "col_9", "col_10",
    "col_13", "col_14", "col_15", "col_16",
]

NUMERIC_COLS = ["col_11", "col_12"]

ALL_FEATURE_COLS = CATEGORY_COLS + NUMERIC_COLS


# --- targets_anonym_encoded.parquet ---
#
# Confirmed directly from the original VTB data-profiling report
# (report.html, not re-derived here) -- unlike the transactions schema
# above, this isn't a duckdb-inferred guess:
#
# id      same client identifier convention as transactions
# col_1   DATE  monthly target period. Dates run Jan 2023 - Feb 2024 but
#         SKIP Nov/Dec 2023 entirely (that gap is real, not missing data
#         we're failing to load) -- target dates for a given client are
#         NOT evenly-spaced consecutive months. Transactions only run
#         through Jan 2024, so late target dates (esp. Feb 2024) have
#         little to no room to leak future transactions into a window
#         ending at them.
# col_2..col_5  4 binary product-purchase-style flags (per the report,
#         probably product-propensity flags given the source project's
#         name). Real class balance, for context when building any
#         downstream eval: col_2 ~45-60% positive (common), col_4 ~7-10%,
#         col_3/col_5 <1.5% (rare) -- col_3/col_5 in particular are
#         heavily imbalanced.
#
# 426,772 distinct clients, 5.34M total target rows (~12.5 rows/client on
# average, but this varies per client -- driven by whatever rows actually
# exist for a client, never hardcoded to an assumed fixed count).
TARGETS_CLIENT_ID_COL = "id"
TARGETS_DATE_COL = "col_1"
TARGET_COLS = ["col_2", "col_3", "col_4", "col_5"]
