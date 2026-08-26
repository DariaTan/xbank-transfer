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
col_12 ~9.7M    continuous, range [0, 1] -- likely a second normalized feature
col_13 6 vals   low-cardinality float in [0, 1] -- looks quantized/binned
col_14 6 vals   same pattern as col_13
col_15 6 vals   same pattern as col_13
col_16 5 vals   same pattern as col_13
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
