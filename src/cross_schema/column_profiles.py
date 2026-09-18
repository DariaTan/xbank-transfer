"""Column passports + dictionary-free schema matching (paper Sec. 5).

One run, `python -m cross_schema.column_profiles`, does the whole part-2
pipeline and writes every artifact under --out-dir (default
data/profiles/ inside the repository):

  1. profile    statistical passports for xbank's anonymous feature columns
                and MBD's named transaction fields. Transactions tables ONLY:
                target files are never read, target columns never enter
                similarity/matching -- only features may be aligned.
  2. similarity pairwise passport similarity in [0, 1], type-constrained
                (numeric<->numeric, categorical<->categorical).
  3. match      rectangular one-to-one assignment per type block; pairs below
                --min-sim are dropped rather than forced, leftovers are
                reported explicitly (xbank: 13 cat + 2 num, MBD: 11 cat +
                1 num -- the partial-mapping rule fixed in advance).
  4. testbed    sanity checks where truth is known by construction:
                "folds"  -- MBD fold A renamed to anonymous slots vs MBD
                            fold B named (client-disjoint, same schema);
                "daily"  -- the daily-aggregated xbank-SHAPED MBD table
                            (built first by `python -m data.mbd_adapter
                            --freq daily`) vs raw hourly MBD; truth is the
                            adapter's own column map.
  5. anchors    the resulting mapping checked against the few known-truth
                correspondences on the xbank side (schema.py / report.html).
  6. freeze     frozen_mapping.json -- the ONLY artifact downstream
                inference / permutation runners read.

Artifacts (all small, human-readable, git-diffable):
  xbank_profiles.csv  mbd_profiles.csv  similarity.csv  mapping.csv
  testbed_recovery.csv  anchors.csv  frozen_mapping.json

Nothing here trains anything or touches checkpoints; every step is
deterministic duckdb aggregation. The permutation runner will import
`assign` / `similarity` / `profile_columns` instead of reimplementing them.
"""

# --------------------------------------------------------------------------
# import
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from data.schema import CATEGORY_COLS, EVENT_TIME_COL, NUMERIC_COLS



# --------------------------------------------------------------------------
# variables
# --------------------------------------------------------------------------

XBANK_TRX_DEFAULT = '/home/stsix/xbank-transfer/data/xbank_data/trans_any_pos_anonym_encoded.parquet'
MBD_ROOT_DEFAULT = '/home/stsix/xbank-transfer/data/mbd'
MBD_ADAPTED_DAILY_DEFAULT = '/home/stsix/xbank-transfer/data/mbd_daily/transactions_adapted_daily.parquet'
OUT_DIR_DEFAULT = '/home/stsix/xbank-transfer/data/profiles'

MBD_TIME_FIELD = "event_time"
MBD_NUMERIC_FIELDS = ("amount",)
MBD_CATEGORICAL_FIELDS = (
    "event_type", "event_subtype", "currency",
    "src_type11", "src_type12", "dst_type11", "dst_type12",
    "src_type21", "src_type22", "src_type31", "src_type32",
)
# stored as DOUBLE in MBD despite being integer-like codes (mirrors
# data/mbd_adapter.py) -- cast to BIGINT so profiles see codes, not floats
MBD_DOUBLE_FIELDS = frozenset(MBD_CATEGORICAL_FIELDS[3:])

N_BINS = 20          # equal-width bins on min-max-scaled numeric columns
TOP_K_VALUES = 1000  # value-count cap for entropy / top-share estimates

# structural pairs that never go through the matcher or permutations: the
# event timestamp's counterpart is unambiguous by construction (dtype DATE
# vs TIMESTAMP), the same way client ids are handled by the loaders
FIXED_PAIRS = {EVENT_TIME_COL: MBD_TIME_FIELD}

# это соответствие, в котором мы уверены до и независимо от матчинга.
ANCHORS = [
    (EVENT_TIME_COL, MBD_TIME_FIELD, "dtype DATE, range 2022-2024 (schema.py)", "hard"),
    ("col_2", "event_type", "report.html: MCC-like, 52 values (schema.py)", "hard"),
    ("col_11", "amount", "one of the only two continuous columns (schema.py)", "soft"),
    ("col_12", "amount", "the other continuous column; which-is-which unknown", "soft"),
]

_PASSPORT_CSV_COLS = [
    "column", "kind", "n_rows", "null_share", "n_unique", "uniqueness",
    "entropy_norm", "top1_share", "top3_share",
    "min", "max", "mean", "std", "zero_share", "q05", "q25", "q50", "q75", "q95",
]



# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def _maybe_sample(sql: str, sample_rows: int) -> str:
    # seeded reservoir: profile_columns scans from_sql several times per
    # column, and an unseeded sample would hand every scan a DIFFERENT row
    # subset -- n_unique from one sample, entropy from another -- making the
    # passport internally inconsistent and non-reproducible
    if sample_rows:
        return f"SELECT * FROM ({sql}) USING SAMPLE {int(sample_rows)} ROWS (reservoir, 42)"
    return sql


def materialize(con: duckdb.DuckDBPyConnection, from_sql: str, tag: str) -> str:
    """Run from_sql ONCE into a temp table, return a cheap SELECT over it.
    profile_columns issues 2-3 queries per column; without materialization
    every one of them re-executes the source -- a full re-scan for plain
    tables (~32 per side), a full-fold GROUP BY for the aggregated testbed
    sources (~25 per side, i.e. ~50 daily aggregations per testbed run)."""
    name = f"_prof_src_{tag}"
    con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {from_sql}")
    return f"SELECT * FROM {name}"


def xbank_source(path: str, sample_rows: int = 0) -> str:
    return _maybe_sample(f"SELECT * FROM read_parquet('{path}')", sample_rows)


def mbd_source(mbd_root: str, folds, sample_rows: int = 0) -> str:
    fold_list = ", ".join(str(f) for f in folds)
    casts = [
        f"CAST({f} AS BIGINT) AS {f}" if f in MBD_DOUBLE_FIELDS else f
        for f in (*MBD_NUMERIC_FIELDS, *MBD_CATEGORICAL_FIELDS)
    ]
    sql = (
        f"SELECT {', '.join(casts)} FROM read_parquet('{mbd_root}/detail/trx/fold=*/*.parquet', "
        f"hive_partitioning=true) WHERE fold IN ({fold_list})"
    )
    return _maybe_sample(sql, sample_rows)


def _mbd_kind(field: str) -> str:
    return "numeric" if field in MBD_NUMERIC_FIELDS else "categorical"


def col_specs_xbank():
    """(out_name, sql_expr, kind) for every xbank FEATURE column -- the
    timestamp stays out of the matcher (FIXED_PAIRS), targets are a
    different file and never appear here."""
    return (
        [(c, c, "categorical") for c in CATEGORY_COLS]
        + [(c, c, "numeric") for c in NUMERIC_COLS]
    )


def col_specs_mbd():
    return [(f, f, _mbd_kind(f)) for f in (*MBD_NUMERIC_FIELDS, *MBD_CATEGORICAL_FIELDS)]



# --------------------------------------------------------------------------
# passports
# --------------------------------------------------------------------------
# сompute entropy = h = -Σ p_i · log2(p_i) + normalize it
def _entropy_norm(counts) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) < 2:
        return 0.0
    h = -sum((c / total) * math.log2(c / total) for c in counts if c > 0)
    return h / math.log2(len(counts))


def profile_columns(con: duckdb.DuckDBPyConnection, from_sql: str, col_specs):
    """One statistical passport per column; deterministic duckdb aggregates
    only. Numeric passports additionally carry an in-memory `_hist` (N_BINS
    equal-width shares of the min-max-scaled column) that fuels the
    histogram-shape similarity -- it is deliberately NOT written to CSV."""
    passports = {}
    for name, expr, kind in col_specs:

        n_rows, n_not_null, n_unique = con.execute(
            f"SELECT COUNT(*), COUNT({expr}), COUNT(DISTINCT {expr}) FROM ({from_sql})"
        ).fetchone()

        p = {
            "column": name,
            "kind": kind,
            "n_rows": int(n_rows or 0),
            "null_share": round(1.0 - n_not_null / n_rows, 4) if n_rows else 1.0, #<-- percentage of empty ones
            "n_unique": int(n_unique or 0),
            "uniqueness": round(n_unique / n_rows, 6) if n_rows else 0.0, #<-- percentage of unique ones
        }

        if n_not_null == 0:
            passports[name] = p
            continue

        if kind == "categorical":
            counts = [r[0] for r in con.execute(
                f"SELECT cnt FROM (SELECT {expr} AS v, COUNT(*) AS cnt FROM ({from_sql}) "
                f"GROUP BY v ORDER BY cnt DESC LIMIT {TOP_K_VALUES})"
            ).fetchall()]
            total = sum(counts)
            p["top1_share"] = round(counts[0] / total, 4) if total else 0.0
            p["top3_share"] = round(sum(counts[:3]) / total, 4) if total else 0.0
            p["entropy_norm"] = round(_entropy_norm(counts), 4)

        else:
            (lo, hi, mean, std, zero_share,
             q05, q25, q50, q75, q95) = con.execute(
                f"SELECT MIN({expr}), MAX({expr}), AVG({expr}), STDDEV({expr}), "
                f"AVG(CASE WHEN {expr} = 0 THEN 1.0 ELSE 0.0 END), "
                f"quantile_cont({expr}, 0.05), quantile_cont({expr}, 0.25), "
                f"quantile_cont({expr}, 0.50), quantile_cont({expr}, 0.75), "
                f"quantile_cont({expr}, 0.95) FROM ({from_sql})"
            ).fetchone()
            span = (hi - lo) or 1.0
            step = span / N_BINS
            bin_exprs = []
            for i in range(N_BINS):
                if i < N_BINS - 1:
                    cond = (f"{expr} >= {lo + i * step!r} AND {expr} < {lo + (i + 1) * step!r}")
                else:  # last bin closed on both ends
                    cond = f"{expr} >= {lo + i * step!r}"
                bin_exprs.append(f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END)")
            bins = con.execute(f"SELECT {', '.join(bin_exprs)} FROM ({from_sql})").fetchone()
            total_bins = sum(bins)
            hist = [round(b / total_bins, 6) for b in bins] if total_bins else [1.0 / N_BINS] * N_BINS
            p.update(
                min=lo, max=hi,
                mean=round(mean, 6) if mean is not None else None,
                std=round(std, 6) if std is not None else None,
                zero_share=round(zero_share, 4),
                q05=q05, q25=q25, q50=q50, q75=q75, q95=q95,
                entropy_norm=round(_entropy_norm(hist), 4),
                _hist=hist,
            )

        passports[name] = p
    return passports


def passports_to_frame(passports: dict) -> pd.DataFrame:
    rows = [{k: v for k, v in p.items() if not k.startswith("_")} for p in passports.values()]
    df = pd.DataFrame(rows)
    return df[[c for c in _PASSPORT_CSV_COLS if c in df.columns]]



# --------------------------------------------------------------------------
# similarity + matching
# --------------------------------------------------------------------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def similarity(a: dict, b: dict) -> float:
    """Passport similarity in [0, 1]; 0 across kinds. Deliberately
    scale-FREE features only (cardinality magnitude, entropy, zero share,
    histogram shape): xbank's amounts are pre-normalized to [0, 1] while
    MBD's are raw rubles, so absolute ranges/min-max carry no signal. The
    weights below are visible constants, not tuned anything."""

    if a["kind"] != b["kind"]:
        return 0.0
    
    delta_un = abs(math.log10(max(a["n_unique"], 1)) - math.log10(max(b["n_unique"], 1)))

    # cardinality gate: a field's value-count class is structural, like its
    # type -- a 6-code column is not the same field as a 53-code one however
    # similar their entropy/top-1 shapes. Identity testbeds cannot catch a
    # missing cardinality signal (their cardinalities match by construction,
    # so the penalty is always ~0 there); the col_2 anchor did: shape-only
    # scores let low-card columns steal high-card fields.
    if delta_un > 0.75:  # >~5.6x value-count mismatch: incompatible fields
        return 0.0
    
    delta_entr = abs(a.get("entropy_norm", 0.0) - b.get("entropy_norm", 0.0))

    if a["kind"] == "categorical":
        delta_top = abs(a.get("top1_share", 0.0) - b.get("top1_share", 0.0))
        return _clip01(1.0 - (0.4 * delta_un / 3.0 + 0.4 * delta_entr + 0.2 * delta_top))
    
    delta_zero = abs(a.get("zero_share", 0.0) - b.get("zero_share", 0.0))
    hist_l1 = sum(abs(x - y) for x, y in zip(a.get("_hist", []), b.get("_hist", []))) / 2.0
    return _clip01(1.0 - (0.25 * delta_un / 3.0 + 0.25 * delta_entr + 0.2 * delta_zero + 0.3 * hist_l1))


def assign(sim: dict, row_names: list, col_names: list, min_sim: float = 0.30):
    """Rectangular one-to-one assignment maximizing total similarity.

    Prefers the optimal Hungarian solution (scipy); falls back to greedy
    descending-similarity if scipy is unavailable. Pairs scoring below
    min_sim are dropped rather than forced -- a column with no plausible
    counterpart must end up in `dropped`, not in a fake pair.
    
    why do we use Hungarian solution? Example:
                event_subtype  currency
    col_3          0.80         0.79
    col_4          0.78         0.10
    Greedy algorithm assigns col_3 to event_subtype (0.80), leaving col_4 as currency (0.10). Total: 0.90.
    Optimal: col_3 --> currency (0.79) + col_4 --> event_subtype (0.78). Total: 1.57.
    """
    matrix = [[sim[r][c] for c in col_names] for r in row_names]
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        ri, ci = linear_sum_assignment(-np.asarray(matrix))
        chosen = list(zip(ri.tolist(), ci.tolist()))
        algorithm = "hungarian"
    except ImportError:
        cells = [(matrix[i][j], i, j)
                 for i in range(len(matrix)) for j in range(len(matrix[0]))]
        cells.sort(reverse=True)
        chosen, taken_rows, taken_cols = [], set(), set()
        for _, i, j in cells:
            if i in taken_rows or j in taken_cols:
                continue
            taken_rows.add(i)
            taken_cols.add(j)
            chosen.append((i, j))
        algorithm = "greedy"
    pairs = [(row_names[i], col_names[j], round(matrix[i][j], 4))
             for i, j in chosen if matrix[i][j] >= min_sim]
    used_rows = {r for r, _, _ in pairs}
    used_cols = {c for _, c, _ in pairs}
    dropped = [r for r in row_names if r not in used_rows]
    unfilled = [c for c in col_names if c not in used_cols]
    return pairs, dropped, unfilled, algorithm


def match_schemas(xbank_passports: dict, mbd_passports: dict, min_sim: float = 0.30):
    """Full xbank->MBD feature matching: one assignment per type block plus
    the structural FIXED_PAIRS. Returns (mapping, pairs, dropped_xbank,
    unfilled_mbd, algorithm, similarity_matrix)."""
    sim = {r: {c: similarity(xbank_passports[r], mbd_passports[c]) for c in mbd_passports}
           for r in xbank_passports}
    pairs, dropped, unfilled, algorithm = [], [], [], None
    for kind in ("numeric", "categorical"):
        rows = [n for n, p in xbank_passports.items() if p["kind"] == kind]
        cols = [n for n, p in mbd_passports.items() if p["kind"] == kind]
        if not rows or not cols:
            continue
        p_, d_, u_, a_ = assign(sim, rows, cols, min_sim)
        pairs += p_
        dropped += d_
        unfilled += u_
        algorithm = a_
    mapping = dict(FIXED_PAIRS)
    mapping.update({r: c for r, c, _ in pairs})
    return mapping, pairs, dropped, unfilled, algorithm, sim 



# --------------------------------------------------------------------------
# testbeds (truth known by construction)
# --------------------------------------------------------------------------

def run_testbed_folds(con, mbd_root, fold_a, fold_b, sample_rows=0, min_sim=0.30):
    """Sanity floor: same schema on both sides, client-disjoint folds. Side
    A's fields are renamed to anonymous col_a<k> slots, the matcher must
    recover the identity mapping. Expected near-perfect recovery -- it
    validates the machinery, not the difficulty of the real task."""
    fields = [f for f in (MBD_TIME_FIELD, *MBD_NUMERIC_FIELDS, *MBD_CATEGORICAL_FIELDS)
              if f != MBD_TIME_FIELD]  # timestamp is a structural pair, mirrors the real task
    truth, alias_select = {}, []

    for k, f in enumerate(fields, start=2):  # col_a1 reserved for the skipped time field
        slot = f"col_a{k}"
        expr = f"CAST({f} AS BIGINT) AS {slot}" if f in MBD_DOUBLE_FIELDS else f"{f} AS {slot}"
        alias_select.append(expr)
        truth[slot] = f
        
    from_a = materialize(con, _maybe_sample(
        f"SELECT {', '.join(alias_select)} FROM read_parquet('{mbd_root}/detail/trx/fold=*/*.parquet', "
        f"hive_partitioning=true) WHERE fold = {fold_a}", sample_rows), "folds_a")
    
    from_b = materialize(con, mbd_source(mbd_root, [fold_b], sample_rows), "folds_b")

    specs_a = [(slot, slot, _mbd_kind(f)) for slot, f in truth.items()]

    passports_a = profile_columns(con, from_a, specs_a)
    passports_b = profile_columns(con, from_b, col_specs_mbd())

    sim = {r: {c: similarity(passports_a[r], passports_b[c]) for c in passports_b}
           for r in passports_a}
    
    pairs, _, _, algorithm = assign(sim, list(passports_a), list(passports_b), min_sim)
    assigned = {r: c for r, c, _ in pairs}
    df = pd.DataFrame([
        {"anon_slot": s, "truth": t, "recovered": assigned.get(s),
         "correct": bool(assigned.get(s) == t)}
        for s, t in truth.items()
    ])
    summary = f"{int(df['correct'].sum())}/{len(df)} exact ({algorithm})"

    return df, summary


def run_testbed_daily_folds(con, mbd_root, fold_a, fold_b, sample_rows=0, min_sim=0.30):
    """Same-granularity sanity at the DAILY level: BOTH sides aggregated to
    xbank's (client, day, category-combination) grain -- side A of fold A
    under anonymous col_a<k> slots, side B of client-disjoint fold B under
    MBD's own field names. Truth is again the identity mapping. Separates
    two questions the daily<->raw testbed conflates: 'does daily grain
    itself break the passports?' (answered here) vs 'does the grain
    MISMATCH between the two sides break matching?' (daily variant).
    Note: --sample-rows samples GROUPS after aggregation, so the daily
    GROUP BY always runs over the full fold -- this testbed is the slowest
    of the three."""
    fields = [f for f in (MBD_TIME_FIELD, *MBD_NUMERIC_FIELDS, *MBD_CATEGORICAL_FIELDS)
              if f != MBD_TIME_FIELD]
    glob = f"'{mbd_root}/detail/trx/fold=*/*.parquet', hive_partitioning=true"

    truth, a_select, b_select = {}, [], []
    group_cols = ["client_id", "DATE_TRUNC('day', event_time)"] + fields

    for k, f in enumerate(fields, start=2):  # col_a1 reserved for the skipped time field
        slot = f"col_a{k}"
        cast = f"CAST({f} AS BIGINT)" if f in MBD_DOUBLE_FIELDS else f
        a_select.append(f"SUM({f}) AS {slot}" if f in MBD_NUMERIC_FIELDS else f"{cast} AS {slot}")
        b_select.append(f"SUM({f}) AS {f}" if f in MBD_NUMERIC_FIELDS else f"{cast} AS {f}")
        truth[slot] = f

    group_by = ", ".join(group_cols)
    from_a = materialize(con, _maybe_sample(
        f"SELECT {', '.join(a_select)} FROM read_parquet({glob}) "
        f"WHERE fold = {fold_a} GROUP BY {group_by}", sample_rows), "daily_folds_a")
    from_b = materialize(con, _maybe_sample(
        f"SELECT {', '.join(b_select)} FROM read_parquet({glob}) "
        f"WHERE fold = {fold_b} GROUP BY {group_by}", sample_rows), "daily_folds_b")

    specs_a = [(slot, slot, _mbd_kind(f)) for slot, f in truth.items()]
    passports_a = profile_columns(con, from_a, specs_a)
    passports_b = profile_columns(con, from_b, col_specs_mbd())

    sim = {r: {c: similarity(passports_a[r], passports_b[c]) for c in passports_b}
           for r in passports_a}
    pairs, _, _, algorithm = assign(sim, list(passports_a), list(passports_b), min_sim)
    assigned = {r: c for r, c, _ in pairs}
    df = pd.DataFrame([
        {"anon_slot": s, "truth": t, "recovered": assigned.get(s),
         "correct": bool(assigned.get(s) == t)}
        for s, t in truth.items()
    ])
    summary = f"{int(df['correct'].sum())}/{len(df)} exact ({algorithm})"
    return df, summary


def run_testbed_daily(con, mbd_root, adapted_path, folds, sample_rows=0, min_sim=0.30, mbd_passports=None):
    """Harder variant: side A is the daily-aggregated xbank-SHAPED MBD table
    (data.mbd_adapter.py's output -- run `python -m data.mbd_adapter --freq
    daily` first), side B is raw hourly MBD. Truth is the adapter's own
    column map, so recovery here measures robustness to the aggregation
    transform, not just machinery. Returns (None, reason) if the adapted
    table has not been materialized yet."""
    if not Path(adapted_path).exists():
        return None, f"skipped: {adapted_path} not found (run `python -m data.mbd_adapter --freq daily`)"
    from data.mbd_adapter import TRX_CATEGORY_MAP, PLACEHOLDER_FEATURE_COLS

    truth = {slot: field for field, slot in TRX_CATEGORY_MAP.items()}
    truth["col_11"] = "amount"

    from_a = materialize(con, _maybe_sample(f"SELECT * FROM read_parquet('{adapted_path}')", sample_rows), "daily_a")

    passports_a = profile_columns(con, from_a, col_specs_xbank())
    # reuse the MBD passports main() already computed over the same folds --
    # re-profiling here would only re-scan the raw table a second time
    passports_b = mbd_passports or profile_columns(
        con, materialize(con, mbd_source(mbd_root, folds, sample_rows), "daily_b"), col_specs_mbd())

    sim = {r: {c: similarity(passports_a[r], passports_b[c]) for c in passports_b}
           for r in passports_a}
    
    pairs, _, _, algorithm = assign(sim, list(passports_a), list(passports_b), min_sim)

    assigned = {r: c for r, c, _ in pairs}

    rows = []

    for slot in (*NUMERIC_COLS, *CATEGORY_COLS):
        if slot in PLACEHOLDER_FEATURE_COLS:
            rows.append({"anon_slot": slot, "truth": "(placeholder)", "recovered": assigned.get(slot),
                         "correct": False})
        else:
            rows.append({"anon_slot": slot, "truth": truth.get(slot), "recovered": assigned.get(slot),
                         "correct": bool(truth.get(slot) is not None and assigned.get(slot) == truth[slot])})

    df = pd.DataFrame(rows)
    scored = df[df["truth"] != "(placeholder)"]
    summary = f"{int(scored['correct'].sum())}/{len(scored)} exact over real fields ({algorithm})"
    return df, summary



# --------------------------------------------------------------------------
# anchors + freezing
# --------------------------------------------------------------------------

def build_mapping_table(mapping, pairs, dropped, unfilled) -> pd.DataFrame:
    """The ONE complete correspondence table (the paper's main mapping
    table): a row per xbank FEATURE column in schema order -- mapped or
    explicitly dropped ('no counterpart'), never silently missing -- plus
    the fixed structural time pair and one row per unfilled MBD slot.
    status: fixed / mapped / dropped / unfilled."""
    pair_sim = {r: s for r, _, s in pairs}
    rows = [{"xbank_col": EVENT_TIME_COL, "kind": "time",
             "mbd_field": MBD_TIME_FIELD, "similarity": "", "status": "fixed"}]
    for col in (*CATEGORY_COLS, *NUMERIC_COLS):
        kind = "numeric" if col in NUMERIC_COLS else "category"
        if col in mapping:
            rows.append({"xbank_col": col, "kind": kind, "mbd_field": mapping[col],
                         "similarity": pair_sim.get(col, ""), "status": "mapped"})
        elif col in dropped:
            rows.append({"xbank_col": col, "kind": kind, "mbd_field": "",
                         "similarity": "", "status": "dropped"})
        else:  # neither mapped nor dropped -- impossible by construction; surface it
            rows.append({"xbank_col": col, "kind": kind, "mbd_field": "",
                         "similarity": "", "status": "UNRESOLVED"})
    for f in unfilled:
        rows.append({"xbank_col": "", "kind": "", "mbd_field": f,
                     "similarity": "", "status": "unfilled"})
    return pd.DataFrame(rows)


def check_anchors(mapping: dict, sim: dict = None, eps: float = 0.05) -> pd.DataFrame:
    """Three verdicts for hard anchors (soft anchors stay INFO):
    OK -- the matcher independently agreed with the trusted pair;
    AMBIGUOUS -- a near-tie: the assigned counterpart scores within eps of
        the expected one, i.e. the passports carry NO signal separating the
        lookalikes; freezing stays allowed, but the pair is flagged for the
        permutation sensitivity analysis (a within-class swap);
    FAIL -- confident disagreement: the matcher clearly preferred something
        else (or dropped a column whose counterpart exists) -- red light,
        freezing must not proceed."""
    rows = []
    for col, expected, why, kind in ANCHORS:
        assigned = mapping.get(col)
        verdict = "INFO"
        if kind == "hard":
            if assigned == expected:
                verdict = "OK"
            elif (sim is not None
                  and assigned in sim.get(col, {})
                  and expected in sim.get(col, {})
                  and abs(sim[col][expected] - sim[col][assigned]) <= eps):
                verdict = "AMBIGUOUS"
            else:
                verdict = "FAIL"
        rows.append({"xbank_col": col, "expected": expected, "why_known": why,
                     "auto_assigned": assigned, "verdict": verdict})
    return pd.DataFrame(rows)


def build_frozen_mapping(mapping, dropped, unfilled, algorithm, args, testbed_summary) -> dict:
    return {
        "mapping": mapping,
        "dropped_xbank": dropped,
        "unfilled_mbd_slots": unfilled,
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profiled_on": {
                "xbank": {"path": args.xbank_path, "sample_rows": args.sample_rows or "full"},
                "mbd": {"root": args.mbd_root, "folds": list(args.mbd_folds),
                        "sample_rows": args.sample_rows or "full"},
            },
            "testbed_recovery": testbed_summary,
            "matcher": {
                "algorithm": algorithm,
                "min_sim_cutoff": args.min_sim,
                "rule": ("rectangular assignment per column type (numeric<->numeric, "
                         "categorical<->categorical); pairs below the cutoff are dropped, "
                         "not forced; event time is a fixed structural pair"),
            },
            "targets": "never profiled, matched or permuted -- feature columns only",
        },
    }



# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Column passports + dictionary-free schema matching")
    parser.add_argument("--xbank-path", default=XBANK_TRX_DEFAULT)                              #<-- файл xbank
    parser.add_argument("--mbd-root", default=MBD_ROOT_DEFAULT)                                 #<-- data/mbd
    parser.add_argument("--mbd-folds", type=int, nargs="+", default=[0],
                        help="labeled folds to profile the named MBD side on (default: fold 0)")#<--на каких фолдах  
    parser.add_argument("--sample-rows", type=int, default=0,
                        help="optional row sample for a quick look (0 = full tables)")          #<--сэмпл для быстрого взгляда; сид зашит, так что сэмпл воспроизводим
    parser.add_argument("--min-sim", type=float, default=0.30,
                        help="pairs below this similarity are dropped, not forced")             #<--cutoff матчера: пары ниже - отбрасываются, а не форсируются
    parser.add_argument("--testbed-variant",
                        choices=["folds", "daily", "daily_folds", "both", "all"], default="both",
                        help="folds: raw<->raw sanity; daily: daily-adapted<->raw; "
                             "daily_folds: daily<->daily sanity; both: folds+daily; all: all three")
    parser.add_argument("--testbed-only", action="store_true",
                        help="run ONLY the testbeds: no xbank profiling, no xbank->MBD "
                             "matching, no frozen mapping -- for fast machinery checks")        #<--какие тестбеды
    parser.add_argument("--no-testbed", action="store_true",
                        help="skip ALL testbeds: only profiles + xbank->MBD match + anchors "
                             "+ frozen mapping (testbeds already validated in earlier runs)")  #<--без тестбедов, только матчинг
    parser.add_argument("--testbed-fold-a", type=int, default=0)                                #<--фолды для тестбеда «folds» = A анонимизированная
    parser.add_argument("--testbed-fold-b", type=int, default=1)                                #<--фолды для тестбеда «folds» = B именованная
    parser.add_argument("--daily-path", default=MBD_ADAPTED_DAILY_DEFAULT)                      #<--где лежит дневная агрегированная таблица
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT)                                   #<--куда писать артефакты
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    xbank_passports = mbd_passports = None
    if args.testbed_only:
        print("testbed-only: skipping xbank/MBD profiling and the main match", flush=True)
    else:
        print("Profiling xbank feature columns ...", flush=True)
        xbank_passports = profile_columns(
            con, materialize(con, xbank_source(args.xbank_path, args.sample_rows), "xbank"),
            col_specs_xbank())
        passports_to_frame(xbank_passports).to_csv(out / "xbank_profiles.csv", index=False)

        print(f"Profiling MBD feature columns (folds={args.mbd_folds}) ...", flush=True)
        mbd_passports = profile_columns(
            con, materialize(con, mbd_source(args.mbd_root, args.mbd_folds, args.sample_rows), "mbd"),
            col_specs_mbd())
        passports_to_frame(mbd_passports).to_csv(out / "mbd_profiles.csv", index=False)

    testbed_frames = []
    testbed_summary = {"note": "skipped by --no-testbed"} if args.no_testbed else {}
    if args.no_testbed:
        print("--no-testbed: skipping testbeds; running match/anchors/freeze only", flush=True)
    if not args.no_testbed and args.testbed_variant in ("folds", "both", "all"):
        print(f"Testbed 'folds': fold {args.testbed_fold_a} anonymized vs fold {args.testbed_fold_b} named ...", flush=True)
        df, summary = run_testbed_folds(con, args.mbd_root, args.testbed_fold_a,
                                        args.testbed_fold_b, args.sample_rows, args.min_sim)
        df["variant"] = "folds"
        testbed_frames.append(df)
        testbed_summary["folds"] = summary
        print(f"  recovery: {summary}", flush=True)
    if not args.no_testbed and args.testbed_variant in ("daily_folds", "all"):
        print(f"Testbed 'daily_folds': daily fold {args.testbed_fold_a} anonymized vs daily fold {args.testbed_fold_b} named ...", flush=True)
        df, summary = run_testbed_daily_folds(con, args.mbd_root, args.testbed_fold_a,
                                              args.testbed_fold_b, args.sample_rows, args.min_sim)
        df["variant"] = "daily_folds"
        testbed_frames.append(df)
        testbed_summary["daily_folds"] = summary
        print(f"  recovery: {summary}", flush=True)
    if not args.no_testbed and args.testbed_variant in ("daily", "both", "all"):
        print("Testbed 'daily': adapted daily table vs raw hourly MBD ...", flush=True)
        df, summary = run_testbed_daily(con, args.mbd_root, args.daily_path, args.mbd_folds,
                                        args.sample_rows, args.min_sim, mbd_passports=mbd_passports)
        if df is None:
            print(f"  {summary}", flush=True)
            testbed_summary["daily"] = summary
        else:
            df["variant"] = "daily"
            testbed_frames.append(df)
            testbed_summary["daily"] = summary
            print(f"  recovery: {summary}", flush=True)
    if testbed_frames:
        pd.concat(testbed_frames, ignore_index=True).to_csv(out / "testbed_recovery.csv", index=False)

    if args.testbed_only:
        print(f"\ntestbed-only run finished; wrote testbed_recovery.csv to {out}")
        return

    print("Matching xbank -> MBD ...", flush=True)
    mapping, pairs, dropped, unfilled, algorithm, sim = match_schemas(
        xbank_passports, mbd_passports, args.min_sim)
    sim_df = pd.DataFrame(sim).T.round(3)
    sim_df.index.name = "column"
    sim_df.to_csv(out / "similarity.csv")

    mapping_table = build_mapping_table(mapping, pairs, dropped, unfilled)
    mapping_table.to_csv(out / "mapping.csv", index=False)

    anchors = check_anchors(mapping, sim)
    anchors.to_csv(out / "anchors.csv", index=False)

    frozen = build_frozen_mapping(mapping, dropped, unfilled, algorithm, args, testbed_summary)
    with open(out / "frozen_mapping.json", "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2, ensure_ascii=False)

    print("\n=== full correspondence table (xbank -> MBD) ===")
    print(mapping_table.to_string(index=False))
    print("\n=== anchors ===")
    print(anchors.to_string(index=False))
    print(f"\nwrote 7 artifacts to {out}; frozen_mapping.json is what inference reads")


if __name__ == "__main__":
    main()
