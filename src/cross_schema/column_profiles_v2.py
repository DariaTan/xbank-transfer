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
  3. match      partial one-to-one assignment per type block; pairs below
                --min-sim are excluded before optimization, leftovers are
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
  testbed_recovery.csv  testbed_metrics.csv  anchors.csv  frozen_mapping.json
  pair_diagnostics.csv  assignment_diagnostics.csv  correlation_diagnostics.json

See MATCHING.md for formulas, limitations, and the quality evaluation protocol.

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
from itertools import combinations
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
OUT_DIR_DEFAULT = '/home/stsix/xbank-transfer/data/profiles_v2'

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

QUANTILE_LEVELS = (0.01, *[i / 20 for i in range(1, 20)], 0.99)
MATCHER_VERSION = "passport-v2"
# Heuristic starting weights, NOT probabilities or fitted/calibrated parameters.
PASSPORT_WEIGHTS = {
    "categorical": {"cardinality": 0.35, "entropy": 0.30, "top1": 0.20, "null": 0.15},
    "numeric": {"quantile": 0.70, "zero": 0.15, "null": 0.15},
}
# Public alias retained for the v2 matcher and existing consumers.
SIMILARITY_WEIGHTS = PASSPORT_WEIGHTS

# structural pairs that never go through the matcher or permutations: the
# event timestamp's counterpart is unambiguous by construction (dtype DATE
# vs TIMESTAMP), the same way client ids are handled by the loaders
FIXED_PAIRS = {EVENT_TIME_COL: MBD_TIME_FIELD}

# Structural anchors are hard; inferred semantic hypotheses stay soft.
ANCHORS = [
    (EVENT_TIME_COL, MBD_TIME_FIELD, "dtype DATE, range 2022-2024 (schema.py)", "hard"),
    ("col_2", "event_type", "hypothesis only: MCC-like does not establish event_type (schema.py)", "soft"),
    ("col_11", "amount", "one of the only two continuous columns (schema.py)", "soft"),
    ("col_12", "amount", "the other continuous column; which-is-which unknown", "soft"),
]

_PASSPORT_CSV_COLS = [
    "column", "kind", "n_rows", "n_valid", "null_share", "n_unique", "uniqueness",
    "entropy_norm", "top1_share", "top3_share",
    "min", "max", "mean", "std", "zero_share", "q05", "q25", "q50", "q75", "q95",
    "numeric_subtype", "integer_share", "quantile_scale", "quantile_fallback",
    "quantile_shape",
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
def _numeric_expr(expr: str) -> str:
    # DuckDB treats NaN as a value, not SQL NULL. Exclude all non-finite values.
    return f"CASE WHEN isfinite(CAST({expr} AS DOUBLE)) THEN CAST({expr} AS DOUBLE) END"


def _quantile_shape(quantiles, lo, hi):
    """Location/positive-scale invariant; central 90% span resists outliers.

    Degenerate central spans fall back to 1--99%, then the full range.
    The fallback is exported so zero-inflated/degenerate columns are visible.
    """
    q = dict(zip(QUANTILE_LEVELS, quantiles))
    scale, fallback = q[0.95] - q[0.05], "none"
    if scale == 0:
        scale, fallback = q[0.99] - q[0.01], "q01_q99"
    if scale == 0:
        scale, fallback = hi - lo, "range"
    if scale == 0:
        return [0.0] * len(quantiles), 0.0, "constant"
    return [(v - q[0.5]) / scale for v in quantiles], scale, fallback


def profile_columns(con: duckdb.DuckDBPyConnection, from_sql: str, col_specs):
    """Feature-only passports; shape statistics are conditional on valid values.

    Missingness is a separate signal. Categorical entropy uses ALL non-null
    counts (no top-k renormalization). Numeric quantiles and shape are exported
    to CSV so the scoring inputs can be inspected.
    """
    passports = {}
    for name, expr, kind in col_specs:
        if kind not in SIMILARITY_WEIGHTS:
            raise ValueError(f"Unsupported column kind: {kind}")
        value = _numeric_expr(expr) if kind == "numeric" else (
            f"CASE WHEN typeof({expr}) IN ('FLOAT', 'DOUBLE') "
            f"AND NOT isfinite(TRY_CAST({expr} AS DOUBLE)) THEN NULL ELSE {expr} END")
        source = f"SELECT {value} AS v FROM ({from_sql})"
        n_rows, n_valid, n_unique = con.execute(
            f"SELECT COUNT(*), COUNT(v), COUNT(DISTINCT v) FROM ({source})"
        ).fetchone()
        p = {
            "column": name, "kind": kind, "n_rows": n_rows, "n_valid": n_valid,
            "null_share": 1.0 - n_valid / n_rows if n_rows else 1.0,
            "n_unique": n_unique,
            "uniqueness": n_unique / n_rows if n_rows else 0.0,
        }
        if not n_valid:
            passports[name] = p
            continue
        valid = f"SELECT v FROM ({source}) WHERE v IS NOT NULL"
        if kind == "categorical":
            # Aggregate counts in DuckDB rather than materializing the dictionary
            # in Python. Missing values are excluded from this distribution.
            entropy, top1, top3 = con.execute(
                f"WITH counts AS (SELECT v, COUNT(*) AS cnt FROM ({valid}) GROUP BY v), "
                "ranked AS (SELECT cnt, ROW_NUMBER() OVER (ORDER BY cnt DESC) AS r FROM counts) "
                f"SELECT -SUM((cnt / {n_valid}) * log2(cnt / {n_valid})), "
                f"MAX(cnt) / {n_valid}, "
                f"SUM(CASE WHEN r <= 3 THEN cnt ELSE 0 END) / {n_valid} FROM ranked"
            ).fetchone()
            p.update(entropy_norm=entropy / math.log2(n_unique) if n_unique > 1 else 0.0,
                     top1_share=top1, top3_share=top3)
        else:
            lo, hi, mean, std, zero, integer, quantiles = con.execute(
                "SELECT MIN(v), MAX(v), AVG(v), STDDEV(v), "
                "AVG(CASE WHEN v = 0 THEN 1.0 ELSE 0.0 END), "
                "AVG(CASE WHEN v = trunc(v) THEN 1.0 ELSE 0.0 END), "
                f"quantile_cont(v, {list(QUANTILE_LEVELS)}) FROM ({valid})"
            ).fetchone()
            shape, scale, fallback = _quantile_shape(quantiles, lo, hi)
            subtype = "constant" if n_unique == 1 else "binary" if n_unique == 2 else "continuous"
            q = dict(zip(QUANTILE_LEVELS, quantiles))
            p.update(min=lo, max=hi, mean=mean, std=std, zero_share=zero,
                     q05=q[0.05], q25=q[0.25], q50=q[0.5], q75=q[0.75], q95=q[0.95],
                     numeric_subtype=subtype, integer_share=integer,
                     quantile_scale=scale, quantile_fallback=fallback,
                     quantile_shape=shape)
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


def _cardinality_distance(a: dict, b: dict):
    """Sample-size slack only when more observations also yield more categories.

    This is a heuristic, not an estimator of the unseen category vocabulary.
    Dividing cardinality by row count would penalize a fixed vocabulary when
    its rows are merely replicated. Numeric cardinality is not used at all.
    """
    du = math.log10(max(a["n_unique"], 1)) - math.log10(max(b["n_unique"], 1))
    dn = math.log10(max(a["n_valid"], 1)) - math.log10(max(b["n_valid"], 1))
    unexplained = max(0.0, abs(du) - (abs(dn) if du * dn > 0 else 0.0))
    return unexplained / 0.75


def distance_components(a: dict, b: dict) -> dict:
    """Weighted passport differences D in [0, 1], with explicit hard gates.

    Rejected pairs have distance 1 as a finite placeholder; callers must also
    check the reason. FGW consumes this distance directly.
    """
    def rejected(reason):
        return {"distance": 1.0, "reason": reason, "distances": {}}

    if a["kind"] != b["kind"]:
        return rejected("different_kinds")
    if not a.get("n_valid", 0) or not b.get("n_valid", 0):
        return rejected("no_valid_values")
    distances = {"null": abs(a["null_share"] - b["null_share"])}
    if a["kind"] == "categorical":
        cardinality = _cardinality_distance(a, b)
        if cardinality > 1.0:
            return rejected("cardinality_gate")
        distances.update(cardinality=cardinality,
                         entropy=abs(a["entropy_norm"] - b["entropy_norm"]),
                         top1=abs(a["top1_share"] - b["top1_share"]))
    else:
        # Integer-valued rubles can become floats under normalization: physical
        # dtype/integer_share is diagnostic only, not a compatibility gate.
        if a["numeric_subtype"] != b["numeric_subtype"]:
            return rejected("different_numeric_subtypes")
        qa, qb = a["quantile_shape"], b["quantile_shape"]
        if len(qa) != len(QUANTILE_LEVELS) or len(qb) != len(QUANTILE_LEVELS):
            raise ValueError("Rebuild passports with the current quantile grid")
        # Mean L1 difference of normalized quantiles, bounded monotonically.
        d = sum(abs(x - y) for x, y in zip(qa, qb)) / len(qa)
        distances.update(quantile=d / (1.0 + d),
                         zero=abs(a["zero_share"] - b["zero_share"]))
    weights = PASSPORT_WEIGHTS[a["kind"]]
    distance = _clip01(sum(weights[k] * distances[k] for k in weights))
    return {"distance": distance, "reason": "compatible", "distances": distances}


def similarity_components(a: dict, b: dict) -> dict:
    """Compatibility interface for the v2 similarity-based matcher."""
    result = distance_components(a, b)
    return {"score": 1.0 - result["distance"], "reason": result["reason"],
            "distances": result["distances"]}


def similarity(a: dict, b: dict) -> float:
    return similarity_components(a, b)["score"]


def assign(sim: dict, row_names: list, col_names: list, min_sim: float = 0.30):
    """Maximum total similarity over admissible edges, with optional abstention.

    Below-cutoff and zero/incompatible edges are removed BEFORE optimization.
    Each row has a zero-utility dummy option. This preserves the sum-of-scores
    objective without forcing an invalid pair. Greedy is an explicit fallback
    when SciPy is absent, and is not guaranteed optimal.
    """
    if not math.isfinite(min_sim) or not 0.0 <= min_sim <= 1.0:
        raise ValueError("min_sim must be between 0 and 1")
    if not row_names or not col_names:
        return [], list(row_names), list(col_names), "empty"
    matrix = [[sim[r][c] for c in col_names] for r in row_names]
    def admissible(score):
        return math.isfinite(score) and score > 0 and score >= min_sim

    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        utility = np.full((len(row_names), len(col_names) + len(row_names)), -1.0)
        utility[:, len(col_names):] = 0.0
        for i, row in enumerate(matrix):
            for j, score in enumerate(row):
                if admissible(score):
                    utility[i, j] = score
        ri, ci = linear_sum_assignment(-utility)
        chosen = [(i, j) for i, j in zip(ri.tolist(), ci.tolist()) if j < len(col_names)]
        algorithm = "hungarian"
    except ImportError:
        cells = [(matrix[i][j], i, j)
                 for i in range(len(matrix)) for j in range(len(matrix[0]))
                 if admissible(matrix[i][j])]
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
             for i, j in chosen if admissible(matrix[i][j])]
    used_rows = {r for r, _, _ in pairs}
    used_cols = {c for _, c, _ in pairs}
    return (pairs, [r for r in row_names if r not in used_rows],
            [c for c in col_names if c not in used_cols], algorithm)


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
        p_, d_, u_, a_ = assign(sim, rows, cols, min_sim)
        pairs += p_
        dropped += d_
        unfilled += u_
        if a_ != "empty":
            algorithm = a_
    mapping = dict(FIXED_PAIRS)
    mapping.update({r: c for r, c, _ in pairs})
    return mapping, pairs, dropped, unfilled, algorithm, sim 



# --------------------------------------------------------------------------
# diagnostics (evidence, not correctness probabilities)
# --------------------------------------------------------------------------

def pair_diagnostics(passports_a, passports_b):
    rows = []
    for r, a in passports_a.items():
        for c, b in passports_b.items():
            result = similarity_components(a, b)
            rows.append({"xbank_col": r, "mbd_field": c, "similarity": result["score"],
                         "reason": result["reason"], **result["distances"]})
    return pd.DataFrame(rows)


def assignment_diagnostics(sim, pairs, min_sim=0.30, ambiguity_margin=0.05):
    """Loss in the global objective when each selected edge is forbidden.

    A zero gap means an equally good alternative mapping exists. Local row/
    column margins additionally expose competition. Gaps require the exact
    solver; the greedy fallback is explicitly marked unchecked.
    """
    rows, cols = list(sim), list(next(iter(sim.values()), {}))
    base = sum(sim[r][c] for r, c, _ in pairs)
    records = []
    for r, c, _ in pairs:
        score = sim[r][c]
        row_best = max((sim[r][k] for k in cols if k != c
                        and sim[r][k] > 0 and sim[r][k] >= min_sim), default=0.0)
        col_best = max((sim[k][c] for k in rows if k != r
                        and sim[k][c] > 0 and sim[k][c] >= min_sim), default=0.0)
        altered = {k: dict(v) for k, v in sim.items()}
        altered[r][c] = 0.0
        alt, _, _, algorithm = assign(altered, rows, cols, min_sim)
        gap = base - sum(sim[x][y] for x, y, _ in alt) if algorithm == "hungarian" else None
        records.append({"xbank_col": r, "mbd_field": c, "similarity": score,
                        "row_margin": score - row_best, "column_margin": score - col_best,
                        "assignment_gap": max(0.0, gap) if gap is not None else None,
                        "status": "UNCHECKED" if gap is None else
                                  "AMBIGUOUS" if gap <= ambiguity_margin else "SEPARATED"})
    return pd.DataFrame(records, columns=["xbank_col", "mbd_field", "similarity",
                        "row_margin", "column_margin", "assignment_gap", "status"])


def correlation_diagnostics(con, from_a, from_b, passports_a, passports_b, mapping,
                            sample_rows=100000, min_pairs=30, warning_delta=0.3):
    """Compare within-dataset Spearman relationships, never categorical codes.

    Independent samples are appropriate: no cross-bank row alignment is used.
    This is a drift warning, not proof of a bad match and not a score penalty.
    """
    matches = [(r, c) for r, c in mapping.items()
               if r in passports_a and c in passports_b
               and passports_a[r]["kind"] == passports_b[c]["kind"] == "numeric"]
    if len(matches) < 2:
        return {"status": "SKIPPED", "reason": "fewer than two matched numeric columns",
                "pairs": []}
    if sample_rows <= 0:
        raise ValueError("correlation sample_rows must be positive")

    def read(source, columns):
        quoted = ['"' + c.replace('"', '""') + '"' for c in columns]
        exprs = [f"{_numeric_expr(c)} AS v{i}" for i, c in enumerate(quoted)]
        return con.execute(_maybe_sample(f"SELECT {', '.join(exprs)} FROM ({source})",
                                         sample_rows)).df()

    a = read(from_a, [r for r, _ in matches])
    b = read(from_b, [c for _, c in matches])
    records = []
    for i, j in combinations(range(len(matches)), 2):
        ca, cb = a[[f"v{i}", f"v{j}"]].dropna(), b[[f"v{i}", f"v{j}"]].dropna()
        result = {"xbank_cols": [matches[i][0], matches[j][0]],
                  "mbd_fields": [matches[i][1], matches[j][1]],
                  "n_a": len(ca), "n_b": len(cb)}
        if min(len(ca), len(cb)) < min_pairs or (ca.nunique() < 2).any() or (cb.nunique() < 2).any():
            result.update(status="SKIPPED", reason="insufficient paired data or constant column")
        else:
            ra = float(ca.corr(method="spearman").iloc[0, 1])
            rb = float(cb.corr(method="spearman").iloc[0, 1])
            delta = abs(ra - rb)
            result.update(rho_a=ra, rho_b=rb, delta=delta,
                          status="WARNING" if delta > warning_delta else "CONSISTENT")
        records.append(result)
    statuses = {r["status"] for r in records}
    return {"status": "WARNING" if "WARNING" in statuses else
                      "SKIPPED" if statuses == {"SKIPPED"} else "PARTIAL" if "SKIPPED" in statuses else "CONSISTENT",
            "method": "spearman", "sample_rows": sample_rows, "min_pairs": min_pairs,
            "warning_delta": warning_delta, "pairs": records}


# --------------------------------------------------------------------------
# testbeds (truth known by construction)
# --------------------------------------------------------------------------

def testbed_metrics(frame: pd.DataFrame) -> dict:
    """Precision on accepted pairs AND recall/coverage; missing truth is excluded.

    Placeholder rows, where provided by a testbed, are known negatives.
    No accepted pairs => precision is undefined, not a perfect score.
    """
    known = frame[frame["truth"].notna()]
    positive = known["truth"] != "(placeholder)"
    accepted = known["recovered"].notna()
    correct = int((positive & accepted & (known["truth"] == known["recovered"])).sum())
    n_accepted, n_positive = int(accepted.sum()), int(positive.sum())
    n_negative = int((~positive).sum())
    return {"correct": correct, "accepted": n_accepted, "true_pairs": n_positive,
            "precision": correct / n_accepted if n_accepted else None,
            "recall": correct / n_positive if n_positive else None,
            "coverage": int((accepted & positive).sum()) / n_positive if n_positive else None,
            "false_matches_on_negatives": int((accepted & ~positive).sum()),
            "known_negatives": n_negative}


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
        the expected one: this score cannot reliably separate the lookalikes;
        the verdict is exported for subsequent sensitivity analysis;
    FAIL -- confident disagreement with a trusted pair. Export remains an
        UNVERIFIED candidate; a failed check must be resolved before using it
        as a validated mapping. Soft hypotheses are never ground truth."""
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
                "version": MATCHER_VERSION,
                "weights": SIMILARITY_WEIGHTS,
                "quantile_levels": list(QUANTILE_LEVELS),
                "quantile_normalization": "(Q - median) / (Q95 - Q05); fallback Q99-Q01, then range",
                "cardinality_gate": "categorical only: unexplained log10 cardinality gap <= 0.75",
                "algorithm": algorithm,
                "min_sim_cutoff": args.min_sim,
                "rule": ("rectangular assignment per column type (numeric<->numeric, "
                         "categorical<->categorical); incompatible and below-cutoff pairs excluded "
                         "before assignment with dummy unmatched options; event time fixed"),
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
        xbank_sql = materialize(con, xbank_source(args.xbank_path, args.sample_rows), "xbank")
        xbank_passports = profile_columns(con, xbank_sql, col_specs_xbank())
        passports_to_frame(xbank_passports).to_csv(out / "xbank_profiles.csv", index=False)

        print(f"Profiling MBD feature columns (folds={args.mbd_folds}) ...", flush=True)
        mbd_sql = materialize(con, mbd_source(args.mbd_root, args.mbd_folds, args.sample_rows), "mbd")
        mbd_passports = profile_columns(con, mbd_sql, col_specs_mbd())
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
        metrics = [{"variant": frame["variant"].iloc[0], **testbed_metrics(frame)}
                   for frame in testbed_frames]
        pd.DataFrame(metrics).to_csv(out / "testbed_metrics.csv", index=False)
        print("\n=== testbed metrics ===")
        print(pd.DataFrame(metrics).to_string(index=False))

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

    pair_diagnostics(xbank_passports, mbd_passports).to_csv(out / "pair_diagnostics.csv", index=False)
    confidence = assignment_diagnostics(sim, pairs, args.min_sim)
    confidence.to_csv(out / "assignment_diagnostics.csv", index=False)
    correlations = correlation_diagnostics(con, xbank_sql, mbd_sql,
                                           xbank_passports, mbd_passports, mapping)
    with open(out / "correlation_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(correlations, f, indent=2, ensure_ascii=False, allow_nan=False)

    frozen = build_frozen_mapping(mapping, dropped, unfilled, algorithm, args, testbed_summary)
    frozen["meta"]["validation"] = {
        "status": "UNVERIFIED",
        "note": "Scores and assignment gaps are not probabilities. Semantic ground truth is required.",
        "ambiguous_pairs": int((confidence["status"] == "AMBIGUOUS").sum()),
        "ambiguity_margin": 0.05,
        "anchor_verdicts": anchors.to_dict(orient="records"),
        "correlation_status": correlations["status"],
    }
    with open(out / "frozen_mapping.json", "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2, ensure_ascii=False)

    print("\n=== full correspondence table (xbank -> MBD) ===")
    print(mapping_table.to_string(index=False))
    print("\n=== anchors ===")
    print(anchors.to_string(index=False))
    print("\n=== assignment diagnostics ===")
    print(confidence.to_string(index=False))
    print(f"Correlation check: {correlations['status']}")
    print(f"\nwrote artifacts to {out}; frozen_mapping.json is an UNVERIFIED candidate for inference")


if __name__ == "__main__":
    main()
