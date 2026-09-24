"""Passport + semi-relaxed Fused Gromov-Wasserstein schema matching.

Direction: xbank DATA -> MBD INPUT FIELDS. Each MBD field independently
selects argmax_i T[i,j]; a source column may serve several destination fields.
The target marginal is fixed, the xbank marginal is free. Thus source reuse
is allowed in the optimization itself, not just in postprocessing.

For Cx, Cy (within-bank dissimilarities), D = weighted passport differences:
  min_T (1-alpha) <D,T> + alpha sum_ijkl (Cx[i,k]-Cy[j,l])^2 T[i,j] T[k,l]
  T >= 0, sum_i T[i,j] = q[j], T[forbidden] = 0.
Targets without admissible candidates are excluded and exported as unfilled.

A small masked Frank-Wolfe solver uses exact quadratic line search and seeded
restarts. No POT dependency is required. This is nonconvex: convergence is
stationarity, not a certificate of global or semantic correctness.

Self-contained module; requires duckdb, numpy, pandas, scikit-learn.
Run: PYTHONPATH=src python -m cross_schema.column_profiles_vFGW --help
Or: python column_profiles_vFGW.py --help
See FGW.md for formulas, outputs, evaluation, and limitations.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

VERSION = "semirelaxed-fgw-v2"


# Schema and passport settings are local: this module has no project imports.
PASSPORT_VERSION = "fgw-passport-v1"

EVENT_TIME_COL = "col_1"

CATEGORY_COLS = [
    "col_2", "col_3", "col_4", "col_5", "col_6",
    "col_7", "col_8", "col_9", "col_10",
    "col_13", "col_14", "col_15", "col_16",
]

NUMERIC_COLS = ["col_11", "col_12"]

XBANK_TRX_DEFAULT = '/home/stsix/xbank-transfer/data/xbank_data/trans_any_pos_anonym_encoded.parquet'

MBD_ROOT_DEFAULT = '/home/stsix/xbank-transfer/data/mbd'

MBD_ADAPTED_DAILY_DEFAULT = '/home/stsix/xbank-transfer/data/mbd_daily/transactions_adapted_daily.parquet'

OUT_DIR_DEFAULT = "/home/stsix/xbank-transfer/data/profiles_vFGW"

MBD_TIME_FIELD = "event_time"

MBD_NUMERIC_FIELDS = ("amount",)

MBD_CATEGORICAL_FIELDS = (
    "event_type", "event_subtype", "currency",
    "src_type11", "src_type12", "dst_type11", "dst_type12",
    "src_type21", "src_type22", "src_type31", "src_type32",
)

MBD_DOUBLE_FIELDS = frozenset(MBD_CATEGORICAL_FIELDS[3:])

QUANTILE_LEVELS = (0.01, *[i / 20 for i in range(1, 20)], 0.99)

PASSPORT_WEIGHTS = {
    "categorical": {"cardinality": 0.35, "entropy": 0.30, "top1": 0.20, "null": 0.15},
    "numeric": {"quantile": 0.70, "zero": 0.15, "null": 0.15},
}

FIXED_PAIRS = {EVENT_TIME_COL: MBD_TIME_FIELD}

_PASSPORT_CSV_COLS = [
    "column", "kind", "n_rows", "n_valid", "null_share", "n_unique", "uniqueness",
    "entropy_norm", "top1_share", "top3_share",
    "min", "max", "mean", "std", "zero_share", "q05", "q25", "q50", "q75", "q95",
    "numeric_subtype", "integer_share", "quantile_scale", "quantile_fallback",
    "quantile_shape",
]

# Ground truth for the existing daily-adapted MBD parquet schema only.
# This is NOT an assumed correspondence for real xbank data.
DAILY_MBD_TO_XBANK = {
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


def materialize(con: duckdb.DuckDBPyConnection, from_sql: str, tag: str) -> str:
    """Run from_sql ONCE into a temp table, return a cheap SELECT over it.
    profile_columns issues 2-3 queries per column; without materialization
    every one of them re-executes the source -- a full re-scan for plain
    tables (~32 per side), a full-fold GROUP BY for the aggregated testbed
    sources (~25 per side, i.e. ~50 daily aggregations per testbed run)."""
    name = f"_prof_src_{tag}"
    con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {from_sql}")
    return f"SELECT * FROM {name}"


def xbank_source(path: str) -> str:
    return f"SELECT * FROM read_parquet('{path}')"


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
        if kind not in PASSPORT_WEIGHTS:
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


def sample_sql(sql, n_rows, seed=42):
    if n_rows < 0:
        raise ValueError("sample_rows must be nonnegative")
    return (f"SELECT * FROM ({sql}) USING SAMPLE {int(n_rows)} ROWS (reservoir, {int(seed)})"
            if n_rows else sql)


def quote_column(name):
    return '"' + name.replace('"', '""') + '"'


def encode_structure_column(series, kind, bins=10, max_categories=64):
    """Discrete codes for NMI, -1 for missing; numeric uses own quantiles.

    Rare categories are pooled to bound NMI's vocabulary/sample-size bias.
    If the cap boundary is tied, the entire tied group is pooled: arbitrary
    category labels cannot decide which equally frequent levels are retained.
    """
    if bins < 2 or max_categories < 2:
        raise ValueError("bins and max_categories must be >= 2")
    if kind == "numeric":
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(values)
        codes = np.full(len(values), -1, dtype=int)
        if valid.any():
            edges = np.unique(np.quantile(values[valid], np.linspace(0, 1, bins + 1)))
            codes[valid] = np.searchsorted(edges[1:-1], values[valid], side="right")
        return codes
    if kind != "categorical":
        raise ValueError(f"Unknown kind: {kind}")
    clean = series.copy()
    if pd.api.types.is_numeric_dtype(clean):
        clean = clean.replace([np.inf, -np.inf], np.nan)
    codes, _ = pd.factorize(clean, sort=False)
    valid = codes >= 0
    if not valid.any():
        return codes
    counts = np.bincount(codes[valid])
    if len(counts) > max_categories:
        # Keep at most max_categories-1 levels, reserve one for the tail.
        cutoff = np.sort(counts)[-(max_categories - 1)]
        retained = np.flatnonzero(counts > cutoff)
        codes[valid & ~np.isin(codes, retained)] = len(counts)
    return codes


def structure_matrix(frame, specs, bins=10, max_categories=64, min_pairs=30):
    """C=1-NMI on pairwise-complete rows; no ordinal meaning for category IDs.

    Unknown/constant relationships use C=1 off-diagonal, diagonal is zero.
    Support and usable flags are exported, so lack of data is not a passed test.
    NMI is an association heuristic, not a guaranteed metric or causal relation.
    """
    if min_pairs < 2:
        raise ValueError("min_pairs must be >= 2")
    names = [name for name, _, _ in specs]
    codes = [encode_structure_column(frame[name], kind, bins, max_categories)
             for name, _, kind in specs]
    c = np.ones((len(names), len(names)))
    np.fill_diagonal(c, 0.0)
    support = np.zeros(c.shape, dtype=int)
    records = []
    for i, left in enumerate(codes):
        support[i, i] = np.count_nonzero(left >= 0)
        for k in range(i + 1, len(codes)):
            right = codes[k]
            valid = (left >= 0) & (right >= 0)
            n = int(valid.sum())
            support[i, k] = support[k, i] = n
            usable = n >= min_pairs and len(np.unique(left[valid])) > 1 and len(np.unique(right[valid])) > 1
            nmi = float(normalized_mutual_info_score(left[valid], right[valid], average_method="arithmetic")) if usable else None
            c[i, k] = c[k, i] = 1.0 - np.clip(nmi, 0, 1) if usable else 1.0
            records.append({"left": names[i], "right": names[k], "n_pairs": n,
                            "usable": usable, "nmi": nmi, "distance": c[i, k]})
    return c, support, pd.DataFrame(records, columns=["left", "right", "n_pairs", "usable", "nmi", "distance"])


def loss_gradient(t, d, loss_tensor, alpha):
    """Full squared-loss objective; valid also for a variable source marginal."""
    contraction = np.einsum("ijkl,kl->ij", loss_tensor, t, optimize=True)
    feature = float(np.sum(d * t))
    structure = float(np.sum(contraction * t))
    objective = (1 - alpha) * feature + alpha * structure
    gradient = (1 - alpha) * d + 2 * alpha * contraction
    return objective, gradient, feature, structure


def solve_fgw(d, cx, cy, allowed, alpha=0.5, n_init=5, max_iter=300, tol=1e-9, seed=42):
    """Masked semi-relaxed FGW; matrices are small (columns, not transactions).

    Column masses are 1 / #active destinations. Unfillable destination columns
    have zero mass. No row marginal/capacity is imposed. Returns the full plan.
    """
    d, cx, cy = (np.asarray(x, dtype=float) for x in (d, cx, cy))
    allowed = np.asarray(allowed, dtype=bool)
    if d.ndim != 2 or allowed.shape != d.shape:
        raise ValueError("d and allowed must be equally shaped matrices")
    n, m = d.shape
    if cx.shape != (n, n) or cy.shape != (m, m):
        raise ValueError("Structure shapes must match the source and destination columns")
    if not all(np.isfinite(x).all() for x in (d, cx, cy)):
        raise ValueError("FGW matrices must be finite")
    if not np.allclose(cx, cx.T) or not np.allclose(cy, cy.T):
        raise ValueError("This solver requires symmetric structures")
    if not 0 <= alpha <= 1 or n_init < 1 or max_iter < 1 or not math.isfinite(tol) or tol <= 0:
        raise ValueError("Require alpha in [0,1], positive n_init, max_iter, tol")
    active = np.flatnonzero(allowed.any(axis=0))
    full_t = np.zeros_like(d)
    if not len(active):
        return full_t, {"status": "NO_CANDIDATES", "objective": None, "restarts": []}
    da, mask, ca = d[:, active], allowed[:, active], cy[np.ix_(active, active)]
    q = np.full(len(active), 1 / len(active))
    loss_tensor = (cx[:, None, :, None] - ca[None, :, None, :]) ** 2
    rng = np.random.default_rng(seed)

    def oracle(gradient):
        selected = np.argmin(np.where(mask, gradient, np.inf), axis=0)
        vertex = np.zeros_like(da)
        vertex[selected, np.arange(len(active))] = q
        return vertex

    best, logs = None, []
    for restart in range(n_init):
        if restart == 0:
            t = mask.astype(float) / mask.sum(axis=0) * q
        elif restart == 1:
            t = oracle(da)
        else:
            t = rng.exponential(size=da.shape) * mask
            t *= q / t.sum(axis=0)
        value, grad, feature, structural = loss_gradient(t, da, loss_tensor, alpha)
        history = [value]
        for iteration in range(max_iter):
            direction = oracle(grad) - t
            slope = float(np.sum(grad * direction))
            gap = max(0.0, -slope)
            if gap <= tol:
                break
            endpoint = loss_gradient(t + direction, da, loss_tensor, alpha)[0]
            quadratic = endpoint - value - slope
            steps = [0.0, 1.0]
            if quadratic > 0:
                steps.append(float(np.clip(-slope / (2 * quadratic), 0, 1)))
            gamma = min(steps, key=lambda g: value + slope * g + quadratic * g * g)
            if gamma == 0:
                break
            t = t + gamma * direction
            value, grad, feature, structural = loss_gradient(t, da, loss_tensor, alpha)
            history.append(value)
        gap = max(0.0, float(np.sum(grad * (t - oracle(grad)))))
        record = {"restart": restart, "objective": value, "feature_loss": feature,
                  "structure_loss": structural, "fw_gap": gap, "converged": gap <= tol,
                  "iterations": len(history) - 1, "history": history,
                  "selected_source_indices": t.argmax(axis=0).tolist()}
        logs.append(record)
        if best is None or value < best[0]:
            best = (value, t.copy(), restart)
    full_t[:, active] = best[1]
    chosen = logs[best[2]]
    return full_t, {"status": "CONVERGED" if chosen["converged"] else "NOT_CONVERGED",
                    "best_restart": best[2], "objective": best[0], "fw_gap": chosen["fw_gap"],
                    "active_targets": active.tolist(), "restarts": logs}


def select_columns(scores, allowed, source_names, target_names, distance, ambiguity_margin=0.05):
    """Independent destination argmax. Repeated source indices are intentional."""
    mapping, rows = {}, []
    for j, target in enumerate(target_names):
        candidates = np.flatnonzero(allowed[:, j])
        if not len(candidates) or scores[candidates, j].sum() <= 0:
            mapping[target] = None
            rows.append({"mbd_field": target, "xbank_col": None, "status": "unfilled",
                         "candidate_count": len(candidates)})
            continue
        # Stable tie break: lower passport distance, then source name.
        ranking = sorted(candidates, key=lambda i: (-scores[i, j], distance[i, j], source_names[i]))
        winner = ranking[0]
        second = float(scores[ranking[1], j]) if len(ranking) > 1 else 0.0
        gap = float(scores[winner, j]) - second
        mapping[target] = source_names[winner]
        rows.append({"mbd_field": target, "xbank_col": source_names[winner], "status": "mapped",
                     "distance": float(distance[winner, j]), "fgw_score": float(scores[winner, j]),
                     "second_score": second, "column_margin": gap, "candidate_count": len(candidates),
                     "ambiguity": "AMBIGUOUS" if gap <= ambiguity_margin else "SEPARATED"})
    reuse_counts = Counter(source for source in mapping.values() if source is not None)
    for row in rows:
        row["source_reuse_count"] = reuse_counts[row["xbank_col"]]
    return mapping, pd.DataFrame(rows)


def match_schemas(xbank_passports, mbd_passports, cx, cy, max_distance=0.70, **solver_options):
    """Build D directly; forbidden pairs remain masked regardless of cost."""
    if not math.isfinite(max_distance) or not 0 <= max_distance <= 1:
        raise ValueError("max_distance must be between 0 and 1")
    source, target = list(xbank_passports), list(mbd_passports)
    distance = np.ones((len(source), len(target)))
    allowed = np.zeros(distance.shape, dtype=bool)
    details = []
    for i, a in enumerate(source):
        for j, b in enumerate(target):
            result = distance_components(xbank_passports[a], mbd_passports[b])
            distance[i, j] = result["distance"]
            compatible = result["reason"] == "compatible"
            allowed[i, j] = compatible and distance[i, j] < 1 and distance[i, j] <= max_distance
            reason = result["reason"] if not compatible or allowed[i, j] else "distance_threshold"
            details.append({"xbank_col": a, "mbd_field": b, "distance": distance[i, j],
                            "allowed": bool(allowed[i, j]), "reason": reason,
                            **result["distances"]})
    t, solver = solve_fgw(distance, cx, cy, allowed, **solver_options)
    mass = t.sum(axis=0)
    scores = np.divide(t, mass[None, :], out=np.zeros_like(t), where=mass[None, :] > 0)
    mapping, diagnostics = select_columns(scores, allowed, source, target, distance)
    return {"mbd_to_xbank": mapping, "source_names": source, "target_names": target,
            "distance": distance, "allowed": allowed, "transport": t, "scores": scores,
            "diagnostics": diagnostics, "pair_diagnostics": pd.DataFrame(details), "solver": solver}


def profile_source(con, sql, specs, tag, args):
    start = perf_counter()
    print(f"[{tag}] materializing source (sample_rows={args.sample_rows}) ...", flush=True)
    source = materialize(con, sample_sql(sql, args.sample_rows, args.seed), tag)
    print(f"[{tag}] profiling columns ...", flush=True)
    profiles = profile_columns(con, source, specs)
    expressions = [f"{expr} AS {quote_column(name)}" for name, expr, _ in specs]
    # All columns share the same sampled rows, otherwise dependencies are lost.
    frame = con.execute(sample_sql(f"SELECT {', '.join(expressions)} FROM ({source})",
                                   args.structure_rows, args.seed)).df()
    print(f"[{tag}] building dependency matrix on {len(frame):,} rows ...", flush=True)
    cx, support, relations = structure_matrix(frame, specs, args.structure_bins,
                                             args.max_categories, args.min_pairs)
    print(f"[{tag}] done in {perf_counter() - start:.1f}s", flush=True)
    return {"passports": profiles, "structure": cx, "support": support, "relations": relations}


def matcher_options(args):
    return {"max_distance": args.max_distance, "alpha": args.alpha, "n_init": args.n_init,
            "max_iter": args.max_iter, "tol": args.tol, "seed": args.seed}


def match_profiled(a, b, args):
    return match_schemas(a["passports"], b["passports"], a["structure"], b["structure"], **matcher_options(args))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def export_result(out, a, b, result, args, fixed_pairs=None):
    out.mkdir(parents=True, exist_ok=True)
    source, target = result["source_names"], result["target_names"]
    mapping = dict(result["mbd_to_xbank"])
    mapping.update(fixed_pairs or {})  # keys are destination fields, values xbank columns
    reverse = {name: [] for name in source}
    for destination, name in mapping.items():
        if name is not None:
            reverse.setdefault(name, []).append(destination)
    for name, key in (("distance", "distance"), ("transport_matrix", "transport"),
                      ("fgw_scores", "scores"), ("allowed_pairs", "allowed")):
        pd.DataFrame(result[key], index=source, columns=target).rename_axis("column").to_csv(out / f"{name}.csv")
    for label, profile in (("xbank", a), ("mbd", b)):
        names = list(profile["passports"])
        passports_to_frame(profile["passports"]).to_csv(out / f"{label}_profiles.csv", index=False)
        pd.DataFrame(profile["structure"], index=names, columns=names).to_csv(out / f"{label}_structure.csv")
        pd.DataFrame(profile["support"], index=names, columns=names).to_csv(out / f"{label}_structure_support.csv")
        profile["relations"].to_csv(out / f"{label}_dependencies.csv", index=False)
    table = result["diagnostics"].copy()
    extra = [{"mbd_field": dst, "xbank_col": src, "status": "fixed"} for dst, src in (fixed_pairs or {}).items()]
    extra += [{"xbank_col": name, "mbd_field": None, "status": "dropped"} for name in source if not reverse[name]]
    pd.concat([table, pd.DataFrame(extra)], ignore_index=True).to_csv(out / "mapping.csv", index=False)
    table.to_csv(out / "assignment_diagnostics.csv", index=False)
    result["pair_diagnostics"].to_csv(out / "pair_diagnostics.csv", index=False)
    write_json(out / "solver.json", result["solver"])
    write_json(out / "frozen_mapping.json", {
        "format_version": VERSION,
        "data_direction": "xbank -> MBD input schema",
        "mapping_convention": "mbd_to_xbank[destination] = source; null means unfilled",
        "mbd_to_xbank": mapping,
        "xbank_to_mbd": reverse,
        "dropped_xbank": [name for name in source if not reverse[name]],
        "unfilled_mbd_slots": [name for name, src in mapping.items() if src is None],
        "meta": {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "parameters": vars(args), "passport_version": PASSPORT_VERSION,
                 "passport_weights": PASSPORT_WEIGHTS,
                 "quantile_levels": list(QUANTILE_LEVELS),
                 "structure": "1 - arithmetic normalized mutual information; numeric quantile bins",
                 "solver_status": result["solver"]["status"],
                 "validation": "UNVERIFIED: transport weights are not correctness probabilities",
                 "note": "Column mapping only; no category-code alignment or inference-table filling"},
    })


def raw_mbd_sql(mbd_root: str, folds) -> str:
    fold_list = ", ".join(str(f) for f in folds)
    casts = [
        f"CAST({f} AS BIGINT) AS {f}" if f in MBD_DOUBLE_FIELDS else f
        for f in (*MBD_NUMERIC_FIELDS, *MBD_CATEGORICAL_FIELDS)
    ]
    sql = (
        f"SELECT {', '.join(casts)} FROM read_parquet('{mbd_root}/detail/trx/fold=*/*.parquet', "
        f"hive_partitioning=true) WHERE fold IN ({fold_list})"
    )
    return sql


def daily_mbd_sql(root, fold):
    cats = MBD_CATEGORICAL_FIELDS
    expressions = [f"CAST({c} AS BIGINT) AS {c}" if c in MBD_DOUBLE_FIELDS else c for c in cats]
    # Amount is aggregated, never a GROUP BY key.
    return (f"SELECT SUM(amount) AS amount, {', '.join(expressions)} "
            f"FROM read_parquet('{root}/detail/trx/fold=*/*.parquet', hive_partitioning=true) "
            f"WHERE fold = {int(fold)} GROUP BY client_id, DATE_TRUNC('day', event_time), {', '.join(cats)}")


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


def run_testbed(con, variant, args):
    """Known truth evaluated by destination, matching the independent selection."""
    fields = [name for name, _, _ in col_specs_mbd()]
    if variant == "daily":
        if not Path(args.daily_path).is_file():
            raise FileNotFoundError(f"Daily testbed requires {args.daily_path}; build it with data.mbd_adapter")
        truth = {**DAILY_MBD_TO_XBANK, "amount": "col_11"}  # destination -> source
        sql_a = f"SELECT * FROM read_parquet('{args.daily_path}')"
        sql_b = raw_mbd_sql(args.mbd_root, args.mbd_folds)
        specs = col_specs_xbank()
    else:
        sql_a = (daily_mbd_sql(args.mbd_root, args.testbed_fold_a) if variant == "daily_folds"
                 else raw_mbd_sql(args.mbd_root, [args.testbed_fold_a]))
        sql_b = (daily_mbd_sql(args.mbd_root, args.testbed_fold_b) if variant == "daily_folds"
                 else raw_mbd_sql(args.mbd_root, [args.testbed_fold_b]))
        truth = {name: f"col_a{i}" for i, name in enumerate(fields, 2)}
        sql_a = f"SELECT {', '.join(f'{name} AS {truth[name]}' for name in fields)} FROM ({sql_a})"
        specs = [(truth[name], truth[name], _mbd_kind(name)) for name in fields]
    a = profile_source(con, sql_a, specs, f"{variant}_a", args)
    b = profile_source(con, sql_b, col_specs_mbd(), f"{variant}_b", args)
    result = match_profiled(a, b, args)
    out = Path(args.out_dir) / f"testbed_{variant}"
    export_result(out, a, b, result, args)
    rows = [{"variant": variant, "mbd_field": field, "truth": truth[field],
             "recovered": result["mbd_to_xbank"][field],
             "correct": result["mbd_to_xbank"][field] == truth[field]} for field in fields]
    recovery = pd.DataFrame(rows)
    metrics = {"variant": variant, **testbed_metrics(recovery)}
    recovery.to_csv(out / "testbed_recovery.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out / "testbed_metrics.csv", index=False)
    print(f"[{variant}] recovery: {metrics['correct']}/{metrics['true_pairs']}; solver={result['solver']['status']}", flush=True)
    # Release testbed data before the next pair; profiles/structures are small.
    for tag in (f"{variant}_a", f"{variant}_b"):
        con.execute(f"DROP TABLE _prof_src_{tag}")
    return recovery, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xbank-path", default=XBANK_TRX_DEFAULT)                  #<-- файл xbank
    parser.add_argument("--mbd-root", default=MBD_ROOT_DEFAULT)                     #<-- data/mbd
    parser.add_argument("--mbd-folds", type=int, nargs="+", default=[0])                      #<--на каких фолдах  
    parser.add_argument("--out-dir", default=OUT_DIR_DEFAULT)  #<-- папка результатов FGW
    parser.add_argument("--sample-rows", type=int, default=0,
                        help="optional row sample for a quick look (0 = full tables)")        #<--сэмпл для быстрого взгляда; сид задаётся через --seed; для воспроизводимости --threads 1
    parser.add_argument("--structure-rows", type=int, default=100000, 
                        help="shared-row subsample for dependencies; 0 = all profiled rows")  #<-- общая выборка строк для связей между признаками; 0 = все строки из паспортной выборки
    parser.add_argument("--structure-bins", type=int, default=10)                             #<-- число квантильных интервалов numeric для расчёта взаимной информации
    parser.add_argument("--max-categories", type=int, default=64, 
                        help="dependency-only vocabulary cap, including pooled rare values")  #<-- лимит групп категорий только для связей: частые отдельно, редкие объединяются
    parser.add_argument("--min-pairs", type=int, default=60)                                  #<-- минимум совместно валидных строк для оценки связи двух признаков
    parser.add_argument("--alpha", type=float, default=0.5, 
                        help="structure weight; 0 = passport-only independent selection")     #<-- вес структуры в FGW: 0 = только паспорта, 1 = только связи; фильтры совместимости сохраняются
    threshold = parser.add_mutually_exclusive_group() 
    threshold.add_argument("--max-distance", type=float, default=0.70,
                           help="maximum passport distance D (default 0.70); hard compatibility gates still apply")  #<-- порог различия паспортов: пары выше исключаются до FGW; это не порог веса T
    threshold.add_argument("--min-sim", type=float,
                           help="legacy threshold: converted once to max_distance = 1 - value")  #<-- совместимость со старыми командами; 0.30 соответствует --max-distance 0.70
    parser.add_argument("--n-init", type=int, default=5)                                      #<-- сколько начальных планов FGW проверить; выбирается результат с минимальной целью
    parser.add_argument("--max-iter", type=int, default=300)                                  #<-- максимум итераций оптимизатора для каждого начального плана
    parser.add_argument("--tol", type=float, default=1e-9)                                    #<-- порог Frank-Wolfe gap для остановки оптимизатора
    parser.add_argument("--seed", type=int, default=42)                                       #<-- сид для выборок строк и случайных начальных планов FGW
    parser.add_argument("--threads", type=int, default=1, help="1 ensures seeded DuckDB samples are reproducible")  #<-- число потоков DuckDB; 1 обеспечивает воспроизводимость seeded-выборки
    parser.add_argument("--temp-dir", help="DuckDB spill directory for large runs")           #<-- куда DuckDB сбрасывает временные данные при нехватке памяти
    parser.add_argument("--memory-limit", help="DuckDB limit, e.g. 16GB")                     #<-- лимит памяти DuckDB, например 16GB; память pandas не ограничивает

    parser.add_argument("--testbed-variant", choices=["folds", "daily_folds", "daily", "both", "all"], default="folds")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--testbed-only", action="store_true")
    mode.add_argument("--no-testbed", action="store_true")
    parser.add_argument("--testbed-fold-a", type=int, default=0)
    parser.add_argument("--testbed-fold-b", type=int, default=1)
    parser.add_argument("--daily-path", default=MBD_ADAPTED_DAILY_DEFAULT)
    args = parser.parse_args(argv)
    # Convert the old CLI option once; all matching and exports use distance D.
    legacy_min_sim = vars(args).pop("min_sim")
    if legacy_min_sim is not None:
        args.max_distance = 1.0 - legacy_min_sim
    
    if (args.sample_rows < 0 or args.structure_rows < 0 or args.structure_bins < 2 or args.max_categories < 2
            or args.min_pairs < 2 or args.n_init < 1 or args.max_iter < 1 or args.threads < 1
            or not 0 <= args.alpha <= 1 or not 0 <= args.max_distance <= 1
            or not math.isfinite(args.tol) or args.tol <= 0 or args.seed < 0):
        parser.error("Invalid sampling, structure, or solver parameters")
    variants = {"both": ["folds", "daily"], "all": ["folds", "daily_folds", "daily"]}.get(args.testbed_variant, [args.testbed_variant])
    if not args.no_testbed and any(v in variants for v in ("folds", "daily_folds")) and args.testbed_fold_a == args.testbed_fold_b:
        parser.error("Testbed folds must differ")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.execute(f"SET threads = {args.threads}")
        if args.temp_dir:
            Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
            con.execute("SET temp_directory = ?", [args.temp_dir])
        if args.memory_limit:
            con.execute("SET memory_limit = ?", [args.memory_limit])
        if not args.no_testbed:
            all_recovery, all_metrics = [], []
            for variant in variants:
                recovery, metrics = run_testbed(con, variant, args)
                all_recovery.append(recovery)
                all_metrics.append(metrics)
            pd.concat(all_recovery, ignore_index=True).to_csv(out / "testbed_recovery.csv", index=False)
            pd.DataFrame(all_metrics).to_csv(out / "testbed_metrics.csv", index=False)
        if args.testbed_only:
            return
        a = profile_source(con, xbank_source(args.xbank_path), col_specs_xbank(), "xbank", args)
        b = profile_source(con, raw_mbd_sql(args.mbd_root, args.mbd_folds), col_specs_mbd(), "mbd", args)
        result = match_profiled(a, b, args)
        fixed = {destination: source for source, destination in FIXED_PAIRS.items()}
        export_result(out, a, b, result, args, fixed)
        print("\n=== xbank -> MBD input fields (source reuse allowed) ===")
        print(result["diagnostics"].to_string(index=False))
        print(f"Solver: {result['solver']['status']}; wrote {out}")


if __name__ == "__main__":
    main()
