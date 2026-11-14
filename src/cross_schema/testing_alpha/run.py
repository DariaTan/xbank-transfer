"""Sweep FGW alpha on clean MBD versus a noisy, anonymously permuted copy.

Run from xbank-transfer:
  PYTHONPATH=src python -m cross_schema.testing_alpha.run --dry-run
  PYTHONPATH=src python -m cross_schema.testing_alpha.run --mbd-root data/mbd

The default grid has 143 matcher calls / at most 715 optimizer starts.
Passports and NMI are cached per noisy copy, never recomputed per alpha.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Also allow: python src/cross_schema/testing_alpha/run.py ...
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import numpy as np
import pandas as pd
import sklearn

from cross_schema import column_profiles_vFGW as fgw
from cross_schema.testing_alpha.noise import NOISE_LEVELS, corrupt_frame

VERSION = "alpha-sweep-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALPHAS = [i / 20 for i in range(21)]


def write_json(path, data):
    """Publish complete JSON files; metrics.json is each run's completion marker."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    fgw.write_json(temporary, data)
    temporary.replace(path)


def experiment_cases(conditions, seeds):
    cases = []
    for name in conditions:
        numeric, categorical = NOISE_LEVELS[name]
        for seed in ([None] if name == "clean" else seeds):
            cases.append({"case_id": name if seed is None else f"{name}/seed_{seed}",
                          "condition": name, "numeric_noise": numeric,
                          "category_noise": categorical, "noise_seed": seed})
    return cases


def anonymous_schema(specs, seed):
    """One fixed permutation for every noise condition, seed and alpha."""
    order = np.random.default_rng(seed).permutation(len(specs))
    truth, source_specs = {}, []
    for i, position in enumerate(order):
        field, _, kind = specs[position]
        anonymous = f"col_a{i:02d}"
        truth[field] = anonymous
        source_specs.append((anonymous, anonymous, kind))
    return truth, source_specs


def profile_frame(con, frame, specs, structure_indices, args):
    con.register("_alpha_frame", frame)
    try:
        passports = fgw.profile_columns(con, "SELECT * FROM _alpha_frame", specs)
        # Finish the final aggregate result before releasing its registered
        # dataframe (required by DuckDB versions with streaming execute()).
        con.fetchall()
    finally:
        con.unregister("_alpha_frame")
    structure, support, relations = fgw.structure_matrix(
        frame.iloc[structure_indices], specs, args.structure_bins, args.max_categories, args.min_pairs)
    return {"passports": passports, "structure": structure, "support": support, "relations": relations}


def save_profile(path, profile):
    path.mkdir(parents=True, exist_ok=True)
    names = list(profile["passports"])
    fgw.passports_to_frame(profile["passports"]).to_csv(path / "profiles.csv", index=False)
    for key in ("structure", "support"):
        pd.DataFrame(profile[key], index=names, columns=names).rename_axis("column").to_csv(path / f"{key}.csv")
    profile["relations"].to_csv(path / "dependencies.csv", index=False)
    write_json(path / "profile.json", {"passports": profile["passports"],
               "structure": profile["structure"].tolist(), "support": profile["support"].tolist()})


def load_profile(path):
    data = json.loads((path / "profile.json").read_text())
    return {"passports": data["passports"], "structure": np.asarray(data["structure"]),
            "support": np.asarray(data["support"]), "relations": pd.read_csv(path / "dependencies.csv")}


def input_source(args):
    if args.input_parquet:
        files = [Path(args.input_parquet).expanduser().resolve()]
        if not files[0].is_file():
            raise FileNotFoundError(files[0])
        expressions = [f"CAST({f} AS BIGINT) AS {f}" if f in fgw.MBD_DOUBLE_FIELDS else f
                       for f, _, _ in fgw.col_specs_mbd()]
        escaped = str(files[0]).replace("'", "''")
        sql = f"SELECT {', '.join(expressions)} FROM read_parquet('{escaped}')"
    else:
        root = Path(args.mbd_root).expanduser().resolve()
        files = []
        for fold in args.mbd_folds:
            found = sorted((root / "detail" / "trx" / f"fold={fold}").glob("*.parquet"))
            if not found:
                raise FileNotFoundError(f"No MBD parquet files in {root}/detail/trx/fold={fold}")
            files.extend(found)
        sql = fgw.raw_mbd_sql(str(root).replace("'", "''"), args.mbd_folds)
    fingerprints = [{"path": str(f), "size": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns}
                    for f in files]
    return sql, fingerprints


def configuration(args, fingerprints):
    # Refuse to mix settings, code versions or modified input files on resume.
    settings = {k: v for k, v in vars(args).items()
                if k not in {"out_dir", "resume", "dry_run", "temp_dir", "memory_limit"}}
    return {"version": VERSION, "settings": settings, "input_files": fingerprints,
            "noise_levels": {k: list(NOISE_LEVELS[k]) for k in args.conditions},
            "passport_weights": fgw.PASSPORT_WEIGHTS,
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), Path(__file__).with_name("noise.py"), Path(fgw.__file__))},
            "packages": {"duckdb": duckdb.__version__, "numpy": np.__version__,
                         "pandas": pd.__version__, "sklearn": sklearn.__version__}}


def setup_output(out, config, cases, alphas):
    manifest = out / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if existing["config"] != config:
            raise ValueError("Output contains another configuration/input/code version; use a new --out-dir")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Output is not empty and has no manifest; use a new --out-dir")
        out.mkdir(parents=True, exist_ok=True)
        n_runs = len(cases) * len(alphas)
        write_json(manifest, {"created": datetime.now(timezone.utc).isoformat(), "config": config,
                             "cases": cases, "expected_runs": n_runs,
                             "max_optimizer_starts": n_runs * config["settings"]["n_init"]})


def evaluate(result, truth, specs):
    fields = [name for name, _, _ in specs]
    kinds = {name: kind for name, _, kind in specs}
    source_positions = {name: i for i, name in enumerate(result["source_names"])}
    target_positions = {name: j for j, name in enumerate(result["target_names"])}
    rows = []
    for name in fields:
        i, j = source_positions[truth[name]], target_positions[name]
        recovered = result["mbd_to_xbank"][name]
        rows.append({"mbd_field": name, "kind": kinds[name], "truth": truth[name],
                     "recovered": recovered, "correct": recovered == truth[name],
                     "truth_allowed": bool(result["allowed"][i, j]),
                     "truth_distance": float(result["distance"][i, j])})
    recovery = pd.DataFrame(rows).merge(result["diagnostics"], on="mbd_field", validate="one_to_one")
    standard = fgw.testbed_metrics(recovery)
    metrics = {**standard, "accuracy": standard["recall"],
               "n_fields": len(fields), "unfilled": len(fields) - standard["accepted"],
               "unfilled_rate": 1 - standard["coverage"],
               "truth_filtered": int((~recovery["truth_allowed"]).sum())}
    for kind in ("categorical", "numeric"):
        group = recovery[recovery["kind"] == kind]
        metrics[f"{kind}_fields"] = len(group)
        metrics[f"{kind}_accuracy"] = float(group["correct"].mean()) if len(group) else None
    return recovery, metrics


def export_run(path, result, recovery, metrics):
    path.mkdir(parents=True, exist_ok=True)
    for filename, key in (("distance", "distance"), ("allowed_pairs", "allowed"),
                          ("transport_matrix", "transport"), ("fgw_scores", "scores")):
        pd.DataFrame(result[key], index=result["source_names"], columns=result["target_names"]).rename_axis(
            "column").to_csv(path / f"{filename}.csv")
    recovery.to_csv(path / "recovery.csv", index=False)
    result["pair_diagnostics"].to_csv(path / "pair_diagnostics.csv", index=False)
    write_json(path / "solver.json", result["solver"])
    write_json(path / "frozen_mapping.json", {"data_direction": "noisy MBD -> clean MBD input fields",
               "mbd_to_noisy": result["mbd_to_xbank"], "known_truth": dict(zip(recovery.mbd_field, recovery.truth))})
    write_json(path / "metrics.json", metrics)


def write_results(out, records):
    if not records:
        return
    frame = pd.DataFrame(records).sort_values(["condition", "noise_seed", "alpha"], na_position="first")
    temporary = out / "results.csv.tmp"
    frame.to_csv(temporary, index=False)
    temporary.replace(out / "results.csv")
    summary = frame.groupby(["condition", "numeric_noise", "category_noise", "alpha"], sort=False).agg(
        runs=("accuracy", "size"), accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        categorical_accuracy_mean=("categorical_accuracy", "mean"),
        categorical_accuracy_sd=("categorical_accuracy", "std"),
        coverage_mean=("coverage", "mean"), unfilled_rate_mean=("unfilled_rate", "mean"),
        truth_filtered_mean=("truth_filtered", "mean"), converged_runs=("converged", "sum"))
    summary.to_csv(out / "summary.csv")


def run(args):
    cases = experiment_cases(args.conditions, args.noise_seeds)
    n_runs = len(cases) * len(args.alphas)
    print(f"Grid: {len(cases)} copies x {len(args.alphas)} alphas = {n_runs} FGW calls; "
          f"up to {n_runs * args.n_init} optimizer starts", flush=True)
    if args.dry_run:
        print(pd.DataFrame(cases).to_string(index=False))
        return
    sql, fingerprints = input_source(args)
    out = Path(args.out_dir).expanduser().resolve()
    if (out / "manifest.json").exists() and not args.resume:
        raise ValueError("Existing experiment: use --resume with the same arguments, or a new --out-dir")
    setup_output(out, configuration(args, fingerprints), cases, args.alphas)
    specs = fgw.col_specs_mbd()
    truth, source_specs = anonymous_schema(specs, args.permutation_seed)
    write_json(out / "ground_truth.json", {"mbd_to_noisy": truth, "source_specs": source_specs})
    records = []
    with duckdb.connect() as con:
        con.execute(f"SET threads = {args.threads}")
        if args.memory_limit:
            con.execute("SET memory_limit = ?", [args.memory_limit])
        if args.temp_dir:
            Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
            con.execute("SET temp_directory = ?", [args.temp_dir])
        frame = structure_indices = None

        def sample_once():
            nonlocal frame, structure_indices
            if frame is None:
                print(f"Reading one MBD sample (up to {args.sample_rows:,} rows) ...", flush=True)
                frame = con.execute(fgw.sample_sql(sql, args.sample_rows, args.sample_seed)).df().reset_index(drop=True)
                if len(frame) < args.min_pairs:
                    raise ValueError(f"Only {len(frame)} rows; need at least --min-pairs={args.min_pairs}")
                count = min(args.structure_rows or len(frame), len(frame))
                structure_indices = np.sort(np.random.default_rng(args.sample_seed).choice(len(frame), count, replace=False))
                write_json(out / "sample.json", {"rows": len(frame), "structure_rows": count,
                           "same_rows_on_both_sides": True, "same_structure_indices": True})
            return frame

        clean_path = out / "clean_profile"
        if (clean_path / "profile.json").exists():
            clean = load_profile(clean_path)
        else:
            print("Profiling clean MBD ...", flush=True)
            sample_once()
            clean = profile_frame(con, frame, specs, structure_indices, args)
            save_profile(clean_path, clean)

        for case in cases:
            case_path = out / "cases" / case["case_id"]
            pending = []
            for alpha in args.alphas:
                path = case_path / f"alpha_{alpha}"
                if (path / "metrics.json").exists():
                    records.append(json.loads((path / "metrics.json").read_text()))
                else:
                    pending.append((alpha, path))
            if not pending:
                print(f"[{case['case_id']}] all alphas complete; skipped", flush=True)
                continue
            profile_path = case_path / "noisy_profile"
            if (profile_path / "profile.json").exists() and (case_path / "noise_audit.csv").exists():
                noisy_profile = load_profile(profile_path)
            else:
                print(f"[{case['case_id']}] corrupting and profiling once ...", flush=True)
                noisy, audit = corrupt_frame(sample_once(), specs, clean["passports"],
                                             case["numeric_noise"], case["category_noise"], case["noise_seed"] or 0)
                masked = noisy[list(truth)].rename(columns=truth)
                noisy_profile = profile_frame(con, masked, source_specs, structure_indices, args)
                save_profile(profile_path, noisy_profile)
                audit.to_csv(case_path / "noise_audit.csv", index=False)
                del noisy, masked
            for alpha, path in pending:
                start = perf_counter()
                result = fgw.match_schemas(noisy_profile["passports"], clean["passports"],
                    noisy_profile["structure"], clean["structure"], max_distance=args.max_distance,
                    alpha=alpha, n_init=args.n_init, max_iter=args.max_iter, tol=args.tol, seed=args.solver_seed)
                elapsed = perf_counter() - start
                recovery, metrics = evaluate(result, truth, specs)
                solver = result["solver"]
                best = solver["restarts"][solver["best_restart"]] if solver["restarts"] else {}
                metrics.update(case, alpha=alpha, run_dir=str(path.relative_to(out)),
                    solver_seed=args.solver_seed, status=solver["status"], converged=solver["status"] == "CONVERGED",
                    objective=solver["objective"], feature_loss=best.get("feature_loss"),
                    structure_loss=best.get("structure_loss"), fw_gap=solver.get("fw_gap"),
                    optimizer_starts=len(solver["restarts"]), match_seconds=elapsed)
                export_run(path, result, recovery, metrics)
                records.append(metrics)
                write_results(out, records)
                print(f"[{len(records)}/{n_runs}] {case['case_id']} alpha={alpha:g}: "
                      f"{metrics['correct']}/{metrics['n_fields']} correct, "
                      f"{metrics['unfilled']} unfilled; {solver['status']}", flush=True)
        write_results(out, records)
    print(f"Completed {len(records)}/{n_runs}; results: {out / 'results.csv'}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mbd-root", default=str(PROJECT_ROOT / "data" / "mbd"))
    source.add_argument("--input-parquet", help="Optional parquet with the original MBD feature names")
    parser.add_argument("--mbd-folds", type=int, nargs="+", default=[0])
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "testing_alpha"))
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--conditions", choices=list(NOISE_LEVELS), nargs="+", default=list(NOISE_LEVELS))
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sample-rows", type=int, default=2_000_000, help="0 = all selected rows (loaded into RAM)")
    parser.add_argument("--structure-rows", type=int, default=200_000, help="0 = all sampled rows")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--permutation-seed", type=int, default=42)
    parser.add_argument("--solver-seed", type=int, default=42, help="Fixed across noise seeds and alpha values")
    parser.add_argument("--structure-bins", type=int, default=10)
    parser.add_argument("--max-categories", type=int, default=64)
    parser.add_argument("--min-pairs", type=int, default=60)
    parser.add_argument("--max-distance", type=float, default=0.70)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", help="DuckDB only; pandas copies also require RAM")
    parser.add_argument("--temp-dir")
    parser.add_argument("--resume", action="store_true", help="Reuse completed profiles/runs for exactly the same configuration")
    parser.add_argument("--dry-run", action="store_true", help="Print the grid without reading data or writing files")
    args = parser.parse_args(argv)
    if (any(not math.isfinite(a) or not 0 <= a <= 1 for a in args.alphas)
            or len(set(args.alphas)) != len(args.alphas)
            or len(set(args.conditions)) != len(args.conditions)
            or len(set(args.noise_seeds)) != len(args.noise_seeds)
            or len(set(args.mbd_folds)) != len(args.mbd_folds)
            or min([*args.noise_seeds, *args.mbd_folds, args.sample_seed, args.permutation_seed, args.solver_seed]) < 0
            or args.sample_rows < 0 or args.structure_rows < 0
            or args.structure_bins < 2 or args.max_categories < 2 or args.min_pairs < 2
            or 0 < args.structure_rows < args.min_pairs or 0 < args.sample_rows < args.min_pairs
            or args.n_init < 1 or args.max_iter < 1 or args.threads < 1
            or not math.isfinite(args.tol) or args.tol <= 0
            or not math.isfinite(args.max_distance) or not 0 <= args.max_distance <= 1):
        parser.error("Invalid/duplicate grid, seed, sampling, structure or solver parameters")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
