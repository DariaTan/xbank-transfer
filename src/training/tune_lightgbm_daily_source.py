"""GPU/CPU probes of daily-pretrained encoders, without touching raw-source runs.

Uses the tested disk-streamed LightGBM engine. MBD uses five client folds;
xbank mappings are evaluated on an identical client-date intersection with
the existing 2023 train/validation and untouched 2024 test protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from training.paths import downstream_dir, embedding_dir, load_data_config
from training.train_downstream import load_config
from training.tune_lightgbm_chronos import run, signature, summarize
from training.tune_lightgbm_xbank_paired import MODELS, TARGETS, VARIANTS, paired_cohort

SOURCE = "mbd_daily"


class EmbeddingsPending(Exception):
    pass


def published_files(model: str, evaluation: str) -> tuple[list[Path], dict]:
    directory = embedding_dir("/app/data/embeds", evaluation, SOURCE, model)
    manifest_path = directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise EmbeddingsPending(f"{evaluation}/{model}: inference manifest not yet published")
    manifest = json.loads(manifest_path.read_text())
    for key, expected in (("model", model), ("evaluation_name", evaluation),
                          ("checkpoint_source", SOURCE)):
        if manifest.get(key) != expected:
            raise ValueError(f"{manifest_path}: wrong {key}")
    dates = manifest["target_dates"]
    if len(dates) != 12 or len(set(dates)) != 12:
        raise ValueError(f"{manifest_path}: expected twelve unique target dates")
    files = sorted(directory.glob("*.parquet"))
    if {p.stem for p in files} - set(dates):
        raise ValueError(f"{directory}: unexpected published dates")
    if {p.stem for p in files} != set(dates):
        raise EmbeddingsPending(f"{evaluation}/{model}: {len(files)}/12 dates published")
    for path in files:
        if pq.ParquetFile(path).metadata.num_rows < 1:
            raise ValueError(f"empty published embedding: {path}")
    return files, {"manifest": manifest, "manifest_file": signature(manifest_path)}


def inputs(model: str, corpus: str) -> tuple[dict, dict]:
    evaluations = VARIANTS if corpus == "xbank" else ("mbd_daily",)
    files, provenance = {}, {}
    for evaluation in evaluations:
        files[evaluation], provenance[evaluation] = published_files(model, evaluation)
    hashes = [p["manifest"]["checkpoint_files_sha256"] for p in provenance.values()]
    if any(h != hashes[0] for h in hashes[1:]):
        raise ValueError("xbank variants use different encoder weights/preprocessors")
    return files, provenance


def paired_summary(model: str, cleanup_cache: bool) -> None:
    roots = {v: downstream_dir("/app/data/downstream", v, SOURCE, model) for v in VARIANTS}
    manifests = {v: json.loads((roots[v] / "lightgbm_hpo_calendar/run_manifest.json").read_text())
                 for v in VARIANTS}
    common = {k: value for k, value in manifests[VARIANTS[0]].items()
              if k not in ("evaluation_name", "embedding_files", "embedding_provenance")}
    other = {k: value for k, value in manifests[VARIANTS[1]].items()
             if k not in ("evaluation_name", "embedding_files", "embedding_provenance")}
    if common != other or "paired_cohort_sha256" not in common:
        raise ValueError("xbank summaries require identical paired cohorts and HPO settings")
    for variant in VARIANTS:
        summarize(variant, model=model, checkpoint_source=SOURCE)
    old, new = [pd.read_csv(roots[v] / "results_all_folds.csv") for v in VARIANTS]
    comparison = old.merge(new, on=["model", "target"], suffixes=("_original", "_fgw_v2"),
                           validate="one_to_one")
    for key in ("n_rows", "n_positive", "prevalence"):
        if not comparison[f"{key}_original"].equals(comparison[f"{key}_fgw_v2"]):
            raise ValueError(f"paired test cohorts disagree in {key}")
    for metric in ("pr_auc", "roc_auc"):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_fgw_v2"] - comparison[f"{metric}_original"]
    output = roots["xbank"].parent.parent.parent / "xbank_mapping_comparison" / f"{SOURCE}_source"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{model}_paired_results.csv"
    temporary = path.with_suffix(".csv.tmp")
    comparison.to_csv(temporary, index=False)
    os.replace(temporary, path)
    if cleanup_cache:
        for variant in VARIANTS:
            summarize(variant, True, model=model, checkpoint_source=SOURCE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--corpus", choices=("mbd_daily", "xbank"), required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("/app/configs"))
    parser.add_argument("--test-fold", type=int, choices=range(5), default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--tune-client-cap", type=int, default=50000)
    parser.add_argument("--ready", action="store_true", help="exit 75 if inference is not complete")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--cleanup-cache", action="store_true")
    args = parser.parse_args()
    trials = args.trials if args.trials is not None else (4 if args.corpus == "mbd_daily" else 3)
    rounds = args.max_rounds if args.max_rounds is not None else (400 if args.corpus == "mbd_daily" else 300)
    if min(trials, rounds, args.threads, args.tune_client_cap) < 1:
        parser.error("search/resource limits must be positive")
    if args.summarize:
        if args.corpus == "xbank":
            paired_summary(args.model, args.cleanup_cache)
        else:
            summarize(args.corpus, args.cleanup_cache, model=args.model, checkpoint_source=SOURCE)
        return
    try:
        files, provenance = inputs(args.model, args.corpus)
    except EmbeddingsPending as error:
        print(error, flush=True)
        raise SystemExit(75) from error
    if args.ready:
        print(f"{args.model}/{args.corpus}: all required embeddings are published", flush=True)
        return
    paired = None
    paired_sources = None
    if args.corpus == "xbank":
        configs = {v: load_data_config(args.config_dir / "data" / f"{v}.yaml") for v in VARIANTS}
        target_paths = {Path(cfg["paths"]["targets"]) for cfg in configs.values()}
        if len(target_paths) != 1:
            raise ValueError("both mappings must share the same targets")
        paired, _, _, _ = paired_cohort(args.model, target_paths.pop(),
                                       load_config(str(args.config_dir / "models/downstream.yaml")), files)
        paired = paired[["id", "col_1", *TARGETS]].copy()
        paired["id"] = paired.id.astype(str)
        paired_sources = {v: {"files": [signature(p) for p in files[v]],
                              "provenance": provenance[v]} for v in VARIANTS}
    for evaluation in files:
        config = args.config_dir / "data" / f"{evaluation}.yaml"
        downstream = args.config_dir / "models" / (
            "downstream_mbd.yaml" if args.corpus == "mbd_daily" else "downstream.yaml")
        run(str(config), str(downstream), args.test_fold, args.device, args.seed,
            trials, args.threads, args.tune_client_cap, rounds,
            255 if args.corpus == "mbd_daily" else 63, model=args.model,
            checkpoint_source=SOURCE, paired_rows=paired, paired_sources=paired_sources,
            embedding_provenance=provenance[evaluation])


if __name__ == "__main__":
    main()
