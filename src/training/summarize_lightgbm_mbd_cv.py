"""Aggregate the five held-out MBD-raw LightGBM evaluations.

Fold 4 is the original holdout run; folds 0-3 are written by
``tune_lightgbm_mbd.py --test-fold``. All runs must use the same frozen
embeddings and HPO settings before a five-fold result is published.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import TARGET_COLS
from training.paths import downstream_dir, resolve_data_path
from training.tune_lightgbm_mbd import ALL_FOLDS, folds_for_test


MODELS = ("coles", "cotic", "thp", "nep", "mlm")
MATCHED_SETTINGS = ("target_file", "embedding_files", "seed", "trials",
                    "threads", "tune_client_cap", "max_rounds")


def fold_output(output_root: str | Path, model: str, test_fold: int) -> Path:
    model_dir = downstream_dir(output_root, "mbd_raw", "mbd", model)
    if test_fold == 4:
        return model_dir / "lightgbm_hpo_holdout"
    return model_dir / "lightgbm_hpo_cv" / f"fold{test_fold}"


def collect(output_root: str | Path, models: tuple[str, ...] = MODELS) -> pd.DataFrame:
    records = []
    for model in models:
        reference = None
        for test_fold in ALL_FOLDS:
            directory = fold_output(output_root, model, test_fold)
            manifest = json.loads((directory / "run_manifest.json").read_text())
            if reference is None:
                reference = {key: manifest[key] for key in MATCHED_SETTINGS}
            elif any(manifest[key] != reference[key] for key in MATCHED_SETTINGS):
                raise ValueError(f"{model} fold {test_fold}: source or HPO settings differ")
            if manifest["model"] != model:
                raise ValueError(f"{directory}: wrong model in manifest")
            if test_fold != 4:
                train_folds, val_fold = folds_for_test(test_fold)
                if (manifest.get("test_fold") != test_fold or
                    manifest.get("val_fold") != val_fold or
                    tuple(manifest.get("train_folds", ())) != train_folds):
                    raise ValueError(f"{directory}: unexpected fold assignment")
            for target in TARGET_COLS:
                result_path = directory / f"{target}_metrics.json"
                result = json.loads(result_path.read_text())
                if result["target"] != target or not (directory / f"{target}_model.txt").is_file():
                    raise ValueError(f"{directory}: incomplete result for {target}")
                metrics = result["test_metrics"]
                if (metrics["n_rows"] <= 0 or
                    not all(np.isfinite(metrics[key]) and 0 <= metrics[key] <= 1
                            for key in ("prevalence", "pr_auc", "roc_auc"))):
                    raise ValueError(f"{result_path}: invalid test metrics")
                records.append({"model": model, "target": target, "test_fold": test_fold,
                                "n_rows": metrics["n_rows"],
                                "prevalence": metrics["prevalence"],
                                "pr_auc": metrics["pr_auc"],
                                "roc_auc": metrics["roc_auc"],
                                "final_rounds": result["final_rounds"]})
    return pd.DataFrame.from_records(records)


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = rows.groupby(["model", "target"], sort=True)
    aggregate = grouped.agg(
        n_folds=("test_fold", "nunique"),
        n_test_rows_total=("n_rows", "sum"),
        prevalence_mean=("prevalence", "mean"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
    ).reset_index()
    if not (aggregate.n_folds == len(ALL_FOLDS)).all():
        raise ValueError("every model/target must have all five test folds")
    macro = aggregate.groupby("model", sort=True).agg(
        mean_roc_auc_across_targets=("roc_auc_mean", "mean"),
        mean_pr_auc_across_targets=("pr_auc_mean", "mean"),
    ).reset_index()
    return aggregate, macro


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/app/data/downstream")
    args = parser.parse_args()
    rows = collect(args.output_root)
    aggregate, macro = summarize(rows)
    output = resolve_data_path(args.output_root) / "mbd_raw" / "mbd_source" / "lightgbm_hpo_cv_summary"
    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(rows, output / "results_all_folds.csv")
    _atomic_csv(aggregate, output / "results_aggregated.csv")
    _atomic_csv(macro, output / "model_macro.csv")
    print(aggregate.to_string(index=False), flush=True)
    print(macro.to_string(index=False), flush=True)
    print(f"Saved five-fold results to {output}", flush=True)


if __name__ == "__main__":
    main()
