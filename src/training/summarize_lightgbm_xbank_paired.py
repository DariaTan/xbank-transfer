"""Compare completed xbank mappings only on verified paired test cohorts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from training.paths import downstream_dir, resolve_data_path


MODELS = ("coles", "cotic", "thp", "nep", "mlm")
TARGETS = ("col_3", "col_4", "col_5")
VARIANTS = ("xbank", "xbank_fgw_v2")


def collect(output_root: str | Path, models: tuple[str, ...] = MODELS,
            targets: tuple[str, ...] = TARGETS) -> pd.DataFrame:
    rows = []
    for model in models:
        runs = {}
        for variant in VARIANTS:
            directory = downstream_dir(output_root, variant, "mbd", model) / "lightgbm_xbank_paired"
            manifest = json.loads((directory / "run_manifest.json").read_text())
            if manifest["model"] != model or manifest["variant"] != variant:
                raise ValueError(f"{directory}: wrong model or variant")
            runs[variant] = (directory, manifest)
        old_manifest = runs[VARIANTS[0]][1]
        new_manifest = runs[VARIANTS[1]][1]
        for key in ("cohort_sha256", "target_file", "split", "seed", "split_seed",
                    "trials", "threads", "max_rounds", "device_type", "max_bin"):
            if old_manifest[key] != new_manifest[key]:
                raise ValueError(f"{model}: comparison invalid; {key} differs")
        for target in targets:
            metrics = {}
            for variant in VARIANTS:
                directory = runs[variant][0]
                if not (directory / f"{target}_model.txt").is_file():
                    raise FileNotFoundError(f"{directory}/{target}_model.txt")
                result = json.loads((directory / f"{target}_metrics.json").read_text())
                if result["model"] != model or result["variant"] != variant or result["target"] != target:
                    raise ValueError(f"{directory}: wrong metrics identity")
                metrics[variant] = result["test_metrics"]
            old, new = (metrics[variant] for variant in VARIANTS)
            if old["n_rows"] != new["n_rows"] or old["prevalence"] != new["prevalence"]:
                raise ValueError(f"{model}/{target}: test cohort or labels differ")
            rows.append({
                "model": model, "target": target, "n_test": old["n_rows"],
                "prevalence": old["prevalence"],
                "old_pr_auc": old["pr_auc"], "new_pr_auc": new["pr_auc"],
                "delta_pr_auc": new["pr_auc"] - old["pr_auc"],
                "old_roc_auc": old["roc_auc"], "new_roc_auc": new["roc_auc"],
                "delta_roc_auc": new["roc_auc"] - old["roc_auc"],
                "old_precision_at_5pct": old["precision@5%"],
                "new_precision_at_5pct": new["precision@5%"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/app/data/downstream")
    args = parser.parse_args()
    frame = collect(args.output_root)
    output = resolve_data_path(args.output_root) / "xbank_mapping_comparison" / "mbd_source"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "lightgbm_paired_results.csv"
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    print(frame.to_string(index=False), flush=True)
    print(f"Saved paired mapping comparison to {path}", flush=True)


if __name__ == "__main__":
    main()
