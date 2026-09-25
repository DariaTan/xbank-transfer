"""Paired, leakage-safe xbank LightGBM probes for two frozen schema mappings.

The two representations of each encoder use exactly the same client-month
rows, client-disjoint 2023 train/validation split, and untouched 2024 test.
Only the three non-conflicting xbank product targets are evaluated.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from data.schema import TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL
from training.paths import downstream_dir, embedding_dir, load_data_config
from training.target_io import load_targets
from training.train_downstream import evaluate, load_config
from training.tune_lightgbm_mbd import candidate_params


VARIANTS = ("xbank", "xbank_fgw_v2")
MODELS = ("coles", "cotic", "thp", "nep", "mlm")
TARGETS = ("col_3", "col_4", "col_5")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _embedding_files(model: str, variant: str) -> list[Path]:
    directory = embedding_dir("/app/data/embeds", variant, "mbd", model)
    files = sorted(directory.glob("*.parquet"))
    if len(files) != 12:
        raise ValueError(f"{variant}/{model}: expected 12 published dates, found {len(files)}")
    return files


def _keys(files: list[Path]) -> pd.DataFrame:
    keys = pd.concat((pd.read_parquet(path, columns=["inn", "date"])
                      for path in files), ignore_index=True)
    keys["date"] = keys["date"].astype(str)
    if keys.duplicated(["inn", "date"]).any():
        raise ValueError("embeddings contain duplicate client-date keys")
    return keys.rename(columns={"inn": TARGETS_CLIENT_ID_COL,
                                "date": TARGETS_DATE_COL})


def paired_cohort(model: str, targets_path: Path, cfg: dict,
                  files_by_variant: dict[str, list[Path]]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    targets = load_targets(str(targets_path), list(TARGETS))
    dates = set(cfg["train"] + cfg["test"])
    targets = targets[targets[TARGETS_DATE_COL].isin(dates)].copy()
    if targets.empty:
        raise ValueError("no xbank targets in the configured train/test dates")
    for variant in VARIANTS:
        targets = targets.merge(_keys(files_by_variant[variant]),
                                on=[TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL],
                                how="inner", validate="one_to_one")
    if targets.empty:
        raise ValueError("the two embedding variants share no labeled client-dates")
    targets.reset_index(drop=True, inplace=True)
    train_pool = np.flatnonzero(targets[TARGETS_DATE_COL].isin(cfg["train"]).to_numpy())
    test = np.flatnonzero(targets[TARGETS_DATE_COL].isin(cfg["test"]).to_numpy())
    if not len(train_pool) or not len(test):
        raise ValueError("train or test period has no paired target rows")
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["val_frac"],
                                 random_state=cfg["seed"])
    in_train, in_val = next(splitter.split(
        targets.iloc[train_pool],
        groups=targets.iloc[train_pool][TARGETS_CLIENT_ID_COL],
    ))
    train, val = train_pool[in_train], train_pool[in_val]
    if set(targets.iloc[train][TARGETS_CLIENT_ID_COL]) & set(
            targets.iloc[val][TARGETS_CLIENT_ID_COL]):
        raise AssertionError("client leaked across training and validation")
    print(f"{model}: paired rows={len(targets)} train={len(train)} "
          f"val={len(val)} test={len(test)}", flush=True)
    return targets, train, val, test


def _features(files: list[Path], targets: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    embeddings = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    embeddings["date"] = embeddings["date"].astype(str)
    cols = sorted((col for col in embeddings if col.startswith("emb_")),
                  key=lambda col: int(col.removeprefix("emb_")))
    if not cols or cols != [f"emb_{i}" for i in range(len(cols))]:
        raise ValueError("embedding columns must be contiguous emb_0..emb_N")
    if embeddings.duplicated(["inn", "date"]).any():
        raise ValueError("embeddings contain duplicate client-date keys")
    frame = targets[[TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL]].merge(
        embeddings[["inn", "date", *cols]],
        left_on=[TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL],
        right_on=["inn", "date"], how="left", sort=False, validate="one_to_one",
    )
    if frame[cols].isna().any().any():
        raise ValueError("a paired client-date lacks finite embeddings")
    values = frame[cols].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain non-finite values")
    del embeddings, frame
    gc.collect()
    return values, cols


def run(model: str, device: str, trials: int, threads: int, max_rounds: int,
        seed: int, output_root: str, config_path: str, gpu_platform_id: int,
        gpu_device_id: int) -> None:
    if model not in MODELS or device not in ("cpu", "gpu"):
        raise ValueError("unsupported model or device")
    if trials < 1 or threads < 1 or max_rounds < 1:
        raise ValueError("trials, threads and max_rounds must be positive")
    cfg = load_config(config_path)
    if tuple(cfg["target_cols"]) != TARGETS:
        raise ValueError(f"xbank probe must use only {TARGETS}")
    configs = {variant: load_data_config(f"/app/configs/data/{variant}.yaml")
               for variant in VARIANTS}
    target_paths = {Path(value["paths"]["targets"]) for value in configs.values()}
    if len(target_paths) != 1:
        raise ValueError("both mappings must use the same xbank targets")
    targets_path = target_paths.pop()
    files = {variant: _embedding_files(model, variant) for variant in VARIANTS}
    targets, train, val, test = paired_cohort(model, targets_path, cfg, files)
    cohort_hash = hashlib.sha256(pd.util.hash_pandas_object(
        targets[[TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL, *TARGETS]],
        index=False).values.tobytes()).hexdigest()
    dev = np.sort(np.concatenate((train, val)))
    common_manifest = {
        "model": model, "targets": list(TARGETS), "target_file": _signature(targets_path),
        "cohort_sha256": cohort_hash,
        "split": {"train_rows": len(train), "val_rows": len(val),
                  "test_rows": len(test), "train_dates": cfg["train"],
                  "test_dates": cfg["test"], "val_frac": cfg["val_frac"]},
        "seed": seed, "split_seed": cfg["seed"], "trials": trials,
        "threads": threads, "max_rounds": max_rounds,
        "device_type": device, "max_bin": 63,
        "selection_metric": "validation PR-AUC",
        "test_access": "only after candidate and round selection",
    }
    params_list = candidate_params(seed, trials, threads)
    for variant in VARIANTS:
        output = downstream_dir(output_root, variant, "mbd", model) / "lightgbm_xbank_paired"
        output.mkdir(parents=True, exist_ok=True)
        manifest = {**common_manifest, "variant": variant,
                    "embedding_files": [_signature(path) for path in files[variant]]}
        manifest_path = output / "run_manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(f"existing run settings differ: {manifest_path}")
        else:
            _atomic_json(manifest_path, manifest)
        if all((output / f"{target}_metrics.json").is_file() and
               (output / f"{target}_model.txt").is_file() for target in TARGETS):
            print(f"{variant}/{model}: complete, skipping", flush=True)
            continue
        X, cols = _features(files[variant], targets)
        print(f"{variant}/{model}: features={X.shape}", flush=True)
        X_train, X_val, X_test = X[train], X[val], X[test]
        del X
        gc.collect()
        for target in TARGETS:
            metrics_path = output / f"{target}_metrics.json"
            model_path = output / f"{target}_model.txt"
            if metrics_path.is_file() and model_path.is_file():
                print(f"{variant}/{model}/{target}: complete, skipping", flush=True)
                continue
            y = targets[target].to_numpy(dtype=np.int8)
            y_train, y_val, y_test = y[train], y[val], y[test]
            if any(len(np.unique(a)) != 2 for a in (y_train, y_val, y_test)):
                raise ValueError(f"{target}: split lacks one of the two label classes")
            train_set = lgb.Dataset(X_train, label=y_train, feature_name=cols,
                                    free_raw_data=False, params={"max_bin": 63})
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set,
                                  free_raw_data=False)
            best_score, best_params, best_round = -1.0, None, None
            trial_rows = []
            for index, base in enumerate(params_list):
                params = {**base, "device_type": device, "max_bin": 63}
                if device == "gpu":
                    params.update(gpu_platform_id=gpu_platform_id,
                                  gpu_device_id=gpu_device_id, gpu_use_dp=False)
                booster = lgb.train(params, train_set, num_boost_round=max_rounds,
                                    valid_sets=[val_set], valid_names=["val"],
                                    callbacks=[lgb.early_stopping(30, verbose=False)])
                rounds = max(1, booster.best_iteration)
                score = float(average_precision_score(
                    y_val, booster.predict(X_val, num_iteration=rounds)))
                trial_rows.append({"trial": index, "val_pr_auc": score,
                                   "best_iteration": rounds, **{
                                       key: base[key] for key in (
                                           "num_leaves", "min_data_in_leaf",
                                           "learning_rate", "feature_fraction", "lambda_l2")}})
                print(f"{variant}/{model}/{target}: trial {index + 1}/{trials} "
                      f"val PR-AUC={score:.6f}, rounds={rounds}", flush=True)
                if score > best_score:
                    best_score, best_params, best_round = score, params, rounds
                del booster
            pd.DataFrame(trial_rows).to_csv(output / f"{target}_trials.csv", index=False)
            del train_set, val_set
            gc.collect()
            # Fit the selected configuration on all pre-test rows, including
            # validation, for exactly the rounds selected without using test.
            X_dev = np.concatenate((X_train, X_val), axis=0)
            y_dev = np.concatenate((y_train, y_val), axis=0)
            final = lgb.train(best_params,
                              lgb.Dataset(X_dev, label=y_dev, feature_name=cols,
                                          params={"max_bin": 63}),
                              num_boost_round=best_round)
            test_metrics = {"n_rows": int(len(y_test)),
                            "prevalence": float(y_test.mean()),
                            **{key: float(value) for key, value in
                               evaluate(y_test, final.predict(X_test)).items()}}
            temporary = model_path.with_suffix(".txt.tmp")
            final.save_model(str(temporary))
            os.replace(temporary, model_path)
            _atomic_json(metrics_path, {
                "model": model, "variant": variant, "target": target,
                "best_params": best_params, "best_rounds": best_round,
                "validation_pr_auc": best_score, "test_metrics": test_metrics,
            })
            print(f"{variant}/{model}/{target}: test PR-AUC="
                  f"{test_metrics['pr_auc']:.6f}, ROC-AUC="
                  f"{test_metrics['roc_auc']:.6f}", flush=True)
            del X_dev, y_dev, final
            gc.collect()
        del X_train, X_val, X_test
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--max-rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="/app/configs/models/downstream.yaml")
    parser.add_argument("--output-root", default="/app/data/downstream")
    parser.add_argument("--gpu-platform-id", type=int, default=0)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    args = parser.parse_args()
    run(args.model, args.device, args.trials, args.threads, args.max_rounds,
        args.seed, args.output_root, args.config, args.gpu_platform_id,
        args.gpu_device_id)


if __name__ == "__main__":
    main()
