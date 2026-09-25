"""Leakage-safe LightGBM tuning on frozen MBD raw or daily embeddings.

For each held-out test fold, the preceding fold validates hyperparameters and
early stopping; the other three folds train the candidates. The selected model
is refit on all four development folds before the test fold is read. The four
product targets are independent binary tasks, not a single multiclass label.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from data.schema import TARGET_COLS, TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL
from training.paths import downstream_dir, embedding_dir, evaluation_name, load_data_config


ALL_FOLDS = tuple(range(5))


def folds_for_test(test_fold: int) -> tuple[tuple[int, ...], int]:
    if test_fold not in ALL_FOLDS:
        raise ValueError(f"test fold must be one of {ALL_FOLDS}, got {test_fold}")
    val_fold = (test_fold - 1) % len(ALL_FOLDS)
    train_folds = tuple(fold for fold in ALL_FOLDS if fold not in (val_fold, test_fold))
    return train_folds, val_fold


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(temporary, path)


def candidate_params(seed: int, trials: int, threads: int) -> list[dict]:
    """Deterministic small search; the first candidate is the baseline."""
    if trials < 1 or threads < 1:
        raise ValueError("trials and threads must be positive")
    rng = np.random.default_rng(seed)
    choices = [(31, 300, 0.05, 1.0, 0.0)]
    space = [(leaves, leaf_min, rate, fraction, l2)
             for leaves in (15, 31, 63)
             for leaf_min in (100, 300, 1000)
             for rate in (0.03, 0.05, 0.08)
             for fraction in (0.7, 0.9, 1.0)
             for l2 in (0.0, 1.0, 10.0)]
    rng.shuffle(space)
    choices.extend(space[:trials - 1])
    return [{
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "num_threads": threads,
        "seed": seed,
        "feature_pre_filter": False,
        "num_leaves": leaves,
        "min_data_in_leaf": leaf_min,
        "learning_rate": rate,
        "feature_fraction": fraction,
        "lambda_l2": l2,
    } for leaves, leaf_min, rate, fraction, l2 in choices]


def _load_joined(targets_path: Path, embeds_dir: Path) -> tuple[pd.DataFrame, list[str], list[Path]]:
    files = sorted(embeds_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no embeddings under {embeds_dir}")
    targets = pd.read_parquet(targets_path,
                              columns=[TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL, "fold", *TARGET_COLS])
    targets[TARGETS_DATE_COL] = targets[TARGETS_DATE_COL].astype(str)
    if targets.duplicated([TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL]).any():
        raise ValueError("MBD targets contain duplicate client-date keys")
    if targets[TARGET_COLS].isna().any().any() or not targets[TARGET_COLS].isin([0, 1]).all().all():
        raise ValueError("MBD target labels must be non-null binary values")
    embeddings = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    cols = sorted((col for col in embeddings if col.startswith("emb_")),
                  key=lambda col: int(col.removeprefix("emb_")))
    if not cols or cols != [f"emb_{i}" for i in range(len(cols))]:
        raise ValueError("embedding columns must be contiguous emb_0..emb_N")
    if embeddings.duplicated(["inn", "date"]).any():
        raise ValueError("embeddings contain duplicate client-date keys")
    joined = targets.merge(embeddings, left_on=[TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL],
                           right_on=["inn", "date"], how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("no target rows match embeddings")
    if joined.groupby(TARGETS_CLIENT_ID_COL)["fold"].nunique().max() != 1:
        raise ValueError("a client occurs in multiple MBD folds")
    if not np.isfinite(joined[cols].to_numpy(dtype=np.float32)).all():
        raise ValueError("embeddings contain non-finite values")
    return joined, cols, files


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    if len(np.unique(labels)) != 2:
        raise ValueError("evaluation partition must contain both target classes")
    predicted = probabilities >= 0.5
    positive_count = int(labels.sum())
    result = {
        "n_rows": int(len(labels)),
        "prevalence": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision_at_0_5": float(precision_score(labels, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, predicted, zero_division=0)),
    }
    order = np.argsort(-probabilities, kind="stable")
    ranked = labels[order]
    for percent in (1, 5, 10):
        top_n = max(1, int(round(len(labels) * percent / 100)))
        true_positives = int(ranked[:top_n].sum())
        result[f"precision_at_top_{percent}pct"] = true_positives / top_n
        result[f"recall_at_top_{percent}pct"] = true_positives / positive_count
    return result


def run(model: str, data_config: str, output_root: str, seed: int,
        trials: int, threads: int, tune_client_cap: int, max_rounds: int,
        test_fold: int = 4, device_type: str = "cpu") -> None:
    import lightgbm as lgb

    if tune_client_cap < 1 or max_rounds < 1:
        raise ValueError("tune-client-cap and max-rounds must be positive")
    if device_type not in ("cpu", "gpu"):
        raise ValueError("device-type must be cpu or gpu")
    train_folds, val_fold = folds_for_test(test_fold)
    data_cfg = load_data_config(data_config)
    eval_name = evaluation_name(data_cfg)
    if eval_name not in ("mbd_raw", "mbd_daily"):
        raise ValueError("HPO requires MBD raw or daily embeddings and targets")
    targets_path = Path(data_cfg["paths"]["targets"])
    embeds_dir = embedding_dir("/app/data/embeds", eval_name, "mbd", model)
    model_output = downstream_dir(output_root, eval_name, "mbd", model)
    legacy_holdout = eval_name == "mbd_raw" and test_fold == 4
    output = (model_output / "lightgbm_hpo_holdout" if legacy_holdout
              else model_output / "lightgbm_hpo_cv" / f"fold{test_fold}")
    output.mkdir(parents=True, exist_ok=True)
    joined, cols, files = _load_joined(targets_path, embeds_dir)
    manifest = {
        "protocol": ("client-disjoint folds 0-2 train, 3 validation, 4 untouched test"
                     if legacy_holdout else
                     f"client-disjoint folds {train_folds} train, {val_fold} validation, "
                     f"{test_fold} untouched test"),
        "model": model,
        "targets": TARGET_COLS,
        "target_file": {"path": str(targets_path), "size": targets_path.stat().st_size,
                        "mtime_ns": targets_path.stat().st_mtime_ns},
        "embedding_files": [{"path": str(p), "size": p.stat().st_size,
                             "mtime_ns": p.stat().st_mtime_ns} for p in files],
        "seed": seed,
        "trials": trials,
        "threads": threads,
        "tune_client_cap": tune_client_cap,
        "max_rounds": max_rounds,
    }
    if not legacy_holdout:
        manifest.update({"test_fold": test_fold, "val_fold": val_fold,
                         "train_folds": list(train_folds)})
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"run settings differ from existing {manifest_path}")
    else:
        _atomic_json(manifest_path, manifest)

    if set(joined.fold.unique()) != set(ALL_FOLDS):
        raise ValueError(f"expected labeled MBD folds {ALL_FOLDS}, got {sorted(joined.fold.unique())}")
    train = joined[joined.fold.isin(train_folds)]
    val = joined[joined.fold == val_fold]
    test = joined[joined.fold == test_fold]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("train, validation and test folds must all be present")
    clients = np.sort(train[TARGETS_CLIENT_ID_COL].unique())
    rng = np.random.default_rng(seed)
    selected = rng.choice(clients, size=min(tune_client_cap, len(clients)), replace=False)
    tune = train[train[TARGETS_CLIENT_ID_COL].isin(selected)]
    print(f"{model}: joined={len(joined)} train={len(train)} tune={len(tune)} "
          f"val={len(val)} test={len(test)} features={len(cols)}", flush=True)
    X_tune = tune[cols].to_numpy(dtype=np.float32)
    X_train = train[cols].to_numpy(dtype=np.float32)
    X_val = val[cols].to_numpy(dtype=np.float32)
    X_test = test[cols].to_numpy(dtype=np.float32)
    labels = {target: (
        tune[target].to_numpy(dtype=np.int8),
        train[target].to_numpy(dtype=np.int8),
        val[target].to_numpy(dtype=np.int8),
        test[target].to_numpy(dtype=np.int8),
    ) for target in TARGET_COLS}
    del joined, train, val, test, tune, targets_path
    gc.collect()

    candidates = candidate_params(seed, trials, threads)
    if device_type == "gpu":
        candidates = [{**params, "device_type": "gpu", "gpu_platform_id": 0,
                       "gpu_device_id": 0} for params in candidates]
    for target in TARGET_COLS:
        result_path = output / f"{target}_metrics.json"
        model_path = output / f"{target}_model.txt"
        if result_path.is_file() and model_path.is_file():
            result = json.loads(result_path.read_text())
            if "precision_at_top_5pct" not in result["test_metrics"]:
                y_test = labels[target][3]
                booster = lgb.Booster(model_file=str(model_path))
                result["test_metrics"] = _metrics(y_test, booster.predict(X_test))
                result.setdefault("training_device", "cpu")
                _atomic_json(result_path, result)
                del booster
                print(f"{target}: backfilled precision/recall from saved model", flush=True)
            else:
                print(f"{target}: completed result exists, skipping", flush=True)
            continue
        y_tune, y_train, y_val, y_test = labels[target]
        if any(len(np.unique(y)) != 2 for y in (y_tune, y_train, y_val)):
            raise ValueError(f"{target}: a development fold lacks one of the binary classes")
        tune_set = lgb.Dataset(X_tune, label=y_tune, feature_name=cols, free_raw_data=False)
        val_set = lgb.Dataset(X_val, label=y_val, reference=tune_set, free_raw_data=False)
        trial_rows = []
        best_score = -1.0
        best_params = None
        for index, params in enumerate(candidates):
            booster = lgb.train(params, tune_set, num_boost_round=max_rounds,
                                valid_sets=[val_set], callbacks=[lgb.early_stopping(40, verbose=False)])
            rounds = max(1, booster.best_iteration)
            score = float(average_precision_score(y_val, booster.predict(X_val, num_iteration=rounds)))
            trial_rows.append({"trial": index, "val_pr_auc": score, "best_iteration": rounds,
                               **{key: params[key] for key in ("num_leaves", "min_data_in_leaf",
                                   "learning_rate", "feature_fraction", "lambda_l2")}})
            pd.DataFrame(trial_rows).to_csv(output / f"{target}_trials.csv", index=False)
            print(f"{model} {target} trial={index + 1}/{len(candidates)} "
                  f"val_pr_auc={score:.6f} rounds={rounds}", flush=True)
            if score > best_score:
                best_score, best_params = score, params
            del booster
        del tune_set, val_set
        gc.collect()

        full_train_set = lgb.Dataset(X_train, label=y_train, feature_name=cols, free_raw_data=False)
        full_val_set = lgb.Dataset(X_val, label=y_val, reference=full_train_set, free_raw_data=False)
        selected_model = lgb.train(best_params, full_train_set, num_boost_round=max_rounds,
                                   valid_sets=[full_val_set], callbacks=[lgb.early_stopping(40, verbose=False)])
        final_rounds = max(1, selected_model.best_iteration)
        final_val = _metrics(y_val, selected_model.predict(X_val, num_iteration=final_rounds))
        del selected_model, full_train_set, full_val_set
        gc.collect()

        X_dev = np.concatenate([X_train, X_val], axis=0)
        y_dev = np.concatenate([y_train, y_val], axis=0)
        final_model = lgb.train(best_params, lgb.Dataset(X_dev, label=y_dev, feature_name=cols),
                                num_boost_round=final_rounds)
        test_metrics = _metrics(y_test, final_model.predict(X_test))
        final_model.save_model(str(model_path))
        _atomic_json(result_path, {
            "target": target,
            "training_device": device_type,
            "best_params": best_params,
            "tune_val_pr_auc": best_score,
            "full_val_metrics": final_val,
            "final_rounds": final_rounds,
            "test_metrics": test_metrics,
            "test_access": "only after all hyperparameters and final rounds were selected",
        })
        print(f"{model} {target}: test PR-AUC={test_metrics['pr_auc']:.6f} "
              f"ROC-AUC={test_metrics['roc_auc']:.6f}", flush=True)
        del X_dev, y_dev, final_model
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["coles", "cotic", "thp", "nep", "mlm", "chronos2"])
    parser.add_argument("--data-config", default="/app/configs/data/mbd.yaml")
    parser.add_argument("--output-root", default="/app/data/downstream")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tune-client-cap", type=int, default=100000)
    parser.add_argument("--max-rounds", type=int, default=600)
    parser.add_argument("--test-fold", type=int, choices=ALL_FOLDS, default=4,
                        help="held-out test fold; raw fold 4 reuses the existing holdout outputs")
    parser.add_argument("--device-type", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    run(args.model, args.data_config, args.output_root, args.seed, args.trials,
        args.threads, args.tune_client_cap, args.max_rounds, args.test_fold,
        args.device_type)


if __name__ == "__main__":
    main()
