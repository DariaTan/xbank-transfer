"""Memory-bounded, resumable LightGBM probes of frozen Chronos-2 embeddings.

The 768-feature MBD panel is streamed into a temporary float32 disk cache.
LightGBM Sequence builds bins in batches, without a full pandas feature panel
or train/validation/test feature copies. Only PR-AUC (average precision) and
ROC-AUC are evaluated. HPO, early stopping and final refit follow the existing
MBD five-fold / xbank calendar protocols; test scores never select a model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import mmap
import os
import shutil
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from training.paths import downstream_dir, embedding_dir, evaluation_name, load_data_config
from training.tune_lightgbm_mbd import candidate_params, folds_for_test


MODEL = "chronos2"
FORMAT_VERSION = 1
KEYS = ["id", "col_1"]


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def check_manifest(path: Path, expected: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"source or settings differ from {path}")
    else:
        atomic_json(path, expected)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    if not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("evaluation needs finite scores and both binary classes")
    return {"n_rows": len(labels), "n_positive": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "pr_auc": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores))}


def _release_pages(values: np.memmap) -> None:
    # Copies returned to LightGBM own their memory; mmap pages can be released
    # under the small server cgroup instead of retaining a 34-GB working set.
    if hasattr(values._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        values._mmap.madvise(mmap.MADV_DONTNEED)


class IndexedSequence(lgb.Sequence):
    def __init__(self, values: np.memmap, rows: np.ndarray, batch_size: int = 8192):
        self.values = values
        self.rows = np.asarray(rows, dtype=np.int64)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        # Sequence's random-access binning samples must be float64 in the
        # LightGBM C API; streamed training/prediction batches remain float32.
        dtype = np.float64 if np.isscalar(index) else np.float32
        result = np.array(self.values[self.rows[index]], dtype=dtype, copy=True)
        if isinstance(index, slice):
            _release_pages(self.values)
        return result


def predict_batches(booster: lgb.Booster, sequence: IndexedSequence) -> np.ndarray:
    scores = np.empty(len(sequence), dtype=np.float64)
    for start in range(0, len(sequence), sequence.batch_size):
        stop = min(start + sequence.batch_size, len(sequence))
        scores[start:stop] = booster.predict(sequence[start:stop])
    return scores


def load_targets(path: Path, target_cols: list[str], is_mbd: bool) -> pd.DataFrame:
    columns = KEYS + target_cols + (["fold"] if is_mbd else [])
    targets = pd.read_parquet(path, columns=columns)
    if targets[columns].isna().any().any():
        raise ValueError("target keys/labels/folds contain nulls")
    targets["id"] = targets["id"].astype(str)
    targets["col_1"] = targets["col_1"].astype(str)
    if not targets[target_cols].isin([0, 1]).all().all():
        raise ValueError("selected target labels must be binary")
    duplicates = targets.duplicated(KEYS, keep=False)
    if duplicates.any():
        if is_mbd or targets.loc[duplicates].groupby(KEYS)[target_cols].nunique().gt(1).any().any():
            raise ValueError("duplicate/conflicting selected client-date targets")
        targets = targets.drop_duplicates(KEYS).copy()
    if is_mbd:
        if set(targets.fold.unique()) != set(range(5)):
            raise ValueError("MBD requires exactly client folds 0..4")
        if targets.groupby("id").fold.nunique().max() != 1:
            raise ValueError("a client occurs in multiple MBD folds")
    return targets


def prepare_cache(cache: Path, targets_path: Path, files: list[Path],
                  target_cols: list[str], is_mbd: bool, batch_size: int = 8192) -> dict:
    source = {"format_version": FORMAT_VERSION, "target_file": signature(targets_path),
              "embedding_files": [signature(p) for p in files], "targets": target_cols,
              "is_mbd": is_mbd, "dtype": "float32"}
    if cache.exists():
        saved = json.loads((cache / "manifest.json").read_text())
        if saved["source"] != source:
            raise ValueError(f"existing feature cache has different inputs: {cache}")
        expected_bytes = saved["capacity"] * saved["n_features"] * 4
        if (cache / "features.f32").stat().st_size != expected_bytes:
            raise ValueError("feature cache is incomplete")
        return saved
    targets = load_targets(targets_path, target_cols, is_mbd)
    dates = sorted(targets.col_1.unique())
    if {p.stem for p in files} != set(dates):
        raise ValueError("published Chronos dates differ from target dates")
    feature_cols = [c for c in pq.ParquetFile(files[0]).schema_arrow.names if c.startswith("emb_")]
    if feature_cols != [f"emb_{i}" for i in range(len(feature_cols))] or not feature_cols:
        raise ValueError("feature columns must be contiguous emb_0..emb_N")
    capacity = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chronos-feature-build-", dir=cache.parent) as directory:
        temporary = Path(directory)
        values = np.memmap(temporary / "features.f32", dtype=np.float32, mode="w+",
                           shape=(capacity, len(feature_cols)))
        offset, digest, metadata = 0, hashlib.sha256(), []
        for path in files:
            parquet = pq.ParquetFile(path)
            if [c for c in parquet.schema_arrow.names if c.startswith("emb_")] != feature_cols:
                raise ValueError(f"inconsistent feature schema: {path}")
            labeled = targets[targets.col_1 == path.stem].set_index("id", drop=False)
            seen = set()
            for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
                keys = batch.select(["inn", "date"]).to_pandas()
                if keys.isna().any().any() or not keys.date.astype(str).eq(path.stem).all():
                    raise ValueError(f"invalid client/date keys: {path}")
                ids = keys.inn.astype(str)
                if ids.duplicated().any() or not seen.isdisjoint(ids):
                    raise ValueError(f"duplicate Chronos client-date keys: {path}")
                seen.update(ids)
                joined_rows = labeled.index.get_indexer(ids)
                keep = joined_rows >= 0
                features = np.column_stack([batch.column(batch.schema.get_field_index(c)).to_numpy()
                                            for c in feature_cols]).astype(np.float32, copy=False)
                if not np.isfinite(features).all():
                    raise ValueError(f"non-finite embedding values: {path}")
                frame = labeled.iloc[joined_rows[keep]].reset_index(drop=True)
                count = len(frame)
                values[offset:offset + count] = features[keep]
                metadata.append(frame)
                digest.update(pd.util.hash_pandas_object(frame[KEYS], index=False).values.tobytes())
                offset += count
                values.flush()
                _release_pages(values)
            print(f"cache {path.stem}: cumulative labeled rows={offset}", flush=True)
        if offset == 0:
            raise ValueError("Chronos embeddings match no targets")
        pd.concat(metadata, ignore_index=True).to_parquet(temporary / "rows.parquet", index=False)
        del values
        saved = {"source": source, "capacity": capacity, "n_rows": offset,
                 "n_features": len(feature_cols), "feature_cols": feature_cols,
                 "cohort_sha256": digest.hexdigest()}
        atomic_json(temporary / "manifest.json", saved)
        os.replace(temporary, cache)
    return saved


def split_rows(rows: pd.DataFrame, is_mbd: bool, test_fold: int, probe: dict,
               seed: int, tune_client_cap: int) -> tuple[np.ndarray, ...]:
    if is_mbd:
        train_folds, val_fold = folds_for_test(test_fold)
        train = np.flatnonzero(rows.fold.isin(train_folds))
        val = np.flatnonzero(rows.fold.eq(val_fold))
        test = np.flatnonzero(rows.fold.eq(test_fold))
        clients = np.sort(rows.iloc[train].id.unique())
        selected = np.random.default_rng(seed).choice(clients, min(tune_client_cap, len(clients)),
                                                      replace=False)
        tune = train[rows.iloc[train].id.isin(selected).to_numpy()]
    else:
        train_dates = pd.period_range(probe["train"]["start"], probe["train"]["end"], freq="M").astype(str)
        test_dates = pd.period_range(probe["test"]["start"], probe["test"]["end"], freq="M").astype(str)
        if max(train_dates) >= min(test_dates):
            raise ValueError("xbank test must be strictly later than training")
        months = rows.col_1.str[:7]
        pool = np.flatnonzero(months.isin(train_dates))
        test = np.flatnonzero(months.isin(test_dates))
        splitter = GroupShuffleSplit(n_splits=1, test_size=probe["val_frac"], random_state=probe["seed"])
        a, b = next(splitter.split(rows.iloc[pool], groups=rows.iloc[pool].id))
        train, val = pool[a], pool[b]
        tune = train
    if min(map(len, (train, val, test, tune))) == 0:
        raise ValueError("empty train/validation/test/tuning partition")
    if set(rows.iloc[train].id) & set(rows.iloc[val].id):
        raise ValueError("client leaked across training and validation")
    dev = np.sort(np.concatenate((train, val)))
    if np.intersect1d(dev, test).size:
        raise ValueError("test rows overlap development rows")
    return train, val, test, tune, dev


def _dataset(sequence: IndexedSequence, labels: np.ndarray, params: dict,
             columns: list[str], reference=None) -> lgb.Dataset:
    return lgb.Dataset(sequence, label=labels, feature_name=columns, reference=reference,
                       params=params, free_raw_data=True).construct()


def run(data_config: str, downstream_config: str, test_fold: int, device: str,
        seed: int, trials: int, threads: int, tune_client_cap: int,
        max_rounds: int, max_bin: int) -> None:
    cfg = load_data_config(data_config)
    eval_name = evaluation_name(cfg)
    if eval_name not in ("mbd_raw", "mbd_daily", "xbank"):
        raise ValueError(f"unsupported evaluation: {eval_name}")
    is_mbd = eval_name.startswith("mbd_")
    probe = yaml.safe_load(Path(downstream_config).read_text())["probe"]
    target_cols = ["col_2", "col_3", "col_4", "col_5"] if is_mbd else list(probe["target_cols"])
    if not is_mbd and target_cols != ["col_3", "col_4", "col_5"]:
        raise ValueError("xbank Chronos must exclude conflicting target col_2")
    output_root = downstream_dir("/app/data/downstream", eval_name, "mbd", MODEL)
    output = output_root / "lightgbm_hpo_cv" / f"fold{test_fold}" if is_mbd else output_root / "lightgbm_hpo_calendar"
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(embedding_dir("/app/data/embeds", eval_name, "mbd", MODEL).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no published Chronos embeddings for {eval_name}")
    base_manifest = {"format_version": FORMAT_VERSION, "model": MODEL,
                     "evaluation_name": eval_name, "target_file": signature(Path(cfg["paths"]["targets"])),
                     "embedding_files": [signature(p) for p in files], "targets": target_cols,
                     "seed": seed, "trials": trials, "threads": threads,
                     "tune_client_cap": tune_client_cap, "max_rounds": max_rounds,
                     "device_type": device, "max_bin": max_bin,
                     "class_weighting": "none, matching existing HPO probes",
                     "negative_sampling": "none", "selection_metric": "validation average precision",
                     "reported_metrics": ["pr_auc", "roc_auc"],
                     "test_fold": test_fold if is_mbd else None,
                     "calendar_probe": None if is_mbd else probe}
    check_manifest(output / "run_manifest.json", base_manifest)
    if all((output / f"{t}_metrics.json").exists() and (output / f"{t}_model.txt").exists()
           for t in target_cols):
        print(f"{eval_name} fold={test_fold}: complete, skipping", flush=True)
        return
    cache = output_root / "_feature_cache"
    saved = prepare_cache(cache, Path(cfg["paths"]["targets"]), files, target_cols, is_mbd)
    rows = pd.read_parquet(cache / "rows.parquet")
    values = np.memmap(cache / "features.f32", dtype=np.float32, mode="r",
                       shape=(saved["capacity"], saved["n_features"]))
    train, val, test, tune, dev = split_rows(rows, is_mbd, test_fold, probe, seed, tune_client_cap)
    atomic_json(output / "cohort.json", {"cohort_sha256": saved["cohort_sha256"],
                "n_features": saved["n_features"], "n_rows": len(rows),
                "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
                "tune_rows": len(tune), "missing_configured_train_months": ([] if is_mbd else
                    sorted(set(pd.period_range(probe["train"]["start"], probe["train"]["end"], freq="M").astype(str))
                           - set(rows.col_1.str[:7])))} )
    print(f"{eval_name} fold={test_fold}: features={saved['n_features']} train={len(train)} "
          f"tune={len(tune)} val={len(val)} test={len(test)} device={device}", flush=True)
    labels = {t: rows[t].to_numpy(dtype=np.int8) for t in target_cols}
    del rows
    gc.collect()
    columns = saved["feature_cols"]
    data_params = {"max_bin": max_bin, "feature_pre_filter": False, "num_threads": threads,
                   "data_random_seed": seed}
    candidates = [{**p, **data_params, "device_type": device}
                  for p in candidate_params(seed, trials, threads)]
    if device == "gpu":
        for p in candidates:
            p.update(gpu_platform_id=0, gpu_device_id=0, gpu_use_dp=False)
    sequences = {name: IndexedSequence(values, ids) for name, ids in
                 (("train", train), ("val", val), ("test", test), ("tune", tune), ("dev", dev))}
    selections = {}
    pending = [t for t in target_cols if not ((output / f"{t}_metrics.json").exists()
                                            and (output / f"{t}_model.txt").exists())]
    for target in pending:
        path = output / f"{target}_selection.json"
        if path.exists():
            selections[target] = json.loads(path.read_text())
    need_hpo = [t for t in pending if t not in selections]
    if need_hpo:
        initial = labels[need_hpo[0]]
        train_set = _dataset(sequences["tune"], initial[tune], data_params, columns)
        val_set = _dataset(sequences["val"], initial[val], data_params, columns, train_set)
        for target in need_hpo:
            y = labels[target]
            for partition in (tune, train, val):
                if set(np.unique(y[partition])) != {0, 1}:
                    raise ValueError(f"{target}: development partition lacks a class")
            train_set.set_label(y[tune]); val_set.set_label(y[val])
            trial_rows, best = [], None
            for i, params in enumerate(candidates):
                booster = lgb.train(params, train_set, num_boost_round=max_rounds,
                                    valid_sets=[val_set], callbacks=[lgb.early_stopping(40 if is_mbd else 30, verbose=False)])
                score = float(booster.best_score["valid_0"]["average_precision"])
                rounds = max(1, booster.best_iteration)
                trial_rows.append({"trial": i, "val_pr_auc": score, "best_iteration": rounds})
                if best is None or score > best["tune_val_pr_auc"]:
                    best = {"best_params": params, "tune_val_pr_auc": score, "tune_rounds": rounds}
                print(f"{eval_name} fold={test_fold} {target} trial={i+1}/{trials} "
                      f"val PR-AUC={score:.6f} rounds={rounds}", flush=True)
                del booster
            pd.DataFrame(trial_rows).to_csv(output / f"{target}_trials.csv", index=False)
            if not is_mbd:
                best["final_rounds"] = best["tune_rounds"]
            selections[target] = best
            atomic_json(output / f"{target}_selection.json", best)
        del train_set, val_set
        gc.collect()
    need_rounds = [t for t in pending if "final_rounds" not in selections[t]]
    if need_rounds:
        y = labels[need_rounds[0]]
        train_set = _dataset(sequences["train"], y[train], data_params, columns)
        val_set = _dataset(sequences["val"], y[val], data_params, columns, train_set)
        for target in need_rounds:
            y = labels[target]
            train_set.set_label(y[train]); val_set.set_label(y[val])
            booster = lgb.train(selections[target]["best_params"], train_set,
                                num_boost_round=max_rounds, valid_sets=[val_set],
                                callbacks=[lgb.early_stopping(40, verbose=False)])
            selections[target]["final_rounds"] = max(1, booster.best_iteration)
            selections[target]["full_val_pr_auc"] = float(booster.best_score["valid_0"]["average_precision"])
            atomic_json(output / f"{target}_selection.json", selections[target])
            print(f"{eval_name} {target}: full-train selected rounds={booster.best_iteration}", flush=True)
            del booster
        del train_set, val_set
        gc.collect()
    if pending:
        dev_set = _dataset(sequences["dev"], labels[pending[0]][dev], data_params, columns)
        for target in pending:
            selection = selections[target]
            dev_set.set_label(labels[target][dev])
            booster = lgb.train(selection["best_params"], dev_set,
                                num_boost_round=selection["final_rounds"])
            scores = predict_batches(booster, sequences["test"])
            result = metrics(labels[target][test], scores)
            tmp = output / f"{target}_model.txt.tmp"
            booster.save_model(str(tmp))
            os.replace(tmp, output / f"{target}_model.txt")
            atomic_json(output / f"{target}_metrics.json", {"target": target, **selection,
                        "training_device": device, "test_metrics": result,
                        "test_access": "scores evaluated only after HPO and round selection"})
            print(f"{eval_name} fold={test_fold} {target}: test PR-AUC={result['pr_auc']:.6f} "
                  f"ROC-AUC={result['roc_auc']:.6f}", flush=True)
            del booster, scores
        del dev_set
    del values
    gc.collect()


def summarize(eval_name: str, cleanup_cache: bool = False) -> None:
    root = downstream_dir("/app/data/downstream", eval_name, "mbd", MODEL)
    is_mbd = eval_name.startswith("mbd_")
    records, reference = [], None
    for fold in range(5) if is_mbd else [None]:
        directory = root / "lightgbm_hpo_cv" / f"fold{fold}" if is_mbd else root / "lightgbm_hpo_calendar"
        manifest = json.loads((directory / "run_manifest.json").read_text())
        settings = {k: v for k, v in manifest.items() if k != "test_fold"}
        if reference is not None and reference != settings:
            raise ValueError("folds use different inputs or HPO settings")
        reference = settings
        for target in manifest["targets"]:
            if not (directory / f"{target}_model.txt").exists():
                raise ValueError(f"missing saved model for {target}")
            result = json.loads((directory / f"{target}_metrics.json").read_text())
            records.append({"model": MODEL, "evaluation_name": eval_name, "target": target,
                            "test_fold": fold, **result["test_metrics"]})
    rows = pd.DataFrame(records)
    aggregate = rows.groupby(["model", "target"], sort=True).agg(
        n_evaluations=("pr_auc", "size"), n_test_rows_total=("n_rows", "sum"),
        prevalence_mean=("prevalence", "mean"), pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"), roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std")).reset_index()
    macro = aggregate.groupby("model", sort=True).agg(
        mean_pr_auc_across_targets=("pr_auc_mean", "mean"),
        mean_roc_auc_across_targets=("roc_auc_mean", "mean")).reset_index()
    for name, frame in (("results_all_folds.csv", rows), ("results_aggregated.csv", aggregate),
                        ("model_macro.csv", macro)):
        tmp = root / (name + ".tmp")
        frame.to_csv(tmp, index=False)
        os.replace(tmp, root / name)
    print(aggregate.to_string(index=False), flush=True)
    print(macro.to_string(index=False), flush=True)
    cache = root / "_feature_cache"
    if cleanup_cache and cache.is_dir():
        saved = json.loads((cache / "manifest.json").read_text())
        if saved["source"]["format_version"] != FORMAT_VERSION:
            raise ValueError("refusing to remove an unknown cache")
        shutil.rmtree(cache)
        print(f"Removed temporary feature cache: {cache}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="/app/configs/data/mbd.yaml")
    parser.add_argument("--downstream-config", default="/app/configs/models/downstream_mbd.yaml")
    parser.add_argument("--test-fold", type=int, choices=range(5), default=0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tune-client-cap", type=int, default=50000)
    parser.add_argument("--max-rounds", type=int, default=400)
    parser.add_argument("--max-bin", type=int, default=63)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--cleanup-cache", action="store_true")
    args = parser.parse_args()
    if min(args.trials, args.threads, args.tune_client_cap, args.max_rounds, args.max_bin) < 1:
        parser.error("resource/search limits must be positive")
    if args.summarize:
        summarize(evaluation_name(load_data_config(args.data_config)), args.cleanup_cache)
    else:
        run(args.data_config, args.downstream_config, args.test_fold, args.device,
            args.seed, args.trials, args.threads, args.tune_client_cap, args.max_rounds, args.max_bin)


if __name__ == "__main__":
    main()
