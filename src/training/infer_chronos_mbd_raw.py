"""Resumable, two-GPU Chronos-2 inference on raw MBD or matched xbank.

Prepare scans the raw adapted parquet once and aggregates every labeled
client's transactions by calendar day. Workers read disjoint cached shards,
embed each monthly history, and publish per-shard parquet files. Finalize
checks every shard and atomically publishes one file per target date.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import duckdb
import numpy as np
import pandas as pd
import torch
import yaml

from data.schema import CLIENT_ID_COL, EVENT_TIME_COL, TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL
from cross_schema.apply_mapping import FrozenSchemaMapping
from training.embedding_io import atomic_parquet, validate_embedding_frame
from training.paths import data_root, embedding_dir, evaluation_name, load_data_config, resolve_data_path


MODEL = "chronos2"
FORMAT_VERSION = 1


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _dates(targets_path: Path) -> list[str]:
    dates = pd.read_parquet(targets_path, columns=[TARGETS_DATE_COL])[TARGETS_DATE_COL]
    return sorted(dates.astype(str).unique().tolist())


def _manifest(transactions: Path, targets: Path, dates: list[str], months: int, shards: int,
              value_col: str = "col_11", mapping_sha256: str | None = None) -> dict:
    manifest = {
        "format_version": FORMAT_VERSION,
        "transactions": str(transactions),
        "transactions_size": transactions.stat().st_size,
        "transactions_mtime_ns": transactions.stat().st_mtime_ns,
        "targets": str(targets),
        "targets_size": targets.stat().st_size,
        "targets_mtime_ns": targets.stat().st_mtime_ns,
        "target_dates": dates,
        "history_window_months": months,
        "n_shards": shards,
        "value_col": value_col,
        "day_cutoff": "strictly_before_target_date",
    }
    if mapping_sha256 is not None:
        manifest["schema_mapping_sha256"] = mapping_sha256
    return manifest


def _check_manifest(path: Path, expected: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest: {path}; run prepare first")
    saved = json.loads(path.read_text())
    if saved != expected:
        raise ValueError(f"run configuration differs from {path}: saved={saved}, expected={expected}")


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, path)


def prepare(cache_dir: Path, out_dir: Path, manifest: dict) -> None:
    cache_manifest = cache_dir / "manifest.json"
    if cache_dir.exists():
        _check_manifest(cache_manifest, manifest)
        print(f"Validated existing daily cache: {cache_dir}", flush=True)
    else:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="chronos2-prepare-", dir=cache_dir.parent) as tmp:
            temporary = Path(tmp)
            shards_dir = temporary / "shards"
            duckdb_tmp = data_root() / "duckdb_tmp" / "chronos2"
            duckdb_tmp.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect()
            con.execute("SET threads = 16")
            con.execute("SET memory_limit = '40GB'")
            con.execute(f"SET temp_directory = '{_sql_path(duckdb_tmp)}'")
            con.execute(
                f"CREATE TEMP TABLE target_clients AS SELECT DISTINCT {TARGETS_CLIENT_ID_COL} AS {CLIENT_ID_COL} "
                f"FROM read_parquet('{_sql_path(Path(manifest['targets']))}')"
            )
            first_day = pd.Timestamp(manifest["target_dates"][0]) - pd.DateOffset(
                months=manifest["history_window_months"]
            )
            last_day = manifest["target_dates"][-1]
            print(
                f"Aggregating raw MBD to client/day: {first_day.date()}..{last_day}, "
                f"{manifest['n_shards']} shards",
                flush=True,
            )
            con.execute(
                f"""
                COPY (
                    SELECT CAST(hash(tx.{CLIENT_ID_COL}) % {manifest['n_shards']} AS INTEGER) AS shard,
                           tx.{CLIENT_ID_COL} AS {CLIENT_ID_COL},
                           CAST(tx.{EVENT_TIME_COL} AS DATE) AS day,
                           SUM(CAST(tx.{manifest['value_col']} AS DOUBLE)) AS value
                    FROM read_parquet('{_sql_path(Path(manifest['transactions']))}') AS tx
                    SEMI JOIN target_clients AS tc USING ({CLIENT_ID_COL})
                    WHERE tx.{EVENT_TIME_COL} > DATE '{first_day.date()}'
                      AND tx.{EVENT_TIME_COL} < DATE '{last_day}'
                    GROUP BY 1, 2, 3
                ) TO '{_sql_path(shards_dir)}'
                (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (shard))
                """
            )
            con.close()
            _write_manifest(temporary / "manifest.json", manifest)
            os.replace(temporary, cache_dir)
            print(f"Published daily cache: {cache_dir}", flush=True)

    out_manifest = out_dir / "run_manifest.json"
    if out_manifest.exists():
        _check_manifest(out_manifest, manifest)
    else:
        _write_manifest(out_manifest, manifest)


def _client_histories(shard_dir: Path) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    if not shard_dir.is_dir():
        return
    daily = pd.read_parquet(shard_dir, columns=[CLIENT_ID_COL, "day", "value"])
    daily["day"] = pd.to_datetime(daily["day"])
    daily.sort_values([CLIENT_ID_COL, "day"], inplace=True)
    for client_id, group in daily.groupby(CLIENT_ID_COL, sort=False):
        yield (
            str(client_id),
            group["day"].to_numpy(dtype="datetime64[D]"),
            group["value"].to_numpy(dtype=np.float32),
        )


def _series_for_date(days: np.ndarray, values: np.ndarray, date: str, months: int) -> np.ndarray | None:
    cutoff = pd.Timestamp(date)
    start = cutoff - pd.DateOffset(months=months)
    lower = np.datetime64(start.date(), "D")
    upper = np.datetime64(cutoff.date(), "D")
    first = np.searchsorted(days, lower, side="right")
    last = np.searchsorted(days, upper, side="left")
    if first == last:
        return None
    length = int((upper - lower) / np.timedelta64(1, "D")) - 1
    series = np.zeros(length, dtype=np.float32)
    positions = ((days[first:last] - lower) / np.timedelta64(1, "D")).astype(int) - 1
    series[positions] = values[first:last]
    return series


def _embed_date(
    histories: list[tuple[str, np.ndarray, np.ndarray]],
    date: str,
    months: int,
    pipeline,
    model_batch_size: int,
    client_batch_size: int,
) -> pd.DataFrame | None:
    from models.chronos2 import extract_reg_embeddings

    all_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    batch_ids: list[str] = []
    batch_series: list[np.ndarray] = []

    def flush() -> None:
        if not batch_ids:
            return
        embeddings = extract_reg_embeddings(pipeline, batch_series, batch_size=model_batch_size)
        if len(embeddings) != len(batch_ids):
            raise ValueError("Chronos returned a different number of embeddings than clients")
        all_ids.extend(batch_ids)
        all_embeddings.append(np.asarray(embeddings, dtype=np.float32))
        batch_ids.clear()
        batch_series.clear()

    for client_id, days, values in histories:
        series = _series_for_date(days, values, date, months)
        if series is None:
            continue
        batch_ids.append(client_id)
        batch_series.append(series)
        if len(batch_ids) >= client_batch_size:
            flush()
    flush()
    if not all_ids:
        return None
    embeddings = np.concatenate(all_embeddings, axis=0)
    frame = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    frame.insert(0, "date", date)
    frame.insert(0, "inn", all_ids)
    validate_embedding_frame(frame, expected_date=date)
    return frame


def worker(
    cache_dir: Path,
    out_dir: Path,
    manifest: dict,
    worker_index: int,
    workers: int,
    batch_size: int,
    client_batch_size: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable inside container; refusing silent CPU fallback")
    from models.chronos2 import load_pipeline

    pipeline = load_pipeline(device_map="cuda")
    print(f"Chronos ready on {torch.cuda.get_device_name(0)}; worker {worker_index}/{workers}", flush=True)
    for shard in range(worker_index, manifest["n_shards"], workers):
        pending_dates = [
            date for date in manifest["target_dates"]
            if not (out_dir / f"{date}.parquet").is_file()
            and not (out_dir / "_shards" / date / f"part_{shard:03d}.parquet").exists()
            and not (out_dir / "_shards" / date / f"part_{shard:03d}.empty").exists()
        ]
        if not pending_dates:
            continue
        shard_dir = cache_dir / "shards" / f"shard={shard}"
        histories = list(_client_histories(shard_dir))
        print(f"shard {shard}: {len(histories)} clients, {len(pending_dates)} dates", flush=True)
        for date in pending_dates:
            part_dir = out_dir / "_shards" / date
            part_dir.mkdir(parents=True, exist_ok=True)
            frame = _embed_date(
                histories, date, manifest["history_window_months"], pipeline,
                batch_size, client_batch_size,
            )
            if frame is None:
                (part_dir / f"part_{shard:03d}.empty").touch()
                print(f"{date}: shard {shard} has no history", flush=True)
            else:
                atomic_parquet(frame, part_dir / f"part_{shard:03d}.parquet")
                print(f"{date}: shard {shard} wrote {len(frame)} embeddings", flush=True)


def finalize(out_dir: Path, manifest: dict) -> None:
    for date in manifest["target_dates"]:
        path = out_dir / f"{date}.parquet"
        if path.is_file():
            print(f"{date}: published output already exists; skipping", flush=True)
            continue
        part_dir = out_dir / "_shards" / date
        parts: list[Path] = []
        for shard in range(manifest["n_shards"]):
            part = part_dir / f"part_{shard:03d}.parquet"
            if part.is_file():
                parts.append(part)
            elif not (part_dir / f"part_{shard:03d}.empty").is_file():
                raise RuntimeError(f"{date}: missing shard {shard}")
        if not parts:
            raise RuntimeError(f"{date}: no client histories")
        frame = pd.concat((pd.read_parquet(part) for part in parts), ignore_index=True)
        validate_embedding_frame(frame, expected_date=date)
        atomic_parquet(frame, path)
        shutil.rmtree(part_dir)
        print(f"{date}: published {len(frame)} embeddings to {path}", flush=True)
    shards_root = out_dir / "_shards"
    if shards_root.is_dir() and not any(shards_root.iterdir()):
        shards_root.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "worker", "finalize"])
    parser.add_argument("--data-config", default="/app/configs/data/mbd.yaml")
    parser.add_argument("--downstream-config", default="/app/configs/models/downstream_mbd.yaml")
    parser.add_argument("--n-shards", type=int, default=32)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.n_shards <= 0 or args.workers <= 0 or not 0 <= args.worker_index < args.workers:
        parser.error("n-shards and workers must be positive; worker-index must be in range")

    data_cfg = load_data_config(args.data_config)
    eval_name = evaluation_name(data_cfg)
    if eval_name not in {"mbd_raw", "mbd_raw_smoke", "xbank"}:
        parser.error("Chronos raw runner requires MBD-raw or xbank data")
    with open(args.downstream_config) as file:
        inf = yaml.safe_load(file)["inference"]
    transactions = Path(data_cfg["paths"]["transactions"])
    targets = Path(data_cfg["paths"]["targets"])
    dates = _dates(targets)
    value_col = "col_11"
    mapping_sha256 = None
    if eval_name == "xbank":
        mapping_path = data_cfg.get("schema_mapping")
        if not mapping_path:
            parser.error("xbank Chronos requires schema_mapping from the cross-schema builder")
        mapping = FrozenSchemaMapping(resolve_data_path(mapping_path))
        matched_amount = [source for source, field in mapping.mapping.items() if field == "amount"]
        if len(matched_amount) != 1:
            parser.error("xbank Chronos requires exactly one field matched to MBD amount")
        value_col = matched_amount[0]
        mapping_sha256 = mapping.sha256
    manifest = _manifest(transactions, targets, dates, inf["history_window_months"], args.n_shards,
                         value_col=value_col, mapping_sha256=mapping_sha256)
    out_dir = embedding_dir(inf["embeds_dir"], eval_name, "mbd", MODEL)
    cache_dir = data_root() / "chronos2_daily_cache" / eval_name

    if args.phase == "prepare":
        prepare(cache_dir, out_dir, manifest)
    else:
        _check_manifest(cache_dir / "manifest.json", manifest)
        _check_manifest(out_dir / "run_manifest.json", manifest)
        if args.phase == "worker":
            worker(
                cache_dir, out_dir, manifest, args.worker_index, args.workers,
                inf["chronos2_batch_size"], inf.get("chronos2_client_batch_size", 2000),
            )
        else:
            finalize(out_dir, manifest)


if __name__ == "__main__":
    main()
