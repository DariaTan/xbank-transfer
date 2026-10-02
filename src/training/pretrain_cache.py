"""Shared, train-only-fitted, disk-backed cache for the new daily arm.

No labels are read. The legacy NEP and raw training paths are not used or
rewritten. Partition once, cap within DuckDB, fit on all retained training
events, then store compact mmap arrays. Preparation resumes per partition.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from data.schema import ALL_FEATURE_COLS, CATEGORY_COLS, NUMERIC_COLS
from ptls.preprocessing import PandasDataPreprocessor
from ptls.preprocessing.pandas.pandas_transformation.pandas_freq_transformer import FrequencyEncoder
from training.common import save_preprocessor
from training.paths import data_root, load_data_config


VERSION = 2
ARRAYS = ("category.npy", "numeric.npy", "time.npy", "mark.npy", "offsets.npy")


def category_keys(values: pd.Series) -> pd.Series:
    # Nullable integer parquet columns may arrive as int in one Arrow batch
    # and float in another. Their category identity must not be "7" vs "7.0".
    return values.astype("string").str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True).fillna("<MISSING>")


class CanonicalFrequencyEncoder(FrequencyEncoder):
    """Stored inside NEW preprocessors only; old checkpoints are unchanged."""
    def transform(self, frame):
        canonical = frame.copy(deep=False)
        canonical[self.col_name_original] = category_keys(frame[self.col_name_original])
        return super().transform(canonical)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_identity(path: Path) -> dict:
    # Detect ordinary file replacements and metadata changes without claiming
    # that a footer hash is a full-file content hash.
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        footer = hashlib.sha256(stream.read()).hexdigest()
    return {"path": str(path.resolve()), **signature(path), "tail_sha256": footer}


def checked_json(path: Path, expected: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"cache/run settings or source changed: {path}")
    else:
        atomic_json(path, expected)


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def partition_files(root: Path, bucket: int) -> list[Path]:
    return sorted((root / f"bucket={bucket}").glob("*.parquet"))


def fixed_split(clients: pd.DataFrame, seed: int, valid_frac: float) -> pd.DataFrame:
    if not 0 < valid_frac < 1 or len(clients) < 2:
        raise ValueError("pretraining needs at least two clients and 0 < valid_frac < 1")
    clients = clients.sort_values("id").reset_index(drop=True).copy()
    if clients.id.duplicated().any():
        raise ValueError("a client appears in multiple cache partitions")
    n_valid = min(len(clients) - 1, max(1, int(len(clients) * valid_frac)))
    clients["valid"] = False
    clients.loc[np.random.RandomState(seed).permutation(len(clients))[:n_valid], "valid"] = True
    return clients.sort_values(["bucket", "id"]).reset_index(drop=True)


def preprocessor_from_counts(counts: dict[str, Counter]) -> PandasDataPreprocessor:
    preprocessor = PandasDataPreprocessor(
        col_id="id", col_event_time="event_time", event_time_transformation="none",
        cols_category=[CanonicalFrequencyEncoder(col_name_original=col) for col in CATEGORY_COLS],
        category_transformation="frequency",
        cols_numerical=NUMERIC_COLS, return_records=True, n_jobs=1,
    )
    for transformer in preprocessor.cts_category:
        column_counts = counts[transformer.col_name_original]
        if not column_counts:
            raise ValueError(f"no training vocabulary for {transformer.col_name_original}")
        values = sorted(column_counts, key=lambda value: (-column_counts[value], value))
        transformer.mapping = {value: index + 1 for index, value in enumerate(values)}
        transformer.other_values_code = len(values) + 1
    return preprocessor


def prepare_cache(source: Path, cache: Path, *, seed: int = 0, valid_frac: float = .05,
                  max_seq_len: int = 500, n_clients: int | None = None,
                  shards: int = 128, threads: int = 6, memory_gb: int = 8) -> dict:
    if max_seq_len < 2 or min(shards, threads, memory_gb) < 1:
        raise ValueError("invalid cache size/resource parameters")
    if n_clients is not None and n_clients < 2:
        raise ValueError("client cap must be at least 2")
    expected = {"format_version": VERSION, "source": source_identity(source),
                "seed": seed, "valid_frac": valid_frac, "max_seq_len": max_seq_len,
                "n_clients": n_clients, "shards": shards, "minimum_train_events": 2,
                "tie_order": ["col_1", *ALL_FEATURE_COLS],
                "time_policy": "calendar_days_no_jitter", "pretraining_scope": "all_source_clients_no_labels",
                "category_policy": "canonical_numeric_integer_strings_explicit_missing"}
    if cache.exists() and not (cache / "manifest.json").exists() and any(cache.iterdir()):
        raise ValueError(f"refusing unowned/nonempty cache directory: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked_json(cache / "manifest.json", expected)
        if (cache / "ready.json").exists():
            validate_cache(cache)
            for name in ("raw", "capped", "spill"):
                intermediate = cache / name
                if intermediate.is_dir():
                    shutil.rmtree(intermediate)
            return json.loads((cache / "ready.json").read_text())
        required_free = max(1024 * 1024, source.stat().st_size * 4)
        if shutil.disk_usage(cache).free < required_free:
            raise OSError("not enough storage for raw partitions, capped rows and mmap cache")
        spill = cache / "spill"
        spill.mkdir(exist_ok=True)
        con = duckdb.connect(config={"threads": threads, "memory_limit": f"{memory_gb}GB",
                                     "temp_directory": str(spill)})
        con.execute("SET enable_progress_bar=false")
        try:
            raw = cache / "raw"
            if not raw.exists():
                with tempfile.TemporaryDirectory(prefix="partition-", dir=cache) as temporary:
                    out = Path(temporary) / "raw"
                    client_filter = ""
                    if n_clients is not None:
                        client_filter = (f"WHERE CAST(id AS VARCHAR) IN (SELECT DISTINCT CAST(id AS VARCHAR) "
                                         f"FROM read_parquet('{sql_path(source)}') "
                                         f"ORDER BY hash(CAST(id AS VARCHAR), {seed}), CAST(id AS VARCHAR) "
                                         f"LIMIT {n_clients})")
                    cols = ", ".join(["col_1", *ALL_FEATURE_COLS])
                    print(f"Partitioning source into {shards} bounded shards", flush=True)
                    con.execute(f"COPY (SELECT CAST(id AS VARCHAR) id, {cols}, "
                                f"hash(CAST(id AS VARCHAR)) % {shards} AS bucket "
                                f"FROM read_parquet('{sql_path(source)}') {client_filter}) "
                                f"TO '{sql_path(out)}' (FORMAT PARQUET, PARTITION_BY (bucket))")
                    os.replace(out, raw)
            capped = cache / "capped"
            capped.mkdir(exist_ok=True)
            for bucket in range(shards):
                output = capped / f"{bucket:03d}.parquet"
                counts_file = capped / f"{bucket:03d}_clients.parquet"
                done = capped / f"{bucket:03d}.json"
                if done.exists():
                    saved = json.loads(done.read_text())
                    if saved != {"rows": signature(output), "clients": signature(counts_file)}:
                        raise ValueError(f"changed prepared partition {bucket}")
                    continue
                files = partition_files(raw, bucket)
                if not files:
                    pd.DataFrame(columns=["id", "col_1", *ALL_FEATURE_COLS]).to_parquet(output, index=False)
                    pd.DataFrame(columns=["id", "n", "bucket", "local_row"]).to_parquet(counts_file, index=False)
                else:
                    inputs = "[" + ",".join(f"'{sql_path(p)}'" for p in files) + "]"
                    descending = ", ".join(f"{col} DESC NULLS LAST" for col in ["col_1", *ALL_FEATURE_COLS])
                    ascending = ", ".join(f"{col} ASC NULLS FIRST" for col in ["col_1", *ALL_FEATURE_COLS])
                    temporary = output.with_suffix(".parquet.tmp")
                    temporary.unlink(missing_ok=True)  # only this run's known incomplete COPY
                    con.execute(f"COPY (SELECT id, col_1, {', '.join(ALL_FEATURE_COLS)} "
                                f"FROM read_parquet({inputs}) "
                                f"QUALIFY row_number() OVER (PARTITION BY id ORDER BY {descending}) <= {max_seq_len} "
                                f"ORDER BY id, {ascending}) TO '{sql_path(temporary)}' (FORMAT PARQUET)")
                    os.replace(temporary, output)
                    frame = con.execute(f"SELECT id, count(*) n FROM read_parquet('{sql_path(output)}') "
                                        "GROUP BY id ORDER BY id").df()
                    frame["bucket"] = bucket
                    frame["local_row"] = np.arange(len(frame))
                    frame.to_parquet(counts_file, index=False)
                atomic_json(done, {"rows": signature(output), "clients": signature(counts_file)})
                print(f"capped partition {bucket + 1}/{shards}", flush=True)
            clients_file = cache / "clients.parquet"
            if not clients_file.exists():
                clients = pd.concat([pd.read_parquet(capped / f"{b:03d}_clients.parquet")
                                     for b in range(shards)], ignore_index=True)
                clients = fixed_split(clients, seed, valid_frac)
                clients.to_parquet(clients_file.with_suffix(".tmp"), index=False)
                os.replace(clients_file.with_suffix(".tmp"), clients_file)
            clients = pd.read_parquet(clients_file)
            if not ((clients.n >= 2) & ~clients.valid).any() or not ((clients.n >= 2) & clients.valid).any():
                raise ValueError("no eligible train or validation clients after minimum length filtering")
            vocab_file = cache / "vocabulary.json"
            if vocab_file.exists():
                vocabulary = json.loads(vocab_file.read_text())
            else:
                counters = {col: Counter() for col in CATEGORY_COLS}
                for bucket in range(shards):
                    eligible = clients[(clients.bucket == bucket) & ~clients.valid & (clients.n >= 2)]
                    train_ids = set(eligible.id)
                    for batch in pq.ParquetFile(capped / f"{bucket:03d}.parquet").iter_batches(
                            batch_size=65536, columns=["id", *CATEGORY_COLS]):
                        frame = batch.to_pandas()
                        selected = frame[frame.id.isin(train_ids)]
                        for col in CATEGORY_COLS:
                            counters[col].update(category_keys(selected[col]).value_counts().to_dict())
                    print(f"train-only vocabulary {bucket + 1}/{shards}", flush=True)
                preprocessor = preprocessor_from_counts(counters)
                save_preprocessor(cache / "preprocessor.pkl", preprocessor)
                marks = sorted(counters["col_2"], key=int)
                # MBD's adapter casts event_type to BIGINT. Refuse a different
                # schema rather than silently map integer 3 to string '3.0'.
                categories = np.array([int(value) for value in marks], dtype=np.int64)
                np.save(cache / "categories.npy", categories)
                vocabulary = {"columns": {ct.col_name_original: ct.mapping for ct in preprocessor.cts_category},
                              "event_codes": {value: index for index, value in enumerate(marks)}}
                atomic_json(vocab_file, vocabulary)
            encoded = cache / "encoded"
            encoded.mkdir(exist_ok=True)
            audit = {"clients_total": len(clients), "events_retained": int(clients.n.sum()),
                     "short_clients_excluded": int((clients.n < 2).sum()),
                     "train_clients": int(((clients.n >= 2) & ~clients.valid).sum()),
                     "valid_clients": int(((clients.n >= 2) & clients.valid).sum()),
                     "unknown_train": {col: 0 for col in CATEGORY_COLS},
                     "unknown_valid": {col: 0 for col in CATEGORY_COLS}, "zero_gap_events": 0,
                     "all_same_day_clients": 0}
            for bucket in range(shards):
                directory = encoded / f"{bucket:03d}"
                done = directory / "ready.json"
                if not done.exists():
                    directory.mkdir(exist_ok=True)
                    frame = pd.read_parquet(capped / f"{bucket:03d}.parquet")
                    local = clients[clients.bucket == bucket].sort_values("local_row")
                    n_rows = len(frame)
                    cats = np.empty((n_rows, len(CATEGORY_COLS)), dtype=np.int32)
                    train_event = np.repeat((~local.valid & (local.n >= 2)).to_numpy(), local.n.to_numpy(dtype=int))
                    valid_event = np.repeat((local.valid & (local.n >= 2)).to_numpy(), local.n.to_numpy(dtype=int))
                    part_audit = {"unknown_train": {}, "unknown_valid": {}}
                    for i, col in enumerate(CATEGORY_COLS):
                        mapping = vocabulary["columns"][col]
                        codes = category_keys(frame[col]).map(mapping)
                        unseen = codes.isna().to_numpy()
                        part_audit["unknown_train"][col] = int((unseen & train_event).sum())
                        part_audit["unknown_valid"][col] = int((unseen & valid_event).sum())
                        cats[:, i] = codes.fillna(len(mapping) + 1).to_numpy(dtype=np.int32)
                    numeric = frame[NUMERIC_COLS].to_numpy(dtype=np.float32)
                    if not np.isfinite(numeric).all() or frame.col_1.isna().any() or frame.id.isna().any():
                        raise ValueError(f"non-finite/null source data in partition {bucket}")
                    times = pd.to_datetime(frame.col_1).astype("datetime64[ns]").astype("int64").to_numpy() // 10**9
                    marks = category_keys(frame.col_2).map(vocabulary["event_codes"]).fillna(-1).to_numpy(dtype=np.int32)
                    offsets = np.r_[0, np.cumsum(local.n.to_numpy(dtype=np.int64))]
                    if int(offsets[-1]) != n_rows:
                        raise ValueError("cache/client offsets differ")
                    part_audit["zero_gap_events"] = int((np.diff(times) == 0).sum()) if len(times) else 0
                    if len(times):
                        # Exclude boundaries between independent clients.
                        boundaries = offsets[1:-1] - 1
                        part_audit["zero_gap_events"] -= int((np.diff(times)[boundaries] == 0).sum())
                    part_audit["all_same_day_clients"] = int(sum(
                        times[offsets[i]] == times[offsets[i + 1] - 1] for i in range(len(local))))
                    known = np.add.reduceat((marks >= 0).astype(np.int32), offsets[:-1]) if len(local) else np.array([], dtype=np.int32)
                    local = local.copy()
                    local["known_events"] = known
                    local.to_parquet(directory / "clients.parquet", index=False)
                    for name, values in zip(ARRAYS, [cats, numeric, times, marks, offsets]):
                        with (directory / (name + ".tmp")).open("wb") as stream:
                            np.save(stream, values)
                        os.replace(directory / (name + ".tmp"), directory / name)
                    atomic_json(done, {"audit": part_audit, "files": {
                        name: signature(directory / name) for name in [*ARRAYS, "clients.parquet"]}})
                    del frame, cats, numeric, times, marks, codes
                    gc.collect()
                saved = json.loads(done.read_text())
                for key in ("unknown_train", "unknown_valid"):
                    for col in CATEGORY_COLS:
                        audit[key][col] += saved["audit"][key][col]
                for key in ("zero_gap_events", "all_same_day_clients"):
                    audit[key] += saved["audit"][key]
                print(f"encoded partition {bucket + 1}/{shards}", flush=True)
            if any(audit["unknown_train"].values()):
                raise ValueError("training vocabulary is incomplete")
            if source_identity(source) != expected["source"]:
                raise ValueError("input data changed during preparation")
            atomic_json(cache / "audit.json", audit)
            ready = {"manifest_sha256": digest(cache / "manifest.json"),
                     "cohort_sha256": digest(clients_file), "vocabulary_sha256": digest(vocab_file),
                     "preprocessor_sha256": digest(cache / "preprocessor.pkl"),
                     "categories_sha256": digest(cache / "categories.npy"), "audit": audit}
            atomic_json(cache / "ready.json", ready)
        finally:
            con.close()
        validate_cache(cache)
        # Only reproducible intermediates owned by this exact cache version.
        for intermediate in (raw, capped, spill):
            if intermediate.is_dir():
                shutil.rmtree(intermediate)
        return ready


def validate_cache(cache: Path) -> dict:
    ready = json.loads((cache / "ready.json").read_text())
    for name, key in (("manifest.json", "manifest_sha256"), ("clients.parquet", "cohort_sha256"),
                      ("vocabulary.json", "vocabulary_sha256"), ("preprocessor.pkl", "preprocessor_sha256"),
                      ("categories.npy", "categories_sha256")):
        if digest(cache / name) != ready[key]:
            raise ValueError(f"cache artifact changed: {cache / name}")
    manifest = json.loads((cache / "manifest.json").read_text())
    for bucket in range(manifest["shards"]):
        directory = cache / "encoded" / f"{bucket:03d}"
        saved = json.loads((directory / "ready.json").read_text())
        if any(signature(directory / name) != value for name, value in saved["files"].items()):
            raise ValueError(f"cache partition changed: {directory}")
    return ready


class DiskRecords(Dataset):
    """At most two mapped shards; returned records own their batch memory."""
    def __init__(self, cache: Path, split: str, model: str = "coles"):
        self.cache, self.model = cache, model
        meta = pd.concat([pd.read_parquet(p / "clients.parquet")
                          for p in sorted((cache / "encoded").iterdir())], ignore_index=True)
        eligible = meta.known_events >= 2 if model in {"cotic", "thp"} else meta.n >= 2
        self.meta = meta[eligible & (meta.valid == (split == "valid"))].reset_index(drop=True)
        if self.meta.empty:
            raise ValueError(f"empty {split} cohort for {model}")
        self.mappings: OrderedDict[int, dict] = OrderedDict()

    def __len__(self):
        return len(self.meta)

    def mapped(self, bucket: int) -> dict:
        if bucket not in self.mappings:
            directory = self.cache / "encoded" / f"{bucket:03d}"
            self.mappings[bucket] = {name: np.load(directory / name, mmap_mode="r") for name in ARRAYS}
            if len(self.mappings) > 2:
                _, old = self.mappings.popitem(last=False)
                for array in old.values():
                    array._mmap.close()
        self.mappings.move_to_end(bucket)
        return self.mappings[bucket]

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        mapped = self.mapped(int(row.bucket))
        start, stop = mapped["offsets.npy"][int(row.local_row):int(row.local_row) + 2]
        times = np.array(mapped["time.npy"][start:stop], copy=True)
        if self.model in {"cotic", "thp"}:
            types = np.array(mapped["mark.npy"][start:stop], dtype=np.int64, copy=True)
            times, types = times[types >= 0], types[types >= 0]
            days = (times - times[0]).astype(np.float64) / 86400.0
            if self.model == "thp":
                return {"time_seqs": days.tolist(), "time_delta_seqs": np.r_[0., np.diff(days)].tolist(),
                        "type_seqs": types.tolist()}
            return torch.from_numpy(days.astype(np.float32)), torch.from_numpy(types)
        cats = np.array(mapped["category.npy"][start:stop], dtype=np.int64, copy=True)
        numeric = np.array(mapped["numeric.npy"][start:stop], copy=True)
        result = {"id": str(row.id), "event_time": torch.from_numpy(times)}
        result.update({col: torch.from_numpy(cats[:, i]) for i, col in enumerate(CATEGORY_COLS)})
        result.update({col: torch.from_numpy(numeric[:, i]) for i, col in enumerate(NUMERIC_COLS)})
        return result


class ShardSampler(Sampler):
    """Shuffle clients and shard order each epoch without random disk thrash."""
    def __init__(self, records: DiskRecords, seed: int):
        self.groups = [group.index.to_numpy() for _, group in records.meta.groupby("bucket", sort=True)]
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return sum(map(len, self.groups))

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        for group in rng.permutation(len(self.groups)):
            yield from rng.permutation(self.groups[group]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="/app/configs/data/mbd_daily.yaml")
    parser.add_argument("--cache-root")
    parser.add_argument("--n-clients", type=int)
    parser.add_argument("--max-seq-len", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--valid-frac", type=float, default=.05)
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-gb", type=int, default=8)
    args = parser.parse_args()
    data = load_data_config(args.data_config)
    if data["name"] != "mbd_daily":
        parser.error("this cache is exclusively for the new MBD-daily pretraining arm")
    root = Path(args.cache_root) if args.cache_root else data_root() / "training_cache" / "mbd_daily" / "v2"
    prepare_cache(Path(data["paths"]["transactions"]), root, seed=args.seed, valid_frac=args.valid_frac,
                  n_clients=args.n_clients, max_seq_len=args.max_seq_len, shards=args.shards,
                  threads=args.threads, memory_gb=args.memory_gb)


if __name__ == "__main__":
    main()
