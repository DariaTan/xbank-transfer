"""Rebuild xbank into the input slots of a model trained on adapted MBD.

Use the real xbank->MBD frozen mapping, not a testbed mapping. Unmatched
categorical fields contain the fitted MBD encoder's other_values_code.
Embedded Parquet metadata lets data.loaders preserve those codes at inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import duckdb

from data.mbd_adapter import TRX_CATEGORY_MAP, PLACEHOLDER_FEATURE_COLS
from data.transfer_metadata import METADATA_KEY, category_encoders, unknown_spec


def quote_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def load_mapping(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = document.get("mbd_to_xbank")
    required = {"event_time", "amount", *TRX_CATEGORY_MAP}
    if not isinstance(mapping, dict) or set(mapping) != required:
        raise ValueError("Expected a real mbd_to_xbank map with event_time, amount and all MBD categories; testbed maps are not valid")
    if any(value is not None and (not isinstance(value, str) or not value) for value in mapping.values()):
        raise ValueError("Mapping values must be source column names or null")
    if mapping["event_time"] is None or mapping["amount"] is None:
        raise ValueError("event_time and amount require a source: other_values_code is categorical only")
    return mapping


def rebuild(xbank_path, mapping_path, preprocessor_path, output_path,
            threads=1, memory_limit=None, temp_dir=None):
    """SQL projection of ALL rows; no sampling, aggregation or vocabulary fit."""
    source, output = Path(xbank_path), Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new path")
    if threads < 1:
        raise ValueError("threads must be positive")
    mapping = load_mapping(mapping_path)
    # This is the same trusted fitted checkpoint used by the inference model.
    checkpoint = Path(preprocessor_path).read_bytes()
    preprocessor = pickle.loads(checkpoint)
    encoders = category_encoders(preprocessor)
    expected_categories = set(TRX_CATEGORY_MAP.values()) | {"col_15", "col_16"}
    if not expected_categories <= set(encoders):
        raise ValueError("Preprocessor must use the adapted MBD model slots col_2..col_16")
    unknown_fields = {}
    expressions = {"id": quote_identifier("id"),
                   "col_1": quote_identifier(mapping["event_time"]),
                   "col_11": quote_identifier(mapping["amount"])}
    for field, slot in TRX_CATEGORY_MAP.items():
        origin = mapping[field]
        if origin is None:
            spec = unknown_spec(encoders[slot])
            unknown_fields[slot] = {**spec, "mbd_field": field}
            expressions[slot] = str(spec["code"])
        else:
            expressions[slot] = quote_identifier(origin)
    # These slots were constants in MBD training; they do not denote MBD fields.
    for slot in PLACEHOLDER_FEATURE_COLS:
        expressions[slot] = "0"
    metadata = {"version": 1, "layout": "adapted_mbd_model_slots",
                "mbd_to_xbank": mapping, "mbd_to_model_slot": dict(TRX_CATEGORY_MAP),
                "unknown_fields": unknown_fields,
                "preprocessor_sha256": hashlib.sha256(checkpoint).hexdigest()}
    columns = ["id", *[f"col_{i}" for i in range(1, 17)]]
    projection = ", ".join(f"{expressions[name]} AS {quote_identifier(name)}" for name in columns)
    read_sql = f"read_parquet({quote_literal(source)})"
    temporary = None
    try:
        with duckdb.connect() as con:
            con.execute(f"SET threads = {int(threads)}")
            if memory_limit:
                con.execute("SET memory_limit = ?", [memory_limit])
            if temp_dir:
                Path(temp_dir).mkdir(parents=True, exist_ok=True)
                con.execute("SET temp_directory = ?", [str(temp_dir)])
            available = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {read_sql}").fetchall()}
            missing = {"id", *[value for value in mapping.values() if value is not None]} - available
            if missing:
                raise ValueError(f"Mapped source columns missing in xbank: {sorted(missing)}")
            output.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".parquet", dir=output.parent)
            os.close(fd)
            print(f"Rebuilding all xbank rows -> {output}", flush=True)
            rows_written = con.execute(
                f"COPY (SELECT {projection} FROM {read_sql}) TO {quote_literal(temporary)} "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, KV_METADATA "
                f"{{{METADATA_KEY}: {quote_literal(json.dumps(metadata, ensure_ascii=False))}}})"
            ).fetchone()[0]
            rows_read = con.execute("SELECT SUM(num_rows) FROM parquet_file_metadata(?)", [str(source)]).fetchone()[0]
            rows_saved = con.execute("SELECT SUM(num_rows) FROM parquet_file_metadata(?)", [temporary]).fetchone()[0]
            if rows_written != rows_read or rows_saved != rows_read:
                raise RuntimeError("Row count changed during rebuild")
        # Publish atomically without overwriting an existing file, even on a race.
        os.link(temporary, output)
        print(f"Saved {rows_written:,} rows; UNK fields: "
              f"{', '.join(spec['mbd_field'] for spec in unknown_fields.values()) or 'none'}", flush=True)
        return {"rows": rows_written, "output": str(output), **metadata}
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbank-path", required=True)  # исходный Parquet xbank, все строки
    parser.add_argument("--mapping", default="data/profiles_vFGW/frozen_mapping.json")  # карта реального xbank -> MBD
    parser.add_argument("--preprocessor", required=True)  # preprocessor.pkl от обученной MBD-модели
    parser.add_argument("--output", default="data/xbank_mbd/transactions.parquet")  # новый Parquet в слотах MBD-модели
    parser.add_argument("--threads", type=int, default=1)  # число потоков DuckDB
    parser.add_argument("--memory-limit")  # лимит памяти DuckDB, например 16GB
    parser.add_argument("--temp-dir")  # каталог для временных данных DuckDB
    args = parser.parse_args(argv)
    rebuild(args.xbank_path, args.mapping, args.preprocessor, args.output,
            args.threads, args.memory_limit, args.temp_dir)


if __name__ == "__main__":
    main()
