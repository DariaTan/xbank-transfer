"""Disposable production-weight inference smoke, with optional real histories.

Never write production embeddings or modify checkpoints. Temporary outputs
are removed on success or failure; the checkpoint tree is read-only via a
symlink. Run locally first, then on the server with --real-data --device cuda.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import duckdb
import numpy as np
import pandas as pd
import yaml

from data.schema import ALL_FEATURE_COLS, CATEGORY_COLS, NUMERIC_COLS
from data.splits import load_windowed_transactions_for_dates, unpack_window_id
from training.embedding_io import validate_embedding_file
from training.paths import load_data_config

REPO = Path(__file__).resolve().parents[2]
MODELS = ("coles", "cotic", "thp", "mlm", "nep")
EVALUATIONS = ("mbd_daily", "xbank", "xbank_fgw_v2")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(root, name, checkpoint_root, real_data, production_batch):
    original = load_data_config(REPO / "configs/data" / f"{name}.yaml")
    if real_data:
        with duckdb.connect() as con:
            ids = con.execute("SELECT DISTINCT id FROM read_parquet(?) ORDER BY hash(id, 0) LIMIT 32",
                              [original["paths"]["targets"]]).df().id.tolist()
            dates = con.execute("SELECT DISTINCT CAST(col_1 AS VARCHAR) d FROM read_parquet(?) ORDER BY d LIMIT 2",
                                [original["paths"]["targets"]]).df().d.tolist()
        frames = [load_windowed_transactions_for_dates(
            original["paths"]["transactions"], [date], 12, 500, ids,
            ["id", "col_1", *ALL_FEATURE_COLS],
            include_cutoff=original.get("include_target_date_transactions", True),
            bounded=True, memory_limit="2GB", threads=2, temp_directory=str(root)) for date in dates]
        frame = pd.concat(frames, ignore_index=True)
        if frame.empty:
            raise ValueError(f"{name}: real sample has no history")
        frame["id"] = frame.id.map(lambda x: unpack_window_id(str(x))[0])
        frame = frame.drop_duplicates()
        ids = [str(x) for x in ids]
    else:
        dates = ["2023-01-01", "2023-02-01"]
        count, length = (256, 500) if production_batch else (8, 24)
        ids = [f"smoke-{i}" for i in range(count)] + ["no-history"]
        cats = np.load(checkpoint_root / "cotic/categories.npy", allow_pickle=True)
        rows = []
        for i, client in enumerate(ids[:-1]):
            for j in range(length):
                row = {"id": client, "col_1": pd.Timestamp("2022-12-01") + pd.Timedelta(days=j % 25)}
                row.update({col: int((i+j) % 3) for col in CATEGORY_COLS})
                row.update({col: float((j % 10) / 10) for col in NUMERIC_COLS})
                # Both frozen xbank mappings have an event mark that can be
                # encoded by the actual TPP checkpoints.
                row["col_2"] = cats[(i+j) % len(cats)]
                if name != "mbd_daily":
                    row["col_10"] = row["col_2"]
                rows.append(row)
        frame = pd.DataFrame(rows)
    transactions, targets = root / f"{name}-transactions.parquet", root / f"{name}-targets.parquet"
    frame.to_parquet(transactions, index=False)
    pd.DataFrame([(client,date) for client in ids for date in dates],
                 columns=["id","col_1"]).to_parquet(targets,index=False)
    # Virtual mount paths are remapped exactly once in the CLI child. An
    # absolute temporary path already below /app/data would otherwise get
    # prefixed a second time by XBANK_DATA_ROOT on the production server.
    original["paths"] = {"transactions": f"/app/data/{name}-transactions.parquet",
                         "targets": f"/app/data/{name}-targets.parquet"}
    original["evaluation_name"] = name
    if original.get("schema_mapping"):
        original["schema_mapping"] = str(REPO / "configs/mappings" / Path(original["schema_mapping"]).name)
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(original))
    return path, dates, set(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--temp-root")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--production-batch", action="store_true")
    parser.add_argument("--device", choices=("cpu","cuda"),default="cpu")
    args = parser.parse_args()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    immutable = {p: sha(p) for p in checkpoint_root.rglob("*") if p.is_file() and p.name != "training.lock"}
    with tempfile.TemporaryDirectory(prefix="daily-inference-smoke-", dir=args.temp_root) as directory:
        root = Path(directory)
        (root/"checkpoints").mkdir()
        (root/"checkpoints/mbd_daily_source").symlink_to(checkpoint_root, target_is_directory=True)
        cfg = yaml.safe_load((REPO/"configs/models/inference_daily_source.yaml").read_text())
        cfg["inference"].update(client_chunk_size=4 if not args.production_batch else 256,
                                batch_size=256 if args.production_batch else 8,
                                duckdb_memory_limit="2GB", duckdb_threads=2)
        downstream = root/"inference.yaml"
        downstream.write_text(yaml.safe_dump(cfg))
        env = os.environ.copy()
        env.update(XBANK_DATA_ROOT=str(root), PYTHONPATH=str(REPO/"src"), PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   MPLCONFIGDIR=str(root/"mpl"), XDG_CACHE_HOME=str(root/"xdg"))
        if args.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        for name in EVALUATIONS:
            data, dates, population = sample(root,name,checkpoint_root,args.real_data,args.production_batch)
            if args.real_data:
                from cross_schema.apply_mapping import FrozenSchemaMapping
                from training.artifact_compat import load_preprocessor
                data_cfg = yaml.safe_load(data.read_text())
                raw = pd.read_parquet(root/f"{name}-transactions.parquet")
                mapping = FrozenSchemaMapping(data_cfg["schema_mapping"]) if data_cfg.get("schema_mapping") else None
                for model in MODELS:
                    aligned = mapping.transform(raw,model) if mapping else raw
                    if model in ("cotic","thp"):
                        cats = np.load(checkpoint_root/model/"categories.npy",allow_pickle=True)
                        unknown = {"col_2": float((pd.Categorical(aligned.col_2,categories=cats).codes < 0).mean())}
                    else:
                        p = load_preprocessor(checkpoint_root/model/"preprocessor.pkl")
                        unknown = {ct.col_name_original: float((ct.transform(
                            aligned[[ct.col_name_original]].copy())[ct.col_name_target] == ct.other_values_code).mean())
                            for ct in p.cts_category}
                    print(f"REAL_SAMPLE_UNKNOWN_RATES {name}/{model}: {json.dumps(unknown,sort_keys=True)}",flush=True)
            for model in MODELS:
                command = [sys.executable,"-m","training.infer_mbd","--model",model,
                           "--data-config",str(data),"--model-config",str(REPO/"configs/models"/f"{model}.yaml"),
                           "--downstream-config",str(downstream),"--checkpoint-source","mbd_daily"]
                if args.device == "cuda":
                    command += ["--require-cuda"]
                result = subprocess.run(command,env=env,capture_output=True,text=True,timeout=600)
                if result.returncode:
                    raise RuntimeError(f"{name}/{model}\n{result.stdout[-5000:]}\n{result.stderr[-5000:]}")
                outputs = root/"embeds"/name/"mbd_daily_source"/model
                before = {}
                for date in dates:
                    path = outputs/f"{date}.parquet"
                    frame = validate_embedding_file(path, date)
                    assert set(frame.inn.astype(str)) <= population
                    assert "no-history" not in set(frame.inn)
                    assert len([c for c in frame if c.startswith("emb_")]) == 128
                    before[path] = sha(path)
                # Actual CLI resume must skip completed dates, not rewrite them.
                repeated = subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
                if repeated.returncode:
                    raise RuntimeError(repeated.stdout+repeated.stderr)
                assert all(sha(p)==h for p,h in before.items())
                assert not (outputs/"_chunks").exists()
                manifest = json.loads((outputs/"run_manifest.json").read_text())
                assert manifest["checkpoint_source"] == "mbd_daily"
                assert manifest["checkpoint_files_sha256"]
                print(f"SMOKE PASSED {name}/{model}: dates={len(dates)} device={args.device} rows={len(frame)} resume=OK",flush=True)
        assert all(sha(p)==h for p,h in immutable.items()), "source checkpoints changed"
    print("ALL 15 DAILY INFERENCE CASES PASSED; temporary outputs removed; checkpoints unchanged",flush=True)


if __name__ == "__main__":
    main()
