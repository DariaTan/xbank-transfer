"""Frozen-checkpoint inference on an adapted MBD corpus.

The checkpoint corpus and evaluation corpus are deliberately independent:
``--checkpoint-source mbd`` selects weights trained on MBD-raw, while
``--data-config`` selects whether those weights see MBD-raw, MBD-daily, or a
local smoke fixture. Results are namespaced by both choices and are written
under ``/app/data`` (the production container's mount of
``/mnt/storage/d.tanyushkina/transactions``).

Inference is restricted to clients present in the targets table and runs in
bounded, resumable client chunks. A final month is published atomically only
after every chunk has completed and passed structural validation.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from data.loaders import build_chronos_series, build_cotic_sequences, build_ptls_records, build_thp_sequences
from data.schema import (
    ALL_FEATURE_COLS,
    CLIENT_ID_COL,
    EVENT_TIME_COL,
    NUMERIC_COLS,
    TARGETS_CLIENT_ID_COL,
    TARGETS_DATE_COL,
)
from data.splits import load_windowed_transactions_for_dates, unpack_window_id
from training.common import load_preprocessor
from training.embedding_io import atomic_parquet, validate_embedding_file, validate_embedding_frame
from training.paths import checkpoint_dir, embedding_dir, evaluation_name, load_data_config


MODEL_CHOICES = ["coles", "cotic", "nep", "mlm", "thp", "chronos2"]


def _required_checkpoint_files(model: str) -> Sequence[str]:
    if model == "chronos2":
        return ()
    if model == "coles":
        return ("best.ckpt", "preprocessor.pkl")
    if model in ("nep", "mlm"):
        return ("best.pt", "preprocessor.pkl")
    if model == "thp":
        return ("best.pt", "categories.npy")
    if model == "cotic":
        return ("best.ckpt", "categories.npy", "normalizer.pkl")
    raise ValueError(model)


def _check_checkpoint(model: str, checkpoint_source: str) -> Path:
    ckpt_dir = checkpoint_dir(checkpoint_source, model)
    missing = [
        str(ckpt_dir / name)
        for name in _required_checkpoint_files(model)
        if not (ckpt_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"checkpoint is incomplete; missing: {missing}")
    return ckpt_dir


def _model_columns(model: str, event_type_col: str, chronos_value_col: str) -> List[str]:
    if model in ("cotic", "thp"):
        return [CLIENT_ID_COL, EVENT_TIME_COL, event_type_col]
    if model == "chronos2":
        return [CLIENT_ID_COL, EVENT_TIME_COL, chronos_value_col]
    return [CLIENT_ID_COL, EVENT_TIME_COL, *ALL_FEATURE_COLS]


def _load_embedder(
    model_name: str,
    model_cfg: Optional[Dict],
    inf: Dict,
    data_cfg: Dict,
    device: torch.device,
    checkpoint_source: str,
) -> Callable[[pd.DataFrame], Tuple[np.ndarray, List[str]]]:
    ckpt_dir = _check_checkpoint(model_name, checkpoint_source)

    if model_name == "coles":
        from models.coles import build_module, extract_embeddings

        preprocessor = load_preprocessor(ckpt_dir / "preprocessor.pkl")
        module = build_module(
            preprocessor.get_category_dictionary_sizes(),
            embedding_dim=model_cfg["embedding_dim"],
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
        )
        ckpt = torch.load(ckpt_dir / "best.ckpt", map_location=device, weights_only=False)
        module.load_state_dict(ckpt["state_dict"])
        module.to(device).eval()

        def embed(windowed: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            records, _ = build_ptls_records(windowed, preprocessor=preprocessor)
            ids = [str(r[CLIENT_ID_COL]) for r in records]
            return extract_embeddings(module, records, batch_size=inf["batch_size"], device=device), ids

        return embed

    if model_name in ("nep", "mlm"):
        from models.mlm import MLM, extract_embeddings as extract_mlm
        from models.nep import NEP, extract_embeddings as extract_nep

        preprocessor = load_preprocessor(ckpt_dir / "preprocessor.pkl")
        cat_sizes = preprocessor.get_category_dictionary_sizes()
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        if model_name == "nep":
            model = NEP(
                cat_sizes,
                NUMERIC_COLS,
                d_model=model_cfg["d_model"],
                num_layers=model_cfg["num_layers"],
                max_position_embeddings=inf["max_seq_len"],
            )
            extract_fn = extract_nep
        else:
            max_pos = ckpt["model"]["backbone.embeddings.position_embeddings.weight"].shape[0]
            model = MLM(
                cat_sizes,
                NUMERIC_COLS,
                d_model=model_cfg["d_model"],
                num_layers=model_cfg["num_layers"],
                max_position_embeddings=max_pos,
            )
            extract_fn = extract_mlm
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        def embed(windowed: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            records, _ = build_ptls_records(windowed, preprocessor=preprocessor)
            ids = [str(r[CLIENT_ID_COL]) for r in records]
            return extract_fn(model, records, batch_size=inf["batch_size"], device=device), ids

        return embed

    if model_name == "thp":
        from models.thp import build_dataloader, build_model, build_tokenizer, extract_embeddings

        categories = np.load(ckpt_dir / "categories.npy", allow_pickle=True)
        num_types = len(categories)
        model = build_model(
            num_types,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            gpu=0 if torch.cuda.is_available() else -1,
        )
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        tokenizer = build_tokenizer(num_types, max_len=inf["max_seq_len"])

        def embed(windowed: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            times, deltas, types, _, _, ids = build_thp_sequences(
                windowed,
                event_type_col=data_cfg.get("event_type_col", "col_2"),
                categories=categories,
            )
            loader = build_dataloader(times, deltas, types, tokenizer, inf["batch_size"], shuffle=False)
            return extract_embeddings(model, loader), [str(x) for x in ids]

        return embed

    if model_name == "cotic":
        from models.cotic import EventDataset, build_module, extract_embeddings

        categories = np.load(ckpt_dir / "categories.npy", allow_pickle=True)
        with open(ckpt_dir / "normalizer.pkl", "rb") as f:
            normalizer = pickle.load(f)
        module = build_module(
            len(categories),
            in_channels=model_cfg["in_channels"],
            nb_filters=model_cfg["nb_filters"],
            nb_layers=model_cfg["nb_layers"],
        )
        ckpt = torch.load(ckpt_dir / "best.ckpt", map_location=device, weights_only=False)
        module.load_state_dict(ckpt["state_dict"])
        net = module.net.to(device).eval()

        def embed(windowed: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            times, types, _, _, ids = build_cotic_sequences(
                windowed,
                event_type_col=data_cfg.get("event_type_col", "col_2"),
                categories=categories,
            )
            dataset = EventDataset(times, types, len(categories))
            dataset.normalize_data(normalizer)
            return (
                extract_embeddings(net, dataset, batch_size=inf["batch_size"], device=device),
                [str(x) for x in ids],
            )

        return embed

    if model_name == "chronos2":
        from models.chronos2 import extract_reg_embeddings, load_pipeline

        pipeline = load_pipeline(device_map="cuda" if torch.cuda.is_available() else "cpu")

        def embed(windowed: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            series, ids = build_chronos_series(windowed, value_col=inf["chronos2_value_col"])
            return (
                extract_reg_embeddings(pipeline, series, batch_size=inf["chronos2_batch_size"]),
                [str(x) for x in ids],
            )

        return embed

    raise ValueError(f"unsupported model: {model_name}")


def _target_population(targets_path: Path, n_clients: Optional[int], seed: int) -> Tuple[List[Any], List[str]]:
    targets = pd.read_parquet(targets_path, columns=[TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL])
    dates = sorted(targets[TARGETS_DATE_COL].astype(str).unique().tolist())
    client_ids = targets[TARGETS_CLIENT_ID_COL].drop_duplicates().tolist()
    if n_clients is not None and n_clients < len(client_ids):
        indices = np.random.RandomState(seed).choice(len(client_ids), size=n_clients, replace=False)
        client_ids = [client_ids[i] for i in sorted(indices)]
    return client_ids, dates


def _chunked(values: Sequence[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        raise ValueError("client_chunk_size must be positive")
    return [list(values[i : i + size]) for i in range(0, len(values), size)]


def _write_or_check_manifest(out_dir: Path, manifest: Dict) -> None:
    path = out_dir / "run_manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != manifest:
            raise ValueError(f"run configuration differs from existing {path}: saved={saved}, current={manifest}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _run_date(
    target_date: str,
    client_chunks: List[List[Any]],
    transactions_path: Path,
    out_dir: Path,
    inf: Dict,
    columns: List[str],
    embed: Callable[[pd.DataFrame], Tuple[np.ndarray, List[str]]],
) -> None:
    out_path = out_dir / f"{target_date}.parquet"
    if out_path.exists():
        completed = validate_embedding_file(out_path, expected_date=target_date)
        print(f"{target_date}: valid existing output ({len(completed)} rows), skipping", flush=True)
        return

    chunks_dir = out_dir / "_chunks" / target_date
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for i, ids in enumerate(client_chunks):
        chunk_path = chunks_dir / f"chunk_{i:05d}.parquet"
        empty_marker = chunks_dir / f"chunk_{i:05d}.empty"
        if chunk_path.exists():
            validate_embedding_file(chunk_path, expected_date=target_date)
            continue
        if empty_marker.exists():
            continue

        windowed = load_windowed_transactions_for_dates(
            str(transactions_path),
            [target_date],
            inf["history_window_months"],
            inf["max_seq_len"],
            client_ids=ids,
            columns=columns,
        )
        if windowed.empty:
            empty_marker.touch()
            print(f"{target_date}: chunk {i + 1}/{len(client_chunks)} has no history", flush=True)
            continue

        emb, window_ids = embed(windowed)
        inns = [unpack_window_id(str(window_id))[0] for window_id in window_ids]
        if len(emb) != len(inns):
            raise ValueError(f"row count mismatch: {len(emb)} embeddings vs {len(inns)} ids")
        frame = pd.DataFrame(emb, columns=[f"emb_{j}" for j in range(emb.shape[1])])
        frame.insert(0, "date", target_date)
        frame.insert(0, "inn", inns)
        validate_embedding_frame(frame, expected_date=target_date)
        atomic_parquet(frame, chunk_path)
        print(f"{target_date}: chunk {i + 1}/{len(client_chunks)} wrote {len(frame)} rows", flush=True)

    done = sum(
        (chunks_dir / f"chunk_{i:05d}.parquet").exists()
        or (chunks_dir / f"chunk_{i:05d}.empty").exists()
        for i in range(len(client_chunks))
    )
    if done != len(client_chunks):
        raise RuntimeError(f"{target_date}: only {done}/{len(client_chunks)} chunks completed")
    paths = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not paths:
        raise RuntimeError(f"{target_date}: no clients had transaction history")
    merged = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    validate_embedding_frame(merged, expected_date=target_date)
    atomic_parquet(merged, out_path)
    shutil.rmtree(chunks_dir)
    if not any(chunks_dir.parent.iterdir()):
        chunks_dir.parent.rmdir()
    print(f"{target_date}: published {len(merged)} rows to {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--data-config", default="/app/configs/data/mbd.yaml")
    parser.add_argument("--downstream-config", default="/app/configs/models/downstream_mbd.yaml")
    parser.add_argument("--model-config", default=None, help="default /app/configs/models/<model>.yaml")
    parser.add_argument("--checkpoint-source", default="mbd")
    cli = parser.parse_args()

    data_cfg = load_data_config(cli.data_config)
    with open(cli.downstream_config) as f:
        inf = yaml.safe_load(f)["inference"]
    transactions_path = Path(data_cfg["paths"]["transactions"])
    targets_path = Path(data_cfg["paths"]["targets"])
    missing = [str(path) for path in (transactions_path, targets_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"adapted data files are missing: {missing}")

    model_cfg: Optional[Dict] = None
    if cli.model != "chronos2":
        model_config_path = cli.model_config or f"/app/configs/models/{cli.model}.yaml"
        with open(model_config_path) as f:
            model_cfg = yaml.safe_load(f)

    eval_name = evaluation_name(data_cfg)
    out_dir = embedding_dir(inf["embeds_dir"], eval_name, cli.checkpoint_source, cli.model)
    client_ids, target_dates = _target_population(targets_path, inf.get("n_clients"), inf.get("seed", 0))
    chunk_size = (
        inf.get("chronos2_chunk_size", 2000)
        if cli.model == "chronos2"
        else inf.get("client_chunk_size", 100000)
    )
    client_chunks = _chunked(client_ids, chunk_size)
    manifest = {
        "model": cli.model,
        "checkpoint_source": cli.checkpoint_source,
        "evaluation_name": eval_name,
        "data_config": str(cli.data_config),
        "transactions": str(transactions_path),
        "targets": str(targets_path),
        "n_target_clients": len(client_ids),
        "target_dates": target_dates,
        "history_window_months": inf["history_window_months"],
        "max_seq_len": inf["max_seq_len"],
        "client_chunk_size": chunk_size,
        "batch_size": inf["batch_size"],
        "model_config": model_cfg,
    }
    _write_or_check_manifest(out_dir, manifest)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"model={cli.model} checkpoint={cli.checkpoint_source}_source eval={eval_name} "
        f"clients={len(client_ids)} dates={len(target_dates)} chunks/date={len(client_chunks)} device={device}",
        flush=True,
    )
    embed = _load_embedder(cli.model, model_cfg, inf, data_cfg, device, cli.checkpoint_source)
    columns = _model_columns(
        cli.model,
        data_cfg.get("event_type_col", "col_2"),
        inf["chronos2_value_col"],
    )
    for target_date in target_dates:
        _run_date(target_date, client_chunks, transactions_path, out_dir, inf, columns, embed)
    print(f"Done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
