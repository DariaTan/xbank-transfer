"""Full-scale COTIC training: all xbank clients, early stopping on
validation log-likelihood, resumable via pytorch-lightning's own
checkpointing. Mirrors train_coles.py.

Run parameters live in configs/models/cotic.yaml, not CLI flags -- the
only flag this script takes is --config, to point at a different one.
"""
import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from data.loaders import build_cotic_sequences, cap_rows_per_client, load_all_raw, load_raw_for_clients, sample_client_ids
from models.cotic import InMemoryEventDataModule, build_datasets, build_module
from training.common import check_or_save_run_config, split_df_by_client

MODEL_NAME = "cotic"


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        return SimpleNamespace(**yaml.safe_load(f))


def load_data_config(data_config_path: str) -> dict:
    with open(data_config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/app/configs/models/cotic.yaml")
    parser.add_argument(
        "--data-config",
        type=str,
        default="/app/configs/data/xbank.yaml",
        help=(
            "which dataset to pretrain on -- default xbank; pass e.g. "
            "/app/configs/data/mbd.yaml (raw, untouched) or "
            "/app/configs/data/mbd_daily.yaml (daily-aggregated) to "
            "pretrain on MBD instead. checkpoint_dir is derived from this "
            "config's own 'name' field (/app/data/checkpoints/<name>_source/"
            f"{MODEL_NAME}), so different corpora never collide."
        ),
    )
    cli = parser.parse_args()
    args = load_config(cli.config)
    data_cfg = load_data_config(cli.data_config)
    args.data_config = cli.data_config
    TRANSACTIONS_PATH = data_cfg["paths"]["transactions"]

    ckpt_dir = Path(f"/app/data/checkpoints/{data_cfg['name']}_source") / MODEL_NAME
    # data_config guarded here too -- resuming a checkpoint against a
    # DIFFERENT dataset than it was started with (e.g. an xbank checkpoint
    # accidentally resumed with --data-config mbd.yaml) would otherwise
    # silently mix two different data sources into one run.
    check_or_save_run_config(ckpt_dir, args, ["seed", "valid_frac", "n_clients", "max_seq_len", "data_config"])

    # Which single category column to use as COTIC's "event type" mark is
    # a property of the DATA, not the architecture -- read from
    # --data-config (e.g. configs/data/xbank.yaml's event_type_col: col_2)
    # rather than hardcoded, since a future data source could reasonably
    # need a different column here (moved off a hardcoded "col_2" literal
    # 2026-09-14).
    EVENT_TYPE_COL = data_cfg["event_type_col"]
    columns = ["id", "col_1", EVENT_TYPE_COL]
    if args.n_clients is not None:
        # Filters to the sampled clients INSIDE DuckDB, before anything
        # becomes a pandas object -- load_all_raw followed by a pandas-side
        # .sample() still reads and materializes every row first, which is
        # what silently OOM-killed train_nep.py against MBD's ~550M-row
        # daily table (2026-09-14, n_clients set but the full table got
        # loaded anyway).
        print(f"Loading transactions table (id, col_1, {EVENT_TYPE_COL}) from {cli.data_config}, capped to {args.n_clients} clients ...", flush=True)
        client_ids = sample_client_ids(TRANSACTIONS_PATH, args.n_clients, seed=args.seed)
        df = load_raw_for_clients(TRANSACTIONS_PATH, client_ids, columns=columns)
        print(f"  {len(df)} rows, {df['id'].nunique()} clients", flush=True)
    else:
        print(f"Loading full transactions table (id, col_1, {EVENT_TYPE_COL}) from {cli.data_config} ...", flush=True)
        df = load_all_raw(TRANSACTIONS_PATH, columns=columns)
        print(f"  {len(df)} rows, {df['id'].nunique()} clients", flush=True)

    df = cap_rows_per_client(df, args.max_seq_len)
    print(f"  {len(df)} rows after capping to {args.max_seq_len}/client", flush=True)

    train_df, valid_df = split_df_by_client(df, "id", args.valid_frac, args.seed)
    print(
        f"  split into train={train_df['id'].nunique()} clients, "
        f"valid={valid_df['id'].nunique()} clients (before building sequences)",
        flush=True,
    )

    print("Building COTIC (times, types) sequences ...", flush=True)
    train_times, train_types, num_types, categories, _ = build_cotic_sequences(train_df, event_type_col=EVENT_TYPE_COL)
    valid_times, valid_types, _, _, _ = build_cotic_sequences(valid_df, event_type_col=EVENT_TYPE_COL, categories=categories)
    print(
        f"  train={len(train_times)} valid={len(valid_times)}, num_types={num_types}",
        flush=True,
    )

    train_dataset, valid_dataset, normalizer = build_datasets(
        train_times, train_types, valid_times, valid_types, num_types
    )

    # Persisted so downstream inference maps event types (col_2) to the
    # same ids, and normalizes inter-event times with the SAME fitted
    # normalizer, this checkpoint was trained against -- see
    # train_thp.py's identical comment on `categories`.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / "categories.npy", categories)
    with open(ckpt_dir / "normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)

    datamodule = InMemoryEventDataModule(
        train_dataset,
        valid_dataset,
        normalizer,
        batch_size_train=args.batch_size,
        batch_size_valid=args.batch_size,
    )

    module = build_module(
        num_types, in_channels=args.in_channels, nb_filters=args.nb_filters, nb_layers=args.nb_layers
    )

    last_ckpt = ckpt_dir / "last.ckpt"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        # save_dir diverges by data_cfg['name'] same as ckpt_dir --
        # otherwise xbank/mbd/mbd_daily runs of this model all write into
        # the SAME TensorBoard log dir and their curves interleave
        # indistinguishably (flagged 2026-09-17).
        logger=TensorBoardLogger(f"/app/data/lightning_logs/{data_cfg['name']}_source", name=MODEL_NAME),
        callbacks=[
            EarlyStopping(monitor="val/log_likelihood", mode="max", patience=args.patience),
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="best",
                monitor="val/log_likelihood",
                mode="max",
                save_last=True,
                save_top_k=1,
            ),
        ],
        num_sanity_val_steps=0,
    )
    resume_path = str(last_ckpt) if last_ckpt.exists() else None
    if resume_path:
        print(f"Resuming from {resume_path}", flush=True)
    trainer.fit(module, datamodule, ckpt_path=resume_path)
    print("Final logged metrics:", trainer.logged_metrics, flush=True)
    print(f"Done. Best checkpoint: {ckpt_dir / 'best.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
