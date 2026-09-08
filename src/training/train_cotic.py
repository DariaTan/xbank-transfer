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


from data.loaders import build_cotic_sequences, cap_rows_per_client, load_all_raw
from models.cotic import InMemoryEventDataModule, build_datasets, build_module
from training.common import check_or_save_run_config, split_df_by_client

XBANK_DATA_CONFIG = "/app/configs/data/xbank.yaml"
with open(XBANK_DATA_CONFIG) as f:
    TRANSACTIONS_PATH = yaml.safe_load(f)["paths"]["transactions"]


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        return SimpleNamespace(**yaml.safe_load(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/app/configs/models/cotic.yaml")
    cli = parser.parse_args()
    args = load_config(cli.config)

    ckpt_dir = Path(args.checkpoint_dir)
    check_or_save_run_config(ckpt_dir, args, ["seed", "valid_frac", "n_clients", "max_seq_len"])

    print("Loading full transactions table (id, col_1, col_2) ...", flush=True)
    df = load_all_raw(TRANSACTIONS_PATH, columns=["id", "col_1", "col_2"])
    print(f"  {len(df)} rows, {df['id'].nunique()} clients", flush=True)

    if args.n_clients is not None:
        keep = df["id"].drop_duplicates().sample(n=args.n_clients, random_state=args.seed)
        df = df[df["id"].isin(keep)]
        print(f"  capped to {args.n_clients} clients for debugging", flush=True)

    df = cap_rows_per_client(df, args.max_seq_len)
    print(f"  {len(df)} rows after capping to {args.max_seq_len}/client", flush=True)

    train_df, valid_df = split_df_by_client(df, "id", args.valid_frac, args.seed)
    print(
        f"  split into train={train_df['id'].nunique()} clients, "
        f"valid={valid_df['id'].nunique()} clients (before building sequences)",
        flush=True,
    )

    print("Building COTIC (times, types) sequences ...", flush=True)
    train_times, train_types, num_types, categories, _ = build_cotic_sequences(train_df)
    valid_times, valid_types, _, _, _ = build_cotic_sequences(valid_df, categories=categories)
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
        logger=TensorBoardLogger("/app/data/lightning_logs", name="cotic"),
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
