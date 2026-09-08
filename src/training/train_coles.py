"""Full-scale CoLES training: all xbank clients, early stopping on
validation recall@k, resumable via pytorch-lightning's own checkpointing
(`ckpt_path="last.ckpt"` picks up optimizer/scheduler/epoch state too, not
just weights).

Run parameters live in configs/models/coles.yaml, not CLI flags -- the
only flag this script takes is --config, to point at a different one.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from data.loaders import build_ptls_records, cap_rows_per_client, load_all_raw
from data.schema import ALL_FEATURE_COLS, CLIENT_ID_COL, EVENT_TIME_COL
from models.coles import build_datamodule, build_module
from training.common import check_or_save_run_config, save_preprocessor, split_df_by_client

XBANK_DATA_CONFIG = "/app/configs/data/xbank.yaml"
with open(XBANK_DATA_CONFIG) as f:
    TRANSACTIONS_PATH = yaml.safe_load(f)["paths"]["transactions"]


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        return SimpleNamespace(**yaml.safe_load(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/app/configs/models/coles.yaml")
    cli = parser.parse_args()
    args = load_config(cli.config)

    ckpt_dir = Path(args.checkpoint_dir)
    check_or_save_run_config(ckpt_dir, args, ["seed", "valid_frac", "n_clients", "max_seq_len"])

    print("Loading full transactions table ...", flush=True)
    df = load_all_raw(TRANSACTIONS_PATH, columns=[CLIENT_ID_COL, EVENT_TIME_COL] + ALL_FEATURE_COLS)
    print(f"  {len(df)} rows, {df[CLIENT_ID_COL].nunique()} clients", flush=True)

    if args.n_clients is not None:
        keep = df[CLIENT_ID_COL].drop_duplicates().sample(n=args.n_clients, random_state=args.seed)
        df = df[df[CLIENT_ID_COL].isin(keep)]
        print(f"  capped to {args.n_clients} clients for debugging", flush=True)

    df = cap_rows_per_client(df, args.max_seq_len)
    print(f"  {len(df)} rows after capping to {args.max_seq_len}/client", flush=True)

    train_df, valid_df = split_df_by_client(df, CLIENT_ID_COL, args.valid_frac, args.seed)
    print(
        f"  split into train={train_df[CLIENT_ID_COL].nunique()} clients, "
        f"valid={valid_df[CLIENT_ID_COL].nunique()} clients (before building records)",
        flush=True,
    )

    print("Building ptls records ...", flush=True)
    train_records, preprocessor = build_ptls_records(train_df)
    cat_sizes = preprocessor.get_category_dictionary_sizes()
    valid_records, _ = build_ptls_records(valid_df, preprocessor=preprocessor)
    print(
        f"  train={len(train_records)} valid={len(valid_records)}, "
        f"category dictionary sizes: {cat_sizes}",
        flush=True,
    )

    # Persisted so downstream inference can transform new data with the
    # SAME category->index mapping this checkpoint's embedding tables were
    # trained against -- fitting a fresh preprocessor on different (e.g.
    # windowed) data would silently produce a different mapping.
    save_preprocessor(ckpt_dir / "preprocessor.pkl", preprocessor)

    module = build_module(
        cat_sizes, embedding_dim=args.embedding_dim, hidden_size=args.hidden_size, num_layers=args.num_layers
    )
    datamodule = build_datamodule(train_records, valid_records, batch_size=args.batch_size)

    last_ckpt = ckpt_dir / "last.ckpt"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=TensorBoardLogger("/app/data/lightning_logs", name="coles"),
        callbacks=[
            EarlyStopping(monitor="valid/recall_top_k", mode="max", patience=args.patience),
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="best",
                monitor="valid/recall_top_k",
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
