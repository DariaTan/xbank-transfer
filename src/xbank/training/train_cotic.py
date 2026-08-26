"""Full-scale COTIC training: all xbank clients, early stopping on
validation log-likelihood, resumable via pytorch-lightning's own
checkpointing. Mirrors scripts/train_coles.py.

Usage (inside the container; pin a GPU via CUDA_VISIBLE_DEVICES on the
`docker exec` call, see TRAINING.md):
    docker exec -e CUDA_VISIBLE_DEVICES=1 xbank-transfer python scripts/train_cotic.py
"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from xbank.data.loaders import build_cotic_sequences, cap_rows_per_client, load_all_raw
from xbank.models.cotic import InMemoryEventDataModule, build_datasets, build_module
from xbank.training.common import split_indices

TRANSACTIONS_PATH = "/app/data/trans_any_pos_anonym_encoded.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clients", type=int, default=None, help="cap for a quick debug run; default uses all clients")
    parser.add_argument("--max-seq-len", type=int, default=500)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--valid-frac", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--nb-filters", type=int, default=128)
    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--nb-layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints/cotic")
    args = parser.parse_args()

    print("Loading full transactions table (id, col_1, col_2) ...", flush=True)
    df = load_all_raw(TRANSACTIONS_PATH, columns=["id", "col_1", "col_2"])
    print(f"  {len(df)} rows, {df['id'].nunique()} clients", flush=True)

    if args.n_clients is not None:
        keep = df["id"].drop_duplicates().sample(n=args.n_clients, random_state=args.seed)
        df = df[df["id"].isin(keep)]
        print(f"  capped to {args.n_clients} clients for debugging", flush=True)

    df = cap_rows_per_client(df, args.max_seq_len)
    print(f"  {len(df)} rows after capping to {args.max_seq_len}/client", flush=True)

    print("Building COTIC (times, types) sequences ...", flush=True)
    times, types, num_types = build_cotic_sequences(df)
    print(f"  {len(times)} client sequences, num_types={num_types}", flush=True)

    train_idx, valid_idx = split_indices(len(times), args.valid_frac, args.seed)
    train_times = [times[i] for i in train_idx]
    train_types = [types[i] for i in train_idx]
    valid_times = [times[i] for i in valid_idx]
    valid_types = [types[i] for i in valid_idx]
    print(f"  train={len(train_times)} valid={len(valid_times)}", flush=True)

    train_dataset, valid_dataset, normalizer = build_datasets(
        train_times, train_types, valid_times, valid_types, num_types
    )
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

    ckpt_dir = Path(args.checkpoint_dir)
    last_ckpt = ckpt_dir / "last.ckpt"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=TensorBoardLogger("outputs/lightning_logs", name="cotic_full"),
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
