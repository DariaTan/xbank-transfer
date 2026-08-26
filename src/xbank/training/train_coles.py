"""Full-scale CoLES training: all xbank clients, early stopping on
validation recall@k, resumable via pytorch-lightning's own checkpointing
(`ckpt_path="last.ckpt"` picks up optimizer/scheduler/epoch state too, not
just weights).

Usage (inside the container; pin a GPU via CUDA_VISIBLE_DEVICES on the
`docker exec` call, see TRAINING.md):
    docker exec -e CUDA_VISIBLE_DEVICES=0 xbank-transfer python scripts/train_coles.py
"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from xbank.data.loaders import build_ptls_records, cap_rows_per_client, load_all_raw
from xbank.data.schema import ALL_FEATURE_COLS, CLIENT_ID_COL, EVENT_TIME_COL
from xbank.models.coles import build_datamodule, build_module
from xbank.training.common import split_indices

TRANSACTIONS_PATH = "/app/data/trans_any_pos_anonym_encoded.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clients", type=int, default=None, help="cap for a quick debug run; default uses all clients")
    parser.add_argument("--max-seq-len", type=int, default=500)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--valid-frac", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints/coles")
    args = parser.parse_args()

    print("Loading full transactions table ...", flush=True)
    df = load_all_raw(TRANSACTIONS_PATH, columns=[CLIENT_ID_COL, EVENT_TIME_COL] + ALL_FEATURE_COLS)
    print(f"  {len(df)} rows, {df[CLIENT_ID_COL].nunique()} clients", flush=True)

    if args.n_clients is not None:
        keep = df[CLIENT_ID_COL].drop_duplicates().sample(n=args.n_clients, random_state=args.seed)
        df = df[df[CLIENT_ID_COL].isin(keep)]
        print(f"  capped to {args.n_clients} clients for debugging", flush=True)

    df = cap_rows_per_client(df, args.max_seq_len)
    print(f"  {len(df)} rows after capping to {args.max_seq_len}/client", flush=True)

    print("Building ptls records ...", flush=True)
    records, preprocessor = build_ptls_records(df)
    cat_sizes = preprocessor.get_category_dictionary_sizes()
    print(f"  {len(records)} client records, category dictionary sizes: {cat_sizes}", flush=True)

    train_idx, valid_idx = split_indices(len(records), args.valid_frac, args.seed)
    train_records = [records[i] for i in train_idx]
    valid_records = [records[i] for i in valid_idx]
    print(f"  train={len(train_records)} valid={len(valid_records)}", flush=True)

    module = build_module(
        cat_sizes, embedding_dim=args.embedding_dim, hidden_size=args.hidden_size, num_layers=args.num_layers
    )
    datamodule = build_datamodule(train_records, valid_records, batch_size=args.batch_size)

    ckpt_dir = Path(args.checkpoint_dir)
    last_ckpt = ckpt_dir / "last.ckpt"

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=TensorBoardLogger("outputs/lightning_logs", name="coles_full"),
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
