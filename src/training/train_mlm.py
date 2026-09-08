"""Full-scale MLM training: all xbank clients, early stopping on
validation loss, resumable checkpointing. Mirrors train_nep.py.

Run parameters live in configs/models/mlm.yaml, not CLI flags -- the
only flag this script takes is --config, to point at a different one.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from ptls.data_load.utils import collate_feature_dict
from torch.utils.tensorboard import SummaryWriter

from data.loaders import build_ptls_records, cap_rows_per_client, load_all_raw
from data.schema import ALL_FEATURE_COLS, CLIENT_ID_COL, EVENT_TIME_COL, NUMERIC_COLS
from models.mlm import MLM
from training.common import (
    EarlyStopper,
    check_or_save_run_config,
    load_checkpoint,
    save_checkpoint,
    save_preprocessor,
    split_df_by_client,
)

XBANK_DATA_CONFIG = "/app/configs/data/xbank.yaml"
with open(XBANK_DATA_CONFIG) as f:
    TRANSACTIONS_PATH = yaml.safe_load(f)["paths"]["transactions"]


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        return SimpleNamespace(**yaml.safe_load(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/app/configs/models/mlm.yaml")
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
    # trained against -- see train_coles.py's identical comment.
    save_preprocessor(ckpt_dir / "preprocessor.pkl", preprocessor)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLM(
        cat_sizes,
        NUMERIC_COLS,
        d_model=args.d_model,
        num_layers=args.num_layers,
        max_position_embeddings=args.max_seq_len,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=args.patience, mode="min")

    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    start_epoch = 0
    if last_path.exists():
        start_epoch = load_checkpoint(str(last_path), model, optimizer, stopper, device)
        print(f"Resuming from checkpoint at epoch {start_epoch} (best={stopper.best:.4f} @ {stopper.best_epoch})", flush=True)

    def batches(recs, batch_size, shuffle):
        order = np.random.permutation(len(recs)) if shuffle else np.arange(len(recs))
        for i in range(0, len(order), batch_size):
            chunk = [recs[j] for j in order[i : i + batch_size]]
            yield collate_feature_dict(chunk).to(device)

    writer = SummaryWriter("/app/data/lightning_logs/mlm")

    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        total, n = 0.0, 0
        for batch in batches(train_records, args.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = model.loss(batch.payload, batch.seq_len_mask)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        train_loss = total / max(n, 1)

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in batches(valid_records, args.batch_size, shuffle=False):
                loss = model.loss(batch.payload, batch.seq_len_mask)
                total += loss.item()
                n += 1
        valid_loss = total / max(n, 1)

        improved = stopper.step(valid_loss, epoch)
        status = "(best)" if improved else f"(no improvement, {stopper.bad_epochs}/{args.patience})"
        print(f"epoch {epoch}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}  {status}", flush=True)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("valid/loss", valid_loss, epoch)

        save_checkpoint(str(last_path), model, optimizer, epoch, stopper)
        if improved:
            save_checkpoint(str(best_path), model, optimizer, epoch, stopper)

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (best valid_loss={stopper.best:.4f} @ epoch {stopper.best_epoch})", flush=True)
            break

    writer.close()
    print(f"Done. Best checkpoint: {best_path} (valid_loss={stopper.best:.4f} @ epoch {stopper.best_epoch})", flush=True)


if __name__ == "__main__":
    main()
