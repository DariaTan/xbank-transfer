"""Full-scale THP training: all xbank clients (not a sample), early
stopping on validation NLL, resumable checkpointing so a killed job (GPU
preempted, tmux session killed, host reboot) can continue instead of
restarting from scratch.

Run parameters live in configs/models/thp.yaml, not CLI flags -- the
only flag this script takes is --config, to point at a different one.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from data.loaders import build_thp_sequences, cap_rows_per_client, load_all_raw
from models.thp import (
    build_dataloader,
    build_model,
    build_tokenizer,
    evaluate,
    train_one_epoch,
)
from training.common import (
    EarlyStopper,
    check_or_save_run_config,
    load_checkpoint,
    save_checkpoint,
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
    parser.add_argument("--config", type=str, default="/app/configs/models/thp.yaml")
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

    print("Building THP (time, time_delta, type) sequences ...", flush=True)
    train_time, train_delta, train_type, num_types, categories, _ = build_thp_sequences(train_df)
    valid_time, valid_delta, valid_type, _, _, _ = build_thp_sequences(valid_df, categories=categories)
    print(
        f"  train={len(train_time)} valid={len(valid_time)}, num_types={num_types}",
        flush=True,
    )

    # Persisted so downstream inference maps event types (col_2) to the
    # SAME ids this checkpoint's layer_type_emb/intensity heads were
    # trained against -- fitting `pd.factorize` fresh on different data
    # would silently produce a different mapping (same class of bug as
    # ptls's category vocabulary, see train_coles.py's identical comment).
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / "categories.npy", categories)

    tokenizer = build_tokenizer(num_types, max_len=args.max_seq_len)
    train_loader = build_dataloader(train_time, train_delta, train_type, tokenizer, args.batch_size, shuffle=True)
    valid_loader = build_dataloader(valid_time, valid_delta, valid_type, tokenizer, args.batch_size, shuffle=False)

    gpu = 0 if torch.cuda.is_available() else -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(num_types, hidden_size=args.hidden_size, num_layers=args.num_layers, gpu=gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=args.patience, mode="min")  # lower NLL is better

    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    start_epoch = 0
    if last_path.exists():
        start_epoch = load_checkpoint(str(last_path), model, optimizer, stopper, device)
        print(f"Resuming from checkpoint at epoch {start_epoch} (best={stopper.best:.4f} @ {stopper.best_epoch})", flush=True)

    writer = SummaryWriter("/app/data/lightning_logs/thp")

    for epoch in range(start_epoch, args.max_epochs):
        train_nll = train_one_epoch(model, train_loader, optimizer)
        valid_nll = evaluate(model, valid_loader)
        improved = stopper.step(valid_nll, epoch)
        status = "(best)" if improved else f"(no improvement, {stopper.bad_epochs}/{args.patience})"
        print(f"epoch {epoch}: train_nll={train_nll:.4f} valid_nll={valid_nll:.4f}  {status}", flush=True)
        writer.add_scalar("train/nll", train_nll, epoch)
        writer.add_scalar("valid/nll", valid_nll, epoch)

        save_checkpoint(str(last_path), model, optimizer, epoch, stopper)
        if improved:
            save_checkpoint(str(best_path), model, optimizer, epoch, stopper)

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (best valid_nll={stopper.best:.4f} @ epoch {stopper.best_epoch})", flush=True)
            break

    writer.close()
    print(f"Done. Best checkpoint: {best_path} (valid_nll={stopper.best:.4f} @ epoch {stopper.best_epoch})", flush=True)


if __name__ == "__main__":
    main()
