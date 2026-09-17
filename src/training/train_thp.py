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

from data.loaders import build_thp_sequences, cap_rows_per_client, load_all_raw, load_raw_for_clients, sample_client_ids
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

MODEL_NAME = "thp"


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        return SimpleNamespace(**yaml.safe_load(f))


def load_data_config(data_config_path: str) -> dict:
    with open(data_config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/app/configs/models/thp.yaml")
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

    # Which single category column to use as THP's "event type" mark is a
    # property of the DATA, not the architecture -- read from
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

    print("Building THP (time, time_delta, type) sequences ...", flush=True)
    train_time, train_delta, train_type, num_types, categories, _ = build_thp_sequences(train_df, event_type_col=EVENT_TYPE_COL)
    valid_time, valid_delta, valid_type, _, _, _ = build_thp_sequences(valid_df, event_type_col=EVENT_TYPE_COL, categories=categories)
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

    # Diverges by data_cfg['name'] same as ckpt_dir -- otherwise xbank/mbd/
    # mbd_daily runs of the same model all write into the SAME TensorBoard
    # log dir and their curves interleave indistinguishably (flagged
    # 2026-09-17).
    writer = SummaryWriter(f"/app/data/lightning_logs/{data_cfg['name']}_source/{MODEL_NAME}")

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
