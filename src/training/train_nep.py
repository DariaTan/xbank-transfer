"""Full-scale NEP training: all xbank clients, early stopping on
validation loss, resumable checkpointing.

Run parameters live in configs/models/nep.yaml, not CLI flags -- the
only flag this script takes is --config, to point at a different one.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from ptls.data_load.utils import collate_feature_dict
from torch.utils.tensorboard import SummaryWriter

from data.loaders import build_ptls_records, cap_rows_per_client, get_all_client_ids, load_raw_for_clients, sample_client_ids
from data.schema import ALL_FEATURE_COLS, CLIENT_ID_COL, EVENT_TIME_COL, NUMERIC_COLS
from models.nep import NEP
from training.common import (
    EarlyStopper,
    check_or_save_run_config,
    load_checkpoint,
    save_checkpoint,
    save_preprocessor,
    split_df_by_client,
)

MODEL_NAME = "nep"


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        return SimpleNamespace(**yaml.safe_load(f))


def load_data_config(data_config_path: str) -> dict:
    with open(data_config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/app/configs/models/nep.yaml")
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

    columns = [CLIENT_ID_COL, EVENT_TIME_COL] + ALL_FEATURE_COLS
    if args.n_clients is not None:
        # Filters to the sampled clients INSIDE DuckDB, before anything
        # becomes a pandas object -- load_all_raw followed by a pandas-side
        # .sample() still reads and materializes every row first, which is
        # what silently OOM-killed this exact script against MBD's
        # ~550M-row daily table (2026-09-14, n_clients=50000 set but the
        # full table got loaded anyway).
        print(f"Loading transactions table from {cli.data_config}, capped to {args.n_clients} clients ...", flush=True)
        client_ids = sample_client_ids(TRANSACTIONS_PATH, args.n_clients, seed=args.seed)
        df = load_raw_for_clients(TRANSACTIONS_PATH, client_ids, columns=columns)
        print(f"  {len(df)} rows, {df[CLIENT_ID_COL].nunique()} clients", flush=True)
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
    else:
        # Full-scale (no cap): stream through ALL clients in bounded
        # chunks rather than load_all_raw's one-shot full materialization
        # -- that's what silently OOM-killed this exact script against
        # MBD's ~550M-row daily table (2026-09-14). Correctness: a client's
        # rows are NEVER split across chunks (each chunk is a disjoint set
        # of whole clients), and train/valid membership is decided ONCE
        # upfront on the cheap id list, before any chunk is loaded -- so
        # this produces the same split build_ptls_records would see from
        # one giant load, just materialized a chunk at a time. The
        # category vocabulary is fit on the FIRST chunk containing train
        # clients (chunks are pre-shuffled, so this is a random ~100K-
        # client sample, not a positionally-biased one) and reused
        # (transform-only) for every subsequent chunk -- fitting on one
        # chunk rather than the full ~1.48M-client population is the same
        # kind of bounded-sample vocabulary the n_clients-capped path
        # above already relies on, just applied to a bigger sample.
        CHUNK_SIZE = getattr(args, "chunk_size", 100_000)
        print(f"Loading transactions table from {cli.data_config} in chunks (streaming, chunk_size={CHUNK_SIZE} clients) ...", flush=True)
        all_ids = get_all_client_ids(TRANSACTIONS_PATH)
        print(f"  {len(all_ids)} distinct clients total", flush=True)

        id_df = pd.DataFrame({CLIENT_ID_COL: all_ids})
        train_id_df, valid_id_df = split_df_by_client(id_df, CLIENT_ID_COL, args.valid_frac, args.seed)
        train_ids, valid_ids = set(train_id_df[CLIENT_ID_COL]), set(valid_id_df[CLIENT_ID_COL])
        print(f"  split into train={len(train_ids)} clients, valid={len(valid_ids)} clients", flush=True)

        shuffled_ids = np.random.RandomState(args.seed).permutation(all_ids).tolist()
        n_chunks = (len(shuffled_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

        train_records, valid_records, preprocessor = [], [], None
        for chunk_i in range(n_chunks):
            batch_ids = shuffled_ids[chunk_i * CHUNK_SIZE : (chunk_i + 1) * CHUNK_SIZE]
            chunk_df = load_raw_for_clients(TRANSACTIONS_PATH, batch_ids, columns=columns)
            chunk_df = cap_rows_per_client(chunk_df, args.max_seq_len)

            chunk_train_df = chunk_df[chunk_df[CLIENT_ID_COL].isin(train_ids)]
            chunk_valid_df = chunk_df[chunk_df[CLIENT_ID_COL].isin(valid_ids)]
            del chunk_df

            if len(chunk_train_df):
                recs, preprocessor = build_ptls_records(chunk_train_df, preprocessor=preprocessor)
                train_records.extend(recs)
            if len(chunk_valid_df) and preprocessor is not None:
                recs, _ = build_ptls_records(chunk_valid_df, preprocessor=preprocessor)
                valid_records.extend(recs)
            del chunk_train_df, chunk_valid_df

            print(
                f"  chunk {chunk_i + 1}/{n_chunks}: train_records={len(train_records)} valid_records={len(valid_records)}",
                flush=True,
            )

        cat_sizes = preprocessor.get_category_dictionary_sizes()
        print(
            f"  done: train={len(train_records)} valid={len(valid_records)}, "
            f"category dictionary sizes: {cat_sizes}",
            flush=True,
        )

    # Persisted so downstream inference can transform new data with the
    # SAME category->index mapping this checkpoint's embedding tables were
    # trained against -- see train_coles.py's identical comment.
    save_preprocessor(ckpt_dir / "preprocessor.pkl", preprocessor)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NEP(
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

    # Diverges by data_cfg['name'] same as ckpt_dir -- otherwise xbank/mbd/
    # mbd_daily runs of the same model all write into the SAME TensorBoard
    # log dir and their curves interleave indistinguishably (flagged
    # 2026-09-17, while an mbd/mbd_daily run of this exact script was
    # already in progress -- this fix only takes effect on the NEXT
    # launch, not the currently-running process).
    writer = SummaryWriter(f"/app/data/lightning_logs/{data_cfg['name']}_source/{MODEL_NAME}")

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
