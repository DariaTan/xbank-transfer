"""Full-scale NEP training: all xbank clients, early stopping on
validation loss, resumable checkpointing.

Usage (inside the container; pin a GPU via CUDA_VISIBLE_DEVICES on the
`docker exec` call, see TRAINING.md):
    docker exec -e CUDA_VISIBLE_DEVICES=0 xbank-transfer python scripts/train_nep.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from ptls.data_load.utils import collate_feature_dict


from xbank.data.loaders import build_ptls_records, cap_rows_per_client, load_all_raw
from xbank.data.schema import ALL_FEATURE_COLS, CLIENT_ID_COL, EVENT_TIME_COL, NUMERIC_COLS
from xbank.models.nep import NEP
from xbank.training.common import EarlyStopper, load_checkpoint, save_checkpoint, split_indices

TRANSACTIONS_PATH = "/app/data/trans_any_pos_anonym_encoded.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clients", type=int, default=None, help="cap for a quick debug run; default uses all clients")
    parser.add_argument("--max-seq-len", type=int, default=500)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--valid-frac", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints/nep")
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NEP(
        cat_sizes,
        NUMERIC_COLS,
        d_model=args.d_model,
        num_layers=args.num_layers,
        max_position_embeddings=args.max_seq_len + 8,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=args.patience, mode="min")

    ckpt_dir = Path(args.checkpoint_dir)
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

        save_checkpoint(str(last_path), model, optimizer, epoch, stopper)
        if improved:
            save_checkpoint(str(best_path), model, optimizer, epoch, stopper)

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (best valid_loss={stopper.best:.4f} @ epoch {stopper.best_epoch})", flush=True)
            break

    print(f"Done. Best checkpoint: {best_path} (valid_loss={stopper.best:.4f} @ epoch {stopper.best_epoch})", flush=True)


if __name__ == "__main__":
    main()
