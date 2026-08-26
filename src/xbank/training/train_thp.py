"""Full-scale THP training: all xbank clients (not a sample), early
stopping on validation NLL, resumable checkpointing so a killed job (GPU
preempted, tmux session killed, host reboot) can continue instead of
restarting from scratch.

Usage (inside the container; pin a GPU via CUDA_VISIBLE_DEVICES on the
`docker exec` call, see TRAINING.md):
    docker exec -e CUDA_VISIBLE_DEVICES=0 xbank-transfer python scripts/train_thp.py
"""
import argparse
from pathlib import Path

import torch


from xbank.data.loaders import build_thp_sequences, cap_rows_per_client, load_all_raw
from xbank.models.thp import (
    build_dataloader,
    build_model,
    build_tokenizer,
    evaluate,
    train_one_epoch,
)
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
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints/thp")
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

    print("Building THP (time, time_delta, type) sequences ...", flush=True)
    time_seqs, time_delta_seqs, type_seqs, num_types = build_thp_sequences(df)
    print(f"  {len(time_seqs)} client sequences, num_types={num_types}", flush=True)

    train_idx, valid_idx = split_indices(len(time_seqs), args.valid_frac, args.seed)

    def subset(lst, ids):
        return [lst[i] for i in ids]

    train_time, train_delta, train_type = (
        subset(time_seqs, train_idx),
        subset(time_delta_seqs, train_idx),
        subset(type_seqs, train_idx),
    )
    valid_time, valid_delta, valid_type = (
        subset(time_seqs, valid_idx),
        subset(time_delta_seqs, valid_idx),
        subset(type_seqs, valid_idx),
    )
    print(f"  train={len(train_time)} valid={len(valid_time)}", flush=True)

    tokenizer = build_tokenizer(num_types, max_len=args.max_seq_len)
    train_loader = build_dataloader(train_time, train_delta, train_type, tokenizer, args.batch_size, shuffle=True)
    valid_loader = build_dataloader(valid_time, valid_delta, valid_type, tokenizer, args.batch_size, shuffle=False)

    gpu = 0 if torch.cuda.is_available() else -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(num_types, hidden_size=args.hidden_size, num_layers=args.num_layers, gpu=gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=args.patience, mode="min")  # lower NLL is better

    ckpt_dir = Path(args.checkpoint_dir)
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    start_epoch = 0
    if last_path.exists():
        start_epoch = load_checkpoint(str(last_path), model, optimizer, stopper, device)
        print(f"Resuming from checkpoint at epoch {start_epoch} (best={stopper.best:.4f} @ {stopper.best_epoch})", flush=True)

    for epoch in range(start_epoch, args.max_epochs):
        train_nll = train_one_epoch(model, train_loader, optimizer)
        valid_nll = evaluate(model, valid_loader)
        improved = stopper.step(valid_nll, epoch)
        status = "(best)" if improved else f"(no improvement, {stopper.bad_epochs}/{args.patience})"
        print(f"epoch {epoch}: train_nll={train_nll:.4f} valid_nll={valid_nll:.4f}  {status}", flush=True)

        save_checkpoint(str(last_path), model, optimizer, epoch, stopper)
        if improved:
            save_checkpoint(str(best_path), model, optimizer, epoch, stopper)

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (best valid_nll={stopper.best:.4f} @ epoch {stopper.best_epoch})", flush=True)
            break

    print(f"Done. Best checkpoint: {best_path} (valid_nll={stopper.best:.4f} @ epoch {stopper.best_epoch})", flush=True)


if __name__ == "__main__":
    main()
