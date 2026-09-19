"""Full-scale THP inference: for every client, one embedding per calendar
date on a fixed monthly grid (1st of each month, 2023-01-01 through
2024-02-01 by default), each built from that client's trailing
`history_window_months` of transaction history up to that date.

The embedding is the encoder's hidden state at the last real event of a
full (non-shifted) forward pass over that window (`extract_embeddings`,
see models/thp.py) -- no future leakage, since the window itself
never extends past the target date.

Reuses the trained checkpoint's OWN `categories` factorization
(`categories.npy`, saved by train_thp.py) to map event types (col_2) --
fitting `pd.factorize` fresh on windowed data would assign different ids
than the ones the checkpoint's `layer_type_emb`/intensity heads were
trained against. See infer_coles.py's identical rationale for the ptls
preprocessor.

Run parameters live in configs/models/downstream.yaml's `inference:`
section (shared across all infer_*.py scripts) and configs/models/thp.yaml

Output: one parquet file per target date under
<embeds_dir>/<checkpoint_source>_source/thp/, columns [inn, date,
emb_0..emb_<hidden_size-1>] -- resumable, a date whose file already
exists is skipped on the next run.

Usage (inside the container):
    python src/training/infer_thp.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data.loaders import build_thp_sequences, sample_client_ids
from data.splits import load_windowed_transactions_for_dates, unpack_window_id
from models.thp import build_dataloader, build_model, build_tokenizer, extract_embeddings

MODEL_NAME = "thp"

XBANK_DATA_CONFIG = "/app/configs/data/xbank.yaml"
with open(XBANK_DATA_CONFIG) as f:
    TRANSACTIONS_PATH = yaml.safe_load(f)["paths"]["transactions"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downstream-config", type=str, default="/app/configs/models/downstream.yaml")
    parser.add_argument("--model-config", type=str, default="/app/configs/models/thp.yaml")
    parser.add_argument(
        "--checkpoint-source",
        type=str,
        default="mbd",
        help=(
            "which pretrained checkpoint to load, by the data_config 'name' it was "
            "trained with -- checkpoint_dir becomes /app/data/checkpoints/<source>_source/"
            f"{MODEL_NAME}. Default mbd (the primary pretraining corpus); pass xbank for "
            "the historical reference-baseline checkpoint, or mbd_daily for that variant."
        ),
    )
    cli = parser.parse_args()

    with open(cli.downstream_config) as f:
        inf = yaml.safe_load(f)["inference"]
    with open(cli.model_config) as f:
        model_cfg = yaml.safe_load(f)

    ckpt_dir = Path(f"/app/data/checkpoints/{cli.checkpoint_source}_source/{MODEL_NAME}")
    out_dir = Path(inf["embeds_dir"]) / f"{cli.checkpoint_source}_source" / MODEL_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(inf["start_date"], inf["end_date"], freq="MS")]
    print(f"{len(target_dates)} target dates: {target_dates[0]} .. {target_dates[-1]}", flush=True)

    print(f"Loading categories from {ckpt_dir / 'categories.npy'} ...", flush=True)
    categories = np.load(ckpt_dir / "categories.npy", allow_pickle=True)
    num_types = len(categories)

    gpu = 0 if torch.cuda.is_available() else -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading THP checkpoint from {ckpt_dir / 'best.pt'} ...", flush=True)
    model = build_model(num_types, hidden_size=model_cfg["hidden_size"], num_layers=model_cfg["num_layers"], gpu=gpu)
    ckpt = torch.load(str(ckpt_dir / "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print("  loaded.", flush=True)

    tokenizer = build_tokenizer(num_types, max_len=inf["max_seq_len"])

    client_ids_filter = None
    if inf["n_clients"] is not None:
        client_ids_filter = sample_client_ids(TRANSACTIONS_PATH, inf["n_clients"], seed=inf["seed"])
        print(f"  capped to {len(client_ids_filter)} clients for debugging", flush=True)

    for target_date in target_dates:
        out_path = out_dir / f"{target_date}.parquet"
        if out_path.exists():
            print(f"{target_date}: already done, skipping", flush=True)
            continue

        print(f"{target_date}: windowing transactions ...", flush=True)
        windowed = load_windowed_transactions_for_dates(
            TRANSACTIONS_PATH,
            [target_date],
            inf["history_window_months"],
            inf["max_seq_len"],
            client_ids=client_ids_filter,
        )
        if len(windowed) == 0:
            print(f"{target_date}: no rows in window, skipping", flush=True)
            continue

        time_seqs, delta_seqs, type_seqs, _, _, window_ids = build_thp_sequences(windowed, categories=categories)
        inns = [unpack_window_id(str(w))[0] for w in window_ids]

        loader = build_dataloader(time_seqs, delta_seqs, type_seqs, tokenizer, inf["batch_size"], shuffle=False)

        print(f"{target_date}: extracting embeddings for {len(window_ids)} clients ...", flush=True)
        emb = extract_embeddings(model, loader)
        assert len(emb) == len(inns), f"row count mismatch: {len(emb)} embeddings vs {len(inns)} ids"

        out = pd.DataFrame(emb, columns=[f"emb_{j}" for j in range(emb.shape[1])])
        out.insert(0, "date", target_date)
        out.insert(0, "inn", inns)
        out.to_parquet(out_path)
        print(f"{target_date}: wrote {len(out)} embeddings to {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
