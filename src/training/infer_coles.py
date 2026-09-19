"""Full-scale CoLES inference: for every client, one embedding per
calendar date on a fixed monthly grid (1st of each month, 2023-01-01
through 2024-02-01 by default), each built from that client's trailing
`history_window_months` of transaction history up to that date.

Reuses the trained checkpoint's OWN preprocessor (`preprocessor.pkl`,
saved by train_coles.py) to transform the windowed data -- fitting a
fresh preprocessor here would assign different category->index codes
than the ones the checkpoint's embedding tables were trained against,
silently corrupting every embedding. If that file doesn't exist (a
checkpoint trained before this was added), see train_coles.py's comment
on where it's produced.

Run parameters live in configs/models/downstream.yaml's `inference:`
section (shared across all infer_*.py scripts: n_clients/dates/
history_window_months/max_seq_len/batch_size/embeds_dir) and
configs/models/coles.yaml 

Output: one parquet file per target date under
<embeds_dir>/<checkpoint_source>_source/coles/, columns [inn, date,
emb_0..emb_<hidden_size-1>] -- resumable the same way infer_chronos2.py's
chunking is: a date whose file already exists is skipped on the next run.

Usage (inside the container):
    python src/training/infer_coles.py
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.loaders import build_ptls_records, sample_client_ids
from data.schema import CLIENT_ID_COL
from data.splits import load_windowed_transactions_for_dates, unpack_window_id
from models.coles import build_module, extract_embeddings
from training.common import load_preprocessor

MODEL_NAME = "coles"

XBANK_DATA_CONFIG = "/app/configs/data/xbank.yaml"
with open(XBANK_DATA_CONFIG) as f:
    TRANSACTIONS_PATH = yaml.safe_load(f)["paths"]["transactions"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downstream-config", type=str, default="/app/configs/models/downstream.yaml")
    parser.add_argument("--model-config", type=str, default="/app/configs/models/coles.yaml")
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

    print(f"Loading preprocessor from {ckpt_dir / 'preprocessor.pkl'} ...", flush=True)
    preprocessor = load_preprocessor(ckpt_dir / "preprocessor.pkl")
    cat_sizes = preprocessor.get_category_dictionary_sizes()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading CoLES checkpoint from {ckpt_dir / 'best.ckpt'} ...", flush=True)
    module = build_module(
        cat_sizes,
        embedding_dim=model_cfg["embedding_dim"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
    )
    ckpt = torch.load(str(ckpt_dir / "best.ckpt"), map_location=device, weights_only=False)
    module.load_state_dict(ckpt["state_dict"])
    module.to(device)
    print("  loaded.", flush=True)

    client_ids = None
    if inf["n_clients"] is not None:
        client_ids = sample_client_ids(TRANSACTIONS_PATH, inf["n_clients"], seed=inf["seed"])
        print(f"  capped to {len(client_ids)} clients for debugging", flush=True)

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
            client_ids=client_ids,
        )
        if len(windowed) == 0:
            print(f"{target_date}: no rows in window, skipping", flush=True)
            continue

        records, _ = build_ptls_records(windowed, preprocessor=preprocessor)
        window_ids = [r[CLIENT_ID_COL] for r in records]
        inns = [unpack_window_id(str(w))[0] for w in window_ids]

        print(f"{target_date}: extracting embeddings for {len(records)} clients ...", flush=True)
        emb = extract_embeddings(module, records, batch_size=inf["batch_size"], device=device)

        out = pd.DataFrame(emb, columns=[f"emb_{j}" for j in range(emb.shape[1])])
        out.insert(0, "date", target_date)
        out.insert(0, "inn", inns)
        out.to_parquet(out_path)
        print(f"{target_date}: wrote {len(out)} embeddings to {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
