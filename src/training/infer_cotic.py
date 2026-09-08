"""Full-scale COTIC inference: for every client, one embedding per
calendar date on a fixed monthly grid (1st of each month, 2023-01-01
through 2024-02-01 by default), each built from that client's trailing
`--history-window-months` of transaction history up to that date.

The embedding is the raw COTIC encoder's (`module.net`) hidden state at
the last real event of a full (non-shifted) forward pass over that
window (`extract_embeddings`, see models/cotic.py) -- no future leakage,
since the window itself never extends past the target date.

Reuses the trained checkpoint's OWN `categories` factorization
(`categories.npy`) AND `normalizer` (`normalizer.pkl`, both saved by
train_cotic.py) -- fitting either fresh on windowed data would assign
different event-type ids / a different inter-event-time scale than the
ones the checkpoint's embedding tables were trained against. See
infer_thp.py's identical rationale for `categories`; the normalizer is
the same idea applied to COTIC's inter-event-time input.

Run parameters live in configs/models/downstream.yaml's `inference:`
section (shared across all infer_*.py scripts) and
configs/models/cotic.yaml (this architecture's own in_channels/
nb_filters/nb_layers/checkpoint_dir -- must match what the checkpoint was
actually trained with). The only CLI flags are --downstream-config/
--model-config, to point at different files.

Output: one parquet file per target date under <embeds_dir>/cotic/,
columns [inn, date, emb_0..emb_<nb_filters-1>] -- resumable, a date whose
file already exists is skipped on the next run.

Usage (inside the container):
    python src/training/infer_cotic.py
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data.loaders import build_cotic_sequences, sample_client_ids
from data.splits import load_windowed_transactions_for_dates, unpack_window_id
from models.cotic import EventDataset, build_module, extract_embeddings

XBANK_DATA_CONFIG = "/app/configs/data/xbank.yaml"
with open(XBANK_DATA_CONFIG) as f:
    TRANSACTIONS_PATH = yaml.safe_load(f)["paths"]["transactions"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downstream-config", type=str, default="/app/configs/models/downstream.yaml")
    parser.add_argument("--model-config", type=str, default="/app/configs/models/cotic.yaml")
    cli = parser.parse_args()

    with open(cli.downstream_config) as f:
        inf = yaml.safe_load(f)["inference"]
    with open(cli.model_config) as f:
        model_cfg = yaml.safe_load(f)

    ckpt_dir = Path(model_cfg["checkpoint_dir"])
    out_dir = Path(inf["embeds_dir"]) / "cotic"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(inf["start_date"], inf["end_date"], freq="MS")]
    print(f"{len(target_dates)} target dates: {target_dates[0]} .. {target_dates[-1]}", flush=True)

    print(f"Loading categories from {ckpt_dir / 'categories.npy'} ...", flush=True)
    categories = np.load(ckpt_dir / "categories.npy", allow_pickle=True)
    num_types = len(categories)

    print(f"Loading normalizer from {ckpt_dir / 'normalizer.pkl'} ...", flush=True)
    with open(ckpt_dir / "normalizer.pkl", "rb") as f:
        normalizer = pickle.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading COTIC checkpoint from {ckpt_dir / 'best.ckpt'} ...", flush=True)
    module = build_module(
        num_types,
        in_channels=model_cfg["in_channels"],
        nb_filters=model_cfg["nb_filters"],
        nb_layers=model_cfg["nb_layers"],
    )
    ckpt = torch.load(str(ckpt_dir / "best.ckpt"), map_location=device, weights_only=False)
    module.load_state_dict(ckpt["state_dict"])
    net = module.net.to(device)
    net.eval()
    print("  loaded.", flush=True)

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

        times, types, _, _, window_ids = build_cotic_sequences(windowed, categories=categories)
        inns = [unpack_window_id(str(w))[0] for w in window_ids]

        dataset = EventDataset(times, types, num_types)
        dataset.normalize_data(normalizer)

        print(f"{target_date}: extracting embeddings for {len(window_ids)} clients ...", flush=True)
        emb = extract_embeddings(net, dataset, batch_size=inf["batch_size"], device=device)
        assert len(emb) == len(inns), f"row count mismatch: {len(emb)} embeddings vs {len(inns)} ids"

        out = pd.DataFrame(emb, columns=[f"emb_{j}" for j in range(emb.shape[1])])
        out.insert(0, "date", target_date)
        out.insert(0, "inn", inns)
        out.to_parquet(out_path)
        print(f"{target_date}: wrote {len(out)} embeddings to {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
