"""Full-scale NEP inference: for every client, one embedding per calendar
date on a fixed monthly grid (1st of each month, 2023-01-01 through
2024-02-01 by default), each built from that client's trailing
`history_window_months` of transaction history up to that date.

The embedding is the last-real-event hidden state from a full causal
encoding of that window (`NEP.encode` + `last_event_embedding`, see
models/nep.py) -- no future leakage, since attention is causal and
the window itself never extends past the target date.

Reuses the trained checkpoint's OWN preprocessor (`preprocessor.pkl`,
saved by train_nep.py) -- see infer_coles.py's identical rationale.

Run parameters live in configs/models/downstream.yaml's `inference:`
section (shared across all infer_*.py scripts) and configs/models/nep.yaml
(this architecture's own d_model/num_layers/checkpoint_dir -- must match
what the checkpoint was actually trained with). The only CLI flags are
--downstream-config/--model-config, to point at different files.

Output: one parquet file per target date under <embeds_dir>/nep/, columns
[inn, date, emb_0..emb_<d_model-1>] -- resumable, a date whose file
already exists is skipped on the next run.

Usage (inside the container):
    python src/training/infer_nep.py
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.loaders import build_ptls_records, sample_client_ids
from data.schema import CLIENT_ID_COL, NUMERIC_COLS
from data.splits import load_windowed_transactions_for_dates, unpack_window_id
from models.nep import NEP, extract_embeddings
from training.common import load_preprocessor

XBANK_DATA_CONFIG = "/app/configs/data/xbank.yaml"
with open(XBANK_DATA_CONFIG) as f:
    TRANSACTIONS_PATH = yaml.safe_load(f)["paths"]["transactions"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downstream-config", type=str, default="/app/configs/models/downstream.yaml")
    parser.add_argument("--model-config", type=str, default="/app/configs/models/nep.yaml")
    cli = parser.parse_args()

    with open(cli.downstream_config) as f:
        inf = yaml.safe_load(f)["inference"]
    with open(cli.model_config) as f:
        model_cfg = yaml.safe_load(f)

    ckpt_dir = Path(model_cfg["checkpoint_dir"])
    out_dir = Path(inf["embeds_dir"]) / "nep"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(inf["start_date"], inf["end_date"], freq="MS")]
    print(f"{len(target_dates)} target dates: {target_dates[0]} .. {target_dates[-1]}", flush=True)

    print(f"Loading preprocessor from {ckpt_dir / 'preprocessor.pkl'} ...", flush=True)
    preprocessor = load_preprocessor(ckpt_dir / "preprocessor.pkl")
    cat_sizes = preprocessor.get_category_dictionary_sizes()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading NEP checkpoint from {ckpt_dir / 'best.pt'} ...", flush=True)
    model = NEP(
        cat_sizes,
        NUMERIC_COLS,
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        max_position_embeddings=inf["max_seq_len"],
    )
    ckpt = torch.load(str(ckpt_dir / "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
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
        emb = extract_embeddings(model, records, batch_size=inf["batch_size"], device=device)

        out = pd.DataFrame(emb, columns=[f"emb_{j}" for j in range(emb.shape[1])])
        out.insert(0, "date", target_date)
        out.insert(0, "inn", inns)
        out.to_parquet(out_path)
        print(f"{target_date}: wrote {len(out)} embeddings to {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
