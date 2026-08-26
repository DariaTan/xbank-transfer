"""Smoke test: load pretrained Chronos-2 zero-shot, build regular daily
series for a small sample of xbank clients, extract [REG]-token
embeddings, and run one real forecast call as a secondary sanity check.

No training happens here -- Chronos-2 is used exactly as intended,
frozen, zero-shot. Downloads the public amazon/chronos-2 checkpoint from
the Hugging Face Hub on first run (cached afterward).

Usage (inside the container):
    python scripts/smoke_chronos2.py --n-clients 500
"""
import argparse

import numpy as np
import torch


from xbank.data.loaders import (
    build_chronos_series,
    cap_rows_per_client,
    load_raw_for_clients,
    sample_client_ids,
)
from xbank.models.chronos2 import extract_reg_embeddings, load_pipeline

TRANSACTIONS_PATH = "/app/data/trans_any_pos_anonym_encoded.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clients", type=int, default=500)
    parser.add_argument("--max-seq-len", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Sampling {args.n_clients} clients ...", flush=True)
    client_ids = sample_client_ids(TRANSACTIONS_PATH, args.n_clients, seed=args.seed)
    print(f"  got {len(client_ids)} distinct clients", flush=True)

    print("Loading raw rows ...", flush=True)
    df = load_raw_for_clients(TRANSACTIONS_PATH, client_ids)
    df = cap_rows_per_client(df, args.max_seq_len)
    print(f"  {len(df)} rows after capping to {args.max_seq_len}/client", flush=True)

    print("Building regular daily series ...", flush=True)
    series_list = build_chronos_series(df, value_col="col_11", freq="D")
    lengths = [len(s) for s in series_list]
    print(
        f"  {len(series_list)} client series, length min/mean/max = "
        f"{min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}",
        flush=True,
    )

    print("Loading pretrained Chronos-2 (amazon/chronos-2) ...", flush=True)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = load_pipeline(device_map=device_map)
    print("  loaded.", flush=True)

    print("Extracting [REG] embeddings ...", flush=True)
    emb = extract_reg_embeddings(pipeline, series_list, batch_size=64)
    norms = np.linalg.norm(emb, axis=1)
    print(f"Embedding shape: {emb.shape}", flush=True)
    print(f"NaNs: {np.isnan(emb).sum()}, Infs: {np.isinf(emb).sum()}", flush=True)
    print(
        f"Per-sample L2 norm min/mean/max: "
        f"{norms.min():.4f} / {norms.mean():.4f} / {norms.max():.4f}",
        flush=True,
    )

    print("\nSanity check: real zero-shot forecast on 3 series ...", flush=True)
    sample = series_list[:3]
    quantiles, mean = pipeline.predict_quantiles(sample, prediction_length=7)
    for i, (q, m) in enumerate(zip(quantiles, mean)):
        print(f"  series {i}: history_len={len(sample[i])}, forecast mean shape={tuple(m.shape)}, "
              f"quantiles shape={tuple(q.shape)}", flush=True)


if __name__ == "__main__":
    main()
