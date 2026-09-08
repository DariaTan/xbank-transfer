"""Full-scale Chronos-2 inference: for every client, one embedding per
calendar date on a fixed monthly grid (1st of each month, 2023-01-01
through 2024-02-01 by default), each built from that client's trailing
`history_window_months` of transaction history up to that date -- same
per-(client, date) design and [inn, date, emb_*] schema as
infer_coles.py/infer_nep.py/infer_mlm.py/infer_thp.py/infer_cotic.py, so
all six models' embeddings are directly comparable (PSI checks,
downstream probes, ...).

Chronos-2 is used zero-shot and frozen (see models.chronos2) -- no
training, so unlike the other five models there is no fitted preprocessor/
categories artifact to reuse, and no configs/models/chronos2.yaml either:
it operates on the raw daily-aggregated amount series directly, not a
learned per-model vocabulary.

Run parameters live in configs/models/downstream.yaml's `inference:`
section, same as the other five scripts, except Chronos-2 is heavier
per-sample so it reads its OWN batch size/chunk size/value column
(`chronos2_batch_size`/`chronos2_chunk_size`/`chronos2_value_col`) instead
of the shared `batch_size`. The only CLI flag is --downstream-config, to
point at a different file.

Within each date, clients are processed in chunks (rather than one
`pipeline.embed()` call over all ~378K clients) so a crash partway
through resumes from the last completed chunk instead of restarting the
whole date: each chunk is written under `<embeds_dir>/chronos2/_chunks/
<date>/`, merged into `<embeds_dir>/chronos2/<date>.parquet` once every
chunk for that date is done, then the chunk files are deleted. A date
whose final merged file already exists is skipped entirely on the next
run.

Usage (inside the container):
    python src/training/infer_chronos2.py
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.loaders import build_chronos_series, sample_client_ids
from data.schema import CLIENT_ID_COL
from data.splits import load_windowed_transactions_for_dates, unpack_window_id
from models.chronos2 import extract_reg_embeddings, load_pipeline

XBANK_DATA_CONFIG = "/app/configs/data/xbank.yaml"
with open(XBANK_DATA_CONFIG) as f:
    TRANSACTIONS_PATH = yaml.safe_load(f)["paths"]["transactions"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downstream-config", type=str, default="/app/configs/models/downstream.yaml")
    cli = parser.parse_args()

    with open(cli.downstream_config) as f:
        inf = yaml.safe_load(f)["inference"]

    out_dir = Path(inf["embeds_dir"]) / "chronos2"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(inf["start_date"], inf["end_date"], freq="MS")]
    print(f"{len(target_dates)} target dates: {target_dates[0]} .. {target_dates[-1]}", flush=True)

    client_ids_filter = None
    if inf["n_clients"] is not None:
        client_ids_filter = sample_client_ids(TRANSACTIONS_PATH, inf["n_clients"], seed=inf["seed"])
        print(f"  capped to {len(client_ids_filter)} clients for debugging", flush=True)

    print("Loading pretrained Chronos-2 (amazon/chronos-2) ...", flush=True)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = load_pipeline(device_map=device_map)
    print("  loaded.", flush=True)

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

        window_ids = windowed[CLIENT_ID_COL].unique().tolist()
        chunk_size = inf["chronos2_chunk_size"]
        chunk_lists = [window_ids[i : i + chunk_size] for i in range(0, len(window_ids), chunk_size)]
        print(f"{target_date}: {len(window_ids)} clients, {len(chunk_lists)} chunks", flush=True)

        chunks_dir = out_dir / "_chunks" / target_date
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk_ids in enumerate(chunk_lists):
            chunk_path = chunks_dir / f"chunk_{i:05d}.parquet"
            if chunk_path.exists():
                continue

            chunk_df = windowed[windowed[CLIENT_ID_COL].isin(chunk_ids)]
            series_list, series_window_ids = build_chronos_series(chunk_df, value_col=inf["chronos2_value_col"])
            inns = [unpack_window_id(str(w))[0] for w in series_window_ids]

            emb = extract_reg_embeddings(pipeline, series_list, batch_size=inf["chronos2_batch_size"])
            out = pd.DataFrame(emb, columns=[f"emb_{j}" for j in range(emb.shape[1])])
            out.insert(0, "date", target_date)
            out.insert(0, "inn", inns)
            out.to_parquet(chunk_path)
            print(f"{target_date}: chunk {i + 1}/{len(chunk_lists)} ({len(inns)} clients)", flush=True)

        merged = pd.concat(
            [pd.read_parquet(p) for p in sorted(chunks_dir.glob("chunk_*.parquet"))], ignore_index=True
        )
        merged.to_parquet(out_path)
        shutil.rmtree(chunks_dir)
        print(f"{target_date}: wrote {len(merged)} embeddings to {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
