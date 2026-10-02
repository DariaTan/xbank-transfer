"""Disposable end-to-end daily-arm smoke; no production checkpoints touched."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
import yaml

from data.schema import CATEGORY_COLS, NUMERIC_COLS
from training.pretrain_cache import prepare_cache
from training.train_daily_encoders import MODELS, probe_production_batch, train


def fixture(root: Path):
    rows = []
    for client in range(64):
        for event in range(1 if client == 63 else 20 + client % 13):
            row = {"id": f"client-{client:03d}",
                   "col_1": pd.Timestamp("2022-01-01") + pd.Timedelta(days=event // 2)}
            row.update({col: (client + event + i) % 4 for i, col in enumerate(CATEGORY_COLS)})
            row.update({col: (event + i) / 100 for i, col in enumerate(NUMERIC_COLS)})
            rows.append(row)
    source = root / "transactions.parquet"
    pd.DataFrame(rows).sample(frac=1, random_state=12).to_parquet(source, index=False)
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"name": "mbd_daily", "event_type_col": "col_2",
                                  "paths": {"transactions": str(source), "targets": str(root / "unused")}}))
    configs = {}
    for model in MODELS:
        cfg = {"seed": 0, "valid_frac": .2, "max_seq_len": 24, "n_clients": None,
               "max_epochs": 2, "patience": 5, "batch_size": 8, "lr": .001,
               "hidden_size": 16, "d_model": 16, "embedding_dim": 4,
               "num_layers": 1, "in_channels": 16, "nb_filters": 16, "nb_layers": 2}
        path = root / f"{model}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        configs[model] = path
    return source, data, configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--real-data-config", help="optional bounded production-data prepare check")
    parser.add_argument("--temp-root", help="place disposable fixtures on storage rather than the root filesystem")
    parser.add_argument("--production-shapes", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="daily-encoder-smoke-", dir=args.temp_root) as temporary:
        root = Path(temporary)
        source, data, configs = fixture(root)
        cache = root / "cache"
        prepare_cache(source, cache, valid_frac=.2, max_seq_len=24, shards=4, threads=2, memory_gb=1)
        results = {}
        for model in MODELS:
            results[model] = train(model, data, configs[model], cache, root / "models" / model,
                                   root / "logs", device=args.device)
            repeated = train(model, data, configs[model], cache, root / "models" / model,
                             root / "logs", device=args.device)
            assert repeated == results[model]
        if args.real_data_config:
            from training.paths import load_data_config
            cfg = load_data_config(args.real_data_config)
            prepare_cache(Path(cfg["paths"]["transactions"]), root / "real-cache",
                          n_clients=128, shards=4, threads=2, memory_gb=2)
        if args.production_shapes:
            for model in MODELS:
                cfg = yaml.safe_load((Path("/app/configs/models") / f"{model}.yaml").read_text())
                probe_production_batch(model, root / "real-cache" if args.real_data_config else cache,
                                       cfg, args.device)
        print("SMOKE PASSED: " + json.dumps({key: {"best_epoch": value["best_epoch"],
                                                   "best_score": value["best_validation_score"]}
                                           for key, value in results.items()}), flush=True)


if __name__ == "__main__":
    main()
