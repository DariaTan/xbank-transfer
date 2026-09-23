"""Validate completeness and numerical integrity of inference outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from data.schema import TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL
from training.embedding_io import validate_embedding_file
from training.paths import embedding_dir, evaluation_name, load_data_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["coles", "cotic", "nep", "mlm", "thp", "chronos2"])
    parser.add_argument("--data-config", default="/app/configs/data/mbd.yaml")
    parser.add_argument("--downstream-config", default="/app/configs/models/downstream_mbd.yaml")
    parser.add_argument("--checkpoint-source", default="mbd")
    parser.add_argument("--min-coverage", type=float, default=0.95)
    args = parser.parse_args()

    data_cfg = load_data_config(args.data_config)
    with open(args.downstream_config) as f:
        inf = yaml.safe_load(f)["inference"]
    targets = pd.read_parquet(
        data_cfg["paths"]["targets"],
        columns=[TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL],
    )
    targets[TARGETS_CLIENT_ID_COL] = targets[TARGETS_CLIENT_ID_COL].astype(str)
    targets[TARGETS_DATE_COL] = targets[TARGETS_DATE_COL].astype(str)
    out_dir = embedding_dir(
        inf["embeds_dir"],
        evaluation_name(data_cfg),
        args.checkpoint_source,
        args.model,
    )

    failed = False
    for date, target_rows in targets.groupby(TARGETS_DATE_COL):
        path = out_dir / f"{date}.parquet"
        if not path.is_file():
            print(f"FAIL {date}: missing {path}")
            failed = True
            continue
        try:
            embeds = validate_embedding_file(path, expected_date=str(date))
        except Exception as exc:
            print(f"FAIL {date}: {exc}")
            failed = True
            continue
        expected = target_rows[TARGETS_CLIENT_ID_COL].nunique()
        actual = embeds["inn"].astype(str).nunique()
        coverage = actual / expected if expected else 1.0
        dim = len([c for c in embeds.columns if c.startswith("emb_")])
        size_mb = path.stat().st_size / (1024 * 1024)
        status = "OK" if coverage >= args.min_coverage else "FAIL"
        print(
            f"{status} {date}: rows={len(embeds)} clients={actual}/{expected} "
            f"coverage={coverage:.2%} dim={dim} size={size_mb:.1f} MiB"
        )
        failed |= coverage < args.min_coverage

    if failed:
        raise SystemExit(1)
    print(f"All embeddings valid: {out_dir}")


if __name__ == "__main__":
    main()
