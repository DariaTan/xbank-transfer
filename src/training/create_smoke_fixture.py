"""Create a tiny adapted-MBD fixture for local end-to-end inference tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.schema import CATEGORY_COLS, NUMERIC_COLS, TARGET_COLS
from training.paths import data_root


def main() -> None:
    root = data_root()
    out_dir = root / "smoke" / "mbd_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    categories_path = root / "checkpoints" / "mbd_source" / "cotic" / "categories.npy"
    event_type = 0
    if categories_path.is_file():
        categories = np.load(categories_path, allow_pickle=True)
        if len(categories):
            event_type = categories[0]

    clients = [f"smoke_{i:03d}" for i in range(20)]
    transaction_clients = clients + ["unlabeled_should_not_appear"]
    rows = []
    for client_i, client in enumerate(transaction_clients):
        for event_i, date in enumerate(pd.date_range("2022-08-01", periods=6, freq="30D")):
            row = {"id": client, "col_1": date}
            for col_i, col in enumerate(CATEGORY_COLS):
                row[col] = event_type if col == "col_2" else (client_i + event_i + col_i) % 3
            for col_i, col in enumerate(NUMERIC_COLS):
                row[col] = float((client_i + event_i + col_i) % 10) / 10.0
            rows.append(row)
    pd.DataFrame(rows).to_parquet(out_dir / "transactions.parquet", index=False)

    target_rows = []
    for client_i, client in enumerate(clients):
        row = {"id": client, "col_1": "2023-01-01", "fold": client_i % 5}
        for target_i, target in enumerate(TARGET_COLS):
            row[target] = int((client_i + target_i) % 2 == 0)
        target_rows.append(row)
    pd.DataFrame(target_rows).to_parquet(out_dir / "targets.parquet", index=False)
    print(f"Wrote smoke fixture to {out_dir} ({len(transaction_clients)} transaction clients, {len(clients)} labeled)")


if __name__ == "__main__":
    main()
