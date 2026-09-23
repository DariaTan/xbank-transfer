import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from training.tune_lightgbm_mbd import candidate_params, run


class MbdLightgbmHpoTests(unittest.TestCase):
    def test_search_is_deterministic_and_starts_with_baseline(self):
        first = candidate_params(42, 3, 2)
        self.assertEqual(first, candidate_params(42, 3, 2))
        self.assertEqual(first[0]["num_leaves"], 31)
        self.assertEqual(first[0]["num_threads"], 2)

    def test_end_to_end_keeps_fold_four_for_final_test(self):
        try:
            import lightgbm  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("LightGBM native library is unavailable locally")
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"XBANK_DATA_ROOT": temp}):
            root = Path(temp)
            targets_path = root / "mbd_data" / "raw_adapted" / "targets.parquet"
            targets_path.parent.mkdir(parents=True)
            rows, embeddings = [], []
            for fold in range(5):
                for client in range(4):
                    for date in ("2023-01-01", "2023-02-01"):
                        client_id = f"f{fold}-c{client}"
                        rows.append({"id": client_id, "col_1": date, "fold": fold,
                                     "col_2": client % 2, "col_3": (client + 1) % 2,
                                     "col_4": client % 2, "col_5": (client + 1) % 2})
                        embeddings.append({"inn": client_id, "date": date,
                                           "emb_0": float(client), "emb_1": float(fold) / 10})
            pd.DataFrame(rows).to_parquet(targets_path, index=False)
            embeds_dir = root / "embeds" / "mbd_raw" / "mbd_source" / "coles"
            embeds_dir.mkdir(parents=True)
            pd.DataFrame(embeddings).to_parquet(embeds_dir / "all.parquet", index=False)
            cfg = root / "mbd.yaml"
            cfg.write_text("evaluation_name: mbd_raw\npaths:\n  targets: /app/data/mbd_data/raw_adapted/targets.parquet\n")
            run("coles", str(cfg), "/app/data/downstream", seed=42, trials=1,
                threads=1, tune_client_cap=10, max_rounds=10)
            result = root / "downstream" / "mbd_raw" / "mbd_source" / "coles" / "lightgbm_hpo_holdout"
            for target in ("col_2", "col_3", "col_4", "col_5"):
                score = json.loads((result / f"{target}_metrics.json").read_text())
                self.assertEqual(score["test_metrics"]["n_rows"], 8)
                self.assertTrue(np.isfinite(score["test_metrics"]["pr_auc"]))
                self.assertTrue((result / f"{target}_model.txt").is_file())


if __name__ == "__main__":
    unittest.main()
