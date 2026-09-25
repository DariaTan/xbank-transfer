import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class XbankLightgbmPairedTests(unittest.TestCase):
    def test_two_variants_use_identical_cohort_and_test_rows(self):
        try:
            from training import tune_lightgbm_xbank_paired as paired
        except (ImportError, OSError):
            self.skipTest("LightGBM native library is unavailable locally")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            targets, old, new = [], [], []
            dates = ("2023-01-01", "2023-02-01", "2024-01-01", "2024-02-01")
            for client in range(10):
                for index, date in enumerate(dates):
                    key = {"id": f"client-{client}", "col_1": date}
                    targets.append({**key, "col_3": (client + index) % 2,
                                    "col_4": (client + index + 1) % 2,
                                    "col_5": index % 2})
                    row = {"inn": key["id"], "date": date,
                           "emb_0": float(client), "emb_1": float(index)}
                    old.append(row)
                    if not (client == 0 and date == "2024-02-01"):
                        new.append({**row, "emb_0": row["emb_0"] + 0.5})
            targets_path = root / "targets.parquet"
            old_path = root / "old.parquet"
            new_path = root / "new.parquet"
            pd.DataFrame(targets).to_parquet(targets_path, index=False)
            pd.DataFrame(old).to_parquet(old_path, index=False)
            pd.DataFrame(new).to_parquet(new_path, index=False)
            files = {"xbank": [old_path], "xbank_fgw_v2": [new_path]}
            cfg = {"train": list(dates[:2]), "test": list(dates[2:]),
                   "target_cols": list(paired.TARGETS), "val_frac": 0.5, "seed": 0}
            cohort, train, val, test = paired.paired_cohort(
                "coles", targets_path, cfg, files)
            self.assertEqual(len(cohort), 39)
            self.assertEqual(len(train) + len(val), 20)
            self.assertEqual(len(test), 19)
            for variant in paired.VARIANTS:
                features, cols = paired._features(files[variant], cohort)
                self.assertEqual(features.shape, (39, 2))
                self.assertEqual(cols, ["emb_0", "emb_1"])

            with (patch.object(paired, "load_config", return_value=cfg),
                  patch.object(paired, "load_data_config",
                               return_value={"paths": {"targets": str(targets_path)}}),
                  patch.object(paired, "_embedding_files",
                               side_effect=lambda _model, variant: files[variant])):
                paired.run("coles", "cpu", trials=1, threads=1, max_rounds=3,
                           seed=42, output_root=str(root / "downstream"),
                           config_path="unused", gpu_platform_id=0, gpu_device_id=0)
                paired.run("coles", "cpu", trials=1, threads=1, max_rounds=3,
                           seed=42, output_root=str(root / "downstream"),
                           config_path="unused", gpu_platform_id=0, gpu_device_id=0)
            manifests = []
            for variant in paired.VARIANTS:
                output = (root / "downstream" / variant / "mbd_source" / "coles" /
                          "lightgbm_xbank_paired")
                manifests.append(json.loads((output / "run_manifest.json").read_text()))
                for target in paired.TARGETS:
                    result = json.loads((output / f"{target}_metrics.json").read_text())
                    self.assertEqual(result["test_metrics"]["n_rows"], 19)
                    self.assertTrue((output / f"{target}_model.txt").is_file())
            self.assertEqual(manifests[0]["cohort_sha256"],
                             manifests[1]["cohort_sha256"])


if __name__ == "__main__":
    unittest.main()
