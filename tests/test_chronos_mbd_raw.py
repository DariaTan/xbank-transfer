import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from training.infer_chronos_mbd_raw import _manifest, _series_for_date, finalize, main, prepare


class ChronosRawTests(unittest.TestCase):
    def test_daily_cli_uses_daily_input_and_bounded_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            transactions = root / "daily.parquet"
            targets = root / "targets.parquet"
            pd.DataFrame({"id": ["a", "a"],
                          "col_1": pd.to_datetime(["2022-12-30", "2023-01-01"]),
                          "col_11": [.25, 10.]}).to_parquet(transactions, index=False)
            pd.DataFrame({"id": ["a"], "col_1": ["2023-01-01"]}).to_parquet(targets, index=False)
            config = root / "daily.yaml"
            config.write_text(f"name: mbd_daily\nevaluation_name: mbd_daily\npaths:\n"
                              f"  transactions: {transactions}\n  targets: {targets}\n")
            downstream = root / "downstream.yaml"
            downstream.write_text("inference:\n  history_window_months: 12\n  embeds_dir: /app/data/embeds\n")
            argv = ["infer_chronos_mbd_raw.py", "prepare", "--data-config", str(config),
                    "--downstream-config", str(downstream), "--n-shards", "1",
                    "--prepare-memory-gb", "1", "--prepare-threads", "1"]
            with patch.dict(os.environ, {"XBANK_DATA_ROOT": temp}), patch("sys.argv", argv):
                main()
            cached = pd.read_parquet(root / "chronos2_daily_cache/mbd_daily/shards")
            self.assertEqual(cached.value.tolist(), [.25])
            self.assertTrue((root / "embeds/mbd_daily/zero_shot/chronos2/run_manifest.json").exists())

    def test_xbank_manifest_uses_matched_amount_column(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            transactions = root / "transactions.parquet"
            targets = root / "targets.parquet"
            pd.DataFrame({
                "id": ["a", "a"],
                "col_1": pd.to_datetime(["2022-12-30", "2022-12-30"]),
                "col_11": [100.0, 200.0],
                "col_12": [1.0, 2.0],
            }).to_parquet(transactions)
            pd.DataFrame({"id": ["a"], "col_1": ["2023-01-01"]}).to_parquet(targets)
            manifest = _manifest(transactions, targets, ["2023-01-01"], months=12, shards=1,
                                 value_col="col_12", mapping_sha256="matching-hash")
            with patch.dict(os.environ, {"XBANK_DATA_ROOT": str(root)}):
                prepare(root / "cache", root / "output", manifest)
            daily = pd.read_parquet(root / "cache" / "shards")
            self.assertEqual(daily.value.tolist(), [3.0])
            self.assertEqual(manifest["schema_mapping_sha256"], "matching-hash")

    def test_prepare_aggregates_full_days_and_excludes_target_day(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            transactions = root / "transactions.parquet"
            targets = root / "targets.parquet"
            rows = [
                {"id": "a", "col_1": "2022-12-30 09:00:00", "col_11": 1.0},
                {"id": "a", "col_1": "2022-12-30 16:00:00", "col_11": 2.0},
                {"id": "a", "col_1": "2023-01-01 10:00:00", "col_11": 100.0},
                {"id": "b", "col_1": "2022-12-31 12:00:00", "col_11": 4.0},
                {"id": "unlabeled", "col_1": "2022-12-31 12:00:00", "col_11": 99.0},
            ]
            pd.DataFrame(rows).assign(col_1=lambda df: pd.to_datetime(df.col_1)).to_parquet(transactions)
            pd.DataFrame({"id": ["a", "b"], "col_1": ["2023-01-01"] * 2}).to_parquet(targets)
            manifest = _manifest(transactions, targets, ["2023-01-01"], months=12, shards=2)
            cache = root / "cache"
            output = root / "output"
            with patch.dict(os.environ, {"XBANK_DATA_ROOT": str(root)}):
                prepare(cache, output, manifest)
                prepare(cache, output, manifest)  # resumable

            self.assertEqual(json.loads((cache / "manifest.json").read_text()), manifest)
            daily = pd.read_parquet(cache / "shards")
            self.assertEqual(set(daily.id), {"a", "b"})
            amounts = dict(zip(daily.id, daily.value))
            self.assertEqual(amounts, {"a": 3.0, "b": 4.0})

            series = _series_for_date(
                np.array(["2022-12-30", "2022-12-31"], dtype="datetime64[D]"),
                np.array([3.0, 4.0], dtype=np.float32),
                "2023-01-01", 12,
            )
            self.assertEqual(series[-2:].tolist(), [3.0, 4.0])
            self.assertEqual(float(series.sum()), 7.0)

    def test_finalize_requires_every_shard_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            date = "2023-01-01"
            part_dir = output / "_shards" / date
            part_dir.mkdir(parents=True)
            manifest = {"target_dates": [date], "n_shards": 2}
            pd.DataFrame({"inn": ["a"], "date": [date], "emb_0": [1.0]}).to_parquet(
                part_dir / "part_000.parquet", index=False
            )
            with self.assertRaisesRegex(RuntimeError, "missing shard 1"):
                finalize(output, manifest)
            (part_dir / "part_001.empty").touch()
            finalize(output, manifest)
            self.assertEqual(pd.read_parquet(output / f"{date}.parquet").inn.tolist(), ["a"])
            self.assertFalse(part_dir.exists())
            finalize(output, manifest)


if __name__ == "__main__":
    unittest.main()
