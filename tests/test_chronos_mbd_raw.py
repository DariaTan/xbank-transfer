import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from training.infer_chronos_mbd_raw import _manifest, _series_for_date, finalize, prepare


class ChronosRawTests(unittest.TestCase):
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
