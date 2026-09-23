import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training.target_io import load_targets


class XbankTargetsTests(unittest.TestCase):
    def test_conflicting_excluded_target_deduplicates_selected_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "targets.parquet"
            pd.DataFrame({
                "id": ["a", "a", "b"],
                "col_1": pd.to_datetime(["2023-01-01"] * 3),
                "col_2": [0, 1, 0],
                "col_3": [1, 1, 0],
                "col_4": [0, 0, 0],
                "col_5": [0, 0, 1],
            }).to_parquet(path)
            targets = load_targets(str(path), ["col_3", "col_4", "col_5"])
            self.assertEqual(len(targets), 2)
            self.assertNotIn("col_2", targets.columns)

    def test_conflict_in_selected_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "targets.parquet"
            pd.DataFrame({
                "id": ["a", "a"],
                "col_1": pd.to_datetime(["2023-01-01"] * 2),
                "col_3": [0, 1],
                "col_4": [0, 0],
                "col_5": [0, 0],
            }).to_parquet(path)
            with self.assertRaisesRegex(ValueError, "conflicting selected labels"):
                load_targets(str(path), ["col_3", "col_4", "col_5"])


if __name__ == "__main__":
    unittest.main()
