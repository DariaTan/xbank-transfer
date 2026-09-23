import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cross_schema.apply_mapping import FrozenSchemaMapping
from data.splits import load_windowed_transactions_for_dates


class CrossSchemaInferenceTests(unittest.TestCase):
    def _mapping(self, root: Path, mapping: dict[str, str]) -> FrozenSchemaMapping:
        path = root / "frozen_mapping.json"
        path.write_text(json.dumps({"mapping": mapping}))
        return FrozenSchemaMapping(path)

    def test_matcher_fields_are_composed_with_checkpoint_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            mapping = self._mapping(Path(temp), {
                "col_1": "event_time",
                "col_2": "dst_type11",
                "col_10": "event_type",
                "col_12": "amount",
            })
            window = pd.DataFrame({
                "id": ["client::2023-01-01"],
                "col_1": pd.to_datetime(["2022-12-31"]),
                "col_2": [81],
                "col_10": [17],
                "col_12": [0.75],
            })
            self.assertEqual(mapping.source_columns("coles"),
                             ["id", "col_1", "col_2", "col_10", "col_12"])
            aligned = mapping.transform(window, "coles")
            self.assertEqual(aligned.at[0, "col_2"], 17)  # event_type
            self.assertEqual(aligned.at[0, "col_7"], 81)  # dst_type11
            self.assertEqual(aligned.at[0, "col_11"], 0.75)  # amount
            self.assertEqual(aligned.at[0, "col_3"], 0)  # unfilled MBD field
            self.assertEqual(aligned.at[0, "id"], "client::2023-01-01")
            self.assertEqual(mapping.source_columns("cotic"), ["id", "col_1", "col_10"])
            self.assertEqual(mapping.transform(window, "cotic").at[0, "col_2"], 17)
            self.assertEqual(mapping.source_columns("chronos2"), ["id", "col_1", "col_12"])
            self.assertEqual(mapping.transform(window, "chronos2").at[0, "col_11"], 0.75)

    def test_invalid_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "one-to-one"):
                self._mapping(root, {"col_1": "event_time", "col_2": "event_type",
                                     "col_3": "event_type"})
            with self.assertRaisesRegex(ValueError, "incompatible types"):
                self._mapping(root, {"col_1": "event_time", "col_2": "amount"})

    def test_xbank_window_excludes_target_day(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transactions.parquet"
            pd.DataFrame({
                "id": ["client", "client"],
                "col_1": pd.to_datetime(["2022-12-31", "2023-01-01"]),
                "col_10": [7, 99],
            }).to_parquet(path)
            selected = load_windowed_transactions_for_dates(
                str(path), ["2023-01-01"], 12, 500,
                client_ids=["client"], columns=["id", "col_1", "col_10"],
                include_cutoff=False,
            )
            self.assertEqual(selected.col_10.tolist(), [7])


if __name__ == "__main__":
    unittest.main()
