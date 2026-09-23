import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from training.embedding_io import atomic_parquet, validate_embedding_file, validate_embedding_frame


class EmbeddingIoTests(unittest.TestCase):
    def test_valid_frame_and_atomic_round_trip(self):
        frame = pd.DataFrame(
            {"inn": ["a", "b"], "date": ["2023-01-01"] * 2, "emb_0": [0.1, 0.2]}
        )
        validate_embedding_frame(frame, "2023-01-01")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.parquet"
            atomic_parquet(frame, path)
            loaded = validate_embedding_file(path, "2023-01-01")
            self.assertEqual(len(loaded), 2)
            self.assertFalse(path.with_suffix(".parquet.tmp").exists())

    def test_duplicate_clients_fail(self):
        frame = pd.DataFrame(
            {"inn": ["a", "a"], "date": ["2023-01-01"] * 2, "emb_0": [0.1, 0.2]}
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_embedding_frame(frame, "2023-01-01")

    def test_non_finite_values_fail(self):
        frame = pd.DataFrame(
            {"inn": ["a"], "date": ["2023-01-01"], "emb_0": [np.nan]}
        )
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_embedding_frame(frame, "2023-01-01")


if __name__ == "__main__":
    unittest.main()
