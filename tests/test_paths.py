import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.paths import embedding_dir, load_data_config, resolve_data_path


class PathTests(unittest.TestCase):
    def test_container_data_path_can_be_remapped_locally(self):
        with patch.dict(os.environ, {"XBANK_DATA_ROOT": "/tmp/xbank-fixture"}):
            self.assertEqual(
                resolve_data_path("/app/data/mbd_data/raw/transactions.parquet"),
                Path("/tmp/xbank-fixture/mbd_data/raw/transactions.parquet"),
            )

    def test_embedding_namespace_separates_eval_and_checkpoint_sources(self):
        with patch.dict(os.environ, {"XBANK_DATA_ROOT": "/tmp/xbank-fixture"}):
            self.assertEqual(
                embedding_dir("/app/data/embeds", "mbd_raw", "mbd", "coles"),
                Path("/tmp/xbank-fixture/embeds/mbd_raw/mbd_source/coles"),
            )
            self.assertNotEqual(
                embedding_dir("/app/data/embeds", "mbd_raw", "mbd", "coles"),
                embedding_dir("/app/data/embeds", "mbd_daily", "mbd", "coles"),
            )

    def test_data_config_paths_are_remapped(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"XBANK_DATA_ROOT": tmp}):
            cfg = load_data_config("configs/data/mbd.yaml")
            self.assertTrue(cfg["paths"]["transactions"].startswith(tmp))
            self.assertEqual(cfg["evaluation_name"], "mbd_raw")


if __name__ == "__main__":
    unittest.main()
