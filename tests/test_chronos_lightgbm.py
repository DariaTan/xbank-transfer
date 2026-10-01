import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from training.tune_lightgbm_chronos import (
    IndexedSequence, load_targets, metrics, predict_batches, prepare_cache,
    run, split_rows, summarize,
)


class ChronosLightgbmTests(unittest.TestCase):
    def fixture(self, root):
        targets, embeddings = [], []
        for date in ("2022-02-01", "2022-03-01"):
            for fold in range(5):
                for client in range(10):
                    identifier = f"f{fold}-c{client}"
                    targets.append({"id": identifier, "col_1": date, "fold": fold,
                                    **{f"col_{i}": client % 2 for i in range(2, 6)}})
                    embeddings.append({"inn": identifier, "date": date,
                                       "emb_0": float(client % 2), "emb_1": float(fold)})
        target_path = root / "targets.parquet"
        pd.DataFrame(targets).to_parquet(target_path, index=False)
        directory = root / "embeds/mbd_raw/zero_shot/chronos2"
        directory.mkdir(parents=True)
        for date, frame in pd.DataFrame(embeddings).groupby("date"):
            frame.to_parquet(directory / f"{date}.parquet", index=False)
        config = root / "mbd.yaml"
        config.write_text(f"evaluation_name: mbd_raw\npaths:\n  targets: {target_path}\n")
        downstream = root / "downstream.yaml"
        downstream.write_text("probe: {}\n")
        return target_path, directory, config, downstream

    def test_cache_preserves_feature_target_alignment_and_source_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            targets, embeds, _, _ = self.fixture(root)
            cache = root / "cache"
            files = sorted(embeds.glob("*.parquet"))
            saved = prepare_cache(cache, targets, files, [f"col_{i}" for i in range(2, 6)], True, 7)
            values = np.memmap(cache / "features.f32", dtype=np.float32, mode="r", shape=(100, 2))
            rows = pd.read_parquet(cache / "rows.parquet")
            self.assertTrue(np.array_equal(values[:, 0], rows.col_2))
            self.assertEqual(saved["n_rows"], 100)
            self.assertEqual(saved, prepare_cache(cache, targets, files, saved["source"]["targets"], True))
            changed = pd.read_parquet(targets)
            changed.to_parquet(targets, index=False)
            with self.assertRaisesRegex(ValueError, "different inputs"):
                prepare_cache(cache, targets, files, saved["source"]["targets"], True)

    def test_all_folds_keep_clients_and_test_rows_out_of_hpo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target_path, _, _, _ = self.fixture(root)
            rows = pd.read_parquet(target_path)
            for fold in range(5):
                train, val, test, tune, dev = split_rows(rows, True, fold, {}, 42, 8)
                self.assertFalse(set(rows.iloc[tune].id) & set(rows.iloc[test].id))
                self.assertFalse(set(rows.iloc[train].id) & set(rows.iloc[val].id))
                self.assertEqual(set(test) | set(dev), set(range(len(rows))))

    def test_xbank_ignores_conflicting_excluded_target_and_splits_by_client(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "targets.parquet"
            data = [{"id": str(i), "col_1": date, "col_2": excluded,
                     "col_3": i % 2, "col_4": i % 2, "col_5": i % 2}
                    for i in range(20) for date in ("2023-01-01", "2024-01-01") for excluded in (0, 1)]
            pd.DataFrame(data).to_parquet(path, index=False)
            rows = load_targets(path, ["col_3", "col_4", "col_5"], False)
            self.assertEqual(len(rows), 40)
            probe = {"train": {"start": "2023-01", "end": "2023-12"},
                     "test": {"start": "2024-01", "end": "2024-02"}, "val_frac": .2, "seed": 0}
            train, val, test, _, _ = split_rows(rows, False, 0, probe, 42, 10)
            self.assertFalse(set(rows.iloc[train].id) & set(rows.iloc[val].id))
            self.assertEqual(set(rows.iloc[test].col_1), {"2024-01-01"})

    def test_sequence_training_and_prediction_match_dense_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            values = np.memmap(Path(temp) / "features", dtype=np.float32, mode="w+", shape=(200, 2))
            values[:] = np.random.default_rng(0).normal(size=(200, 2))
            labels = (values[:, 0] > 0).astype(np.int8)
            seq = IndexedSequence(values, np.arange(200), 17)
            params = {"objective": "binary", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 2}
            streamed = lgb.train(params, lgb.Dataset(seq, label=labels), num_boost_round=10)
            dense = lgb.train(params, lgb.Dataset(np.array(values), label=labels), num_boost_round=10)
            self.assertTrue(np.allclose(predict_batches(streamed, seq), dense.predict(np.array(values))))

    def test_five_fold_run_resume_summary_and_cache_cleanup(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"XBANK_DATA_ROOT": temp}):
            root = Path(temp)
            _, _, config, downstream = self.fixture(root)
            for fold in range(5):
                run(str(config), str(downstream), fold, "cpu", 42, 1, 1, 10, 3, 63)
            output = root / "downstream/mbd_raw/zero_shot/chronos2"
            result = output / "lightgbm_hpo_cv/fold0/col_2_metrics.json"
            before = result.stat().st_mtime_ns
            run(str(config), str(downstream), 0, "cpu", 42, 1, 1, 10, 3, 63)
            self.assertEqual(before, result.stat().st_mtime_ns)
            self.assertEqual(set(json.loads(result.read_text())["test_metrics"]),
                             {"n_rows", "n_positive", "prevalence", "pr_auc", "roc_auc"})
            summarize("mbd_raw", cleanup_cache=True)
            self.assertEqual(len(pd.read_csv(output / "results_all_folds.csv")), 20)
            self.assertTrue((pd.read_csv(output / "results_aggregated.csv").n_evaluations == 5).all())
            self.assertFalse((output / "_feature_cache").exists())
            with self.assertRaisesRegex(ValueError, "settings differ"):
                run(str(config), str(downstream), 0, "cpu", 42, 2, 1, 10, 3, 63)

    def test_auc_metrics_do_not_calculate_precision_or_recall(self):
        result = metrics(np.array([0, 1, 0, 1]), np.array([.1, .9, .2, .8]))
        self.assertEqual(result["pr_auc"], 1.)
        self.assertEqual(result["roc_auc"], 1.)
        self.assertFalse(any("precision" in key or "recall" in key for key in result))

    def test_calendar_run_refits_without_mbd_folds(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"XBANK_DATA_ROOT": temp}):
            root = Path(temp)
            target_path, old_embeds, _, downstream = self.fixture(root)
            dates = {"2022-02-01": "2023-01-01", "2022-03-01": "2024-01-01"}
            targets = pd.read_parquet(target_path).drop(columns="fold")
            targets.col_1 = targets.col_1.map(dates)
            targets.to_parquet(target_path, index=False)
            embeds = root / "embeds/xbank/zero_shot/chronos2"
            embeds.mkdir(parents=True)
            for path in old_embeds.glob("*.parquet"):
                frame = pd.read_parquet(path)
                frame.date = frame.date.map(dates)
                frame.to_parquet(embeds / f"{dates[path.stem]}.parquet", index=False)
            cfg = root / "xbank.yaml"
            cfg.write_text(f"evaluation_name: xbank\npaths:\n  targets: {target_path}\n")
            downstream.write_text("probe:\n  target_cols: [col_3, col_4, col_5]\n"
                                  "  train: {start: 2023-01, end: 2023-12}\n"
                                  "  test: {start: 2024-01, end: 2024-02}\n"
                                  "  val_frac: 0.4\n  seed: 0\n")
            run(str(cfg), str(downstream), 0, "cpu", 42, 1, 1, 10, 3, 63)
            summarize("xbank", cleanup_cache=True)
            output = root / "downstream/xbank/zero_shot/chronos2"
            aggregate = pd.read_csv(output / "results_aggregated.csv")
            self.assertEqual(set(aggregate.target), {"col_3", "col_4", "col_5"})
            self.assertTrue((aggregate.n_test_rows_total == 50).all())
            self.assertFalse((output / "_feature_cache").exists())


if __name__ == "__main__":
    unittest.main()
