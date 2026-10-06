import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from training.tune_lightgbm_chronos import predict_batches, run, summarize
from training.tune_lightgbm_daily_source import EmbeddingsPending, inputs, paired_summary, published_files
from training.tune_lightgbm_xbank_paired import paired_cohort
from training.train_downstream import load_config


class DailySourceLightgbmTests(unittest.TestCase):
    def fixture(self, root, evaluation, dates, n=10, exclude=None, reverse=False):
        targets, embeddings = [], []
        rng = np.random.default_rng(42)
        for date in dates:
            for fold in range(5):
                for client in range(n):
                    identifier = f"f{fold}-c{client:04}"
                    labels = {f"col_{i}": (client + i) % 2 for i in range(2, 6)}
                    targets.append({"id": identifier, "col_1": date, "fold": fold, **labels})
                    if identifier == exclude:
                        continue
                    embeddings.append({"inn": identifier, "date": date,
                                       "emb_0": float(client % 2),
                                       **{f"emb_{i}": float(rng.normal()) for i in range(1, 32)}})
        target_path = root / f"{evaluation}_targets.parquet"
        pd.DataFrame(targets).to_parquet(target_path, index=False)
        directory = root / "embeds" / evaluation / "mbd_daily_source/cotic"
        directory.mkdir(parents=True)
        for date, frame in pd.DataFrame(embeddings).groupby("date"):
            if reverse:
                frame = frame.iloc[::-1]
            frame.to_parquet(directory / f"{date}.parquet", index=False)
        manifest = {"model": "cotic", "evaluation_name": evaluation, "checkpoint_source": "mbd_daily",
                    "target_dates": dates, "checkpoint_files_sha256": {"best.ckpt": "verified-fixture"}}
        (directory / "run_manifest.json").write_text(json.dumps(manifest))
        config = root / f"{evaluation}.yaml"
        config.write_text(f"evaluation_name: {evaluation}\npaths:\n  targets: {target_path}\n")
        downstream = root / "downstream.yaml"
        downstream.write_text("probe:\n  target_cols: [col_3, col_4, col_5]\n"
                              "  train: {start: 2023-01, end: 2023-12}\n"
                              "  test: {start: 2024-01, end: 2024-02}\n"
                              "  val_frac: 0.15\n  seed: 0\n")
        return target_path, directory, config, downstream

    def test_readiness_rejects_partial_and_wrong_source(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"XBANK_DATA_ROOT": temp}):
            root = Path(temp)
            dates = pd.date_range("2022-02-01", periods=12, freq="MS").strftime("%Y-%m-%d").tolist()
            _, directory, _, _ = self.fixture(root, "mbd_daily", dates)
            files, _ = published_files("cotic", "mbd_daily")
            self.assertEqual(len(files), 12)
            files[-1].unlink()
            with self.assertRaises(EmbeddingsPending):
                inputs("cotic", "mbd_daily")
            path = directory / "run_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["checkpoint_source"] = "mbd"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "wrong checkpoint_source"):
                inputs("cotic", "mbd_daily")

    def test_complete_daily_and_paired_runs_preserve_old_namespace(self):
        # The same end-to-end smoke runs on the actual server GPU when requested.
        device = os.environ.get("DAILY_SOURCE_SMOKE_DEVICE", "cpu")
        n = 500 if device == "gpu" else 10
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"XBANK_DATA_ROOT": temp}):
            root = Path(temp)
            old = root / "downstream/mbd_daily/mbd_source/cotic/sentinel.txt"
            old.parent.mkdir(parents=True)
            old.write_text("old results must survive")
            _, _, config, downstream = self.fixture(root, "mbd_daily", ["2022-02-01", "2022-03-01"], n)
            for fold in range(5):
                run(str(config), str(downstream), fold, device, 42, 1, 1, 500, 3, 255,
                    model="cotic", checkpoint_source="mbd_daily")
            result = root / "downstream/mbd_daily/mbd_daily_source/cotic"
            stamp = (result / "lightgbm_hpo_cv/fold0/col_2_metrics.json").stat().st_mtime_ns
            run(str(config), str(downstream), 0, device, 42, 1, 1, 500, 3, 255,
                model="cotic", checkpoint_source="mbd_daily")
            self.assertEqual(stamp, (result / "lightgbm_hpo_cv/fold0/col_2_metrics.json").stat().st_mtime_ns)
            summarize("mbd_daily", True, model="cotic", checkpoint_source="mbd_daily")
            self.assertEqual(len(pd.read_csv(result / "results_all_folds.csv")), 20)
            self.assertEqual(old.read_text(), "old results must survive")
            self.assertFalse((result / "_feature_cache").exists())
            dates = ["2023-01-01", "2024-01-01"]
            target_path, original, xcfg, downstream = self.fixture(root, "xbank", dates, n)
            _, mapped, fcfg, _ = self.fixture(root, "xbank_fgw_v2", dates, n, exclude="f0-c0000", reverse=True)
            fcfg.write_text(f"evaluation_name: xbank_fgw_v2\npaths:\n  targets: {target_path}\n")
            files = {"xbank": sorted(original.glob("*.parquet")), "xbank_fgw_v2": sorted(mapped.glob("*.parquet"))}
            cohort, train, val, test = paired_cohort("cotic", target_path, load_config(str(downstream)), files)
            self.assertFalse(set(cohort.iloc[train].id) & set(cohort.iloc[val].id))
            self.assertEqual(len(test), n * 5 - 1)
            def aligned_prediction(booster, sequence):
                # The mapped parquet has reversed row order and a missing client.
                # Inspect actual disk feature offsets, not just metadata counts.
                features = sequence[:len(sequence)]
                np.testing.assert_array_equal(features[:, 0], cohort.iloc[test].col_4.to_numpy())
                return predict_batches(booster, sequence)

            with patch("training.tune_lightgbm_chronos.predict_batches", side_effect=aligned_prediction):
                for cfg in (xcfg, fcfg):
                    run(str(cfg), str(downstream), 0, device, 42, 1, 1, 500, 3, 63,
                        model="cotic", checkpoint_source="mbd_daily", paired_rows=cohort,
                        paired_sources={"fixture": "same source guard for both variants"})
            paired_summary("cotic", True)
            comparison = pd.read_csv(root / "downstream/xbank_mapping_comparison/mbd_daily_source/cotic_paired_results.csv")
            self.assertEqual(set(comparison.target), {"col_3", "col_4", "col_5"})
            self.assertTrue((comparison.n_rows_original == n * 5 - 1).all())
            self.assertTrue((comparison.n_rows_original == comparison.n_rows_fgw_v2).all())
            self.assertFalse(any("precision" in c or "recall" in c for c in comparison))
            if device == "gpu":
                self.assertTrue((comparison.roc_auc_original > .95).all())
                self.assertTrue((comparison.roc_auc_fgw_v2 > .95).all())
            for variant in files:
                self.assertFalse((root / "downstream" / variant / "mbd_daily_source/cotic/_feature_cache").exists())
                manifest = root / "downstream" / variant / "mbd_daily_source/cotic/lightgbm_hpo_calendar/run_manifest.json"
                self.assertEqual(json.loads(manifest.read_text())["checkpoint_source"], "mbd_daily")
            with self.assertRaisesRegex(ValueError, "settings differ"):
                run(str(config), str(downstream), 0, device, 42, 2, 1, 500, 3, 255,
                    model="cotic", checkpoint_source="mbd_daily")


if __name__ == "__main__":
    unittest.main()
