import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from training.summarize_lightgbm_mbd_cv import collect, fold_output, summarize
from training.tune_lightgbm_mbd import candidate_params, folds_for_test, run


class MbdLightgbmHpoTests(unittest.TestCase):
    def test_search_is_deterministic_and_starts_with_baseline(self):
        first = candidate_params(42, 3, 2)
        self.assertEqual(first, candidate_params(42, 3, 2))
        self.assertEqual(first[0]["num_leaves"], 31)
        self.assertEqual(first[0]["num_threads"], 2)

    def test_every_rotation_keeps_test_out_of_training_and_validation(self):
        for test_fold in range(5):
            train_folds, val_fold = folds_for_test(test_fold)
            self.assertEqual(len(train_folds), 3)
            self.assertEqual(len({*train_folds, val_fold, test_fold}), 5)
        self.assertEqual(folds_for_test(4), ((0, 1, 2), 3))
        self.assertEqual(folds_for_test(0), ((1, 2, 3), 4))
        with self.assertRaises(ValueError):
            folds_for_test(5)

    def test_five_fold_summary_uses_existing_fold_four(self):
        with tempfile.TemporaryDirectory() as temp:
            for fold in range(5):
                for eval_name in ("mbd_raw", "mbd_daily"):
                    directory = fold_output(temp, "coles", fold, eval_name)
                    directory.mkdir(parents=True)
                    manifest = {"model": "coles", "target_file": {"path": "targets"},
                                "embedding_files": [{"path": "embeddings"}], "seed": 42,
                                "trials": 4, "threads": 6, "tune_client_cap": 50000,
                                "max_rounds": 400}
                    if fold != 4 or eval_name == "mbd_daily":
                        train_folds, val_fold = folds_for_test(fold)
                        manifest.update(test_fold=fold, val_fold=val_fold,
                                        train_folds=train_folds)
                    (directory / "run_manifest.json").write_text(json.dumps(manifest))
                    for target in ("col_2", "col_3", "col_4", "col_5"):
                        (directory / f"{target}_model.txt").write_text("model")
                        (directory / f"{target}_metrics.json").write_text(json.dumps({
                            "target": target, "final_rounds": 10,
                            "test_metrics": {"n_rows": 8, "prevalence": 0.125,
                                             "pr_auc": 0.2 + fold * 0.01,
                                             "roc_auc": 0.6 + fold * 0.01},
                        }))
            for eval_name in ("mbd_raw", "mbd_daily"):
                rows = collect(temp, models=("coles",), eval_name=eval_name)
                aggregate, macro = summarize(rows)
                self.assertEqual(len(rows), 20)
                self.assertTrue((aggregate.n_folds == 5).all())
                self.assertTrue(np.allclose(aggregate.roc_auc_mean, 0.62))
                self.assertAlmostEqual(macro.mean_roc_auc_across_targets.iloc[0], 0.62)

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

            run("coles", str(cfg), "/app/data/downstream", seed=42, trials=1,
                threads=1, tune_client_cap=10, max_rounds=10, test_fold=0)
            fold_zero = result.parent / "lightgbm_hpo_cv" / "fold0"
            manifest = json.loads((fold_zero / "run_manifest.json").read_text())
            self.assertEqual(manifest["train_folds"], [1, 2, 3])
            self.assertEqual(manifest["val_fold"], 4)
            self.assertEqual(manifest["test_fold"], 0)
            for target in ("col_2", "col_3", "col_4", "col_5"):
                score = json.loads((fold_zero / f"{target}_metrics.json").read_text())
                self.assertEqual(score["test_metrics"]["n_rows"], 8)
                self.assertTrue(np.isfinite(score["test_metrics"]["roc_auc"]))

            daily_embeds = root / "embeds" / "mbd_daily" / "mbd_source" / "coles"
            daily_embeds.mkdir(parents=True)
            pd.DataFrame(embeddings).to_parquet(daily_embeds / "all.parquet", index=False)
            daily_cfg = root / "mbd_daily.yaml"
            daily_cfg.write_text(
                "evaluation_name: mbd_daily\npaths:\n"
                "  targets: /app/data/mbd_data/raw_adapted/targets.parquet\n"
            )
            run("coles", str(daily_cfg), "/app/data/downstream", seed=42, trials=1,
                threads=1, tune_client_cap=10, max_rounds=10, test_fold=4)
            daily_fold_four = root / "downstream/mbd_daily/mbd_source/coles/lightgbm_hpo_cv/fold4"
            manifest = json.loads((daily_fold_four / "run_manifest.json").read_text())
            self.assertEqual(manifest["test_fold"], 4)
            self.assertTrue((daily_fold_four / "col_2_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
