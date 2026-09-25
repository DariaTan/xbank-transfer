import json
import tempfile
import unittest
from pathlib import Path

from training.summarize_lightgbm_xbank_paired import collect


class XbankLightgbmSummaryTests(unittest.TestCase):
    def test_paired_comparison_and_cohort_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for variant, pr_auc in (("xbank", 0.03), ("xbank_fgw_v2", 0.04)):
                directory = root / variant / "mbd_source" / "coles" / "lightgbm_xbank_paired"
                directory.mkdir(parents=True)
                manifest = {"model": "coles", "variant": variant,
                            "cohort_sha256": "same", "target_file": {"path": "t"},
                            "split": {"test_rows": 10}, "seed": 42, "split_seed": 0,
                            "trials": 3, "threads": 6, "max_rounds": 300,
                            "device_type": "gpu", "max_bin": 63}
                (directory / "run_manifest.json").write_text(json.dumps(manifest))
                (directory / "col_3_model.txt").write_text("model")
                (directory / "col_3_metrics.json").write_text(json.dumps({
                    "model": "coles", "variant": variant, "target": "col_3",
                    "test_metrics": {"n_rows": 10, "prevalence": 0.1,
                                     "pr_auc": pr_auc, "roc_auc": 0.7,
                                     "precision@5%": 0.2},
                }))
            result = collect(root, models=("coles",), targets=("col_3",))
            self.assertEqual(len(result), 1)
            self.assertAlmostEqual(result.delta_pr_auc.iloc[0], 0.01)
            newer = root / "xbank_fgw_v2/mbd_source/coles/lightgbm_xbank_paired/run_manifest.json"
            manifest = json.loads(newer.read_text())
            manifest["cohort_sha256"] = "different"
            newer.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "cohort_sha256"):
                collect(root, models=("coles",), targets=("col_3",))


if __name__ == "__main__":
    unittest.main()
