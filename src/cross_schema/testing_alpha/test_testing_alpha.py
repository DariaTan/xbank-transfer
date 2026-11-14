"""Small synthetic checks; no bank data or checkpoints required."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

from cross_schema import column_profiles_vFGW as fgw
from cross_schema.testing_alpha.noise import corrupt_frame
from cross_schema.testing_alpha.run import (
    DEFAULT_ALPHAS, anonymous_schema, evaluate, experiment_cases, parse_args, run,
)


def synthetic_mbd(rows=240):
    rng = np.random.default_rng(23)
    frame = pd.DataFrame({"amount": rng.lognormal(1, 0.8, rows)})
    frame.loc[::11, "amount"] = 0
    for i, name in enumerate(fgw.MBD_CATEGORICAL_FIELDS):
        weights = np.arange(1, i + 3, dtype=float)
        weights /= weights.sum()
        frame[name] = rng.choice(len(weights), rows, p=weights)
    return frame


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "amount": [0.0, np.nan, np.inf, -np.inf, *np.linspace(-20, 30, 496)],
            "category": pd.Series([None, *(["a", "b", "c"] * 166), "a"], dtype="string"),
            "constant": [7] * 500,
        })
        self.specs = [("amount", "amount", "numeric"), ("category", "category", "categorical"),
                      ("constant", "constant", "categorical")]
        with duckdb.connect() as con:
            con.register("fixture", self.frame)
            self.passports = fgw.profile_columns(con, "SELECT * FROM fixture", self.specs)

    def corrupt(self, numeric, category, seed=5):
        return corrupt_frame(self.frame, self.specs, self.passports, numeric, category, seed)

    def test_zero_noise_is_exact_copy_and_source_is_never_mutated(self):
        original = self.frame.copy(deep=True)
        clean, audit = self.corrupt(0, 0)
        pd.testing.assert_frame_equal(clean, original)
        self.assertEqual(audit.n_changed.sum(), 0)
        noisy, _ = self.corrupt(0.3, 1)
        pd.testing.assert_frame_equal(self.frame, original)
        self.assertEqual(noisy.amount.iloc[0], 0)
        np.testing.assert_array_equal(noisy.amount.iloc[:4], original.amount.iloc[:4])
        self.assertTrue(noisy.category.isna().equals(original.category.isna()))
        pd.testing.assert_series_equal(noisy.constant, original.constant)

    def test_replacement_is_different_and_stays_in_original_vocabulary(self):
        noisy, audit = self.corrupt(0, 1)
        valid = self.frame.category.notna()
        self.assertTrue((noisy.loc[valid, "category"] != self.frame.loc[valid, "category"]).all())
        self.assertEqual(set(noisy.category.dropna()), set(self.frame.category.dropna()))
        row = audit.set_index("field").loc["category"]
        self.assertEqual(row.n_changed, valid.sum())

    def test_paired_noise_is_reproducible_and_streams_are_independent(self):
        low, _ = self.corrupt(0.15, 0.15)
        again, _ = self.corrupt(0.15, 0.15)
        high, _ = self.corrupt(0.30, 0.30)
        num, _ = self.corrupt(0.15, 0)
        cat, _ = self.corrupt(0, 0.15)
        other, _ = self.corrupt(0.15, 0.15, seed=6)
        pd.testing.assert_frame_equal(low, again)
        pd.testing.assert_series_equal(low.amount, num.amount)
        pd.testing.assert_series_equal(low.category, cat.category)
        np.testing.assert_allclose(high.amount.iloc[4:] - self.frame.amount.iloc[4:],
                                   2 * (low.amount.iloc[4:] - self.frame.amount.iloc[4:]))
        changed = (low.category != self.frame.category).fillna(False)
        pd.testing.assert_series_equal(high.loc[changed, "category"], low.loc[changed, "category"])
        self.assertFalse(low.equals(other))


class SweepTests(unittest.TestCase):
    def test_default_budget(self):
        args = parse_args([])
        cases = experiment_cases(args.conditions, args.noise_seeds)
        self.assertEqual(len(cases), 13)
        self.assertEqual(len(cases) * len(DEFAULT_ALPHAS), 143)
        self.assertEqual(sum(c["condition"] == "clean" for c in cases), 1)
        self.assertEqual(args.n_init, 5)

    def test_accuracy_counts_unfilled_fields_and_exposes_filter_failures(self):
        specs = [("amount", "amount", "numeric"), ("cat", "cat", "categorical")]
        truth, source_specs = anonymous_schema(specs, 42)
        names = [s[0] for s in source_specs]
        result = {
            "source_names": names, "target_names": ["amount", "cat"],
            "mbd_to_xbank": {"amount": truth["amount"], "cat": None},
            "distance": np.zeros((2, 2)), "allowed": np.array([[True, False], [True, False]]),
            "diagnostics": pd.DataFrame({"mbd_field": ["amount", "cat"], "status": ["mapped", "unfilled"]}),
        }
        _, metrics = evaluate(result, truth, specs)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["categorical_accuracy"], 0)
        self.assertEqual(metrics["truth_filtered"], 1)

    def test_parquet_sweep_resume_and_configuration_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root / "input.parquet", root / "results"
            data = synthetic_mbd()
            with duckdb.connect() as con:
                con.register("fixture", data)
                con.execute(f"COPY fixture TO '{source}' (FORMAT PARQUET)")
            argv = ["--input-parquet", str(source), "--out-dir", str(out),
                    "--sample-rows", "180", "--structure-rows", "120", "--min-pairs", "10",
                    "--conditions", "clean", "mixed_015", "categorical_015", "numeric_015",
                    "--noise-seeds", "7", "8", "--alphas", "0", "0.5", "1", "--max-iter", "50"]
            with contextlib.redirect_stdout(io.StringIO()):
                run(parse_args(argv))
            results = pd.read_csv(out / "results.csv")
            self.assertEqual(len(results), 21)
            self.assertEqual(len(results[results.condition == "clean"]), 3)
            self.assertTrue(results.accuracy.between(0, 1).all())
            self.assertTrue((results.n_fields == 12).all())
            self.assertTrue((results.numeric_fields == 1).all())
            sample = json.loads((out / "sample.json").read_text())
            self.assertEqual(sample["rows"], 180)
            self.assertEqual(sample["structure_rows"], 120)
            for relative in results.run_dir:
                recovered = pd.read_csv(out / relative / "recovery.csv")
                self.assertEqual(len(recovered), 12)
                self.assertEqual(recovered.truth.nunique(), 12)
            module = "cross_schema.testing_alpha.run"
            with patch(module + ".profile_frame", side_effect=AssertionError("reprofiled")), \
                    patch(module + ".corrupt_frame", side_effect=AssertionError("re-noised")), \
                    patch.object(fgw, "match_schemas", side_effect=AssertionError("re-solved")), \
                    contextlib.redirect_stdout(io.StringIO()):
                run(parse_args(argv + ["--resume"]))
            missing = out / results.iloc[-1].run_dir / "metrics.json"
            missing.unlink()
            with patch(module + ".profile_frame", side_effect=AssertionError("reprofiled")), \
                    patch(module + ".corrupt_frame", side_effect=AssertionError("re-noised")), \
                    patch.object(fgw, "match_schemas", wraps=fgw.match_schemas) as matcher, \
                    contextlib.redirect_stdout(io.StringIO()):
                run(parse_args(argv + ["--resume"]))
            self.assertEqual(matcher.call_count, 1)
            self.assertEqual(len(pd.read_csv(out / "results.csv")), 21)
            with self.assertRaisesRegex(ValueError, "another configuration"):
                with contextlib.redirect_stdout(io.StringIO()):
                    run(parse_args(argv + ["--resume", "--solver-seed", "99"]))


if __name__ == "__main__":
    unittest.main()
