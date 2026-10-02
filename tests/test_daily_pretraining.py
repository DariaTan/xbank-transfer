import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from training.common import (EarlyStopper, load_checkpoint, save_checkpoint,
                             seed_training, validation_rng)
from training.pretrain_cache import DiskRecords, fixed_split, prepare_cache, validate_cache
from training.smoke_daily_encoders import fixture
from training.train_daily_encoders import NonSingletonBatches, masked_events, train
from data.schema import ALL_FEATURE_COLS


class DailyPretrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_cache(self, root):
        source, data, configs = fixture(root)
        cache = root / "cache"
        prepare_cache(source, cache, valid_frac=.2, max_seq_len=24, shards=4, threads=2, memory_gb=1)
        return source, data, configs, cache

    def test_fixed_split_does_not_depend_on_source_order(self):
        frame = pd.DataFrame({"id": [f"c{i}" for i in range(100)], "bucket": [i % 4 for i in range(100)]})
        first = fixed_split(frame, 0, .05)
        second = fixed_split(frame.sample(frac=1, random_state=12), 0, .05)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(int(first.valid.sum()), 5)

    def test_non_singleton_batches_keep_all_clients(self):
        for n in range(2, 40):
            batches = NonSingletonBatches(list(range(n)), 8)
            values = list(batches)
            self.assertTrue(all(len(batch) >= 2 for batch in values))
            self.assertEqual(sum(values, []), list(range(n)))
            self.assertEqual(len(values), len(batches))

    def test_nan_is_not_a_best_score(self):
        with self.assertRaises(FloatingPointError):
            EarlyStopper().step(float("nan"), 0)

    def test_validation_rng_does_not_consume_training_randomness(self):
        seed_training(4)
        expected = torch.rand(4)
        seed_training(4)
        with validation_rng(42):
            first = torch.rand(4)
        actual = torch.rand(4)
        with validation_rng(42):
            second = torch.rand(4)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_restores_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.Adam(model.parameters())
            stopper = EarlyStopper()
            stopper.step(1., 0)
            seed_training(3)
            path = str(Path(temporary) / "last.pt")
            save_checkpoint(path, model, optimizer, 0, stopper)
            expected = (torch.rand(4), np.random.random(4))
            seed_training(100)
            self.assertEqual(load_checkpoint(path, model, optimizer, stopper, torch.device("cpu")), 1)
            self.assertTrue(torch.equal(expected[0], torch.rand(4)))
            np.testing.assert_array_equal(expected[1], np.random.random(4))
            self.assertFalse(Path(path + ".tmp").exists())

    def test_masks_supervise_each_short_sequence_without_padding(self):
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        selected = masked_events(mask, 0.)
        self.assertTrue(selected.any(dim=1).all())
        self.assertFalse((selected & ~mask).any())

    def test_cache_train_only_vocab_cap_order_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data, configs, cache = self.make_cache(root)
            ready = validate_cache(cache)
            self.assertEqual(ready["audit"]["short_clients_excluded"], 1)
            self.assertFalse(any(ready["audit"]["unknown_train"].values()))
            clients = pd.read_parquet(cache / "clients.parquet")
            raw = pd.read_parquet(source)
            train_ids = set(clients.loc[~clients.valid & (clients.n >= 2), "id"])
            capped = raw.sort_values(["id", "col_1", *ALL_FEATURE_COLS])
            capped = capped.groupby("id", sort=False).tail(24)
            counts = capped[capped.id.isin(train_ids)].col_2.astype(str).value_counts().to_dict()
            expected = {value: i + 1 for i, value in enumerate(sorted(counts, key=lambda v: (-counts[v], v)))}
            self.assertEqual(json.loads((cache / "vocabulary.json").read_text())["columns"]["col_2"], expected)
            records = DiskRecords(cache, "train", "coles")
            for i in range(len(records)):
                record = records[i]
                self.assertLessEqual(len(record["event_time"]), 24)
                self.assertGreaterEqual(len(record["event_time"]), 2)
                self.assertTrue((record["event_time"][1:] >= record["event_time"][:-1]).all())
            repeat = prepare_cache(source, cache, valid_frac=.2, max_seq_len=24, shards=4, threads=2, memory_gb=1)
            self.assertEqual(ready, repeat)
            self.assertFalse((cache / "raw").exists())
            self.assertFalse((cache / "capped").exists())
            with self.assertRaises(ValueError):
                prepare_cache(source, cache, valid_frac=.2, max_seq_len=25, shards=4, threads=2, memory_gb=1)

    def test_mlm_resume_matches_uninterrupted_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data, configs, cache = self.make_cache(root)
            full, resumed = root / "full", root / "resumed"
            train("mlm", data, configs["mlm"], cache, full, root / "logs-full", device="cpu")
            from training.train_daily_encoders import plain_epoch
            calls = 0

            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("simulated interruption after epoch 0")
                return plain_epoch(*args, **kwargs)

            with patch("training.train_daily_encoders.plain_epoch", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train("mlm", data, configs["mlm"], cache, resumed, root / "logs-resume", device="cpu")
            train("mlm", data, configs["mlm"], cache, resumed, root / "logs-resume", device="cpu")
            first = torch.load(full / "last.pt", weights_only=False)
            second = torch.load(resumed / "last.pt", weights_only=False)
            for key, value in first["model"].items():
                self.assertTrue(torch.equal(value, second["model"][key]), key)
            self.assertEqual(first["best"], second["best"])

    def test_nep_and_legacy_runs_are_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data, configs, cache = self.make_cache(root)
            with self.assertRaisesRegex(ValueError, "NEP"):
                train("nep", data, configs["mlm"], cache, device="cpu")
            output = root / "legacy"
            output.mkdir()
            (output / "best.pt").write_bytes(b"legacy")
            with self.assertRaisesRegex(ValueError, "legacy"):
                train("mlm", data, configs["mlm"], cache, output, device="cpu")
            self.assertEqual((output / "best.pt").read_bytes(), b"legacy")

    def test_masked_only_prediction_matches_dense_loss_and_gradients(self):
        from models.mlm import MLM
        model = MLM({"col_2": 5}, ["col_11"], d_model=16, num_layers=1)
        model.eval()
        payload = {"col_2": torch.tensor([[1, 2, 3], [2, 1, 0]]),
                   "col_11": torch.rand(2, 3)}
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        selected = torch.tensor([[False, True, False], [True, False, False]])
        inputs = model.trx_embedding(payload, mask)
        hidden = model.backbone(inputs_embeds=torch.where(selected.unsqueeze(-1), model.mask_embedding, inputs),
                                attention_mask=mask).last_hidden_state
        logits, numeric = model.heads(hidden)
        dense = model.heads.loss(logits, numeric, payload, selected.float())
        dense.backward()
        gradients = {key: value.grad.clone() for key, value in model.named_parameters() if value.grad is not None}
        model.zero_grad(set_to_none=True)
        compact = model.loss(payload, mask, event_mask=selected)
        compact.backward()
        torch.testing.assert_close(compact, dense)
        for key, value in model.named_parameters():
            if key in gradients:
                torch.testing.assert_close(value.grad, gradients[key], rtol=1e-4, atol=1e-6)

    def test_validation_exclusive_category_is_not_fitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data, configs = fixture(root)
            clients = pd.DataFrame({"id": [f"client-{i:03d}" for i in range(64)], "bucket": 0})
            selected = fixed_split(clients, 0, .2)
            raw = pd.read_parquet(source)
            raw.loc[raw.id.isin(selected.loc[selected.valid, "id"]), "col_3"] = 999
            raw.to_parquet(source, index=False)
            ready = prepare_cache(source, root / "cache", valid_frac=.2, max_seq_len=24,
                                  n_clients=None, shards=4, threads=2, memory_gb=1)
            vocabulary = json.loads((root / "cache" / "vocabulary.json").read_text())
            self.assertNotIn("999", vocabulary["columns"]["col_3"])
            self.assertGreater(ready["audit"]["unknown_valid"]["col_3"], 0)
            self.assertEqual(ready["audit"]["unknown_train"]["col_3"], 0)

    def test_lightning_completed_checkpoint_recovers_missing_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data, configs, cache = self.make_cache(root)
            output = root / "coles"
            first = train("coles", data, configs["coles"], cache, output, root / "logs", device="cpu")
            (output / "complete.json").unlink()
            recovered = train("coles", data, configs["coles"], cache, output, root / "logs", device="cpu")
            self.assertEqual(first, recovered)


if __name__ == "__main__":
    unittest.main()
