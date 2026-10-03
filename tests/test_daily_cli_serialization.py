"""Cross-process tests of the actual Docker CLI and old on-disk pickles."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import torch

from training.artifact_compat import load_preprocessor, load_torch_checkpoint
from training.common import load_preprocessor as ordinary_load_preprocessor
from training.pretrain_cache import digest, validate_cache
from training.smoke_daily_encoders import fixture


class DailyCLISerializationTests(unittest.TestCase):
    def child(self, root, *args):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(root / "mpl-cache"),
                   XDG_CACHE_HOME=str(root / "xdg-cache"))
        result = subprocess.run([sys.executable, *map(str, args)], env=env,
                                text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout[-5000:] + result.stderr[-5000:])
        return result.stdout

    def test_actual_cli_prepare_then_train_in_separate_processes(self):
        with tempfile.TemporaryDirectory(prefix="daily-cli-test-") as temporary:
            root = Path(temporary)
            _, data, configs = fixture(root)
            cache = root / "cache"
            print("CLI test: preparation subprocess", flush=True)
            self.child(root, "-m", "training.daily_cli", "prepare", "--data-config", data,
                       "--cache-root", cache, "--valid-frac", ".2", "--max-seq-len", "24",
                       "--shards", "4", "--threads", "2", "--memory-gb", "1")
            validate_cache(cache)
            before = digest(cache / "preprocessor.pkl")
            preprocessor = ordinary_load_preprocessor(cache / "preprocessor.pkl")
            self.assertTrue(all(ct.__class__.__module__ == "training.pretrain_cache"
                                for ct in preprocessor.cts_category))
            for model in ("mlm", "coles", "cotic"):
                print(f"CLI test: {model} training subprocess", flush=True)
                output = root / "models" / model
                stdout = self.child(root, "-m", "training.daily_cli", "train", "--model", model,
                                    "--data-config", data, "--model-config", configs[model],
                                    "--cache-root", cache, "--output-root", output,
                                    "--log-root", root / "logs", "--device", "cpu")
                self.assertIn(f"COMPLETE {model}", stdout)
                complete = json.loads((output / "complete.json").read_text())
                checkpoint = load_torch_checkpoint(output / complete["best_checkpoint"])
                state = checkpoint.get("state_dict", checkpoint.get("model"))
                self.assertTrue(all(torch.isfinite(value).all() for value in state.values()
                                    if torch.is_tensor(value)))
                if model == "cotic":
                    loss = checkpoint["hyper_parameters"]["joined_head"].downstream_head.event_type_loss
                    self.assertEqual(type(loss).__module__, "training.train_daily_encoders")
                else:
                    recovered = load_preprocessor(output / "preprocessor.pkl")
                    self.assertEqual(recovered.get_category_dictionary_sizes(),
                                     preprocessor.get_category_dictionary_sizes())
            self.assertEqual(before, digest(cache / "preprocessor.pkl"))

    def test_original_main_pickles_load_without_rewriting(self):
        with tempfile.TemporaryDirectory(prefix="daily-legacy-test-") as temporary:
            root = Path(temporary)
            # Changing class metadata is confined to this disposable child;
            # reproduces exactly the original `python -m` pickle references.
            script = """
import __main__, sys, torch
from collections import Counter
from pathlib import Path
from data.schema import CATEGORY_COLS
from training.pretrain_cache import CanonicalFrequencyEncoder, preprocessor_from_counts
from training.train_daily_encoders import FlatCrossEntropy
from training.common import save_preprocessor
root=Path(sys.argv[1])
for cls in (CanonicalFrequencyEncoder, FlatCrossEntropy):
    cls.__module__='__main__'
    setattr(__main__,cls.__name__,cls)
preprocessor=preprocessor_from_counts({col:Counter({'1':3,'<MISSING>':1}) for col in CATEGORY_COLS})
save_preprocessor(root/'legacy.pkl',preprocessor)
torch.save({'loss':FlatCrossEntropy(ignore_index=0),'weight':torch.tensor([1.,2.])},root/'legacy.pt')
"""
            self.child(root, "-c", script, root)
            before = [digest(root / name) for name in ("legacy.pkl", "legacy.pt")]
            preprocessor = load_preprocessor(root / "legacy.pkl")
            transformed = preprocessor.cts_category[0].transform(pd.DataFrame({"col_2": [1., float("nan")]}))
            self.assertEqual(transformed.col_2.tolist(), [1, 2])
            checkpoint = load_torch_checkpoint(root / "legacy.pt")
            torch.testing.assert_close(checkpoint["weight"], torch.tensor([1., 2.]))
            self.assertEqual(type(checkpoint["loss"]).__name__, "FlatCrossEntropy")
            self.assertEqual(before, [digest(root / name) for name in ("legacy.pkl", "legacy.pt")])
            # The stable CLI also supplies old aliases to the unchanged
            # Lightning loader, then restores __main__ when it exits.
            stdout = self.child(root, "-c", """
import sys, torch
from pathlib import Path
from training.daily_cli import legacy_main_aliases
from training.common import load_preprocessor
with legacy_main_aliases():
    p=load_preprocessor(Path(sys.argv[1])/'legacy.pkl')
    c=torch.load(Path(sys.argv[1])/'legacy.pt',weights_only=False)
    assert p.get_category_dictionary_sizes()['col_2']==4
    assert c['weight'].tolist()==[1.,2.]
print('OLD_ARTIFACTS_OK')
""", root)
            self.assertIn("OLD_ARTIFACTS_OK", stdout)


if __name__ == "__main__":
    unittest.main()
