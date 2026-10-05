import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data.splits import load_windowed_transactions_for_dates
from training.infer_mbd import _run_date


class BoundedDailyInferenceTests(unittest.TestCase):
    def test_cap_and_ties_stable_across_projection_and_parquet_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = pd.DataFrame({"id": ["a"]*20, "col_1": pd.to_datetime(["2022-12-31"]*20),
                                   "col_2": list(range(20)), "col_3": list(reversed(range(20)))})
            seen = []
            for seed in (1,2):
                p = root/f"{seed}.parquet"
                source.sample(frac=1,random_state=seed).to_parquet(p,index=False)
                full = load_windowed_transactions_for_dates(str(p),["2023-01-01"],12,5,bounded=True)
                small = load_windowed_transactions_for_dates(str(p),["2023-01-01"],12,5,
                            columns=["id","col_1","col_2"],bounded=True)
                self.assertEqual(full.col_2.tolist(),small.col_2.tolist())
                self.assertEqual(full.col_2.tolist(),[15,16,17,18,19])
                seen.append(full.col_2.tolist())
            self.assertEqual(seen[0],seen[1])

    def test_bounded_xbank_excludes_cutoff_and_handles_empty_population(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"input.parquet"
            pd.DataFrame({"id":["a","a"],"col_1":pd.to_datetime(["2022-12-31","2023-01-01"]),
                          "col_2":[3,99]}).to_parquet(p,index=False)
            frame=load_windowed_transactions_for_dates(str(p),["2023-01-01"],12,500,bounded=True,include_cutoff=False)
            self.assertEqual(frame.col_2.tolist(),[3])
            self.assertTrue(load_windowed_transactions_for_dates(str(p),["2023-01-01"],12,500,client_ids=[],bounded=True).empty)

    def test_resume_partial_chunk_and_publish_without_duplicate_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            p=root/"input.parquet"
            pd.DataFrame({"id":["a","b"],"col_1":pd.to_datetime(["2022-12-31"]*2),"col_2":[1,2]}).to_parquet(p,index=False)
            cfg={"history_window_months":12,"max_seq_len":500,"bounded_windows":True}
            calls=[]
            def embed(frame):
                client=frame.id.iloc[0]
                calls.append(client)
                if client.startswith("b") and len(calls)==2:
                    raise RuntimeError("simulated interruption")
                return np.ones((1,2)),[client]
            out=root/"out"
            out.mkdir()
            with self.assertRaisesRegex(RuntimeError,"simulated interruption"):
                _run_date("2023-01-01",[["a"],["b"]],p,out,cfg,["id","col_1","col_2"],embed)
            _run_date("2023-01-01",[["a"],["b"]],p,out,cfg,["id","col_1","col_2"],embed)
            self.assertEqual(sum(x.startswith("a") for x in calls),1)
            self.assertFalse((out/"_chunks").exists())
            with patch("training.infer_mbd.load_windowed_transactions_for_dates",side_effect=AssertionError("unexpected rescan")):
                _run_date("2023-01-01",[["a"],["b"]],p,out,cfg,["id","col_1","col_2"],embed)


if __name__ == "__main__":
    unittest.main()
