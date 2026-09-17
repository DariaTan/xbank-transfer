"""Serial (non-Dask) replacement for pytorch-lifestream==0.7.0's
multithread_dispatcher.py -- see environments/patches/README.md.

The real `DaskDispatcher` creates a persistent `DaskServer` singleton
(`LocalCluster(processes=False, n_workers=4, threads_per_worker=4,
memory_limit='auto')`) -- `processes=False` means its "workers" are just
threads inside THIS SAME process, so whatever memory they retain shows up
directly as this process's own RSS. Confirmed directly (2026-09-14): RSS
grew ~16GB -> 37GB over 6 chunks of a chunked full-scale training run
that reuses one `PandasDataPreprocessor` across many repeated
`fit_transform`/transform calls (`data/loaders.py`'s `build_ptls_records`,
called once per chunk) -- exactly matching Dask's own "Unmanaged memory
use is high... may not be released to the OS" warnings. This project
only ever transforms a handful of category/numerical columns per call;
serial execution is cheap enough that Dask's cross-column parallelism
isn't worth a persistent cluster's memory cost, especially under a reuse
pattern Dask's own LocalCluster apparently isn't built to release memory
back to the OS under.

Same public interface as the original (`evaluate`/`shutdown`/
`transformation_func`), so `ptls.preprocessing.base.data_preprocessor.
DataPreprocessor` needs no changes -- it calls `multithread_dispatcher.
evaluate(individuals=..., objective_func=...)` in two shapes: `objective_func`
a dict of {col_name: transformer} (per-column fit_transform/transform,
called with `individuals[col_name]` wrapped in `pd.DataFrame(...)` -- the
ORIGINAL does this same wrapping, kept identical here) returning a list of
per-column results in dict-iteration order (matches the original's
`list(map(...))` + `dask.compute(*results)`, which is a no-op passthrough
for already-eager results); or a single transformer (aggregate-column
step, called directly on the given DataFrame, no dict). Verified
equivalent to the original Dask dispatcher on an identical input sample
before deploying (2026-09-14) -- same category_dictionary_sizes, same
record values.
"""
from typing import Callable, Dict, List, Union

import pandas as pd


class DaskDispatcher:
    def __init__(self, n_jobs: int = -1):
        self.transformation_func = "fit_transform"

    def shutdown(self):
        pass

    def _evaluate_single(self, data: Union[pd.DataFrame, pd.Series], eval_func: Callable):
        if self.transformation_func == "fit_transform":
            return eval_func.fit_transform(data)
        return eval_func.transform(data)

    def evaluate(
        self,
        individuals: Union[List, Dict, pd.DataFrame],
        objective_func: Union[Callable, Dict],
    ):
        if individuals is None:
            return None
        if isinstance(objective_func, dict):
            return [
                self._evaluate_single(pd.DataFrame(individuals[func_name]), func_impl)
                for func_name, func_impl in objective_func.items()
            ]
        return self._evaluate_single(individuals, objective_func)
