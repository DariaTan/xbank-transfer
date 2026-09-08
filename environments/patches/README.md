# Patches

Workarounds for real bugs in third-party packages, applied by the
Dockerfile after `pip install`. Not our code -- upstream issues, kept
here (rather than fixed live in a running container) so a fresh image
build reproduces the same working state.

## `ptls_preprocessing_init.py`

Replaces `ptls/preprocessing/__init__.py` in the installed
`pytorch-lifestream==0.7.0` package.

Two independent problems in the same import chain, both triggered just by
`from ptls.preprocessing import PandasDataPreprocessor` -- this project
only ever uses the pandas backend, never dask or pyspark:

1. `ptls.preprocessing.dask.dask_preprocessor` imports `dask.distributed`,
   which isn't a transitive dependency of the `dask` package alone.
   Fixed by adding `distributed` to `requirements.txt` -- a normal pip
   dependency, no patch needed for this half.
2. `ptls.preprocessing.dask.dask_transformation` (a subpackage
   `dask_preprocessor.py` imports from) is entirely missing from the
   published PyPI wheel -- confirmed by comparing against the upstream
   git source, which has the files but no `__init__.py` in that
   directory, so `setuptools.find_packages()` silently drops it when
   building the wheel. This one can't be fixed by adding a pip package;
   there's nothing to install.

The patched `__init__.py` wraps both the `dask` and `pyspark` sub-imports
in `try/except ImportError`, since we don't use either backend. If a
future model adapter needs the Spark preprocessor, install `pyspark` as a
real dependency instead of removing this guard.

## `dt_to_timestamp` -- not patched here, worked around in our own code

A separate, unrelated bug in the same version: `PandasDataPreprocessor`'s
internal dispatcher wraps every unitary-transform column in
`pd.DataFrame(...)` before calling the transformer
(`multithread_dispatcher.evaluate_single`), but
`DatetimeToTimestamp.transform` (used when
`event_time_transformation="dt_to_timestamp"`) calls `pd.to_datetime()`
directly on that DataFrame instead of re-extracting the Series first --
pandas then tries to assemble a date from year/month/day *columns* and
raises `ValueError: to assemble mappings requires at least that [year,
month, day] be specified`. `ColIdentityEncoder` (used by
`event_time_transformation="none"`) re-extracts the Series correctly, so
`data.loaders.build_ptls_records` precomputes the numeric event
time itself with `ptls.preprocessing.util.dt_to_timestamp` and passes
`event_time_transformation="none"`. No site-packages patch needed for
this one -- see the docstring in `src/data/loaders.py` for the
full explanation.
