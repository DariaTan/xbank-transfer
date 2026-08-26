try:
    from .dask.dask_preprocessor import DaskDataPreprocessor
except ImportError:
    # `ptls.preprocessing.dask.dask_transformation` is missing from the
    # published pytorch-lifestream==0.7.0 PyPI wheel: the subpackage has no
    # __init__.py, so setuptools.find_packages() silently drops it from the
    # build, but dask_preprocessor.py still imports from it unconditionally.
    # This project only uses PandasDataPreprocessor, so DaskDataPreprocessor
    # is patched to None rather than crashing the whole `ptls.preprocessing`
    # package on import. See environments/README.md for how this patch is
    # (re)applied in the Docker image.
    DaskDataPreprocessor = None
from .pandas.pandas_preprocessor import PandasDataPreprocessor
try:
    from .pyspark.pyspark_preprocessor import PysparkDataPreprocessor
except ImportError:
    # pyspark isn't installed -- this project doesn't use the Spark backend.
    PysparkDataPreprocessor = None
