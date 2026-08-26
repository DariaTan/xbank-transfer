# CUDA 12.4 matches the host driver (550.144.03, max CUDA 12.4) -- do not bump
# to a cuda12.6/13.0 base without also confirming the host driver was upgraded,
# newer CUDA runtimes are not backward-compatible with an older driver.
# Pinned by digest (not just tag) for full reproducibility -- tags can move,
# digests can't.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installs from the full resolved lock file, not requirements.txt, so a
# rebuild is byte-identical. See requirements.txt for the human-edited
# direct-dependency list and how to regenerate the lock after changing it.
COPY environments/requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt

# Workarounds for real bugs in pytorch-lifestream==0.7.0 -- see
# environments/patches/README.md for what's actually wrong upstream.
COPY environments/patches/ptls_preprocessing_init.py \
    /opt/conda/lib/python3.11/site-packages/ptls/preprocessing/__init__.py

CMD ["/bin/bash"]
