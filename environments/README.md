# environments/

Docker setup for the xbank-transfer project.

## Build

```bash
./environments/dbuild.sh
```

## Run

```bash
./environments/drun.sh    # starts the container as a persistent background service
./environments/dexec.sh   # attaches an interactive shell to it
```

The container runs `tail -f /dev/null` to stay alive rather than an interactive
shell directly, so a training job survives an SSH disconnect -- `dexec.sh` is
how you get a shell in it, any number of times.

Mounts the repo root at `/app` and the transaction data
(`/mnt/storage/d.tanyushkina/transactions`) read-write at `/app/data`
(read-write, not read-only, since downstream embeds/checkpoints/logs are
all written back under there -- be careful not to touch the raw parquet
files directly), and `/mnt/storage/d.tanyushkina/hf_cache` (read-write) at
`/hf_cache` -- Chronos-2 downloads a real pretrained checkpoint from the
HF Hub (public weights, Apache-2.0, not proprietary data) on first use;
without this mount it silently re-downloads every time the container is
recreated. `HF_HOME` is set to match. Runs as the host user
(`--user "$(id -u):$(id -g)"`, added 2026-09-14) rather than root, so
every file it creates is already owned correctly on the host -- `HOME` is
set to `/tmp` since the base image's baked-in `HOME=/root` isn't writable
by a non-root uid. `PYTHONPATH` is set to `/app/src` so
`data.*`/`models.*`/`training.*` are importable from anywhere in the
container -- the smoke/train entry-point scripts live at `src/training/`,
not a separate top-level `scripts/`.
Publishes port 6006 for TensorBoard -- see TRAINING.md for the access
command (it isn't reachable from outside the host on its own; forward it
over SSH).

Both host GPUs (RTX A5000, 24GB each) are shared with other users' containers on
this box -- pass `GPUS='"device=0"'` before `drun.sh` to grab only one:

```bash
GPUS='"device=0"' ./environments/drun.sh
```

## What's in the image vs. what isn't

Pip-installed in the image: `pytorch-lifestream` (CoLES), `easy-tpp` (THP / NHP /
AttNHP / Intensity-Free TPP), `transformers` + `chronos-forecasting` (Chronos-2),
plus the usual data/ML stack (pandas, duckdb, scikit-learn, lightgbm, ...).

**Not** pip-installable, cloned as source under `third_party/` instead (untracked,
see `.gitignore`) so the image doesn't need rebuilding every time the upstream
code changes:

```bash
git clone https://github.com/VladislavZh/COTIC.git third_party/COTIC
```

The Sber "NEP" causal-transformer baseline has no released code or weights --
it's built from scratch on top of `transformers`/`torch`, already covered above.

## Patches

`environments/patches/` fixes real bugs in `pytorch-lifestream==0.7.0` that
otherwise break `from ptls.preprocessing import PandasDataPreprocessor`
(a missing `dask.distributed` dependency, and a subpackage silently dropped
from the published PyPI wheel) -- applied automatically by the Dockerfile
after `pip install`. See `environments/patches/README.md` for what's
actually wrong upstream and why. Verified against a from-scratch rebuild,
not just a live-patched container -- `docker exec xbank-transfer python
src/training/train_coles.py --n-clients 500 --max-epochs 1` (or any
of the other `train_*.py` scripts, which hit the same
`PandasDataPreprocessor` patch path) works right after `dbuild.sh` +
`drun.sh` with no manual steps.
