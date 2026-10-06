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

## Daily-pretrained LightGBM queue

Run **on the server host**, after pulling the tested revision:

```bash
bash environments/run_daily_source_lightgbm.sh 1
tail -f /mnt/storage/d.tanyushkina/transactions/logs/daily_source_lightgbm_gpu1_controller.log
tmux attach -t daily-source-lightgbm-gpu1
```

Detach with Ctrl-b, then d. The queue uses physical GPU 1 (OpenCL device 0
inside its single-GPU container), 6 CPU threads, and a 24-GiB RAM limit.
It waits for a free GPU and at least 40 GiB available host RAM. It does not
stop other jobs or use GPU 0. Complete runs resume without overwriting results;
changed inputs/settings abort rather than silently reusing an incompatible run.

This queue evaluates CoLES, COTIC, THP, retained daily NEP, and MLM from
`embeds/{mbd_daily,xbank,xbank_fgw_v2}/mbd_daily_source/`. It waits for all 12
published dates of a representation; pending MLM does not block other encoders.
Partial `_chunks` files are never inputs. Inference manifests and weight hashes
are recorded in downstream provenance; both xbank mappings must use the same
encoder checkpoint/preprocessor.

MBD uses four independent binary targets, five client-disjoint test folds,
four HPO candidates on at most 50,000 training clients, up to 400 rounds,
`max_bin=255`. Xbank excludes target `col_2`, intersects the two mappings'
client-date cohorts, uses client-disjoint training/validation in 2023 and an
untouched 2024 test, three HPO candidates, up to 300 rounds, `max_bin=63`.
Both use seed 42, no resampling/class weighting, validation average precision
for selection, and refit on development rows before accessing test scores.
Reported metrics are PR-AUC (average precision) and ROC-AUC only.

Outputs are separate from all `mbd_source` and `zero_shot` results:

```text
downstream/mbd_daily/mbd_daily_source/<model>/lightgbm_hpo_cv/fold{0..4}/
downstream/{xbank,xbank_fgw_v2}/mbd_daily_source/<model>/lightgbm_hpo_calendar/
downstream/xbank_mapping_comparison/mbd_daily_source/<model>_paired_results.csv
```

Each model directory also receives `results_all_folds.csv`,
`results_aggregated.csv` and `model_macro.csv`. Temporary streamed feature caches
live on the storage volume and are removed only after successful final summaries;
models, metrics, selections, manifests and logs remain for reproducibility.
The retained NEP was not retrained with the new four-encoder training protocol,
so label it as a retained baseline when reporting comparisons.
