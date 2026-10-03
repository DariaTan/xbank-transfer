# Full-scale training

Five full-scale training scripts, `src/training/train_{coles,cotic,thp,nep,mlm}.py`
-- one per trainable model, living inside `src/training/` itself (not a
separate top-level `scripts/`) so they import sibling packages (`data.*`,
`models.*`, `training.*`) directly, no sys.path hack -- the container sets
`PYTHONPATH=/app/src` (see `drun.sh`).
Chronos-2 has nothing to train (zero-shot only, see
`src/training/infer_chronos2.py`). Each script takes two CLI flags:
`--config` (default `configs/models/<model>.yaml` -- every architecture/
run parameter lives in that file, not on the command line) and
`--data-config` (default `configs/data/xbank.yaml`, kept as the flag's
default for backward compatibility, but see the standing decision below),
which selects the pretraining CORPUS -- pass `configs/data/mbd.yaml`
to pretrain on MBD-raw. For the new MBD-daily institutional-shift arm,
use the safe runner below, not these legacy training scripts. Always pass
`--data-config configs/data/mbd.yaml` explicitly for legacy raw training.
`checkpoint_dir` is derived from `--data-config`'s own `name`.

- Loads **all** clients from the configured corpus, not a sample.
  Set `n_clients` in `--config` to cap it for a
  quick debug run.
- Trains with early stopping on a validation metric (`patience`, default
  5 epochs of no improvement) under a generous `max_epochs` cap (default
  100). There's no prior estimate of how many epochs full-scale
  convergence needs for any of these models, so early stopping decides
  rather than a guessed fixed count.
- Checkpoints every epoch to `checkpoint_dir` (`last.*` for resuming,
  `best.*` for the best validation epoch so far) -- re-running the exact
  same command picks up where a killed job left off automatically, no
  extra flags needed. xbank-pretrained checkpoints live under
  `/app/data/checkpoints/xbank_source/<model>/`; MBD-pretrained ones
  should go under the sibling `/app/data/checkpoints/mbd_source/<model>/`.


## MBD-daily pretraining (2026-10-02)

The approved new arm trains **CoLES, COTIC, THP and MLM**, while keeping
the existing daily NEP unchanged. From the **Docker host**:

```bash
bash environments/run_mbd_daily_pretraining.sh
tmux ls
tail -f /mnt/storage/d.tanyushkina/transactions/logs/mbd-daily-pretrain_controller.log
tail -f /mnt/storage/d.tanyushkina/transactions/logs/mbd_daily_pretrain_prepare.log
tail -f /mnt/storage/d.tanyushkina/transactions/logs/mlm_mbd_daily_v2.log
nvidia-smi
```

One CPU preparation precedes two queues: GPU0 MLM then CoLES; GPU1 THP
then COTIC. Idle GPUs and available RAM are checked before starting;
each disposable container is limited to 24 GiB RAM and six CPUs. A
full-vocabulary, configured-batch, length-500 GPU preflight runs in a
separate process before each model starts. Failure stops that queue, not
the other GPU, and is logged; rerunning the launcher resumes saved epochs.
Closing SSH does not stop the host tmux worker. Do not git pull or change
training source/configs during an active run.

Serialization repair (2026-10-03): production scripts now invoke
`python -m training.daily_cli prepare ...` / `... train ...`, importing
the original implementations so helper classes get stable pickle module
names. Do not invoke the implementation modules directly with `python -m`.
The compatible inference loader also reads the two original `__main__`
helper references; it does not rewrite data, weights or manifests. Only
load trusted project artifacts, as with ordinary pickle/Torch checkpoints.
The original training/preparation/common source files remain unchanged so
the provenance checks of existing THP and active COTIC still match.

If GPU0 stopped before MLM, recover **only** its queue while GPU1 keeps
running; this reuses the frozen complete cache and appends to model logs:

```bash
bash environments/run_mbd_daily_pretraining.sh --gpu0-only
tail -f /mnt/storage/d.tanyushkina/transactions/logs/mbd-daily-pretrain-gpu0_controller.log
```

Recovery has its own host tmux session `mbd-daily-pretrain-gpu0`. The old
controller may still report its historical GPU0 failure when GPU1 ends;
use the recovery log and per-model `complete.json` for current status.

Preparation stores only reusable encoded arrays and provenance under
`/app/data/training_cache/mbd_daily/v2`; intermediate parquet/spill files
are removed once the cache is validated. No labels are loaded. The cache
uses all source clients, exact 95/5 client split (seed 0, sorted IDs),
latest 500 events per client capped in DuckDB **before** pandas, and
vocabularies fitted on **all eligible train events only**. Deterministic
same-day ties use the feature tuple, not invented within-day timestamps.
New preprocessors canonicalize nullable numeric category codes (`7` and
`7.0` are the same category; missing values use an explicit token), both
when fitting the cache and during subsequent inference. This does not
alter category handling inside historical raw or NEP preprocessors.
Clients with fewer than two events are excluded from new pretraining;
TPP validation additionally drops unknown event marks. `audit.json` and
each model's `cohort.json` record exclusions and unknown codes.

Architectures and existing model configs are retained: early stopping
patience 5, ceiling 100 epochs. Training fixes Python/NumPy/Torch/CUDA
seeds, requires deterministic operations, isolates fixed validation RNG,
rejects non-finite losses/gradients, and atomically saves `best` and `last`.
There is no new gradient clipping or timestamp jitter. MLM computes heads
only at supervised masked positions (same loss, less memory), with at
least one mask per sequence. Resume reuses frozen preprocessing and checks
data/config/code identity. An epoch interrupted mid-way is replayed from
the previous saved epoch; it is not resumed at the exact batch.

Weights: `/app/data/checkpoints/mbd_daily_source/{coles,cotic,thp,mlm}`.
Logs: `/app/data/logs/*_mbd_daily_v2.log`, TensorBoard:
`/app/data/lightning_logs/mbd_daily_source/<model>`.
`complete.json` is the success marker, including best epoch/score/hash.
Nothing writes into raw checkpoints, existing embeddings, downstream
outputs or `/app/data/checkpoints/mbd_daily_source/nep`.

Smoke testing (disposable files, automatically removed):

```bash
python -m training.smoke_daily_encoders --device cuda \
  --temp-root /app/data --real-data-config /app/configs/data/mbd_daily.yaml \
  --production-shapes
```

Scientific scope: like the historical raw run, this is **transductive
unlabelled pretraining on the whole source corpus**, not a strict
unseen-client/causal forecasting experiment. Downstream train/validation/
test labels remain separated, but this does not undo pretraining exposure
to test-client histories. Existing NEP uses the older preparation/split;
it is retained for time reasons and is **not a matched rerun** of this
four-model protocol. Cross-arm results should disclose that limitation.
Institutional transfer still requires the agreed frozen xbank feature
mapping; vocabulary semantics across banks are a separate research risk.

## Legacy raw launching in tmux

Run tmux inside the persistent container, but invoke it from the Docker
host. One session per model means a closed SSH connection does not kill the
job. Always pass the data config explicitly; the training entry points keep
their historical xbank default for backward compatibility.

```bash
docker exec xbank-transfer tmux new-session -d -s coles \
  'cd /app && CUDA_VISIBLE_DEVICES=0 python src/training/train_coles.py --data-config /app/configs/data/mbd.yaml 2>&1 | tee -a /app/data/logs/coles_mbd_raw.log'
```

That's two models per GPU as a starting guess (CoLES+THP+MLM on GPU 0,
COTIC+NEP on GPU 1) -- a starting guess, not a hard rule. COTIC/THP's
convolution/attention over 500-length sequences are heavier per-sample
than CoLES/NEP/MLM's GRU/transformer at the same batch size. If jobs
sharing a GPU OOM or visibly slow each other down, run them one-at-a-time
per GPU instead and queue the rest -- rerun the same tmux command for a
queued model once a GPU frees up (a finished or early-stopped job releases
its memory on process exit).

**Watching progress:**

```bash
docker exec xbank-transfer tail -f /app/data/logs/coles_mbd_raw.log
docker exec -it xbank-transfer tmux attach -t coles  # Ctrl-b d to detach
```

**Managing sessions:**

```bash
docker exec xbank-transfer tmux ls
docker exec xbank-transfer tmux kill-session -t coles
```

## Watching curves in TensorBoard

All five models log to `/app/data/lightning_logs/<data_cfg_name>_source/<model>`
(CoLES/COTIC via `TensorBoardLogger`, THP/NEP/MLM via a plain
`SummaryWriter` -- `train/loss`+`valid/loss`, or `train/nll`+`valid/nll`
for THP) -- diverges by `--data-config`'s own `name` field, same
`<name>_source` convention as `checkpoint_dir` (fixed 2026-09-17: it used
to be a flat `/app/data/lightning_logs/<model>` regardless of
`--data-config`, so an xbank run and an MBD run of the same model wrote
into the SAME TensorBoard log dir and their curves interleaved
indistinguishably). Launch it inside the container, backgrounded:

```bash
docker exec -d xbank-transfer tensorboard --logdir /app/data/lightning_logs --host 0.0.0.0 --port 6006
```

The container publishes port 6006 to the host (`-p 6006:6006` in
`drun.sh`), but the host itself isn't reachable from your laptop --
forward it over SSH and open the local end in a browser:

```bash
ssh -L 6006:localhost:6006 d.tanyushkina@10.16.84.4
```

Then browse to `http://localhost:6006`. Stop it with
`docker exec xbank-transfer pkill -f tensorboard` when done -- it keeps
running (and holding the port) until then.

## If a full-population load runs out of host RAM

`load_all_raw` pulls every row into pandas in one shot (~91.5M rows across
all clients). The host had 106GB free as of 2026-08-26, which should be
enough for any single model's load (COTIC/THP project down to 3 columns;
CoLES/NEP/MLM need all 16) -- but if a job OOMs: rerun it with `n_clients`
(in `configs/models/<model>.yaml`) set to some large-but-bounded number as
a stopgap, and flag it. A real
chunked/streaming loader would be the actual fix, not built yet since it
wasn't needed at smoke-test scale.

## Inference

Six inference scripts, `src/training/infer_{coles,nep,mlm,thp,cotic,chronos2}.py`
-- one per model, producing frozen per-(client, date) embeddings on a fixed
monthly grid (1st of each month, 2023-01-01 through 2024-02-01 by default)
from a client's trailing history window up to that date. Chronos-2 is
zero-shot (no checkpoint, no `configs/models/chronos2.yaml`); the other
five reuse their own trained checkpoint's fitted preprocessor/categories/
normalizer artifact -- never refit fresh on windowed data, since that
would silently produce a different vocabulary than the one the
checkpoint's embedding tables were trained against.

Each script takes only `--downstream-config` (default
`configs/models/downstream.yaml`, `inference:` section -- shared params:
`n_clients`/`start_date`/`end_date`/`history_window_months`/`max_seq_len`/
`batch_size`/`embeds_dir`) and, for the five checkpoint-based models,
`--model-config` (default `configs/models/<model>.yaml` -- that
architecture's own `hidden_size`/`num_layers`/etc, which must match what
the checkpoint was actually trained with):

```bash
python src/training/infer_coles.py
python src/training/infer_nep.py
python src/training/infer_mlm.py
python src/training/infer_thp.py
python src/training/infer_cotic.py
python src/training/infer_chronos2.py
```

Output: one parquet file per target date under the source-namespaced output
directory, columns `[inn, date, emb_0..emb_D]`
-- resumable, a date whose file already exists is skipped on the next
per-client than the other five, via `chronos2_chunk_size`/
`chronos2_batch_size` in the same `inference:` section); see that
script's own docstring for the chunk-merge-cleanup mechanics.

### Cross-institution transfer (MBD)

One consolidated script, `src/training/infer_mbd.py`, runs the SAME
frozen checkpoints zero-shot over MBD (Sber's public benchmark --
`data/mbd_adapter.py` reshapes it into xbank's own column convention
first; run its adapter once before this):

MBD-raw inference with MBD-raw checkpoints is the current production path.
From the Docker host, launch at most two models at a time with the helper:

```bash
./environments/run_infer_mbd_raw.sh coles cotic
```

To run the complete MBD-raw then MBD-daily matrix unattended, start the two
sequential GPU queues:

```bash
./environments/run_infer_mbd_queues.sh
```

GPU 0 runs CoLES/THP/MLM and GPU 1 runs COTIC/NEP. Each queue finishes its
raw jobs before continuing with the same models on daily input. A failed job
stops only its own queue so the error is not hidden by later jobs.

The equivalent command inside the container is:

```bash
python src/training/infer_mbd.py \
  --model coles \
  --data-config /app/configs/data/mbd.yaml \
  --downstream-config /app/configs/models/downstream_mbd.yaml \
  --checkpoint-source mbd
```

The job only processes clients present in MBD targets, runs in resumable
client chunks, and writes atomically to
`/app/data/embeds/mbd_raw/mbd_source/<model>/`. Since `/app/data` is the
container bind mount, these files physically live under
`/mnt/storage/d.tanyushkina/transactions/embeds/` on the server.

For a local smoke test, set `XBANK_DATA_ROOT` to a writable fixture root and
use `configs/data/mbd_smoke.yaml` plus
`configs/models/downstream_mbd_smoke.yaml`; smoke outputs never share the
production directory.

Chronos-2 is a separate zero-shot control on MBD-raw. Its source transactions
are aggregated to daily amount sums once, then 32 independent client shards
are split between two GPUs. Unlike the five event models, Chronos sees the
full 12-month daily history, with no 500-transaction cap. On the Docker host:

```bash
./environments/run_chronos_mbd_raw.sh
```

The resumable job first writes its daily cache under
`/app/data/chronos2_daily_cache/mbd_raw`, then publishes embeddings under
`/app/data/embeds/mbd_raw/zero_shot/chronos2`. Logs are
`/app/data/logs/chronos_mbd_raw_{prepare,gpu0,gpu1,finalize}.log`.

Reads the same two-config split as the xbank scripts above, just against
`configs/models/downstream_mbd.yaml`'s `inference:` section instead.
Output: `<embeds_dir>/<evaluation_name>/<checkpoint_source>_source/<model>/<date>.parquet`,
one file per month that actually
appears in MBD's own targets file -- not xbank's calendar grid (MBD's
target months and xbank's aren't on the same calendar axis; see
RESEARCH_PLAN.md's "ID / date splitting (transfer, MBD)").

## Downstream classification probe

Two scripts evaluate how well each model's frozen embeddings predict real
binary targets, via an independent LightGBM + MLP per target column
(`roc_auc`/`pr_auc`/`precision@k`/`recall@k`, optional negative
undersampling) -- every tunable parameter lives in `configs/models/
downstream.yaml`'s (or `downstream_mbd.yaml`'s) `probe:` section:

```bash
# xbank in-domain: train on all of 2023, held-out test on 2024-01/02
python src/training/train_downstream.py --model coles
python src/training/train_downstream.py --model coles --eval-only   # re-score saved checkpoints, no retraining

# MBD transfer: out-of-fold rotation across every client-disjoint fold
# present (the paper's own protocol -- it names no single canonical test
# fold, see RESEARCH_PLAN.md), reported as mean +/- std across rotations
python src/training/train_downstream_mbd.py \
  --model coles \
  --data-config /app/configs/data/mbd.yaml \
  --checkpoint-source mbd
```

Output: `lgbm_<target>.txt`/`mlp_<target>.pt` per target column, plus
`results.csv` (xbank) or `results_all_folds.csv`/`results_aggregated.csv`
(MBD, under `/app/data/downstream/<evaluation>/<source>/<model>/`).

## Chronos-2 LightGBM (2026-10-01)

Start the memory-capped queue from the Docker host:

```bash
bash environments/run_chronos_lightgbm.sh 0
```

The host tmux session `chronos-lightgbm-gpu0` uses one GPU and a 24-GiB
container limit. It runs xbank's calendar probe, MBD-raw's five client-fold
rotations, then produces the missing MBD-daily Chronos embeddings and runs
the same five rotations there. Jobs wait for at least 32 GiB available RAM.
A failure stops the queue; rerunning the launcher resumes completed targets
and their saved hyperparameter selections.

Chronos has 768 features, so the ordinary pandas-concatenation HPO loader is
too large for the shared server. `tune_lightgbm_chronos.py` streams parquets
into a temporary float32 disk cache and builds LightGBM bins through Sequence
batches. The cache is removed after a successful corpus summary; checkpoints,
metrics and manifests remain. Precision/recall are not calculated or backfilled.

The MBD protocol is identical to the completed HPO runs: seed 42, four
candidates, 50,000 tuning clients, at most 400 rounds; a preceding fold
validates candidates/early stopping, three folds train, and all four
development folds refit before the untouched test fold is scored. GPU bins
use 255 values, matching the old MBD default. Xbank uses three candidates,
300 rounds, 63 bins, the configured client-grouped 2023 train/validation
split, and 2024-01/02 test; conflicting target `col_2` is excluded.

Existing HPO probes used no class weighting or negative sampling; the Chronos
probe follows that actual protocol. The manuscript's statement that every
probe applies class weighting therefore needs to be corrected separately.
PR-AUC is sklearn average precision and ROC-AUC is sklearn ROC AUC. MBD
summaries give per-target mean/std across folds and an equal-weight macro
average over four targets; xbank is a single calendar test over three targets.

Outputs: `/app/data/downstream/<evaluation>/zero_shot/chronos2/`, with
per-fold (or `lightgbm_hpo_calendar`) models/metrics plus
`results_all_folds.csv`, `results_aggregated.csv`, `model_macro.csv`.
Controller log: `/app/data/logs/chronos_lightgbm_controller.log`;
per-stage logs: `/app/data/logs/chronos_lightgbm_*.log`.

Chronos uses a strict pre-target-day cutoff and a full 12-month daily amount
series. The five event encoders' existing MBD outputs use an inclusive cutoff
and a 500-event cap; these differences must be disclosed in the comparison.
Raw and daily amount normalization differs, so daily Chronos embeddings are
computed separately. On xbank both frozen mappings select `col_12` as amount;
the existing Chronos amount-only embeddings can serve both mapping references.
