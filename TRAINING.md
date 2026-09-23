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
to pretrain on MBD-raw. `configs/data/mbd_daily.yaml` is now an
inference-only aggregation-shift corpus; no new MBD-daily pretraining is
planned. Always pass `--data-config configs/data/mbd.yaml` explicitly for
the production training run.
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


## Launching in tmux

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
