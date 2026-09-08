# Full-scale training

Five full-scale training scripts, `src/training/train_{coles,cotic,thp,nep,mlm}.py`
-- one per trainable model, living inside `src/training/` itself (not a
separate top-level `scripts/`) so they import sibling packages (`data.*`,
`models.*`, `training.*`) directly, no sys.path hack -- the container sets
`PYTHONPATH=/app/src` (see `drun.sh`).
Chronos-2 has nothing to train (zero-shot only, see
`src/training/infer_chronos2.py`). Each script takes no CLI flags except
`--config` (default `configs/models/<model>.yaml`) -- every run parameter
lives in that file, not on the command line:

- Loads **all** clients from `trans_any_pos_anonym_encoded.parquet`, not a
  sample (`smoke_*.py` sampled 500 to prove the pipeline works;
  these load the real ~378K-client corpus). Set `n_clients` in the config
  to cap it for a quick debug run.
- Trains with early stopping on a validation metric (`patience`, default
  5 epochs of no improvement) under a generous `max_epochs` cap (default
  100). There's no prior estimate of how many epochs full-scale
  convergence needs for any of these models, so early stopping decides
  rather than a guessed fixed count.
- Checkpoints every epoch to `checkpoint_dir` (default
  `/app/data/checkpoints/<model>/`; `last.*` for resuming, `best.*` for
  the best validation epoch so far) -- re-running the exact same command
  picks up where a killed job left off automatically, no extra flags
  needed.

## Picking a GPU

Both host GPUs (RTX A5000, 24GB each) are shared with other users'
containers. Check current load before launching:

```bash
./environments/pick_gpu.sh
```

Pin a job to one GPU via `CUDA_VISIBLE_DEVICES` on the `docker exec` call
-- the container itself sees both GPUs (`--gpus all` in `drun.sh`), so this
env var is what actually restricts a given process to one of them.

```bash
mkdir -p /app/data/logs
CUDA_VISIBLE_DEVICES=0 xbank-transfer python src/training/train_coles.py 2>&1 | tee /app/data/logs/coles.log
CUDA_VISIBLE_DEVICES=1 xbank-transfer python src/training/train_cotic.py 2>&1 | tee /app/data/logs/cotic.log
СUDA_VISIBLE_DEVICES=0 xbank-transfer python src/training/train_thp.py   2>&1 | tee /app/data/logs/thp.log
CUDA_VISIBLE_DEVICES=1 xbank-transfer python src/training/train_nep.py   2>&1 | tee /app/data/logs/nep.log
CUDA_VISIBLE_DEVICES=0 xbank-transfer python src/training/train_mlm.py   2>&1 | tee /app/data/logs/mlm.log
```

That's two models per GPU as a starting guess (CoLES+THP+MLM on GPU 0,
COTIC+NEP on GPU 1) -- a starting guess, not a hard rule. COTIC/THP's
convolution/attention over 500-length sequences are heavier per-sample
than CoLES/NEP/MLM's GRU/transformer at the same batch size. If jobs
sharing a GPU OOM or visibly slow each other down, run them one-at-a-time
per GPU instead and queue the rest -- rerun the same tmux command for a
queued model once a GPU frees up (a finished or early-stopped job releases
its memory on process exit).

## Watching curves in TensorBoard

All five models log to `/app/data/lightning_logs/<model>` (CoLES/COTIC
via `TensorBoardLogger`, THP/NEP/MLM via a plain `SummaryWriter` --
`train/loss`+`valid/loss`, or `train/nll`+`valid/nll` for THP). Launch it
inside the container, backgrounded:

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

Output: one parquet file per target date under `<embeds_dir>/<model>/`
(default `/app/data/embeds/<model>/`), columns `[inn, date, emb_0..emb_D]`
-- resumable, a date whose file already exists is skipped on the next
run. Chronos-2 additionally chunks clients within a date (heavier
per-client than the other five, via `chronos2_chunk_size`/
`chronos2_batch_size` in the same `inference:` section); see that
script's own docstring for the chunk-merge-cleanup mechanics.

### Cross-institution transfer (MBD)

One consolidated script, `src/training/infer_mbd.py`, runs the SAME
frozen checkpoints zero-shot over MBD (Sber's public benchmark --
`data/mbd_adapter.py` reshapes it into xbank's own column convention
first; run its adapter once before this):

```bash
python -m data.mbd_adapter          # once, materializes the adapted parquet files
python src/training/infer_mbd.py --model coles
python src/training/infer_mbd.py --model nep
python src/training/infer_mbd.py --model mlm
python src/training/infer_mbd.py --model thp
python src/training/infer_mbd.py --model chronos2
```

Reads the same two-config split as the xbank scripts above, just against
`configs/models/downstream_mbd.yaml`'s `inference:` section instead.
Output: `<embeds_dir>/<model>/<date>.parquet` (default
`/app/data/embeds_mbd/<model>/`), one file per month that actually
appears in MBD's own targets file -- not xbank's calendar grid (MBD's
target months and xbank's aren't on the same calendar axis; see
RESEARCH_PLAN.md's "ID / date splitting (transfer, MBD)"). COTIC isn't
wired into this consolidated dispatch yet (see infer_mbd.py's docstring).

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
python src/training/train_downstream_mbd.py --model coles
```

Output: `lgbm_<target>.txt`/`mlp_<target>.pt` per target column, plus
`results.csv` (xbank, under `/app/data/downstream/<model>/` by default) or
`results_all_folds.csv`/`results_aggregated.csv` (MBD, under
`/app/data/downstream_mbd/<model>/` by default).
