# Full-scale training

Five full-scale training scripts, `src/xbank/training/train_{coles,cotic,thp,nep,mlm}.py`
-- one per trainable model, living inside the `xbank` package itself (not a
separate top-level `scripts/`) so they import `xbank.*` directly, no
sys.path hack -- the container sets `PYTHONPATH=/app/src` (see `drun.sh`).
Chronos-2 has nothing to train (zero-shot only, see
`src/xbank/training/smoke_chronos2.py`). Each script:

- Loads **all** clients from `trans_any_pos_anonym_encoded.parquet`, not a
  sample (`smoke_*.py` sampled 500 to prove the pipeline works;
  these load the real ~378K-client corpus). Pass `--n-clients N` to cap it
  for a quick debug run.
- Trains with early stopping on a validation metric (`--patience`, default
  5 epochs of no improvement) under a generous `--max-epochs` cap (default
  100). There's no prior estimate of how many epochs full-scale
  convergence needs for any of these models, so early stopping decides
  rather than a guessed fixed count.
- Checkpoints every epoch to `outputs/checkpoints/<model>/` (`last.*` for
  resuming, `best.*` for the best validation epoch so far) -- re-running
  the exact same command picks up where a killed job left off
  automatically, no extra flags needed.

## Picking a GPU

Both host GPUs (RTX A5000, 24GB each) are shared with other users'
containers. Check current load before launching:

```bash
./environments/pick_gpu.sh
```

Pin a job to one GPU via `CUDA_VISIBLE_DEVICES` on the `docker exec` call
-- the container itself sees both GPUs (`--gpus all` in `drun.sh`), so this
env var is what actually restricts a given process to one of them.

## Launching in tmux

One tmux session per model, so a closed SSH connection doesn't kill
training and you can reattach any time to watch progress. `tee` writes the
log to a file while still showing it live if you're attached at launch
time:

```bash
mkdir -p outputs/logs

tmux new-session -d -s coles 'docker exec -e CUDA_VISIBLE_DEVICES=0 xbank-transfer python src/xbank/training/train_coles.py 2>&1 | tee outputs/logs/coles.log'
tmux new-session -d -s cotic 'docker exec -e CUDA_VISIBLE_DEVICES=1 xbank-transfer python src/xbank/training/train_cotic.py 2>&1 | tee outputs/logs/cotic.log'
tmux new-session -d -s thp   'docker exec -e CUDA_VISIBLE_DEVICES=0 xbank-transfer python src/xbank/training/train_thp.py   2>&1 | tee outputs/logs/thp.log'
tmux new-session -d -s nep   'docker exec -e CUDA_VISIBLE_DEVICES=1 xbank-transfer python src/xbank/training/train_nep.py   2>&1 | tee outputs/logs/nep.log'
tmux new-session -d -s mlm   'docker exec -e CUDA_VISIBLE_DEVICES=0 xbank-transfer python src/xbank/training/train_mlm.py   2>&1 | tee outputs/logs/mlm.log'
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
tail -f outputs/logs/coles.log      # simplest, no attach needed
tmux attach -t coles                # or attach directly; Ctrl-b d to detach without killing it
```

**Managing sessions:**

```bash
tmux ls                             # list running sessions
tmux kill-session -t coles          # stop one (job keeps its last checkpoint, resumable later)
```

## If a full-population load runs out of host RAM

`load_all_raw` pulls every row into pandas in one shot (~91.5M rows across
all clients). The host had 106GB free as of 2026-08-26, which should be
enough for any single model's load (COTIC/THP project down to 3 columns;
CoLES/NEP/MLM need all 16) -- but if a job OOMs: rerun it with `--n-clients`
set to some large-but-bounded number as a stopgap, and flag it. A real
chunked/streaming loader would be the actual fix, not built yet since it
wasn't needed at smoke-test scale.
