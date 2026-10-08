# Frozen-embedding downstream extension

Launch on the server host only, after CPU and GPU smoke tests:

```bash
bash environments/run_extended_downstream.sh
```

Two tmux workers (`extended-downstream-gpu0/1`) sequentially dispatch disposable
containers, one physical GPU each, 6 CPUs and a hard 24-GiB RAM limit, no swap.
They wait for GPU availability and host RAM; they never stop other tasks or
restart the existing service container. A failed job stops its worker; rerunning
resumes targets whose manifest/model/metrics are complete. Input or code changes
are rejected instead of silently mixing experiments. Do not change code or inputs
while this queue is running.
The larger 768-feature Chronos MBD panels are assigned to different GPUs;
Xbank Chronos MLP goes to GPU1 while GPU0 owns Chronos FGW LightGBM.

GPU0 first fits the missing Chronos FGW LightGBM probe, with exactly the seed,
search budget and calendar split of the existing original-mapping baseline.
Current mappings both feed Chronos `event_time <- col_1, amount <- col_12`.
Categorical changes do not affect Chronos. After source-manifest validation,
FGW uses hardlinks to the byte-identical original embeddings, documented by
`embedding_reuse.json`. This is not an independent Chronos representation or
evidence of an FGW benefit. Original results are not modified.

MLP covers all existing representations: 17 MBD dataset/source/encoder groups
(5 folds x 4 targets = 340 evaluations), and 11 paired Xbank encoder/source groups
(2 mappings x 3 targets = 66 evaluations). Chronos is external frozen/zero-shot;
learned encoders use MBD-raw or MBD-daily checkpoints. The daily NEP is the retained
existing checkpoint, not a new pretraining run. No encoder is trained here.

Protocol:

- Identical outer splits to LightGBM: MBD five client-disjoint folds, validation
  fold preceding the test fold; Xbank client-disjoint 2023 train/validation and
  untouched January/February 2024 test. Each mapping pair uses the intersection
  of available labeled client-months. Xbank `col_2` target is excluded.
  The two mappings receive identical logical minibatches/seed, independently
  of physical parquet order; sorted reads are restored to logical batch order.
- Three fixed MLP candidates (see `configs/models/downstream_mlp.yaml`), two ReLU
  hidden layers, dropout/weight decay, AdamW. HPO caps training at 50,000 clients
  and evaluates full validation; best parameters are retrained on full training
  to select duration, then reinitialized/refit on train+validation.
- Maximum 16 epochs, validation average-precision early stopping (patience 4).
  Standardization fits full train for selection, train+validation for refit;
  test statistics/labels never select parameters, epochs or scaling. No negative
  undersampling. Class weighting is a validation-selected candidate.
- Disk-backed feature cache and block-local shuffling visit every training row
  once per epoch, keeping RAM bounded. Cache is deleted only after a group's
  complete summary. BF16 training on compatible GPUs; float32 evaluation avoids
  score quantization in ranking metrics. Seed 42; no best-of-test run selection.
- Report PR-AUC (average precision) and ROC-AUC on natural-prevalence test rows,
  per target and macro across targets; MBD includes per-target fold mean/std.
  Xbank has one calendar test, not five-fold uncertainty. Individual saved
  models carry train+validation standardization for reproducible predictions.

Results: `/mnt/storage/d.tanyushkina/transactions/downstream/<dataset>/<source>/<encoder>/mlp_hpo/`
(`source` is `mbd_source`, `mbd_daily_source` or `zero_shot`). Summaries:
`results_all_folds.csv`, `results_aggregated.csv`, `model_macro.csv`;
paired comparison: original mapping's `mapping_comparison.csv`.
LightGBM FGW Chronos uses the existing LightGBM namespace in `xbank_fgw_v2/zero_shot/chronos2`.
Logs: `/mnt/storage/d.tanyushkina/transactions/logs/extended_downstream_*.log`.

Read-only status:

```bash
tmux ls
nvidia-smi
tail -n 30 /mnt/storage/d.tanyushkina/transactions/logs/extended_downstream_gpu{0,1}_queue.log
tail -n 30 /mnt/storage/d.tanyushkina/transactions/logs/extended_downstream_lightgbm_chronos_fgw.log
```

Tests use temporary synthetic inputs and leave no production data artifacts:

```bash
PYTHONPATH=src python -m pytest -q tests/test_mlp_downstream.py tests/test_chronos_fgw.py
# Without pytest (production image), use the standalone test runners:
PYTHONPATH=src MLP_SMOKE_DEVICE=gpu python tests/test_mlp_downstream.py
PYTHONPATH=src python tests/test_chronos_fgw.py
# GPU tests must run inside a fresh --gpus device=N container.
```
