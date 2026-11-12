# xbank-transfer

Benchmark and code for cross-bank transfer of frozen transaction-sequence
foundation models: pretrained on MBD, evaluated frozen on MBD-raw,
MBD-daily, and xbank. See `docs/research_plan.md` for the full design.

Five self-supervised architectures (CoLES, NEP, MLM, THP, COTIC) plus
Chronos-2 as an external zero-shot reference are pretrained once on MBD
and evaluated without fine-tuning across the three corpora, separating
aggregation shift (raw vs. daily-aggregated MBD) from institution shift
(MBD vs. xbank). Crossing into xbank uses a dictionary-free schema-alignment
method (semi-relaxed Fused Gromov-Wasserstein; see `src/cross_schema/FGW.md`)
instead of a hand-built field mapping.

## Layout

- `src/data/` -- MBD/xbank adapters, schema definitions, client-disjoint splits
- `src/models/` -- model implementations (CoLES, COTIC, MLM, NEP, THP, Chronos-2 wrapper)
- `src/training/` -- pretraining, inference, and LightGBM/MLP downstream probes, one script per stage
- `src/cross_schema/` -- column-profile schema matching and the FGW mapping solver (`FGW.md`, `REBUILD.md`)
- `configs/` -- per-model and per-corpus YAML configs, plus frozen xbank<->MBD column mappings under `configs/mappings/`
- `environments/` -- Docker build/run scripts and environment notes (`environments/README.md`)
- `tests/` -- unit tests, plus the host-run worker/queue shell scripts used for full-scale jobs
- `docs/` -- `research_plan.md` (design), `training.md` (pretraining), `downstream.md` (frozen-embedding evaluation)
- `third_party/COTIC` -- cloned upstream source, not pip-installable (see `environments/README.md`)

## Quick start

```bash
./environments/dbuild.sh     # build the image
./environments/drun.sh       # start the container as a persistent background service
./environments/dexec.sh      # attach an interactive shell
```

Entry points live in `src/training/` (not a top-level `scripts/`) and
import as `data.*` / `models.*` / `training.*`, since the container sets
`PYTHONPATH=/app/src`. See `docs/training.md` and `docs/downstream.md`
for the full pretraining and evaluation workflows, and
`environments/README.md` for data/HF-cache mounts, GPU allocation, and
what's baked into the image vs. cloned under `third_party/`.

## Tests

```bash
./environments/dexec.sh
pytest tests/
```

Long-running full-scale jobs (training, inference, LightGBM/MLP sweeps)
are driven by the `run_*.sh` / `*_worker.sh` scripts under `tests/`,
meant to be launched on the server host under `tmux`, not as part of the
pytest suite.
