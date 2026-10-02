"""Safe daily-arm pretraining. NEP is deliberately excluded (legacy retained)."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import gc

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.io import TorchCheckpointIO
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from data.schema import NUMERIC_COLS
from training.common import (EarlyStopper, atomic_torch_save, finite_gradients, load_checkpoint,
                             restore_rng, rng_state, save_checkpoint, seed_training, validation_rng)
from training.paths import data_root, load_data_config
from training.pretrain_cache import (DiskRecords, ShardSampler, atomic_json, checked_json, digest,
                                    source_identity, validate_cache)


MODELS = ("coles", "cotic", "thp", "mlm")


class NonSingletonBatches:
    """Keep every client; append a final singleton to the preceding batch."""
    def __init__(self, sampler, batch_size):
        if len(sampler) < 2 or batch_size < 2:
            raise ValueError("training/validation needs at least two clients per cohort")
        self.sampler, self.batch_size = sampler, batch_size

    def __iter__(self):
        previous, batch = None, []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                if previous is not None:
                    yield previous
                previous, batch = batch, []
        if previous is None:
            yield batch
        elif len(batch) == 1:
            yield previous + batch
        else:
            yield previous
            if batch:
                yield batch

    def __len__(self):
        n = len(self.sampler)
        return math.ceil(n / self.batch_size) - int(n > self.batch_size and n % self.batch_size == 1)


class FixedValidationCoLES(ColesDataset):
    def __getitem__(self, index):
        previous = np.random.get_state()
        try:
            np.random.seed(1_000_000 + index)
            return super().__getitem__(index)
        finally:
            np.random.set_state(previous)


def cotic_collate(batch, normalizer):
    times, types = zip(*batch)
    # Only a batch is padded; never the entire 1.5M-client population.
    padded_times = torch.nn.utils.rnn.pad_sequence(times, batch_first=True)
    padded_types = torch.nn.utils.rnn.pad_sequence([value + 1 for value in types], batch_first=True)
    return normalizer.normalize(padded_times), padded_types


def interval_blocks(cache: Path):
    for directory in sorted((cache / "encoded").iterdir()):
        import pandas as pd
        meta = pd.read_parquet(directory / "clients.parquet")
        times = np.load(directory / "time.npy", mmap_mode="r")
        offsets = np.load(directory / "offsets.npy", mmap_mode="r")
        if not len(times):
            continue
        selected = np.repeat((~meta.valid & (meta.n >= 2)).to_numpy(), meta.n.to_numpy(dtype=int))
        selected[offsets[:-1]] = False
        deltas = np.diff(times, prepend=times[0]).astype(np.float64) / 86400.0
        values = deltas[selected]
        if (values < 0).any() or not np.isfinite(values).all():
            raise ValueError("invalid train time intervals")
        yield values


def fit_normalizer(cache: Path):
    from models.cotic import ExponentialNormalizerP99
    limit, samples, size = 16_000_000, [], 0
    for values in interval_blocks(cache):
        part = values[:limit - size]
        samples.append(part.astype(np.float32))
        size += len(part)
        if size >= limit:
            break
    if not size:
        raise ValueError("no train intervals for COTIC")
    # Preserve the upstream p99/truncated-exponential recipe, but avoid
    # holding all padded intervals in RAM. Only train contributes.
    q99 = float(torch.quantile(torch.from_numpy(np.concatenate(samples)), .99))
    if not np.isfinite(q99) or q99 <= 0:
        raise ValueError("COTIC train p99 interval is zero; cannot fit time normalizer without inventing timestamps")
    del samples
    total, count = 0., 0
    for values in interval_blocks(cache):
        retained = values[values <= q99]
        total += float(retained.sum())
        count += len(retained)
    mean = total / count
    if mean <= 0:
        raise ValueError("COTIC train truncated mean interval is zero")
    value = float(ExponentialNormalizerP99.solve_for_lambda(mean, q99, 1 / q99))
    if not np.isfinite(value) or value <= 0:
        raise FloatingPointError("invalid fitted COTIC normalizer")
    predicted = 1 / value - q99 / np.expm1(value * q99)
    if not np.isclose(predicted, mean, rtol=1e-4, atol=1e-8):
        raise FloatingPointError("COTIC normalizer solver failed to converge")
    return ExponentialNormalizerP99(value)


class StreamingModule(pl.LightningDataModule):
    def __init__(self, train_records, valid_records, model, batch_size, seed, normalizer=None):
        super().__init__()
        self.sampler = ShardSampler(train_records, seed)
        self.normalizer = normalizer
        self.train_data, self.valid_data = train_records, valid_records
        self.batch_size, self.model = batch_size, model
        if model == "coles":
            splitter = SampleSlices(split_count=5, cnt_min=15, cnt_max=150)
            self.train_data = ColesDataset(train_records, splitter)
            self.valid_data = FixedValidationCoLES(valid_records, splitter)
            self.collate = ColesDataset.collate_fn
        else:
            from functools import partial
            self.collate = partial(cotic_collate, normalizer=normalizer)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_sampler=NonSingletonBatches(self.sampler, self.batch_size),
                          collate_fn=self.collate, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.valid_data,
                          batch_sampler=NonSingletonBatches(SequentialSampler(self.valid_data), self.batch_size),
                          collate_fn=self.collate, num_workers=0)


class AtomicCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if storage_options is not None:
            raise ValueError("only local atomic checkpoint storage is supported")
        atomic_torch_save(checkpoint, path)


class SafetyCallback(Callback):
    def __init__(self, seed):
        self.seed, self.loaded_rng, self.previous_rng = seed, None, None

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.loaded_rng = checkpoint.get("pretrain_rng")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["pretrain_rng"] = rng_state()

    def on_train_start(self, trainer, pl_module):
        if self.loaded_rng is not None:
            restore_rng(self.loaded_rng)
            self.loaded_rng = None

    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.sampler.epoch = trainer.current_epoch

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        finite_gradients(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            print(f"epoch={trainer.current_epoch} batch={batch_idx}/{trainer.num_training_batches} "
                  f"global_step={trainer.global_step} loss={float(loss.detach())}", flush=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.previous_rng = rng_state()
        seed_training(self.seed + 1_000_000)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.previous_rng is not None:
            restore_rng(self.previous_rng)
            self.previous_rng = None

    def on_validation_end(self, trainer, pl_module):
        for name, metric in trainer.callback_metrics.items():
            if torch.is_tensor(metric) and not torch.isfinite(metric).all():
                raise FloatingPointError(f"non-finite Lightning metric: {name}")
        if not trainer.sanity_checking:
            scores = {name: float(value) for name, value in trainer.callback_metrics.items()
                      if (name.startswith("valid/") or name.startswith("val/")) and torch.is_tensor(value)}
            print(f"validation epoch={trainer.current_epoch}: {scores}", flush=True)


def ensure_finite_weights(model):
    if any(not torch.isfinite(value).all() for value in model.state_dict().values() if torch.is_tensor(value)):
        raise FloatingPointError("non-finite model parameters/buffers")


def masked_events(mask, probability):
    chosen = (torch.rand_like(mask, dtype=torch.float32) < probability) & mask.bool()
    empty = ~chosen.any(dim=1)
    if empty.any():
        lengths = mask.sum(dim=1).long()
        positions = (torch.rand(len(lengths), device=mask.device) * lengths).long()
        chosen[torch.arange(len(lengths), device=mask.device)[empty], positions[empty]] = True
    return chosen


def build_plain_loaders(train_records, valid_records, model, cfg, seed, num_types):
    sampler = ShardSampler(train_records, seed)
    args = {"batch_sampler": NonSingletonBatches(sampler, cfg["batch_size"]), "num_workers": 0}
    valid_args = {"batch_sampler": NonSingletonBatches(SequentialSampler(valid_records), cfg["batch_size"]),
                  "num_workers": 0}
    if model == "mlm":
        return (DataLoader(train_records, collate_fn=collate_feature_dict, **args),
                DataLoader(valid_records, collate_fn=collate_feature_dict, **valid_args), sampler)
    from easy_tpp.preprocess.dataset import get_data_loader
    from models.thp import build_tokenizer
    tokenizer = build_tokenizer(num_types, cfg["max_seq_len"])
    return (get_data_loader(train_records, "torch", tokenizer, **args),
            get_data_loader(valid_records, "torch", tokenizer, **valid_args), sampler)


def plain_epoch(model, loader, name, device, optimizer=None):
    model.train(optimizer is not None)
    total, weight = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch_index, batch in enumerate(loader):
            if name == "mlm":
                batch = batch.to(device)
                event_mask = masked_events(batch.seq_len_mask, model.mask_prob)
                loss = model.loss(batch.payload, batch.seq_len_mask, event_mask=event_mask)
                n_events = int(event_mask.sum())
            else:
                batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
                loss, _ = model.loglike_loss(batch)
                n_events = int(batch["seq_non_pad_mask"][:, 1:].sum())
            if n_events <= 0 or not torch.isfinite(loss):
                raise FloatingPointError("empty supervised batch or non-finite training/validation loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                finite_gradients(model)
                optimizer.step()
                if batch_index % 100 == 0:
                    print(f"{name} batch={batch_index}/{len(loader)} loss={float(loss.detach())}", flush=True)
            total += float(loss.detach()) * (n_events if name == "mlm" else 1)
            weight += n_events
    if not weight:
        raise ValueError("empty supervised epoch")
    return total / weight


def probe_production_batch(model_name, cache, cfg, device="cuda"):
    """One worst-length synthetic batch at the configured real architecture.

    Uses the cache's vocabulary, but does NOT train/save production weights.
    Run in a separate process before starting the actual training process.
    """
    import pandas as pd
    from training.common import load_preprocessor
    seed_training(cfg["seed"])
    target_device = torch.device(device)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU preflight requires CUDA")
        torch.cuda.reset_peak_memory_stats()
    categories = np.load(cache / "categories.npy")
    records = DiskRecords(cache, "train", model_name)
    record = records[int(records.meta.n.idxmax())]
    length = cfg["max_seq_len"]
    if model_name in {"coles", "mlm"}:
        record = {key: (value.repeat(math.ceil(length / len(value)))[:length]
                        if torch.is_tensor(value) else value) for key, value in record.items()}
        record["event_time"] = torch.arange(length) // 2 * 86400
    elif model_name == "cotic":
        record = (torch.arange(length).float() // 2,
                  torch.arange(length).long() % len(categories))
    else:
        times = (np.arange(length) // 2).astype(float)
        record = {"time_seqs": times.tolist(), "time_delta_seqs": np.r_[0., np.diff(times)].tolist(),
                  "type_seqs": (np.arange(length) % len(categories)).tolist()}

    class ProbeRecords:
        def __init__(self):
            self.meta = pd.DataFrame({"bucket": [0] * cfg["batch_size"]})

        def __len__(self):
            return len(self.meta)

        def __getitem__(self, index):
            return record

    data = ProbeRecords()
    if model_name in {"coles", "cotic"}:
        if model_name == "coles":
            from models.coles import build_module
            preprocessor = load_preprocessor(cache / "preprocessor.pkl")
            module = build_module(preprocessor.get_category_dictionary_sizes(), embedding_dim=cfg["embedding_dim"],
                                  hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"])
            normalizer = None
        else:
            from models.cotic import build_module, ExponentialNormalizerP99
            module = build_module(len(categories), in_channels=cfg["in_channels"],
                                  nb_filters=cfg["nb_filters"], nb_layers=cfg["nb_layers"])
            normalizer = ExponentialNormalizerP99(1.)
        datamodule = StreamingModule(data, data, model_name, cfg["batch_size"], cfg["seed"], normalizer)
        trainer = pl.Trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=0,
                             num_sanity_val_steps=0, accelerator="gpu" if device == "cuda" else "cpu", devices=1,
                             deterministic=True, callbacks=[SafetyCallback(cfg["seed"])], logger=False,
                             enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(module, datamodule=datamodule)
    else:
        if model_name == "mlm":
            from models.mlm import MLM
            preprocessor = load_preprocessor(cache / "preprocessor.pkl")
            module = MLM(preprocessor.get_category_dictionary_sizes(), NUMERIC_COLS, d_model=cfg["d_model"],
                         num_layers=cfg["num_layers"], max_position_embeddings=length).to(target_device)
        else:
            from models.thp import build_model
            module = build_model(len(categories), hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
                                 gpu=0 if device == "cuda" else -1)
        loader, _, _ = build_plain_loaders(data, data, model_name, cfg, cfg["seed"], len(categories))
        optimizer = torch.optim.Adam(module.parameters(), lr=cfg["lr"])
        plain_epoch(module, loader, model_name, target_device, optimizer)
    ensure_finite_weights(module)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None
    print(f"PREFLIGHT PASSED {model_name}: batch={cfg['batch_size']} max_length={length} peak_allocated_GiB={peak}", flush=True)
    del module, data, records
    gc.collect()


def train(model_name: str, data_config: Path, model_config: Path, cache: Path,
          output: Path | None = None, log_root: Path | None = None, *, device: str = "cuda"):
    if model_name not in MODELS:
        raise ValueError("NEP is retained and must not be retrained by this runner")
    data = load_data_config(data_config)
    if data["name"] != "mbd_daily":
        raise ValueError("new runner is exclusively for MBD-daily")
    cfg = yaml.safe_load(model_config.read_text())
    ready = validate_cache(cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    if source_identity(Path(data["paths"]["transactions"])) != manifest["source"]:
        raise ValueError("prepared data identity differs from current input")
    for key in ("seed", "valid_frac", "max_seq_len", "n_clients"):
        if cfg[key] != manifest[key]:
            raise ValueError(f"model/cache setting mismatch: {key}")
    if cfg["max_epochs"] < 1 or cfg["patience"] < 1:
        raise ValueError("positive epoch ceiling and patience required")
    output = output or data_root() / "checkpoints" / "mbd_daily_source" / model_name
    logs = (log_root or data_root() / "lightning_logs" / "mbd_daily_source") / model_name
    expected = {"format_version": 2, "model": model_name, "config": cfg,
                "cache": {key: value for key, value in ready.items() if key != "audit"},
                "code": {str(path.relative_to(Path(__file__).parents[1])): digest(path) for path in [
                    Path(__file__), Path(__file__).with_name("pretrain_cache.py"),
                    Path(__file__).with_name("common.py"),
                    Path(__file__).parents[1] / "models" / f"{model_name}.py"]},
                "torch": torch.__version__, "lightning": pl.__version__,
                "seed_policy": "all_rng_strict_deterministic_fixed_validation",
                "pretraining_scope": manifest["pretraining_scope"],
                "time_policy": manifest["time_policy"], "sampler": "epoch_shard_then_client_shuffle",
                "minimum_train_events": 2, "gradient_clipping": "none_finite_check_only"}
    if output.exists() and any(output.iterdir()) and not (output / "training_manifest.json").exists():
        raise ValueError(f"refusing to overwrite a legacy/untracked training run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked_json(output / "training_manifest.json", expected)
        if (output / "complete.json").exists():
            complete = json.loads((output / "complete.json").read_text())
            best = output / complete["best_checkpoint"]
            if digest(best) != complete["best_sha256"]:
                raise ValueError("completed checkpoint was changed")
            print(f"Already complete: {model_name}", flush=True)
            return complete
        seed_training(cfg["seed"])
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU required, refusing silent CPU fallback")
        target_device = torch.device(device)
        # Always copy/reuse the frozen cache artifact; never refit on resume.
        for name in (["preprocessor.pkl"] if model_name in {"coles", "mlm"} else ["categories.npy"]):
            destination = output / name
            if destination.exists() and digest(destination) != digest(cache / name):
                raise ValueError(f"checkpoint preprocessing artifact changed: {destination}")
            if not destination.exists():
                temporary = destination.with_name(name + ".tmp")
                shutil.copyfile(cache / name, temporary)
                os.replace(temporary, destination)
        from training.common import load_preprocessor
        categories = np.load(cache / "categories.npy")
        train_records, valid_records = DiskRecords(cache, "train", model_name), DiskRecords(cache, "valid", model_name)
        atomic_json(output / "cohort.json", {"train_clients": len(train_records), "valid_clients": len(valid_records),
                                             "cache_audit": ready["audit"]})
        print(f"{model_name}: train={len(train_records)} valid={len(valid_records)} device={device}", flush=True)
        if model_name in {"coles", "cotic"}:
            normalizer = None
            if model_name == "coles":
                from models.coles import build_module
                preprocessor = load_preprocessor(output / "preprocessor.pkl")
                module = build_module(preprocessor.get_category_dictionary_sizes(), embedding_dim=cfg["embedding_dim"],
                                      hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"])
                monitor, mode = "valid/recall_top_k", "max"
            else:
                import pickle
                from models.cotic import build_module
                normalizer_file = output / "normalizer.pkl"
                normalizer_meta = output / "normalizer.json"
                if normalizer_file.exists():
                    if not normalizer_meta.exists() or digest(normalizer_file) != json.loads(normalizer_meta.read_text())["sha256"]:
                        raise ValueError("COTIC normalizer changed or incomplete")
                    with normalizer_file.open("rb") as stream:
                        normalizer = pickle.load(stream)
                else:
                    normalizer = fit_normalizer(cache)
                    with normalizer_file.with_suffix(".tmp").open("wb") as stream:
                        pickle.dump(normalizer, stream)
                    os.replace(normalizer_file.with_suffix(".tmp"), normalizer_file)
                    atomic_json(normalizer_meta, {"sha256": digest(normalizer_file), "lambda": normalizer.lambda_value})
                module = build_module(len(categories), in_channels=cfg["in_channels"],
                                      nb_filters=cfg["nb_filters"], nb_layers=cfg["nb_layers"])
                monitor, mode = "val/log_likelihood", "max"
            datamodule = StreamingModule(train_records, valid_records, model_name, cfg["batch_size"], cfg["seed"], normalizer)
            checkpoint = ModelCheckpoint(dirpath=str(output), filename="best", monitor=monitor, mode=mode,
                                         save_last=True, save_top_k=1, enable_version_counter=False)
            trainer = pl.Trainer(max_epochs=cfg["max_epochs"], accelerator="gpu" if device == "cuda" else "cpu",
                                 devices=1, deterministic=True, num_sanity_val_steps=2,
                                 logger=TensorBoardLogger(str(logs.parent), name=model_name),
                                 callbacks=[SafetyCallback(cfg["seed"]),
                                            EarlyStopping(monitor=monitor, mode=mode, patience=cfg["patience"], check_finite=True),
                                            checkpoint], plugins=[AtomicCheckpointIO()], enable_progress_bar=False)
            last = output / "last.ckpt"
            if last.exists():
                saved = torch.load(last, map_location="cpu", weights_only=False)
                finished = saved["epoch"] + 1 >= cfg["max_epochs"] or any(
                    isinstance(value, dict) and value.get("wait_count", 0) >= cfg["patience"]
                    for key, value in saved.get("callbacks", {}).items() if "EarlyStopping" in key)
            else:
                finished = False
            if not finished:
                # Own local checkpoints include optimizer and RNG objects;
                # PyTorch's newer weights-only default cannot resume these.
                trainer.fit(module, datamodule=datamodule, ckpt_path=str(last) if last.exists() else None,
                            weights_only=False)
                ensure_finite_weights(module)
            best_file = output / "best.ckpt"
            if not best_file.exists():
                raise FileNotFoundError("no best Lightning checkpoint was saved")
            saved = torch.load(best_file, map_location="cpu", weights_only=False)
            best_epoch = int(saved["epoch"])
            scores = [value["best_model_score"] for key, value in saved.get("callbacks", {}).items()
                      if "ModelCheckpoint" in key and value.get("best_model_score") is not None]
            if not scores:
                raise ValueError("best checkpoint does not contain its monitored score")
            best_score = float(scores[0])
        else:
            if model_name == "mlm":
                from models.mlm import MLM
                preprocessor = load_preprocessor(output / "preprocessor.pkl")
                module = MLM(preprocessor.get_category_dictionary_sizes(), NUMERIC_COLS, d_model=cfg["d_model"],
                             num_layers=cfg["num_layers"], max_position_embeddings=cfg["max_seq_len"]).to(target_device)
            else:
                from models.thp import build_model
                module = build_model(len(categories), hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
                                     gpu=0 if device == "cuda" else -1)
            optimizer = torch.optim.Adam(module.parameters(), lr=cfg["lr"])
            stopper = EarlyStopper(cfg["patience"], "min")
            train_loader, valid_loader, sampler = build_plain_loaders(train_records, valid_records, model_name,
                                                                    cfg, cfg["seed"], len(categories))
            last, best_file = output / "last.pt", output / "best.pt"
            start = load_checkpoint(str(last), module, optimizer, stopper, target_device) if last.exists() else 0
            writer = SummaryWriter(str(logs))
            try:
                for epoch in range(start, cfg["max_epochs"]):
                    if stopper.should_stop:
                        break
                    sampler.epoch = epoch
                    train_loss = plain_epoch(module, train_loader, model_name, target_device, optimizer)
                    with validation_rng(cfg["seed"] + 1_000_000):
                        valid_loss = plain_epoch(module, valid_loader, model_name, target_device)
                    ensure_finite_weights(module)
                    improved = stopper.step(valid_loss, epoch)
                    print(f"{model_name} epoch {epoch}: train={train_loss:.6f} valid={valid_loss:.6f} "
                          f"best={stopper.best:.6f} bad_epochs={stopper.bad_epochs}", flush=True)
                    writer.add_scalar("train/loss", train_loss, epoch)
                    writer.add_scalar("valid/loss", valid_loss, epoch)
                    if improved:
                        save_checkpoint(str(best_file), module, optimizer, epoch, stopper)
                    save_checkpoint(str(last), module, optimizer, epoch, stopper)
            finally:
                writer.close()
            best_epoch, best_score = stopper.best_epoch, stopper.best
        complete = {"model": model_name, "best_checkpoint": best_file.name, "best_epoch": best_epoch,
                    "best_validation_score": best_score, "best_sha256": digest(best_file),
                    "training_manifest_sha256": digest(output / "training_manifest.json")}
        if not np.isfinite(best_score):
            raise FloatingPointError("non-finite best score")
        atomic_json(output / "complete.json", complete)
        print(f"COMPLETE {model_name}: {complete}", flush=True)
        return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--data-config", default="/app/configs/data/mbd_daily.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--cache-root")
    parser.add_argument("--output-root")
    parser.add_argument("--log-root")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight", action="store_true", help="test one full-sized batch without writing weights")
    args = parser.parse_args()
    cache = Path(args.cache_root) if args.cache_root else data_root() / "training_cache" / "mbd_daily" / "v2"
    config = Path(args.model_config) if args.model_config else Path("/app/configs/models") / f"{args.model}.yaml"
    if args.preflight:
        validate_cache(cache)
        probe_production_batch(args.model, cache, yaml.safe_load(config.read_text()), args.device)
        return
    train(args.model, Path(args.data_config), config, cache,
          Path(args.output_root) if args.output_root else None,
          Path(args.log_root) if args.log_root else None, device=args.device)


if __name__ == "__main__":
    main()
