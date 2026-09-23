"""Downstream classification probe for MBD (cross-institution transfer
target): frozen per-(client, date) embeddings from one xbank-pretrained
model -- run zero-shot over MBD via infer_mbd.py -- evaluated against
MBD's own adapted targets (data/mbd_adapter.py's build_mbd_targets
output).

Reuses the exact same probe architectures/metrics as train_downstream.py
(LightGBM + independent MLP, roc_auc/pr_auc/precision@k/recall@k, optional
negative undersampling) -- imported directly from that module rather than
duplicated, so both stay in sync automatically.

The one real difference is the train/test split: MBD's own paper (Sec 4.1)
specifies an "out-of-fold validation protocol" -- 5 CLIENT-DISJOINT folds
(client_split/fold=0..4), 4 for training and 1 held out, but does NOT
designate any one fold as canonical. So rather than picking one fold
arbitrarily, this script ROTATES through every fold present in the data
as the held-out test set in turn (train on the rest, test on that one),
matching the paper's actual protocol, and reports both the full per-fold
breakdown and an aggregate (mean +/- std across rotations) per
(target, arch) -- not a single arbitrarily-chosen fold's number (see
RESEARCH_PLAN.md's "ID / date splitting (transfer, MBD)"). Since fold
assignment is per-CLIENT (not per-row), filtering by fold already
guarantees no client straddles train/test in any rotation -- no
additional client-id split is needed at that boundary, only for carving
val out of each rotation's training folds.

Requires infer_mbd.py's source- and evaluation-namespaced embeddings and
data/mbd_adapter.py's adapted targets to already exist.

Usage (inside the container):
    python -m data.mbd_adapter          # once
    python src/training/infer_mbd.py --model coles
    python src/training/train_downstream_mbd.py --model coles
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit

from data.schema import TARGET_COLS, TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL
from training.paths import downstream_dir, embedding_dir, evaluation_name, load_data_config
from training.train_downstream import (
    MLP,
    emb_cols,
    evaluate,
    load_embeddings,
    train_lightgbm,
    train_mlp,
    undersample_indices,
)


def load_config(config_path: str) -> Dict:
    """Reads the `probe:` section of configs/models/downstream_mbd.yaml --
    that file's `inference:` section is a separate pipeline stage, read
    by infer_mbd.py instead (see its own docstring).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["probe"]
    return {
        "max_neg_ratio": cfg.get("max_neg_ratio"),
        "embeds_dir": cfg.get("embeds_dir", "/app/data/embeds"),
        "checkpoint_dir": cfg.get("checkpoint_dir"),
        "val_frac": cfg.get("val_frac", 0.15),
        "seed": cfg.get("seed", 0),
        "max_epochs": cfg.get("max_epochs", 100),
        "patience": cfg.get("patience", 10),
        "batch_size": cfg.get("batch_size", 4096),
        "hidden": cfg.get("hidden", 256),
        "lr": cfg.get("lr", 1e-3),
    }


def load_targets(targets_path: str) -> pd.DataFrame:
    targets = pd.read_parquet(targets_path)
    targets[TARGETS_DATE_COL] = targets[TARGETS_DATE_COL].astype(str)
    return targets


def run_rotation(
    test_fold,
    joined: pd.DataFrame,
    cfg: Dict,
    mlp_args: SimpleNamespace,
    device: torch.device,
    fold_ckpt_dir: Path,
    eval_only: bool,
) -> List[dict]:
    """One out-of-fold rotation: train on every fold except `test_fold`,
    test on it. Returns result rows tagged with `fold=test_fold`.
    """
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_val = joined[joined["fold"] != test_fold]
    test = joined[joined["fold"] == test_fold]
    print(
        f"  train+val folds={sorted(train_val['fold'].unique())} ({len(train_val)} rows)  "
        f"test fold={test_fold} ({len(test)} rows)",
        flush=True,
    )

    # Only a val split is carved out here -- train/test are already
    # client-disjoint by MBD's own fold construction (see module docstring).
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["val_frac"], random_state=cfg["seed"])
    train_idx, val_idx = next(splitter.split(train_val, groups=train_val[TARGETS_CLIENT_ID_COL]))
    train_df = train_val.iloc[train_idx]
    val_df = train_val.iloc[val_idx]
    print(
        f"  train={len(train_df)} ({train_df[TARGETS_CLIENT_ID_COL].nunique()} clients)  "
        f"val={len(val_df)} ({val_df[TARGETS_CLIENT_ID_COL].nunique()} clients) -- split by unique id",
        flush=True,
    )

    cols = emb_cols(joined)
    X_train = train_df[cols].to_numpy(dtype=np.float32)
    X_val = val_df[cols].to_numpy(dtype=np.float32)
    X_test = test[cols].to_numpy(dtype=np.float32)

    results = []
    for target in TARGET_COLS:
        y_train = train_df[target].to_numpy()
        y_val = val_df[target].to_numpy()
        y_test = test[target].to_numpy()

        print(f"\n=== fold {test_fold} / {target} (train pos rate={y_train.mean():.4f}) ===", flush=True)

        if eval_only:
            X_train_t, y_train_t = None, None  # unused -- eval-only skips both training calls below
        elif cfg["max_neg_ratio"] is not None:
            keep = undersample_indices(y_train, cfg["max_neg_ratio"], cfg["seed"])
            X_train_t, y_train_t = X_train[keep], y_train[keep]
            print(
                f"  undersampled train: {len(y_train)} -> {len(y_train_t)} rows "
                f"(neg:pos capped at {cfg['max_neg_ratio']}, new pos rate={y_train_t.mean():.4f})",
                flush=True,
            )
        else:
            X_train_t, y_train_t = X_train, y_train

        lgbm_path = fold_ckpt_dir / f"lgbm_{target}.txt"
        if eval_only:
            print(f"  loading saved LightGBM from {lgbm_path} ...", flush=True)
            if not lgbm_path.exists():
                raise FileNotFoundError(f"--eval-only given but {lgbm_path} doesn't exist -- run without --eval-only first")
            booster = lgb.Booster(model_file=str(lgbm_path))
        else:
            print("  training LightGBM ...", flush=True)
            booster = train_lightgbm(X_train_t, y_train_t, X_val, y_val, cfg["seed"])
            booster.save_model(str(lgbm_path))
        metrics = evaluate(y_test, booster.predict(X_test))
        results.append({"fold": test_fold, "target": target, "arch": "lightgbm", **metrics})
        print(f"    test roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}", flush=True)

        mlp_path = fold_ckpt_dir / f"mlp_{target}.pt"
        if eval_only:
            print(f"  loading saved MLP from {mlp_path} ...", flush=True)
            if not mlp_path.exists():
                raise FileNotFoundError(f"--eval-only given but {mlp_path} doesn't exist -- run without --eval-only first")
            mlp = MLP(X_train.shape[1], hidden=mlp_args.hidden).to(device)
            mlp.load_state_dict(torch.load(str(mlp_path), map_location=device))
        else:
            print("  training MLP ...", flush=True)
            mlp = train_mlp(X_train_t, y_train_t, X_val, y_val, device, mlp_args)
            torch.save(mlp.state_dict(), mlp_path)
        mlp.eval()
        with torch.no_grad():
            test_score = mlp(torch.from_numpy(X_test).to(device)).cpu().numpy()
        metrics = evaluate(y_test, test_score)
        results.append({"fold": test_fold, "target": target, "arch": "mlp", **metrics})
        print(f"    test roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["coles", "cotic", "nep", "mlm", "thp", "chronos2"])
    parser.add_argument("--config", type=str, default="/app/configs/models/downstream_mbd.yaml")
    parser.add_argument("--data-config", type=str, default="/app/configs/data/mbd.yaml")
    parser.add_argument("--checkpoint-source", type=str, default="mbd")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "skip training -- load each fold/target's already-saved lgbm_<target>.txt/"
            "mlp_<target>.pt from checkpoint_dir/fold<k>/ and just re-score its test "
            "fold. Errors if a checkpoint is missing."
        ),
    )
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    data_cfg = load_data_config(cli.data_config)
    eval_name = evaluation_name(data_cfg)
    mlp_args = SimpleNamespace(
        hidden=cfg["hidden"],
        lr=cfg["lr"],
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        batch_size=cfg["batch_size"],
    )

    ckpt_dir = downstream_dir(
        cfg["checkpoint_dir"] or "/app/data/downstream",
        eval_name,
        cli.checkpoint_source,
        cli.model,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"max_neg_ratio: {cfg['max_neg_ratio']}  (from {cli.config})", flush=True)

    print(f"Loading {cli.model} MBD embeddings ...", flush=True)
    embeds_path = embedding_dir(cfg["embeds_dir"], eval_name, cli.checkpoint_source, cli.model)
    embeds = load_embeddings(embeds_path.parent, cli.model)
    print(f"  {len(embeds)} rows across {embeds['date'].nunique()} dates", flush=True)

    print("Loading MBD targets ...", flush=True)
    targets = load_targets(data_cfg["paths"]["targets"])

    joined = targets.merge(
        embeds,
        left_on=[TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL],
        right_on=["inn", "date"],
        how="inner",
    )
    print(f"  joined on (id, date): {len(joined)} rows", flush=True)
    fold_counts = joined["fold"].value_counts().sort_index()
    print("  rows per fold: " + ", ".join(f"{k}={v}" for k, v in fold_counts.items()), flush=True)

    folds = sorted(joined["fold"].unique())
    print(f"\nRotating out-of-fold validation across {len(folds)} folds: {folds}", flush=True)

    all_results: List[dict] = []
    for test_fold in folds:
        print(f"\n{'=' * 60}\nFOLD {test_fold} (held out as test)\n{'=' * 60}", flush=True)
        all_results.extend(
            run_rotation(test_fold, joined, cfg, mlp_args, device, ckpt_dir / f"fold{test_fold}", cli.eval_only)
        )

    per_fold_df = pd.DataFrame(all_results)
    per_fold_df.to_csv(ckpt_dir / "results_all_folds.csv", index=False)

    metric_cols = [c for c in per_fold_df.columns if c not in ("fold", "target", "arch")]
    agg = per_fold_df.groupby(["target", "arch"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(ckpt_dir / "results_aggregated.csv", index=False)

    print(f"\n=== {cli.model} MBD downstream results, aggregated across {len(folds)} folds ===")
    print(agg.to_string(index=False))
    print(f"\nSaved per-fold models to {ckpt_dir}/fold<k>/, full breakdown to results_all_folds.csv, "
          f"aggregate to results_aggregated.csv", flush=True)


if __name__ == "__main__":
    main()
