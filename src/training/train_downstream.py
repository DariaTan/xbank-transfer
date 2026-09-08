"""Downstream classification probe: frozen per-(client, date) embeddings
from one pretrained model (CoLES/NEP/MLM/THP/Chronos-2) against the 4
binary product-propensity targets in targets_anonym_encoded.parquet (see
data/schema.py for column semantics).

Trains two things per target column: an independent LightGBM classifier
and an independent small MLP (multi-label, not mutually exclusive;
imbalance differs per column so each gets its own scale_pos_weight/
pos_weight). configs/models/downstream.yaml's (`probe:` section) max_neg_ratio
optionally undersamples negatives in the per-target training set for both.

(A joint multi-head MLP sharing one trunk across all targets was tried
and removed -- on CoLES it was clearly worse on the common target
(col_2) and only a coin-flip on the rarer ones, not worth the added
complexity of a shared training trajectory.)

Every run parameter except which embedding model to evaluate lives in
configs/models/downstream.yaml's `probe:` section -- train/test periods
(month granularity, inclusive), max_neg_ratio, embeds_dir/checkpoint_dir, and the
val_frac/seed/max_epochs/patience/batch_size/hidden/lr MLP
hyperparameters. --model stays a CLI flag (`--model
{coles,nep,mlm,thp,chronos2}`) since it's what actually varies between
invocations of this same script. Test is held out entirely from training
and model selection -- early stopping / LightGBM's own early stopping
both use a validation split carved out of the train period only. Within
the train period, train/val is split by unique client id (not by row) so
the same client's several monthly rows never straddle the split.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from data.schema import TARGET_COLS, TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL
from training.common import EarlyStopper

XBANK_DATA_CONFIG = "/app/configs/data/xbank.yaml"
with open(XBANK_DATA_CONFIG) as f:
    TARGETS_PATH = yaml.safe_load(f)["paths"]["targets"]


def load_embeddings(embeds_dir: Path, model: str) -> pd.DataFrame:
    paths = sorted((embeds_dir / model).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no embedding files under {embeds_dir / model}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def load_targets() -> pd.DataFrame:
    targets = pd.read_parquet(TARGETS_PATH)
    targets[TARGETS_DATE_COL] = targets[TARGETS_DATE_COL].astype(str)
    return targets


def month_range(start: str, end: str) -> List[str]:
    """Every first-of-month date string ("YYYY-MM-01") from `start` to
    `end` ("YYYY-MM"), inclusive of both ends -- matches the fixed
    calendar-date grid target rows (and the embeddings built for them)
    are always stamped on.
    """
    months = pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M")
    return [m.strftime("%Y-%m-01") for m in months]


def load_config(config_path: str) -> Dict:
    """Everything the run needs except which embedding model to evaluate
    (that stays a CLI flag, --model, since it's what you're actually
    choosing between on any given invocation) and the config path itself
    (structurally can't live inside the file it points to). Reads the
    `probe:` section of configs/models/downstream.yaml -- that file's
    `inference:` section is a separate pipeline stage, read by the
    infer_*.py scripts instead (see their own docstrings).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["probe"]
    return {
        "train": month_range(cfg["train"]["start"], cfg["train"]["end"]),
        "test": month_range(cfg["test"]["start"], cfg["test"]["end"]),
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


def check_required_months(targets: pd.DataFrame, required_dates: List[str]) -> None:
    """The target data is known to skip some months entirely for every
    client (e.g. Nov/Dec 2023, per data/schema.py) -- not missing data
    we're failing to load, but a real gap. Warns (doesn't raise) if any
    month configs/models/downstream.yaml asks for isn't in the target data at
    all, since silently training/testing on fewer months than configured
    is easy to miss otherwise.
    """
    available = set(targets[TARGETS_DATE_COL].unique())
    missing = sorted(set(required_dates) - available)
    if missing:
        print(
            f"WARNING: {len(missing)} configured month(s) have no rows at all in "
            f"{TARGETS_PATH}: {missing}",
            flush=True,
        )


def emb_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def undersample_indices(y: np.ndarray, max_neg_ratio: float, seed: int) -> np.ndarray:
    """Row indices keeping every positive plus at most `max_neg_ratio`
    negatives per positive -- a no-op (returns all indices, unsorted-safe)
    if the column is already at or below that ratio, so this is safe to
    apply uniformly across targets with very different base rates (e.g.
    col_2 ~50% positive vs. col_3/col_5 <1.5%). Only ever applied to a
    TRAIN split -- val/test must stay at the true, natural distribution
    since they're what early stopping and final metrics are judged
    against; a val/test set only sees numbers that would occur for real.
    """
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_keep = int(len(pos_idx) * max_neg_ratio)
    if n_keep >= len(neg_idx):
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    neg_keep = rng.choice(neg_idx, size=n_keep, replace=False)
    return np.sort(np.concatenate([pos_idx, neg_keep]))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


PRECISION_RECALL_K_FRACS = (0.01, 0.05, 0.10)


def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fracs=PRECISION_RECALL_K_FRACS) -> Dict[str, float]:
    """Precision/recall among the top `k_frac` of the population by
    predicted score -- the "if we contact the N% most-likely clients,
    what fraction of them actually convert (precision), and what fraction
    of all true converters did we catch (recall)" framing, more directly
    decision-relevant for a propensity-targeting use case than ROC-AUC/
    PR-AUC alone (a model can have a good ROC-AUC yet very low PR-AUC on
    a rare target -- see the col_3/col_5 discussion this was added for --
    precision@k makes the practical "how many of the flagged clients are
    actually real" number explicit at a stated operating point).
    """
    n = len(y_true)
    total_pos = y_true.sum()
    if n == 0 or total_pos == 0:
        return {f"{m}@{int(f * 100)}%": float("nan") for f in k_fracs for m in ("precision", "recall")}

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)

    out: Dict[str, float] = {}
    for frac in k_fracs:
        k = max(1, int(round(frac * n)))
        tp = cum_pos[k - 1]
        out[f"precision@{int(frac * 100)}%"] = tp / k
        out[f"recall@{int(frac * 100)}%"] = tp / total_pos
    return out


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        nan_pr_k = {f"{m}@{int(f * 100)}%": float("nan") for f in PRECISION_RECALL_K_FRACS for m in ("precision", "recall")}
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), **nan_pr_k}
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        **precision_recall_at_k(y_true, y_score),
    }


def train_lightgbm(X_train, y_train, X_val, y_val, seed: int) -> lgb.Booster:
    pos = y_train.sum()
    neg = len(y_train) - pos
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "binary",
        "metric": "auc",
        "scale_pos_weight": neg / max(pos, 1),
        "num_leaves": 63,
        "learning_rate": 0.05,
        "seed": seed,
        "verbose": -1,
    }
    return lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    args: SimpleNamespace,
) -> MLP:
    model = MLP(X_train.shape[1], hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    stopper = EarlyStopper(patience=args.patience, mode="max")

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).to(device)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(args.max_epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        total, n = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        train_loss = total / max(n, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy()
        val_metrics = evaluate(y_val, val_logits)
        val_auc = val_metrics["roc_auc"]

        improved = stopper.step(val_auc, epoch)
        if improved:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        status = "(best)" if improved else f"(no improvement, {stopper.bad_epochs}/{args.patience})"
        print(f"    epoch {epoch}: train_loss={train_loss:.4f} val_auc={val_auc:.4f} {status}", flush=True)

        if stopper.should_stop:
            print(f"    early stopping at epoch {epoch} (best val_auc={stopper.best:.4f} @ {stopper.best_epoch})", flush=True)
            break

    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["coles", "nep", "mlm", "thp", "chronos2"])
    parser.add_argument("--config", type=str, default="/app/configs/models/downstream.yaml")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "skip training -- load each target's already-saved lgbm_<target>.txt/"
            "mlp_<target>.pt from checkpoint_dir and just re-score the test set "
            "(e.g. to compute a newly-added metric without retraining). Errors if "
            "a checkpoint is missing."
        ),
    )
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    mlp_args = SimpleNamespace(
        hidden=cfg["hidden"],
        lr=cfg["lr"],
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        batch_size=cfg["batch_size"],
    )

    ckpt_dir = Path(cfg["checkpoint_dir"] or f"/app/data/downstream/{cli.model}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(
        f"Train period: {cfg['train'][0]}..{cfg['train'][-1]}  "
        f"Test period: {cfg['test'][0]}..{cfg['test'][-1]}  "
        f"max_neg_ratio: {cfg['max_neg_ratio']}  (from {cli.config})",
        flush=True,
    )

    print(f"Loading {cli.model} embeddings ...", flush=True)
    embeds = load_embeddings(Path(cfg["embeds_dir"]), cli.model)
    print(f"  {len(embeds)} rows across {embeds['date'].nunique()} dates", flush=True)

    print("Loading targets ...", flush=True)
    targets = load_targets()
    check_required_months(targets, cfg["train"] + cfg["test"])

    joined = targets.merge(
        embeds,
        left_on=[TARGETS_CLIENT_ID_COL, TARGETS_DATE_COL],
        right_on=["inn", "date"],
        how="inner",
    )
    print(f"  joined on (id, date): {len(joined)} rows", flush=True)

    train_val = joined[joined[TARGETS_DATE_COL].isin(cfg["train"])]
    test = joined[joined[TARGETS_DATE_COL].isin(cfg["test"])]
    print(f"  train period (train+val)={len(train_val)}  test period (held out)={len(test)}", flush=True)

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

        print(f"\n=== {target} (train pos rate={y_train.mean():.4f}) ===", flush=True)

        if cli.eval_only:
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

        lgbm_path = ckpt_dir / f"lgbm_{target}.txt"
        if cli.eval_only:
            print(f"  loading saved LightGBM from {lgbm_path} ...", flush=True)
            if not lgbm_path.exists():
                raise FileNotFoundError(f"--eval-only given but {lgbm_path} doesn't exist -- run without --eval-only first")
            booster = lgb.Booster(model_file=str(lgbm_path))
        else:
            print("  training LightGBM ...", flush=True)
            booster = train_lightgbm(X_train_t, y_train_t, X_val, y_val, cfg["seed"])
            booster.save_model(str(lgbm_path))
        metrics = evaluate(y_test, booster.predict(X_test))
        results.append({"target": target, "arch": "lightgbm", **metrics})
        print(f"    test roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}", flush=True)

        mlp_path = ckpt_dir / f"mlp_{target}.pt"
        if cli.eval_only:
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
        results.append({"target": target, "arch": "mlp", **metrics})
        print(f"    test roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}", flush=True)

    results_df = pd.DataFrame(results)
    print(
        f"\n=== {cli.model} downstream results "
        f"(train={cfg['train'][0]}..{cfg['train'][-1]}, "
        f"test={cfg['test'][0]}..{cfg['test'][-1]}) ==="
    )
    print(results_df.to_string(index=False))
    results_df.to_csv(ckpt_dir / "results.csv", index=False)
    print(f"\nSaved models + results to {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
