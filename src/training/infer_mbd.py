"""Full-scale MBD (cross-institution transfer-target) inference: for every
labeled MBD client, one embedding per calendar month that actually appears
in MBD's own adapted targets file (data/mbd_adapter.py), each built from
that client's trailing `history_window_months` of transaction history up
to that date -- reusing the exact xbank windowing/embedding-extraction
machinery (data/splits.py's load_windowed_transactions_for_dates, each
model's own extract_embeddings) completely UNCHANGED, over MBD
transactions that data/mbd_adapter.py has already reshaped into xbank's
own (id, col_1, col_2..col_16) column convention.

Every xbank-pretrained checkpoint is used FROZEN here -- no fine-tuning,
no re-fitting a preprocessor/categories factorization on MBD data (that
would defeat the whole point of a zero-shot transfer test; see
train_*.py's identical rationale for why the checkpoint's OWN
preprocessor/categories artifact is reused rather than refit). See
RESEARCH_PLAN.md's "MBD Integration" section for why category-vocabulary
transfer is expected to be lossy there, and why event timestamps are kept
at MBD's native hour-of-day precision rather than truncated to match
xbank's daily aggregation.

Supports the same architectures as xbank's infer_*.py scripts
(--model {coles,nep,mlm,thp,chronos2}); COTIC now has its own xbank
inference script (infer_cotic.py) but isn't yet wired into this
consolidated --model dispatch.

Run parameters live in configs/models/downstream_mbd.yaml's `inference:`
section (shared across every --model) and configs/models/<model>.yaml
(that architecture's own hidden_size/num_layers/etc -- must match what
the checkpoint was actually trained with; chronos2 has no such file,
it's zero-shot). checkpoint_dir is DERIVED from --checkpoint-source as
/app/data/checkpoints/<source>_source/<model> -- not read from
configs/models/<model>.yaml, which no longer has that key (updated
2026-09-19). UNLIKE the xbank infer_{model}.py scripts (which now default
--checkpoint-source to mbd, since MBD is the primary pretraining corpus
as of 2026-09-14), THIS script defaults to xbank -- its established job is
running some checkpoint zero-shot against MBD, historically always the
xbank-pretrained one; pass --checkpoint-source mbd or mbd_daily instead
for the newer in-domain-MBD checks (RESEARCH_PLAN.md §4's "no-shift"
baselines: pretrain-on-MBD, eval-on-MBD-with-no-institution-change).

Requires data/mbd_adapter.py's output to already exist (run
`python -m data.mbd_adapter` first) -- this script only consumes the
adapted parquet files, it doesn't materialize them, since materializing
is a one-time step shared across every --model run.

Output: one parquet file per target month under
<embeds_dir>/<checkpoint_source>_source/<model>/ (chronos2, having no
checkpoint at all, keeps the flat <embeds_dir>/chronos2/ instead), columns
[inn, date, emb_0..emb_D] -- same shape as xbank's own embeds under
/app/data/embeds/, so the same downstream-probe code can be pointed at
either. Resumable: a date whose file already exists is skipped on the
next run.

Usage (inside the container):
    python -m data.mbd_adapter                    # once
    python src/training/infer_mbd.py --model coles
"""
import argparse
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from data.loaders import build_chronos_series, build_ptls_records, build_thp_sequences, sample_client_ids
from data.schema import CLIENT_ID_COL, NUMERIC_COLS, TARGETS_DATE_COL
from data.splits import load_windowed_transactions_for_dates, unpack_window_id
from training.common import load_preprocessor

MBD_DATA_CONFIG = "/app/configs/data/mbd.yaml"
with open(MBD_DATA_CONFIG) as f:
    _mbd_paths = yaml.safe_load(f)["paths"]
MBD_TRANSACTIONS_PATH = _mbd_paths["transactions"]
MBD_TARGETS_PATH = _mbd_paths["targets"]


def _require_adapted_files() -> None:
    missing = [p for p in (MBD_TRANSACTIONS_PATH, MBD_TARGETS_PATH) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing} not found -- run `python -m data.mbd_adapter` first to "
            "materialize MBD's adapted transactions/targets (see data/mbd_adapter.py)."
        )


def _target_dates() -> List[str]:
    """MBD's own monthly target dates (from its adapted targets file), NOT
    xbank's calendar grid -- MBD's targets run 2022-02..2023-01, on a
    different calendar axis than xbank's 2023-01..2024-02 (RESEARCH_PLAN.md's
    "ID / date splitting (transfer, MBD)" explains why the two protocols
    aren't calendar-aligned).
    """
    targets = pd.read_parquet(MBD_TARGETS_PATH, columns=[TARGETS_DATE_COL])
    return sorted(targets[TARGETS_DATE_COL].astype(str).unique())


def _load_embedder(
    model_name: str, model_cfg: Dict, inf: Dict, device: torch.device, checkpoint_source: str
) -> Callable[[pd.DataFrame], Tuple[np.ndarray, List[str]]]:
    """Loads the checkpoint for `model_name` and returns a closure that
    maps one windowed dataframe -> (embeddings, window_ids). Chronos-2 is
    handled separately in main() (it needs per-date chunking, unlike the
    other four -- see infer_chronos2.py's identical reasoning).
    """
    ckpt_dir = Path(f"/app/data/checkpoints/{checkpoint_source}_source/{model_name}")

    if model_name == "coles":
        from models.coles import build_module, extract_embeddings

        preprocessor = load_preprocessor(ckpt_dir / "preprocessor.pkl")
        cat_sizes = preprocessor.get_category_dictionary_sizes()
        module = build_module(
            cat_sizes,
            embedding_dim=model_cfg["embedding_dim"],
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
        )
        ckpt = torch.load(str(ckpt_dir / "best.ckpt"), map_location=device, weights_only=False)
        module.load_state_dict(ckpt["state_dict"])
        module.to(device)

        def embed(windowed: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            records, _ = build_ptls_records(windowed, preprocessor=preprocessor)
            window_ids = [r[CLIENT_ID_COL] for r in records]
            emb = extract_embeddings(module, records, batch_size=inf["batch_size"], device=device)
            return emb, window_ids

        return embed

    if model_name in ("nep", "mlm"):
        from models.nep import NEP, extract_embeddings as extract_nep
        from models.mlm import MLM, extract_embeddings as extract_mlm

        preprocessor = load_preprocessor(ckpt_dir / "preprocessor.pkl")
        cat_sizes = preprocessor.get_category_dictionary_sizes()
        ckpt = torch.load(str(ckpt_dir / "best.pt"), map_location=device, weights_only=False)

        if model_name == "nep":
            model = NEP(
                cat_sizes,
                NUMERIC_COLS,
                d_model=model_cfg["d_model"],
                num_layers=model_cfg["num_layers"],
                max_position_embeddings=inf["max_seq_len"],
            )
            extract_fn = extract_nep
        else:
            # MLM's BERT backbone has a real learned position-embedding
            # table -- read its size directly off the checkpoint rather
            # than from max_seq_len, same rationale as infer_mlm.py.
            ckpt_max_pos = ckpt["model"]["backbone.embeddings.position_embeddings.weight"].shape[0]
            model = MLM(
                cat_sizes,
                NUMERIC_COLS,
                d_model=model_cfg["d_model"],
                num_layers=model_cfg["num_layers"],
                max_position_embeddings=ckpt_max_pos,
            )
            extract_fn = extract_mlm

        model.load_state_dict(ckpt["model"])
        model.to(device)

        def embed(windowed: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            records, _ = build_ptls_records(windowed, preprocessor=preprocessor)
            window_ids = [r[CLIENT_ID_COL] for r in records]
            emb = extract_fn(model, records, batch_size=inf["batch_size"], device=device)
            return emb, window_ids

        return embed

    if model_name == "thp":
        from models.thp import build_dataloader, build_model, build_tokenizer, extract_embeddings

        categories = np.load(ckpt_dir / "categories.npy", allow_pickle=True)
        num_types = len(categories)
        gpu = 0 if torch.cuda.is_available() else -1
        model = build_model(num_types, hidden_size=model_cfg["hidden_size"], num_layers=model_cfg["num_layers"], gpu=gpu)
        ckpt = torch.load(str(ckpt_dir / "best.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        tokenizer = build_tokenizer(num_types, max_len=inf["max_seq_len"])

        def embed(windowed: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
            time_seqs, delta_seqs, type_seqs, _, _, window_ids = build_thp_sequences(windowed, categories=categories)
            loader = build_dataloader(time_seqs, delta_seqs, type_seqs, tokenizer, inf["batch_size"], shuffle=False)
            emb = extract_embeddings(model, loader)
            return emb, window_ids

        return embed

    raise ValueError(f"no embedder for --model {model_name!r} (chronos2 is handled separately in main())")


def _run_chronos2(inf: Dict, out_dir: Path, target_dates: List[str], client_ids_filter) -> None:
    """Same per-date chunking as infer_chronos2.py -- Chronos-2 inference
    is heavy enough per client that a crash partway through a date should
    resume from the last completed chunk, not restart the whole date.
    """
    from models.chronos2 import extract_reg_embeddings, load_pipeline

    print("Loading pretrained Chronos-2 (amazon/chronos-2) ...", flush=True)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = load_pipeline(device_map=device_map)
    print("  loaded.", flush=True)

    for target_date in target_dates:
        out_path = out_dir / f"{target_date}.parquet"
        if out_path.exists():
            print(f"{target_date}: already done, skipping", flush=True)
            continue

        print(f"{target_date}: windowing transactions ...", flush=True)
        windowed = load_windowed_transactions_for_dates(
            MBD_TRANSACTIONS_PATH,
            [target_date],
            inf["history_window_months"],
            inf["max_seq_len"],
            client_ids=client_ids_filter,
        )
        if len(windowed) == 0:
            print(f"{target_date}: no rows in window, skipping", flush=True)
            continue

        window_ids = windowed[CLIENT_ID_COL].unique().tolist()
        chunk_size = inf["chronos2_chunk_size"]
        chunk_lists = [window_ids[i : i + chunk_size] for i in range(0, len(window_ids), chunk_size)]
        print(f"{target_date}: {len(window_ids)} clients, {len(chunk_lists)} chunks", flush=True)

        chunks_dir = out_dir / "_chunks" / target_date
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk_ids in enumerate(chunk_lists):
            chunk_path = chunks_dir / f"chunk_{i:05d}.parquet"
            if chunk_path.exists():
                continue
            chunk_df = windowed[windowed[CLIENT_ID_COL].isin(chunk_ids)]
            series_list, series_window_ids = build_chronos_series(chunk_df, value_col=inf["chronos2_value_col"])
            inns = [unpack_window_id(str(w))[0] for w in series_window_ids]

            emb = extract_reg_embeddings(pipeline, series_list, batch_size=inf["chronos2_batch_size"])
            out = pd.DataFrame(emb, columns=[f"emb_{j}" for j in range(emb.shape[1])])
            out.insert(0, "date", target_date)
            out.insert(0, "inn", inns)
            out.to_parquet(chunk_path)
            print(f"{target_date}: chunk {i + 1}/{len(chunk_lists)} ({len(inns)} clients)", flush=True)

        merged = pd.concat([pd.read_parquet(p) for p in sorted(chunks_dir.glob("chunk_*.parquet"))], ignore_index=True)
        merged.to_parquet(out_path)
        shutil.rmtree(chunks_dir)
        print(f"{target_date}: wrote {len(merged)} embeddings to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["coles", "nep", "mlm", "thp", "chronos2"])
    parser.add_argument("--downstream-config", type=str, default="/app/configs/models/downstream_mbd.yaml")
    parser.add_argument("--model-config", type=str, default=None, help="default /app/configs/models/<model>.yaml")
    parser.add_argument(
        "--checkpoint-source",
        type=str,
        default="xbank",
        help=(
            "which pretrained checkpoint to load, by the data_config 'name' it was "
            "trained with -- checkpoint_dir becomes /app/data/checkpoints/<source>_source/"
            "<model>. Default xbank (this script's established zero-shot-transfer role); "
            "pass mbd or mbd_daily for the in-domain-MBD checks instead."
        ),
    )
    cli = parser.parse_args()

    with open(cli.downstream_config) as f:
        inf = yaml.safe_load(f)["inference"]

    model_cfg: Optional[Dict] = None
    if cli.model != "chronos2":
        model_config_path = cli.model_config or f"/app/configs/models/{cli.model}.yaml"
        with open(model_config_path) as f:
            model_cfg = yaml.safe_load(f)

    _require_adapted_files()

    # Diverges by checkpoint_source same as ckpt_dir -- otherwise embeddings
    # from an xbank-pretrained and an mbd-pretrained checkpoint would
    # silently land in the SAME output path (2026-09-19). Chronos-2 is
    # zero-shot (no checkpoint at all, see its own module docstring), so
    # it has no checkpoint_source to diverge by -- keeps its own flat path.
    out_subdir = cli.model if cli.model == "chronos2" else f"{cli.checkpoint_source}_source/{cli.model}"
    out_dir = Path(inf["embeds_dir"]) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dates = _target_dates()
    print(f"{len(target_dates)} target dates (from MBD's own targets): {target_dates[0]} .. {target_dates[-1]}", flush=True)

    client_ids_filter = None
    if inf["n_clients"] is not None:
        client_ids_filter = sample_client_ids(MBD_TRANSACTIONS_PATH, inf["n_clients"], seed=inf["seed"])
        print(f"  capped to {len(client_ids_filter)} clients for debugging", flush=True)

    if cli.model == "chronos2":
        _run_chronos2(inf, out_dir, target_dates, client_ids_filter)
        print("Done.", flush=True)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading {cli.model} checkpoint ...", flush=True)
    embed = _load_embedder(cli.model, model_cfg, inf, device, cli.checkpoint_source)
    print("  loaded.", flush=True)

    for target_date in target_dates:
        out_path = out_dir / f"{target_date}.parquet"
        if out_path.exists():
            print(f"{target_date}: already done, skipping", flush=True)
            continue

        print(f"{target_date}: windowing transactions ...", flush=True)
        windowed = load_windowed_transactions_for_dates(
            MBD_TRANSACTIONS_PATH,
            [target_date],
            inf["history_window_months"],
            inf["max_seq_len"],
            client_ids=client_ids_filter,
        )
        if len(windowed) == 0:
            print(f"{target_date}: no rows in window, skipping", flush=True)
            continue

        print(f"{target_date}: extracting embeddings for up to {windowed[CLIENT_ID_COL].nunique()} clients ...", flush=True)
        emb, window_ids = embed(windowed)
        inns = [unpack_window_id(str(w))[0] for w in window_ids]
        assert len(emb) == len(inns), f"row count mismatch: {len(emb)} embeddings vs {len(inns)} ids"

        out = pd.DataFrame(emb, columns=[f"emb_{j}" for j in range(emb.shape[1])])
        out.insert(0, "date", target_date)
        out.insert(0, "inn", inns)
        out.to_parquet(out_path)
        print(f"{target_date}: wrote {len(out)} embeddings to {out_path}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
