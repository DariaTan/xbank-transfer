"""THP adapter: build easy_tpp's THP model/dataloaders directly from
in-memory xbank sequences.

`easy_tpp` is a normal pip dependency (unlike COTIC), but its own Runner/
Config/YAML orchestration is built around named datasets/experiments on
disk. Same approach as the COTIC adapter: reuse the actual model and data
classes (`THP`, `TPPDataset`, `EventTokenizer`, `get_data_loader`)
directly, construct their plain-kwargs `Config` objects ourselves, and
skip the Runner/YAML layer entirely.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from easy_tpp.config_factory.data_config import DataSpecConfig
from easy_tpp.config_factory.model_config import ModelConfig
from easy_tpp.model.baselayer import EncoderLayer, MultiHeadAttention
from easy_tpp.model.thp import THP
from easy_tpp.preprocess.data_loader import get_data_loader
from easy_tpp.preprocess.dataset import TPPDataset
from easy_tpp.preprocess.event_tokenizer import EventTokenizer


def build_tokenizer(num_types: int, max_len: int) -> EventTokenizer:
    """pad_token_id = num_types: real event ids are 0..num_types-1, the
    next free index is reserved for padding (num_event_types_pad =
    num_types + 1) -- EasyTPP's own convention, not ours.

    truncation_strategy=None -> easy_tpp.preprocess.dataset.get_data_loader
    resolves this to `truncation=False` in the collator, so no sequence is
    ever truncated here regardless of length -- the actual cap comes from
    `cap_rows_per_client` upstream, which already keeps at most `max_len`
    of each client's most recent rows before sequences are built. No
    truncation_side to set for the same reason (there's deliberately no
    truncation_side kwarg here -- it would sit unused).
    """
    data_spec = DataSpecConfig(
        num_event_types=num_types,
        pad_token_id=num_types,
        padding_side="right",
        padding_strategy=None,
        truncation_strategy=None,
        max_len=max_len,
        model_input_names=None,
    )
    return EventTokenizer(data_spec)


def build_dataloader(
    time_seqs: List[List[float]],
    time_delta_seqs: List[List[float]],
    type_seqs: List[List[int]],
    tokenizer: EventTokenizer,
    batch_size: int,
    shuffle: bool,
):
    dataset = TPPDataset(
        {
            "time_seqs": time_seqs,
            "time_delta_seqs": time_delta_seqs,
            "type_seqs": type_seqs,
        }
    )
    return get_data_loader(
        dataset,
        backend="torch",
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def build_model(
    num_types: int,
    hidden_size: int = 128,
    time_emb_size: int = 16,
    num_layers: int = 4,
    num_heads: int = 2,
    dropout_rate: float = 0.1,
    use_ln: bool = True,
    loss_integral_num_sample_per_step: int = 20,
    gpu: int = -1,
) -> THP:
    model_config = ModelConfig(
        hidden_size=hidden_size,
        time_emb_size=time_emb_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        use_ln=use_ln,
        use_mc_samples=True,
        loss_integral_num_sample_per_step=loss_integral_num_sample_per_step,
        num_event_types=num_types,
        num_event_types_pad=num_types + 1,
        event_pad_index=num_types,
        thinning=None,  # only needed for autoregressive sampling/generation
        gpu=gpu,
    )
    model = THP(model_config)
    # BaseModel.__init__ calls self.to(self.device) partway through its own
    # constructor, before THP.__init__ (the caller) creates stack_layers/
    # feed_forward/etc. -- those submodules are added afterward and stay on
    # CPU otherwise. Moving again here catches everything.
    model = model.to(model.device)

    if use_ln:
        # easy_tpp.model.thp.THP.__init__ reads model_config.use_ln into
        # self.use_norm (thp.py:22) but never actually consults it --
        # self.stack_layers is built two lines later with use_residual=
        # False hardcoded (thp.py:52), and use_residual is what gates
        # whether EncoderLayer wraps its sublayers in SublayerConnection,
        # which is where the actual nn.LayerNorm lives
        # (easy_tpp/model/baselayer.py). So use_ln=True silently did
        # nothing upstream -- confirmed by tracing exactly where
        # self.use_norm is (never) read again after being set.
        #
        # Root cause matters here beyond cosmetics: THP's hidden states
        # come out with an anomalously huge, unnormalized L2 norm
        # (~500-700 vs ~10-15 for the other models at matched size), and
        # ScaledSoftplus (baselayer.py) switches to an effectively linear
        # response once its scaled input exceeds a fixed threshold --
        # so unnormalized hidden states push the intensity computation
        # into a regime dominated by raw magnitude rather than learned
        # structure. Observed in practice as erratic train NLL and a
        # validation NLL that barely moves across epochs, i.e. not just
        # a cross-model embedding-scale mismatch but a plausible cause of
        # THP not training well at all.
        #
        # Rebuilt here to mirror THP.__init__'s own construction exactly
        # (down to sharing one `feed_forward` module across every layer,
        # matching upstream's existing -- if unusual -- behavior) with
        # use_residual=True instead, so use_ln does what its name claims.
        model.stack_layers = nn.ModuleList(
            [
                EncoderLayer(
                    model.d_model,
                    MultiHeadAttention(
                        model.n_head, model.d_model, model.d_model, model.dropout, output_linear=False
                    ),
                    use_residual=True,
                    feed_forward=model.feed_forward,
                    dropout=model.dropout,
                )
                for _ in range(model.n_layers)
            ]
        ).to(model.device)

    return model


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_one_epoch(model: THP, loader, optimizer) -> float:
    model.train()
    total_loss, total_events = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, model.device)
        optimizer.zero_grad()
        loss, num_events = model.loglike_loss(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_events += num_events
    return total_loss / max(total_events, 1)


@torch.no_grad()
def evaluate(model: THP, loader) -> float:
    model.eval()
    total_loss, total_events = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, model.device)
        loss, num_events = model.loglike_loss(batch)
        total_loss += loss.item()
        total_events += num_events
    return total_loss / max(total_events, 1)


@torch.no_grad()
def extract_embeddings(model: THP, loader) -> np.ndarray:
    """One embedding per sequence in `loader` (built with shuffle=False,
    so output rows are in the same order the caller's client/window id
    list is in): the encoder's hidden state at each sequence's last real
    event.

    Unlike `loglike_loss` (which encodes `[:, :-1]` -- the model predicts
    event i+1 from events 0..i, so the last event is never itself an
    encoder input, only a loss target), this runs the FULL given sequence
    through `model.forward`, matching NEP/MLM's `.encode()` semantics: no
    shift, the entire window is real input, safe from future leakage only
    because the window itself is already truncated to the target cutoff
    date upstream. `get_logits_at_last_step` (inherited from `BaseModel`)
    is the same last-real-position gather NEP/MLM's `last_event_embedding`
    does, just easy_tpp's own version of it.
    """
    model.eval()
    out = []
    for batch in loader:
        batch = _to_device(batch, model.device)
        enc_out = model.forward(batch["time_seqs"], batch["type_seqs"], batch["attention_mask"])
        last = model.get_logits_at_last_step(enc_out, batch["seq_non_pad_mask"])
        out.append(last.cpu().numpy())
    return np.concatenate(out, axis=0)
