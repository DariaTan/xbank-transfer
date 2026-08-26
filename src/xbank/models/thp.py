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

import torch
from easy_tpp.config_factory.data_config import DataSpecConfig
from easy_tpp.config_factory.model_config import ModelConfig
from easy_tpp.model.thp import THP
from easy_tpp.preprocess.data_loader import get_data_loader
from easy_tpp.preprocess.dataset import TPPDataset
from easy_tpp.preprocess.event_tokenizer import EventTokenizer


def build_tokenizer(num_types: int, max_len: int) -> EventTokenizer:
    """pad_token_id = num_types: real event ids are 0..num_types-1, the
    next free index is reserved for padding (num_event_types_pad =
    num_types + 1) -- EasyTPP's own convention, not ours.
    """
    data_spec = DataSpecConfig(
        num_event_types=num_types,
        pad_token_id=num_types,
        padding_side="right",
        truncation_side="right",
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
    use_ln: bool = False,
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
    return model.to(model.device)


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
