"""Apply the frozen xbank-to-MBD feature match to MBD-trained input slots.

The matcher names native MBD fields, while the checkpoints were trained on
``mbd_adapter``'s anonymized ``col_N`` layout.  Composing those two mappings
is essential: a matched xbank feature must land in the slot used for that
MBD field during pretraining.  Targets are never read or transformed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.mbd_adapter import TRX_CATEGORY_MAP
from data.schema import ALL_FEATURE_COLS, CLIENT_ID_COL, EVENT_TIME_COL, NUMERIC_COLS


MBD_SLOT_BY_FIELD = {**TRX_CATEGORY_MAP, "amount": "col_11", "event_time": EVENT_TIME_COL}
XBANK_FEATURES = set(ALL_FEATURE_COLS) | {EVENT_TIME_COL}


class FrozenSchemaMapping:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        document: dict[str, Any] = json.loads(raw)
        mapping = document.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"{self.path}: missing nonempty 'mapping' object")
        if mapping.get(EVENT_TIME_COL) != "event_time":
            raise ValueError(f"{self.path}: expected fixed col_1 -> event_time pair")
        unknown_sources = set(mapping) - XBANK_FEATURES
        unknown_fields = set(mapping.values()) - set(MBD_SLOT_BY_FIELD)
        if unknown_sources or unknown_fields:
            raise ValueError(
                f"{self.path}: unknown xbank columns {sorted(unknown_sources)} "
                f"or MBD fields {sorted(unknown_fields)}"
            )
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f"{self.path}: mapping must be one-to-one")
        for source, field in mapping.items():
            target_slot = MBD_SLOT_BY_FIELD[field]
            if (source in NUMERIC_COLS) != (target_slot in NUMERIC_COLS):
                if source != EVENT_TIME_COL or target_slot != EVENT_TIME_COL:
                    raise ValueError(f"{self.path}: incompatible types for {source} -> {field}")
        self.mapping: dict[str, str] = mapping

    def source_columns(self, model: str) -> list[str]:
        if model in ("cotic", "thp"):
            required_fields = {"event_type"}
        elif model == "chronos2":
            required_fields = {"amount"}
        else:
            required_fields = set(MBD_SLOT_BY_FIELD) - {"event_time"}
        available = {field: source for source, field in self.mapping.items()}
        if model in ("cotic", "thp", "chronos2"):
            missing = required_fields - set(available)
            if missing:
                raise ValueError(f"{self.path}: {model} requires matched MBD field(s) {sorted(missing)}")
        sources = [source for source, field in self.mapping.items() if field in required_fields]
        return [CLIENT_ID_COL, EVENT_TIME_COL, *sources]

    def transform(self, windowed: pd.DataFrame, model: str) -> pd.DataFrame:
        """Return the exact feature layout the MBD checkpoint was trained on.

        Unfilled MBD fields use the same zero placeholder convention as the
        MBD adapter's own absent columns; no vocabulary is refitted here.
        """
        required = self.source_columns(model)
        missing = set(required) - set(windowed.columns)
        if missing:
            raise ValueError(f"window is missing matched xbank columns {sorted(missing)}")
        aligned = windowed[[CLIENT_ID_COL, EVENT_TIME_COL]].copy()
        if model in ("cotic", "thp"):
            output_slots = [MBD_SLOT_BY_FIELD["event_type"]]
        elif model == "chronos2":
            output_slots = [MBD_SLOT_BY_FIELD["amount"]]
        else:
            output_slots = ALL_FEATURE_COLS
        for slot in output_slots:
            aligned[slot] = 0.0 if slot in NUMERIC_COLS else 0
        for source, field in self.mapping.items():
            slot = MBD_SLOT_BY_FIELD[field]
            if slot in output_slots:
                if slot in NUMERIC_COLS:
                    aligned[slot] = windowed[source].to_numpy()
                else:
                    # MBD adapter cast its integer-like category codes to
                    # BIGINT before pretraining. Xbank stores many of those
                    # codes as DOUBLE (e.g. 3.0). The frozen PTLS
                    # FrequencyEncoder stringifies inputs, where "3.0" is
                    # *not* the trained key "3". Preserve the matched
                    # integer code, not a new/refitted vocabulary.
                    values = pd.to_numeric(windowed[source], errors="raise")
                    if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
                        raise ValueError(f"{source}: infinite category code")
                    fractional = values.notna() & (values % 1 != 0)
                    if fractional.any():
                        raise ValueError(f"{source}: non-integer category code")
                    aligned[slot] = values.astype("Int64")
        return aligned
