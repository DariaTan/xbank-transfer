"""Cell-wise corruption; missing values and numeric zeros are preserved.

Streams depend on (noise seed, feature position, operation), not alpha or
noise strength. Within a seed, stronger noise uses the same Gaussian draws
and a nested categorical replacement mask. Source frames are never mutated.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


NOISE_LEVELS = {
    "clean": (0.0, 0.0),
    "mixed_015": (0.15, 0.15),
    "mixed_030": (0.30, 0.30),
    "categorical_015": (0.0, 0.15),
    "numeric_015": (0.15, 0.0),
}


def corrupt_frame(frame, specs, passports, numeric_noise, category_noise, seed):
    """Gaussian numeric noise and uniform replacement by a DIFFERENT category.

    x'[r,j] = x[r,j] + lambda * s[j] * epsilon[r,j], epsilon ~ N(0,1).
    s[j] is the CLEAN passport's Q95-Q05 scale (same fallbacks as FGW).
    No clipping: amounts may be signed. Constant and binary numeric fields
    are left intact to preserve the matcher's subtype; MBD amount is continuous.
    Categorical replacement samples uniformly from the OTHER observed valid
    categories. It therefore changes both marginals and dependencies.
    """
    if (not math.isfinite(numeric_noise) or numeric_noise < 0
            or not math.isfinite(category_noise) or not 0 <= category_noise <= 1
            or seed < 0):
        raise ValueError("Require lambda >= 0, p in [0,1], and a nonnegative seed")
    noisy = frame.copy(deep=True)
    audit = []
    for column_id, (name, _, kind) in enumerate(specs):
        original = frame[name]
        passport = passports[name]
        rng = np.random.default_rng(np.random.SeedSequence([seed, column_id, 0]))
        record = {"field": name, "kind": kind, "n_rows": len(frame),
                  "numeric_noise": numeric_noise, "category_noise": category_noise,
                  "n_changed": 0, "scale": None, "skip_reason": ""}
        if kind == "numeric":
            values = pd.to_numeric(original, errors="raise").to_numpy(dtype=float, na_value=np.nan)
            valid = np.isfinite(values)
            eligible = valid & (values != 0)
            scale = float(passport.get("quantile_scale", 0.0))
            record.update(n_valid=int(valid.sum()), n_eligible=int(eligible.sum()), scale=scale)
            if passport.get("numeric_subtype") != "continuous" or scale == 0:
                record["skip_reason"] = "non_continuous_or_zero_scale"
            elif numeric_noise > 0:
                changed = values.copy()
                changed[eligible] += numeric_noise * scale * rng.standard_normal(int(eligible.sum()))
                if not np.isfinite(changed[eligible]).all():
                    raise ValueError(f"Numeric noise overflow in {name}; reduce lambda")
                noisy[name] = changed
                record["n_changed"] = int(np.count_nonzero(changed[eligible] != values[eligible]))
        elif kind == "categorical":
            clean = original.copy()
            if pd.api.types.is_numeric_dtype(clean):
                clean = clean.replace([np.inf, -np.inf], np.nan)
            codes, categories = pd.factorize(clean, sort=False)
            positions = np.flatnonzero(codes >= 0)
            n_categories = len(categories)
            record.update(n_valid=len(positions), n_eligible=len(positions) if n_categories > 1 else 0)
            if n_categories < 2:
                record["skip_reason"] = "fewer_than_two_categories"
            elif category_noise > 0:
                selected = rng.random(len(positions)) < category_noise
                # Draw replacements for ALL valid rows: the same seed gives
                # the same replacement whenever a row is corrupted at two p's.
                replacement_rng = np.random.default_rng(np.random.SeedSequence([seed, column_id, 1]))
                replacements = replacement_rng.integers(0, n_categories - 1, size=len(positions))
                replacements += replacements >= codes[positions]
                values = original.copy()
                values.iloc[positions[selected]] = categories.take(replacements[selected]).to_numpy()
                noisy[name] = values
                record["n_changed"] = int(selected.sum())
        else:
            raise ValueError(f"Unsupported column kind: {kind}")
        record["changed_fraction"] = (record["n_changed"] / record["n_eligible"]
                                      if record["n_eligible"] else 0.0)
        audit.append(record)
    return noisy, pd.DataFrame(audit)
