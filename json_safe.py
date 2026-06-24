"""Tiny helper so pandas rows survive the trip across an API boundary as
clean JSON. Two separate gotchas, both arising from how pandas stores
values internally, and both invisible until you actually try to
`json.dumps` the result:

1. Missing values (e.g. no detectable color, no rating) come back as the
   float NaN, not None. Starlette's JSONResponse uses `allow_nan=False`,
   so a stray NaN crashes the response with a 500, not a clean error.
2. Numeric columns (price_inr, rating_count...) come back as numpy scalar
   types (np.int64, np.float64), which the stdlib json encoder does not
   know how to serialize at all (`TypeError: Object of type int64 is not
   JSON serializable`), independent of the NaN issue above.

`safe_dict` fixes both by walking each field, unwrapping numpy scalars to
native Python types via `.item()`, then mapping NaN/NaT to None.
"""
from __future__ import annotations

import pandas as pd


def _clean_value(v):
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()  # numpy scalar (np.int64, np.float64, np.bool_...) -> native Python type
        except (ValueError, AttributeError):
            pass
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass  # pd.isna() can't evaluate some types (e.g. arrays) -- not our case here, but don't crash
    return v


def safe_dict(row: pd.Series) -> dict:
    return {k: _clean_value(v) for k, v in row.items()}
