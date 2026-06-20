from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from indicator_testing.data_loader import has_volume
from indicator_testing.models import Check, IndicatorResult, ValidationResult
from indicator_testing.pandas_indicators import PandasIndicatorRegistry
from indicator_testing.validator import compute_warmup_bars, validate_indicator_result


def _to_output_dict(result: pd.Series | pd.DataFrame) -> dict[str, np.ndarray]:
    if isinstance(result, pd.Series):
        return {result.name or "real": result.to_numpy(dtype=np.float64)}
    return {col: result[col].to_numpy(dtype=np.float64) for col in result.columns}


def run_pandas_indicator(
    name: str,
    df: pd.DataFrame,
    registry: PandasIndicatorRegistry | None = None,
    params: dict[str, Any] | None = None,
    *,
    validate: bool = True,
) -> IndicatorResult:
    reg = registry or PandasIndicatorRegistry()
    info = reg.get(name)
    params_used = dict(info.default_params)
    if params:
        params_used.update(params)

    if info.requires_volume and not has_volume(df):
        return IndicatorResult(
            name=info.name,
            group=info.group,
            params_used=params_used,
            outputs={},
            warmup_bars=None,
            status="skipped",
            message="Volume column required but not present in CSV.",
        )

    try:
        raw = info.func(df, **params_used)
        outputs = _to_output_dict(raw)
        warmup = compute_warmup_bars(outputs, is_pattern=False)
        result = IndicatorResult(
            name=info.name,
            group=info.group,
            params_used=params_used,
            outputs=outputs,
            warmup_bars=warmup,
            status="success",
            message="OK",
        )
        if validate:
            result.validation = validate_indicator_result(result, len(df))
        return result
    except Exception as exc:
        return IndicatorResult(
            name=info.name,
            group=info.group,
            params_used=params_used,
            outputs={},
            warmup_bars=None,
            status="error",
            message=str(exc),
        )


def run_pandas_batch(
    df: pd.DataFrame,
    registry: PandasIndicatorRegistry | None = None,
    *,
    validate: bool = True,
) -> list[IndicatorResult]:
    reg = registry or PandasIndicatorRegistry()
    return [
        run_pandas_indicator(info.name, df, reg, validate=validate)
        for info in reg.iter_ordered()
    ]
