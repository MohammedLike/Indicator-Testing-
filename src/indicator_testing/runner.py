from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from talib import abstract

from indicator_testing.models import IndicatorResult
from indicator_testing.registry import IndicatorRegistry
from indicator_testing.validator import compute_warmup_bars, validate_indicator_result


def _normalize_outputs(
    raw: Any,
    output_names: list[str],
) -> dict[str, np.ndarray]:
    if isinstance(raw, (list, tuple)):
        if len(output_names) != len(raw):
            keys = output_names or [f"output_{i}" for i in range(len(raw))]
        else:
            keys = output_names
        return {k: np.asarray(v, dtype=np.float64) for k, v in zip(keys, raw)}

    key = output_names[0] if output_names else "real"
    return {key: np.asarray(raw, dtype=np.float64)}


def run_indicator(
    name: str,
    df: pd.DataFrame,
    registry: IndicatorRegistry | None = None,
    params: dict[str, Any] | None = None,
    *,
    validate: bool = True,
) -> IndicatorResult:
    reg = registry or IndicatorRegistry()
    info = reg.get(name)
    params_used = reg.resolve_params(name, params)

    can_run, skip_reason = reg.can_run(name, df)
    if not can_run:
        return IndicatorResult(
            name=info.name,
            group=info.group,
            params_used=params_used,
            outputs={},
            warmup_bars=None,
            status="skipped",
            message=skip_reason,
        )

    try:
        func = abstract.Function(info.name)
        func.set_input_arrays(reg.build_input_arrays(df, indicator=info.name))
        raw = func(**params_used)
        outputs = _normalize_outputs(raw, info.output_names)
        warmup = compute_warmup_bars(outputs, is_pattern=info.is_pattern)
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


def run_batch(
    df: pd.DataFrame,
    registry: IndicatorRegistry | None = None,
    *,
    indicator: str | None = None,
    group: str | None = None,
    run_all: bool = False,
    validate: bool = True,
) -> list[IndicatorResult]:
    reg = registry or IndicatorRegistry()

    if indicator:
        names = [reg.get(indicator).name]
    elif group:
        names = reg.names_by_group(group)
    elif run_all:
        names = [info.name for info in reg.iter_ordered()]
    else:
        raise ValueError("Specify indicator, group, or run_all=True.")

    results: list[IndicatorResult] = []
    for name in names:
        results.append(run_indicator(name, df, reg, validate=validate))
    return results


def filter_results(
    results: Iterable[IndicatorResult],
    *,
    status: str | None = None,
) -> list[IndicatorResult]:
    if status is None:
        return list(results)
    return [r for r in results if r.status == status]
