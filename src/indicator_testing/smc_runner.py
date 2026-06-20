from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from indicator_testing.models import IndicatorResult
from indicator_testing.smc_indicators import SmcIndicatorRegistry
from indicator_testing.validator import compute_warmup_bars, validate_indicator_result


def _to_output_dict(result: pd.Series | pd.DataFrame) -> dict[str, np.ndarray]:
    if isinstance(result, pd.Series):
        name = result.name or "real"
        if result.dtype == object:
            return {name: result.to_numpy()}
        return {name: pd.to_numeric(result, errors="coerce").to_numpy(dtype=np.float64)}
    outputs: dict[str, np.ndarray] = {}
    for col in result.columns:
        if result[col].dtype == object:
            outputs[col] = result[col].to_numpy()
        else:
            outputs[col] = pd.to_numeric(result[col], errors="coerce").to_numpy(dtype=np.float64)
    return outputs


def _validate_smc(result: IndicatorResult, input_length: int) -> Any:
    """SMC outputs may include string labels — validate numeric columns only."""
    from indicator_testing.models import Check, ValidationResult

    numeric_outputs = {k: v for k, v in result.outputs.items() if v.dtype != object}

    if not numeric_outputs:
        return ValidationResult(
            passed=True,
            checks=[Check("outputs_present", True, "Label-only SMC outputs (no numeric validation).")],
        )

    numeric_result = IndicatorResult(
        name=result.name,
        group=result.group,
        params_used=result.params_used,
        outputs=numeric_outputs,
        warmup_bars=result.warmup_bars,
        status=result.status,
        message=result.message,
    )
    base = validate_indicator_result(numeric_result, input_length)
    if base.passed:
        return base

    has_signal = False
    for arr in numeric_outputs.values():
        finite = arr[np.isfinite(arr)]
        if len(finite) and np.any(finite != 0):
            has_signal = True
            break

    if has_signal:
        checks = [c for c in base.checks if c.name not in {"post_warmup_data", "not_all_nan"}]
        checks.extend([
            Check("post_warmup_data", True, "SMC event/signal columns present."),
            Check("not_all_nan", True, "SMC outputs valid (sparse events expected)."),
        ])
        passed = all(c.passed for c in checks if c.name not in {"range"})
        return ValidationResult(passed=passed, checks=checks)
    return base


def run_smc_indicator(
    name: str,
    df: pd.DataFrame,
    registry: SmcIndicatorRegistry | None = None,
    params: dict[str, Any] | None = None,
    *,
    validate: bool = True,
) -> IndicatorResult:
    reg = registry or SmcIndicatorRegistry()
    info = reg.get(name)
    params_used = dict(info.default_params)
    if params:
        params_used.update(params)

    try:
        raw = info.func(df, **params_used)
        outputs = _to_output_dict(raw)
        warmup = compute_warmup_bars(
            {k: v for k, v in outputs.items() if v.dtype != object},
            is_pattern=False,
        )
        if warmup is None:
            warmup = 0
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
            result.validation = _validate_smc(result, len(df))
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


def run_smc_batch(
    df: pd.DataFrame,
    registry: SmcIndicatorRegistry | None = None,
    *,
    validate: bool = True,
) -> list[IndicatorResult]:
    reg = registry or SmcIndicatorRegistry()
    return [run_smc_indicator(info.name, df, reg, validate=validate) for info in reg.iter_ordered()]
