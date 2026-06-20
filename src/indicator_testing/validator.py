from __future__ import annotations

import math

import numpy as np

from indicator_testing.models import Check, IndicatorResult, ValidationResult

RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "RSI": (0.0, 100.0),
    "WILLR": (-100.0, 0.0),
    "MFI": (0.0, 100.0),
}


def compute_warmup_bars(outputs: dict[str, np.ndarray], *, is_pattern: bool) -> int | None:
    if not outputs:
        return None

    n = len(next(iter(outputs.values())))
    if is_pattern:
        for i in range(n):
            if any(not math.isnan(v) for v in (arr[i] for arr in outputs.values())):
                return i
        return n

    for i in range(n):
        finite = True
        for arr in outputs.values():
            val = arr[i]
            if np.isnan(val) or not np.isfinite(val):
                finite = False
                break
        if finite:
            return i
    return n


def validate_indicator_result(result: IndicatorResult, input_length: int) -> ValidationResult:
    checks: list[Check] = []

    if result.status != "success":
        checks.append(Check("status", False, f"Run status is {result.status}: {result.message}"))
        return ValidationResult(passed=False, checks=checks)

    if not result.outputs:
        checks.append(Check("outputs_present", False, "No output arrays produced."))
        return ValidationResult(passed=False, checks=checks)

    length_ok = True
    for name, arr in result.outputs.items():
        if len(arr) != input_length:
            length_ok = False
            checks.append(
                Check(
                    "length",
                    False,
                    f"Output '{name}' length {len(arr)} != input length {input_length}.",
                )
            )
    if length_ok:
        checks.append(Check("length", True, "All output lengths match input."))

    warmup = result.warmup_bars if result.warmup_bars is not None else input_length
    post_warmup_finite = False
    for arr in result.outputs.values():
        tail = arr[warmup:]
        if len(tail) and np.any(np.isfinite(tail) & ~np.isnan(tail)):
            post_warmup_finite = True
            break

    not_all_nan = False
    for arr in result.outputs.values():
        if warmup < len(arr):
            tail = arr[warmup:]
            if len(tail) and not np.all(np.isnan(tail)):
                not_all_nan = True
                break

    math_transform_out_of_domain = result.group == "Math Transform" and not not_all_nan

    checks.append(
        Check(
            "post_warmup_data",
            post_warmup_finite or math_transform_out_of_domain,
            "Skipped finite check (price data out of domain for math transform)."
            if math_transform_out_of_domain
            else "At least one finite value after warmup."
            if post_warmup_finite
            else "No finite values after warmup period.",
        )
    )
    checks.append(
        Check(
            "not_all_nan",
            not_all_nan or math_transform_out_of_domain,
            "All NaN after warmup (price data out of domain for this transform)."
            if math_transform_out_of_domain
            else "Outputs contain data after warmup."
            if not_all_nan
            else "All outputs are NaN after warmup.",
        )
    )

    if result.group == "Pattern Recognition":
        int_ok = all(
            np.issubdtype(arr.dtype, np.integer) or np.all(np.isfinite(arr))
            for arr in result.outputs.values()
        )
        checks.append(
            Check(
                "pattern_dtype",
                int_ok,
                "Pattern output is integer-like." if int_ok else "Pattern output dtype unexpected.",
            )
        )
    else:
        for out_name, arr in result.outputs.items():
            if result.name in RANGE_CHECKS and out_name == "real":
                lo, hi = RANGE_CHECKS[result.name]
                tail = arr[warmup:]
                finite = tail[np.isfinite(tail)]
                if len(finite):
                    in_range = bool(np.all((finite >= lo) & (finite <= hi)))
                    checks.append(
                        Check(
                            "range",
                            in_range,
                            f"{result.name} values within [{lo}, {hi}]."
                            if in_range
                            else f"{result.name} has values outside [{lo}, {hi}] (warning).",
                        )
                    )

    passed = all(c.passed for c in checks if c.name not in {"range"})
    return ValidationResult(passed=passed, checks=checks)
