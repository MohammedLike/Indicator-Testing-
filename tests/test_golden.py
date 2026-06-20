from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from indicator_testing.data_loader import load_monthly_ohlc
from indicator_testing.golden import (
    GOLDEN_INDICATORS,
    compare_with_golden,
    default_golden_dir,
    load_snapshot,
)
from indicator_testing.models import IndicatorResult
from indicator_testing.validator import validate_indicator_result


def _sample_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "sample_monthly.csv"


@pytest.fixture
def df():
    return load_monthly_ohlc(_sample_csv(), warn_short=False)


@pytest.mark.parametrize("indicator", GOLDEN_INDICATORS)
def test_golden_checkpoints(df, indicator: str):
    golden_path = default_golden_dir() / f"{indicator}.json"
    if not golden_path.exists():
        pytest.skip(f"No golden file for {indicator}")

    ok, errors = compare_with_golden(df, indicator)
    assert ok, errors


def test_golden_files_exist():
    existing = list(default_golden_dir().glob("*.json"))
    assert len(existing) >= 10


def test_validator_fails_on_bad_length():
    result = IndicatorResult(
        name="RSI",
        group="Momentum Indicators",
        params_used={"timeperiod": 14},
        outputs={"real": np.array([1.0, 2.0])},
        warmup_bars=0,
        status="success",
        message="OK",
    )
    validation = validate_indicator_result(result, input_length=10)
    assert not validation.passed
    assert any(c.name == "length" and not c.passed for c in validation.checks)


def test_validator_passes_good_result(df):
    from indicator_testing.runner import run_indicator

    result = run_indicator("RSI", df)
    validation = validate_indicator_result(result, len(df))
    assert validation.passed
