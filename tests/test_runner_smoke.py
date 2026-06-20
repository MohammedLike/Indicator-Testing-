from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from indicator_testing.data_loader import load_monthly_ohlc
from indicator_testing.registry import IndicatorRegistry
from indicator_testing.runner import run_batch, run_indicator


def _sample_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "sample_monthly.csv"


@pytest.fixture
def df():
    return load_monthly_ohlc(_sample_csv(), warn_short=False)


@pytest.fixture
def registry():
    return IndicatorRegistry()


def test_run_single_rsi(df, registry):
    result = run_indicator("RSI", df, registry)
    assert result.status == "success"
    assert "real" in result.outputs
    assert len(result.outputs["real"]) == len(df)
    assert result.validation is not None
    assert result.validation.passed


def test_run_macd_multi_output(df, registry):
    result = run_indicator("MACD", df, registry)
    assert result.status == "success"
    assert set(result.outputs.keys()) == {"macd", "macdsignal", "macdhist"}


def test_volume_indicator_skipped_without_volume(df, registry):
    df_no_vol = df.drop(columns=["volume"])
    result = run_indicator("OBV", df_no_vol, registry)
    assert result.status == "skipped"


def test_run_all_indicators_smoke(df, registry):
    results = run_batch(df, registry, run_all=True, validate=True)
    assert len(results) == len(registry.all_names())

    errors = [r for r in results if r.status == "error"]
    assert not errors, f"Errors: {[(r.name, r.message) for r in errors]}"

    successes = [r for r in results if r.status == "success"]
    assert len(successes) > 100

    validation_failures = [
        r for r in successes if r.validation and not r.validation.passed
    ]
    assert not validation_failures, [
        (r.name, [c.message for c in r.validation.checks if not c.passed])
        for r in validation_failures
    ]
