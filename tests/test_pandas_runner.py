from __future__ import annotations

from pathlib import Path

import pytest

from indicator_testing.data_loader import load_ohlc
from indicator_testing.pandas_indicators import PandasIndicatorRegistry
from indicator_testing.pandas_runner import run_pandas_batch, run_pandas_indicator


def _sample_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "sample_monthly.csv"


@pytest.fixture
def df():
    return load_ohlc(_sample_csv(), warn_short=False)


@pytest.fixture
def registry():
    return PandasIndicatorRegistry()


def test_registry_count(registry):
    assert len(registry.all_names()) == 37


def test_run_rsi(df, registry):
    result = run_pandas_indicator("RSI", df, registry)
    assert result.status == "success"
    assert "rsi" in result.outputs
    assert result.validation and result.validation.passed


def test_run_macd_multi_output(df, registry):
    result = run_pandas_indicator("MACD", df, registry)
    assert result.status == "success"
    assert set(result.outputs.keys()) == {"macd", "macdsignal", "macdhist"}


def test_volume_skipped_without_volume(df, registry):
    df_no_vol = df.drop(columns=["volume"])
    result = run_pandas_indicator("OBV", df_no_vol, registry)
    assert result.status == "skipped"


def test_all_pandas_indicators_smoke(df, registry):
    results = run_pandas_batch(df, registry)
    assert len(results) == len(registry.all_names())
    errors = [r for r in results if r.status == "error"]
    assert not errors, [(r.name, r.message) for r in errors]
    val_fail = [r for r in results if r.status == "success" and r.validation and not r.validation.passed]
    assert not val_fail, [(r.name, [c.message for c in r.validation.checks if not c.passed]) for r in val_fail]
