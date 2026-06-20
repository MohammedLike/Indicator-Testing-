from __future__ import annotations

from pathlib import Path

import pytest

from indicator_testing.data_loader import load_ohlc
from indicator_testing.smc_indicators import SmcIndicatorRegistry
from indicator_testing.smc_runner import run_smc_batch, run_smc_indicator


def _sample_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "sample_monthly.csv"


@pytest.fixture
def df():
    return load_ohlc(_sample_csv(), warn_short=False)


@pytest.fixture
def registry():
    return SmcIndicatorRegistry()


def test_registry_count(registry):
    assert len(registry.all_names()) == 17


def test_fvg(df, registry):
    result = run_smc_indicator("FVG", df, registry)
    assert result.status == "success"
    assert "fvg_bull" in result.outputs


def test_swing_highs(df, registry):
    result = run_smc_indicator("SWING_HIGH", df, registry)
    assert result.status == "success"
    assert result.outputs["is_swing_high"].sum() >= 0


def test_bos(df, registry):
    result = run_smc_indicator("BOS", df, registry)
    assert result.status == "success"
    assert "bos_bull" in result.outputs


def test_all_smc_smoke(df, registry):
    results = run_smc_batch(df, registry)
    assert len(results) == len(registry.all_names())
    errors = [r for r in results if r.status == "error"]
    assert not errors, [(r.name, r.message) for r in errors]
