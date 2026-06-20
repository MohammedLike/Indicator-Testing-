from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from indicator_testing.data_loader import DataLoadError, load_monthly_ohlc
from indicator_testing.models import IndicatorResult
from indicator_testing.registry import IndicatorRegistry
from indicator_testing.validator import validate_indicator_result


def _sample_csv() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "sample_monthly.csv"


def test_load_sample_csv():
    df = load_monthly_ohlc(_sample_csv(), warn_short=False)
    assert len(df) >= 60
    assert list(df.columns[:4]) == ["open", "high", "low", "close"]
    assert df.index.is_monotonic_increasing


def test_column_aliases(tmp_path: Path):
    csv = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29"],
            "O": [100, 102],
            "H": [105, 108],
            "L": [98, 101],
            "C": [102, 106],
        }
    ).to_csv(csv, index=False)
    df = load_monthly_ohlc(csv, warn_short=False)
    assert len(df) == 2
    assert "close" in df.columns


def test_invalid_ohlc_raises(tmp_path: Path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-31"],
            "open": [100],
            "high": [90],
            "low": [98],
            "close": [102],
        }
    ).to_csv(csv, index=False)
    with pytest.raises(DataLoadError):
        load_monthly_ohlc(csv, warn_short=False)


def test_sort_order(tmp_path: Path):
    csv = tmp_path / "unsorted.csv"
    pd.DataFrame(
        {
            "date": ["2020-03-31", "2020-01-31", "2020-02-29"],
            "open": [100, 100, 100],
            "high": [105, 105, 105],
            "low": [98, 98, 98],
            "close": [102, 102, 102],
        }
    ).to_csv(csv, index=False)
    df = load_monthly_ohlc(csv, warn_short=False)
    assert df.index[0] < df.index[-1]
