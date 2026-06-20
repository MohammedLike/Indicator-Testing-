from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "datetime", "time", "timestamp", "month"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "adj_close", "adj close"],
    "volume": ["volume", "vol", "v"],
}

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close"}
MIN_RECOMMENDED_BARS = 60

RESAMPLE_RULES: dict[str, str] = {
    "monthly": "ME",
    "daily": "D",
    "weekly": "W-FRI",
}


class DataLoadError(ValueError):
    pass


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                rename[lower_map[alias]] = canonical
                break

    df = df.rename(columns=rename)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataLoadError(
            f"Missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def _validate_ohlc(df: pd.DataFrame) -> None:
    if (df[["open", "high", "low", "close"]] < 0).any().any():
        raise DataLoadError("OHLC values must be non-negative.")

    high_ok = df["high"] >= df[["open", "close", "low"]].max(axis=1)
    low_ok = df["low"] <= df[["open", "close", "high"]].min(axis=1)
    if not high_ok.all():
        bad = df.index[~high_ok][:3]
        raise DataLoadError(f"high must be >= open, close, low. Bad rows: {list(bad)}")
    if not low_ok.all():
        bad = df.index[~low_ok][:3]
        raise DataLoadError(f"low must be <= open, close, high. Bad rows: {list(bad)}")


def _filter_symbol(df: pd.DataFrame, symbol: str | None) -> pd.DataFrame:
    if "symbol" not in df.columns:
        return df
    symbols = df["symbol"].dropna().unique()
    if symbol:
        mask = df["symbol"].astype(str).str.upper() == symbol.upper()
        if not mask.any():
            raise DataLoadError(
                f"Symbol '{symbol}' not found. Available: {list(symbols)}"
            )
        return df.loc[mask].drop(columns=["symbol"])
    if len(symbols) == 1:
        return df.drop(columns=["symbol"])
    raise DataLoadError(
        f"Multiple symbols in CSV: {list(symbols)}. Pass symbol= to filter."
    )


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    resampled = df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    _validate_ohlc(resampled)
    return resampled


def _infer_bar_seconds(index: pd.DatetimeIndex) -> float | None:
    if len(index) < 2:
        return None
    return abs((index[1] - index[0]).total_seconds())


def _detect_intraday(csv_path: Path, symbol: str | None) -> bool:
    raw = pd.read_csv(csv_path)
    if raw.empty:
        return False
    tmp = _filter_symbol(raw, symbol)
    tmp = _normalize_columns(tmp)
    tmp["date"] = pd.to_datetime(tmp["date"], utc=True, errors="coerce")
    tmp = tmp.sort_values("date")
    if len(tmp) < 2:
        return False
    delta = (tmp["date"].iloc[1] - tmp["date"].iloc[0]).total_seconds()
    return abs(delta) < 86400


def has_volume(df: pd.DataFrame) -> bool:
    return "volume" in df.columns and df["volume"].notna().any()


def load_ohlc(
    csv_path: str | Path,
    *,
    resample: str | None = None,
    symbol: str | None = None,
    min_bars: int = MIN_RECOMMENDED_BARS,
    warn_short: bool = True,
) -> pd.DataFrame:
    """Load OHLC CSV, optionally filter symbol and resample to daily/monthly bars."""
    path = Path(csv_path)
    if not path.exists():
        raise DataLoadError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise DataLoadError("CSV is empty.")

    df = _filter_symbol(df, symbol)
    df = _normalize_columns(df)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    if df["date"].isna().any():
        raise DataLoadError("One or more date values could not be parsed.")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise DataLoadError("OHLC columns contain non-numeric values.")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    dup_count = df["date"].duplicated(keep="last").sum()
    if dup_count:
        warnings.warn(
            f"Dropping {dup_count} duplicate timestamp(s), keeping last occurrence.",
            stacklevel=2,
        )
        df = df.drop_duplicates(subset="date", keep="last")

    df = df.sort_values("date").set_index("date")
    _validate_ohlc(df)

    if resample:
        key = resample.lower()
        if key not in RESAMPLE_RULES:
            raise DataLoadError(
                f"Unknown resample '{resample}'. Use: {', '.join(RESAMPLE_RULES)}"
            )
        df = _resample_ohlc(df, RESAMPLE_RULES[key])

    bar_seconds = _infer_bar_seconds(df.index)
    if bar_seconds is not None and bar_seconds < 86400 and resample is None:
        warnings.warn(
            f"Data looks intraday (~{int(bar_seconds)}s bars). "
            "Use --resample monthly or --resample daily for aggregated testing.",
            stacklevel=2,
        )

    if warn_short and len(df) < min_bars:
        freq_hint = f" ({resample})" if resample else ""
        warnings.warn(
            f"Only {len(df)} bars loaded{freq_hint}; {min_bars}+ bars recommended "
            "for most TA-Lib default lookbacks.",
            stacklevel=2,
        )

    return df


def load_monthly_ohlc(
    csv_path: str | Path,
    *,
    resample: str | None = None,
    symbol: str | None = None,
    min_bars: int = MIN_RECOMMENDED_BARS,
    warn_short: bool = True,
) -> pd.DataFrame:
    """Load OHLC CSV; auto-resamples intraday QuestDB exports to monthly bars."""
    path = Path(csv_path)
    auto_resample = resample
    if auto_resample is None and _detect_intraday(path, symbol):
        auto_resample = "monthly"
    return load_ohlc(
        path,
        resample=auto_resample,
        symbol=symbol,
        min_bars=min_bars,
        warn_short=warn_short,
    )


def describe_ohlc(df: pd.DataFrame) -> dict:
    """Return summary stats for loaded OHLC data."""
    bar_seconds = _infer_bar_seconds(df.index)
    freq = "unknown"
    if bar_seconds is not None:
        if bar_seconds < 120:
            freq = "intraday (minute/sub-minute)"
        elif bar_seconds < 86400:
            freq = "intraday"
        elif bar_seconds < 86400 * 8:
            freq = "daily"
        else:
            freq = "weekly or longer"

    return {
        "bars": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "inferred_frequency": freq,
        "has_volume": has_volume(df),
        "close_min": float(df["close"].min()),
        "close_max": float(df["close"].max()),
    }
