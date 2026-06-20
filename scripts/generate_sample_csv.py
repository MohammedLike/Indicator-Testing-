"""Generate sample monthly OHLC CSV for tests and golden snapshots."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_monthly(n_bars: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_bars, freq="ME")

    close = np.empty(n_bars, dtype=float)
    close[0] = 100.0
    for i in range(1, n_bars):
        close[i] = close[i - 1] * (1 + rng.normal(0.005, 0.03))

    trend = np.sin(np.linspace(0, 4 * math.pi, n_bars)) * 5
    close = close + trend

    open_ = close * (1 + rng.normal(0, 0.005, n_bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.005, 0.02, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.005, 0.02, n_bars))
    volume = rng.integers(500_000, 2_000_000, n_bars).astype(float)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "sample_monthly.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_monthly()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
