"""Regenerate all golden reference snapshots from sample CSV."""

from pathlib import Path

from indicator_testing.data_loader import load_monthly_ohlc
from indicator_testing.golden import default_golden_dir, update_golden


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "sample_monthly.csv"
    df = load_monthly_ohlc(csv_path, warn_short=False)
    paths = update_golden(df, run_all=True, golden_dir=default_golden_dir())
    print(f"Wrote {len(paths)} golden snapshot(s) to {default_golden_dir()}")


if __name__ == "__main__":
    main()
