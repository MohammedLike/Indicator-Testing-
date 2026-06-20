# TA-Lib Monthly OHLC Indicator Testing

Test every [TA-Lib](https://github.com/TA-Lib/ta-lib-python) indicator one-by-one against your **monthly OHLC CSV** data. The toolkit validates outputs, generates charts for visual review, and keeps golden reference snapshots for regression testing.

## Prerequisites

### 1. Python 3.10+

### 2. TA-Lib C library (Windows)

The Python package requires the TA-Lib C library.

1. Download the Windows installer or binaries from [TA-Lib releases](https://github.com/TA-Lib/ta-lib/releases).
2. Install the C library (add `bin` to your PATH if needed).
3. Install the Python wrapper:

```powershell
pip install TA-Lib
```

If building from source fails, try a prebuilt wheel matching your Python version from [Christoph Gohlke's wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) or use WSL/Docker.

## Install

```powershell
cd "D:\Projects\Indicator Testing"
pip install -e ".[dev]"
```

## CSV format

Place your files in `data/user/` (gitignored). Required columns (case-insensitive):

| Column | Aliases | Required |
|--------|---------|----------|
| date | datetime, time, timestamp, month | Yes |
| open | o | Yes |
| high | h | Yes |
| low | l | Yes |
| close | c, adj_close | Yes |
| volume | vol, v | Optional (needed for volume indicators) |

Example:

```csv
date,open,high,low,close,volume
2020-01-31,100,105,98,102,1000000
2020-02-29,102,108,101,106,1100000
```

**Recommendations:**

- Use **60+ monthly bars** so default lookback periods (e.g. RSI 14, SMA 20) produce meaningful output after warmup.
- Dates are sorted ascending; duplicate dates keep the last row.
- OHLC sanity is checked (`high >= open/close/low`, etc.).

## Quick start

```powershell
# List all indicators by group
indicator-testing list

# Run a single indicator
indicator-testing run --csv data/user/your_symbol.csv --indicator RSI

# Run every indicator (validate only)
indicator-testing run --csv data/user/your_symbol.csv --all

# Run a group
indicator-testing run --csv data/user/your_symbol.csv --group "Momentum Indicators"

# Plot one indicator
indicator-testing plot --csv data/user/your_symbol.csv --indicator MACD

# Run all + generate charts
indicator-testing run --csv data/sample_monthly.csv --all --plot
```

Reports are written to `outputs/reports/` as JSON and CSV. Charts go to `outputs/charts/{group}/{indicator}.png`.

## Warmup NaNs on monthly data

TA-Lib fills the first *N* bars with `NaN` where *N* depends on the indicator lookback (e.g. SMA 20 → first 19 values are NaN). On monthly data with short history, some indicators may not produce finite values before the series ends — the validator reports this clearly.

Volume indicators (`OBV`, `AD`, `ADOSC`, etc.) are **skipped** if your CSV has no volume column.

## Parameter overrides

Edit [`config/defaults.yaml`](config/defaults.yaml) to override TA-Lib defaults per indicator:

```yaml
RSI:
  timeperiod: 14

BBANDS:
  timeperiod: 20
  nbdevup: 2.0
  nbdevdn: 2.0
```

## Golden reference tests

A fixed [`data/sample_monthly.csv`](data/sample_monthly.csv) (120 bars) drives regression tests. Golden snapshots live in `outputs/golden/`.

```powershell
# Regenerate golden files after TA-Lib upgrade or intentional changes
python scripts/generate_golden.py

# Or via CLI
indicator-testing golden update --all

# Run tests
pytest -v
```

To add a new golden indicator, add its name to `GOLDEN_INDICATORS` in `src/indicator_testing/golden.py`, then run `golden update --indicator NAME`.

## Project layout

```
config/defaults.yaml       # parameter overrides
data/sample_monthly.csv    # fixed test dataset
data/user/                 # your CSVs
src/indicator_testing/     # core package
outputs/charts/            # generated plots
outputs/reports/           # run summaries
outputs/golden/            # reference JSON snapshots
tests/                     # pytest suite
```

## Example session

```powershell
indicator-testing list --group "Overlap Studies"
indicator-testing run --csv data/sample_monthly.csv --indicator BBANDS
indicator-testing plot --csv data/sample_monthly.csv --indicator RSI
indicator-testing report --csv data/sample_monthly.csv --all
pytest -v
```

## Jupyter notebook (interactive one-by-one testing)

Open [`notebooks/indicator_tester.ipynb`](notebooks/indicator_tester.ipynb) for **one runnable cell per indicator** (158 blocks, no dropdown).

Regenerate after TA-Lib updates:

```powershell
python scripts/generate_indicator_notebook.py
```

### Pandas indicators (no TA-Lib)

Open [`notebooks/pandas_indicator_tester.ipynb`](notebooks/pandas_indicator_tester.ipynb) — **37 indicators** built with pure pandas/numpy (SMA, EMA, RSI, MACD, BBANDS, OBV, VWAP, etc.), one runnable cell each.

```powershell
python scripts/generate_pandas_indicator_notebook.py
jupyter lab notebooks/pandas_indicator_tester.ipynb
```

```powershell
pip install -e ".[notebook]"
jupyter lab notebooks/indicator_tester.ipynb
```

## Notes

- **Math Transform** indicators (e.g. `ACOS`, `ASIN`) applied to stock prices often return all `NaN` because prices are outside `[-1, 1]` — this is expected and validation treats it as out-of-domain, not a failure.
- **MAVP** uses a synthetic constant `periods` array (20) when not provided in CSV.
- Pattern recognition outputs are integers: `0` = no pattern, non-zero = pattern detected.
