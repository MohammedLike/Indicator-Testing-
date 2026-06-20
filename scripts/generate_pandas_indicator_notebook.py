"""Generate pandas_indicator_tester.ipynb — one cell per pandas indicator."""

from __future__ import annotations

import json
from pathlib import Path

from indicator_testing.pandas_indicators import PandasIndicatorRegistry


def _md(lines: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in lines]}


def _code(lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def build_notebook() -> dict:
    registry = PandasIndicatorRegistry()
    groups = registry.groups()

    toc_lines = ["## Table of contents", ""]
    for group in groups:
        anchor = group.lower().replace(" ", "-")
        toc_lines.append(f"- [{group}](#{anchor})")

    cells = [
        _md([
            "# Pandas Indicator Tester (One-by-One)",
            "",
            "Pure **pandas/numpy** technical indicators — no TA-Lib required.",
            "",
            "Run setup cells once, then run any indicator cell below.",
            "",
            "```bash",
            "pip install -e \".[notebook]\"",
            "jupyter lab notebooks/pandas_indicator_tester.ipynb",
            "```",
        ]),
        _code([
            "%matplotlib inline",
            "",
            "import sys",
            "from pathlib import Path",
            "",
            "import matplotlib.pyplot as plt",
            "import pandas as pd",
            "from IPython.display import display",
            "",
            "PROJECT_ROOT = Path.cwd()",
            "if PROJECT_ROOT.name == \"notebooks\":",
            "    PROJECT_ROOT = PROJECT_ROOT.parent",
            "if str(PROJECT_ROOT / \"src\") not in sys.path:",
            "    sys.path.insert(0, str(PROJECT_ROOT / \"src\"))",
            "",
            "from indicator_testing.data_loader import describe_ohlc, load_ohlc",
            "from indicator_testing.pandas_indicators import PandasIndicatorRegistry",
            "from indicator_testing.pandas_runner import run_pandas_indicator",
            "",
            "print(f\"Project root: {PROJECT_ROOT}\")",
        ]),
        _md(["## Configuration"]),
        _code([
            "CSV_PATH = PROJECT_ROOT / \"questdb-query-1781940224994.csv\"",
            "RESAMPLE = \"none\"   # \"none\" | \"daily\" | \"monthly\" | \"weekly\"",
            "SYMBOL = None",
        ]),
        _code([
            "resample_arg = None if RESAMPLE == \"none\" else RESAMPLE",
            "df = load_ohlc(CSV_PATH, resample=resample_arg, symbol=SYMBOL, warn_short=False)",
            "registry = PandasIndicatorRegistry()",
            "info = describe_ohlc(df)",
            "",
            "print(f\"Loaded {info['bars']} bars | {info['inferred_frequency']}\")",
            "print(f\"Pandas indicators: {len(registry.all_names())}\")",
            "df.tail(3)",
        ]),
        _md(["## Helpers"]),
        _code([
            "def show_validation(result):",
            "    meta = registry.get(result.name)",
            "    print(f\"{'='*60}\")",
            "    print(f\"{result.name}  |  {result.group}  (pandas)\")",
            "    print(f\"Status: {result.status}  |  Warmup: {result.warmup_bars}\")",
            "    print(f\"Outputs: {meta.output_names}  |  Params: {result.params_used}\")",
            "    if result.message and result.status != \"success\":",
            "        print(f\"Message: {result.message}\")",
            "    if result.validation:",
            "        tag = \"PASS\" if result.validation.passed else \"FAIL\"",
            "        print(f\"Validation: {tag}\")",
            "        for c in result.validation.checks:",
            "            mark = \"OK\" if c.passed else \"X\"",
            "            print(f\"  [{mark}] {c.name}: {c.message}\")",
            "    print(f\"{'='*60}\")",
            "",
            "",
            "def result_to_frame(result, n_tail=15):",
            "    if result.status != \"success\" or not result.outputs:",
            "        return pd.DataFrame()",
            "    out = pd.DataFrame({\"close\": df[\"close\"]})",
            "    for k, arr in result.outputs.items():",
            "        out[k] = arr",
            "    out.index = df.index",
            "    return out.tail(n_tail)",
            "",
            "",
            "def plot_pandas(result):",
            "    if result.status != \"success\" or not result.outputs:",
            "        print(\"Nothing to plot.\")",
            "        return",
            "    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)",
            "    axes[0].plot(df.index, df[\"close\"], label=\"Close\", color=\"black\", lw=1)",
            "    axes[0].set_ylabel(\"Price\")",
            "    axes[0].legend(loc=\"upper left\")",
            "    for key, arr in result.outputs.items():",
            "        axes[1].plot(df.index, arr, label=key, lw=1)",
            "    axes[1].set_ylabel(result.name)",
            "    axes[1].legend(loc=\"upper left\")",
            "    fig.suptitle(f\"{result.name} (pandas) — {result.params_used}\")",
            "    fig.tight_layout()",
            "    plt.show()",
            "",
            "",
            "def test_pandas_indicator(name, params=None):",
            "    result = run_pandas_indicator(name, df, registry, params=params)",
            "    show_validation(result)",
            "    display(result_to_frame(result))",
            "    plot_pandas(result)",
            "    return result",
        ]),
        _md(toc_lines),
    ]

    for group, names in groups.items():
        anchor = group.lower().replace(" ", "-")
        cells.append(_md([f"<a id=\"{anchor}\"></a>", f"## {group} ({len(names)})"]))
        for name in names:
            info = registry.get(name)
            vol = " · requires volume" if info.requires_volume else ""
            cells.append(
                _md([
                    f"### {name}",
                    f"Outputs: `{info.output_names}` · Defaults: `{info.default_params}`{vol}",
                ])
            )
            cells.append(_code([f'test_pandas_indicator("{name}")']))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "notebooks" / "pandas_indicator_tester.ipynb"
    notebook = build_notebook()
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
