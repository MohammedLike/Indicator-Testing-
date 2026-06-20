"""Generate indicator_tester.ipynb with one runnable cell per TA-Lib indicator."""

from __future__ import annotations

import json
from pathlib import Path

import talib
from talib import abstract


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
    groups = talib.get_function_groups()
    toc_lines = ["## Table of contents", ""]
    for group in sorted(groups.keys()):
        anchor = group.lower().replace(" ", "-")
        toc_lines.append(f"- [{group}](#{anchor})")

    cells = [
        _md([
            "# TA-Lib Indicator Tester (One-by-One)",
            "",
            "Run the **Setup** cells once, then run any indicator cell below on its own.",
            "",
            "**Install:**",
            "```bash",
            "pip install -e \".[notebook]\"",
            "jupyter lab notebooks/indicator_tester.ipynb",
            "```",
            "",
            "1. Run **Imports** → **Configuration** → **Load data** → **Helpers**",
            "2. Scroll to an indicator (e.g. `RSI`) and run **only that cell**",
            "3. Use Ctrl+F to jump to an indicator name",
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
            "from indicator_testing.plotter import plot_indicator",
            "from indicator_testing.registry import IndicatorRegistry",
            "from indicator_testing.runner import run_indicator",
            "",
            "print(f\"Project root: {PROJECT_ROOT}\")",
        ]),
        _md(["## Configuration", "", "Edit below, then re-run this cell."]),
        _code([
            "CSV_PATH = PROJECT_ROOT / \"questdb-query-1781940224994.csv\"",
            "# CSV_PATH = PROJECT_ROOT / \"data\" / \"sample_monthly.csv\"",
            "",
            "RESAMPLE = \"daily\"   # \"none\" | \"daily\" | \"monthly\" | \"weekly\"",
            "SYMBOL = None        # e.g. \"NIFTY\" if multiple symbols",
        ]),
        _code([
            "resample_arg = None if RESAMPLE == \"none\" else RESAMPLE",
            "df = load_ohlc(CSV_PATH, resample=resample_arg, symbol=SYMBOL, warn_short=False)",
            "registry = IndicatorRegistry()",
            "info = describe_ohlc(df)",
            "",
            "print(f\"Loaded {info['bars']} bars | {info['inferred_frequency']}\")",
            "print(f\"Range: {info['start']} -> {info['end']}\")",
            "print(f\"Close: {info['close_min']:.2f} - {info['close_max']:.2f} | Volume: {info['has_volume']}\")",
            "df.tail(3)",
        ]),
        _md(["## Helpers", "", "Run once after loading data."]),
        _code([
            "def show_validation(result):",
            "    info = registry.get(result.name)",
            "    print(f\"{'='*60}\")",
            "    print(f\"{result.name}  |  {result.group}\")",
            "    print(f\"Status: {result.status}  |  Warmup bars: {result.warmup_bars}\")",
            "    print(f\"Inputs: {info.input_names}  |  Outputs: {info.output_names}\")",
            "    print(f\"Params: {result.params_used}\")",
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
            "    warmup = result.warmup_bars or 0",
            "    out[\"_warmup\"] = [i < warmup for i in range(len(out))]",
            "    return out.tail(n_tail)",
            "",
            "",
            "def plot_inline(result):",
            "    if result.status != \"success\" or not result.outputs:",
            "        print(\"Nothing to plot.\")",
            "        return",
            "    chart_path = PROJECT_ROOT / \"outputs\" / \"charts\" / \"notebook_preview.png\"",
            "    chart_path.parent.mkdir(parents=True, exist_ok=True)",
            "    plot_indicator(df, result, chart_path, show=False)",
            "    img = plt.imread(chart_path)",
            "    plt.figure(figsize=(12, 5))",
            "    plt.imshow(img)",
            "    plt.axis(\"off\")",
            "    plt.title(result.name)",
            "    plt.show()",
            "",
            "",
            "def test_indicator(name, params=None):",
            "    result = run_indicator(name, df, registry, params=params)",
            "    show_validation(result)",
            "    display(result_to_frame(result))",
            "    plot_inline(result)",
            "    return result",
        ]),
        _md(toc_lines),
    ]

    for group in sorted(groups.keys()):
        names = sorted(groups[group])
        anchor = group.lower().replace(" ", "-")
        cells.append(_md([f"<a id=\"{anchor}\"></a>", f"## {group} ({len(names)})"]))
        for name in names:
            info = abstract.Function(name)
            inputs = list(info.input_names.values()) if hasattr(info.input_names, "values") else []
            cells.append(_md([f"### {name}", f"Inputs: `{inputs}` · Outputs: `{list(info.output_names)}`"]))
            cells.append(_code([f"test_indicator(\"{name}\")"]))

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
    out = root / "notebooks" / "indicator_tester.ipynb"
    notebook = build_notebook()
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
    print(f"Wrote {out}")
    print(f"Cells: {len(notebook['cells'])} ({n_md} markdown, {n_code} code)")


if __name__ == "__main__":
    main()
