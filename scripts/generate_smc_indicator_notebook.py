"""Generate smc_indicator_tester.ipynb — one cell per SMC indicator."""

from __future__ import annotations

import json
from pathlib import Path

from indicator_testing.smc_indicators import SmcIndicatorRegistry


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
    registry = SmcIndicatorRegistry()
    groups = registry.groups()

    toc_lines = ["## Table of contents", ""]
    for group in groups:
        anchor = group.lower().replace(" ", "-").replace("/", "")
        toc_lines.append(f"- [{group}](#{anchor})")

    cells = [
        _md([
            "# SMC Indicator Tester (Smart Money Concepts)",
            "",
            "Test **Smart Money Concepts** indicators one cell at a time.",
            "",
            "Includes: swing points, BOS, CHoCH, FVG, order blocks, liquidity sweeps, premium/discount.",
            "",
            "```bash",
            "pip install -e \".[notebook]\"",
            "jupyter lab notebooks/smc_indicator_tester.ipynb",
            "```",
            "",
            "1. Run setup cells once",
            "2. Jump to any SMC block (Ctrl+F e.g. `FVG`, `BOS`)",
            "3. Run only that cell",
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
            "from indicator_testing.smc_indicators import SmcIndicatorRegistry",
            "from indicator_testing.smc_runner import run_smc_indicator",
            "",
            "print(f\"Project root: {PROJECT_ROOT}\")",
        ]),
        _md(["## Configuration"]),
        _code([
            "CSV_PATH = PROJECT_ROOT / \"questdb-query-1781940224994.csv\"",
            "RESAMPLE = \"daily\"   # SMC works well on daily+; try \"none\" for 1-min",
            "SYMBOL = None",
            "SWING_LENGTH = 2     # bars each side for swing detection",
        ]),
        _code([
            "resample_arg = None if RESAMPLE == \"none\" else RESAMPLE",
            "df = load_ohlc(CSV_PATH, resample=resample_arg, symbol=SYMBOL, warn_short=False)",
            "registry = SmcIndicatorRegistry()",
            "info = describe_ohlc(df)",
            "",
            "print(f\"Loaded {info['bars']} bars | {info['inferred_frequency']}\")",
            "print(f\"SMC indicators: {len(registry.all_names())}\")",
            "df.tail(3)",
        ]),
        _md(["## Helpers"]),
        _code([
            "def show_validation(result):",
            "    meta = registry.get(result.name)",
            "    print(f\"{'='*60}\")",
            "    print(f\"{result.name}  |  {result.group}  (SMC)\")",
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
            "def result_to_frame(result, n_tail=20):",
            "    if result.status != \"success\" or not result.outputs:",
            "        return pd.DataFrame()",
            "    out = pd.DataFrame({\"close\": df[\"close\"]})",
            "    for k, arr in result.outputs.items():",
            "        out[k] = arr",
            "    out.index = df.index",
            "    # show rows with any signal",
            "    numeric = out.select_dtypes(include=\"number\")",
            "    if len(numeric.columns):",
            "        hits = numeric[(numeric != 0).any(axis=1) | numeric.notna().all(axis=1)]",
            "        if len(hits):",
            "            return hits.tail(n_tail)",
            "    return out.tail(n_tail)",
            "",
            "",
            "def plot_smc(result):",
            "    if result.status != \"success\" or not result.outputs:",
            "        print(\"Nothing to plot.\")",
            "        return",
            "    fig, ax = plt.subplots(figsize=(12, 5))",
            "    ax.plot(df.index, df[\"close\"], color=\"black\", lw=1, label=\"Close\")",
            "    for key, arr in result.outputs.items():",
            "        if arr.dtype == object:",
            "            continue",
            "        mask = pd.Series(arr, index=df.index).fillna(0) != 0",
            "        if mask.any():",
            "            ax.scatter(df.index[mask], df[\"close\"][mask], s=30, label=key)",
            "        elif \"top\" in key or \"bottom\" in key or \"high\" in key or \"low\" in key or key in {\"equilibrium\", \"range_high\", \"range_low\"}:",
            "            ax.plot(df.index, arr, lw=0.8, alpha=0.7, label=key)",
            "    ax.legend(loc=\"upper left\", fontsize=8)",
            "    ax.set_title(f\"{result.name} (SMC)\")",
            "    fig.tight_layout()",
            "    plt.show()",
            "",
            "",
            "def test_smc_indicator(name, params=None):",
            "    p = dict(params or {})",
            "    if \"length\" not in p and name not in {\"FVG\", \"FVG_BULL\", \"FVG_BEAR\"}:",
            "        p.setdefault(\"length\", SWING_LENGTH)",
            "    result = run_smc_indicator(name, df, registry, params=p)",
            "    show_validation(result)",
            "    display(result_to_frame(result))",
            "    plot_smc(result)",
            "    return result",
        ]),
        _md(toc_lines),
    ]

    for group, names in groups.items():
        anchor = group.lower().replace(" ", "-").replace("/", "")
        cells.append(_md([f"<a id=\"{anchor}\"></a>", f"## {group} ({len(names)})"]))
        for name in names:
            info = registry.get(name)
            cells.append(
                _md([
                    f"### {name}",
                    f"Outputs: `{info.output_names}` · Defaults: `{info.default_params}`",
                ])
            )
            cells.append(_code([f'test_smc_indicator("{name}")']))

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
    out = root / "notebooks" / "smc_indicator_tester.ipynb"
    notebook = build_notebook()
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
