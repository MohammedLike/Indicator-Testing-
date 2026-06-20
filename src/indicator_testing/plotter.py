from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from indicator_testing.models import IndicatorResult
from indicator_testing.runner import run_indicator

OVERLAP_GROUP = "Overlap Studies"
MOMENTUM_GROUP = "Momentum Indicators"
VOLATILITY_GROUP = "Volatility Indicators"
VOLUME_GROUP = "Volume Indicators"
PATTERN_GROUP = "Pattern Recognition"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_charts_dir() -> Path:
    return _project_root() / "outputs" / "charts"


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name)


def _format_params(params: dict) -> str:
    if not params:
        return ""
    parts = [f"{k}={v}" for k, v in params.items()]
    return ", ".join(parts)


def plot_indicator(
    df: pd.DataFrame,
    result: IndicatorResult,
    output_path: Path | None = None,
    *,
    show: bool = False,
) -> Path:
    if result.status != "success" or not result.outputs:
        raise ValueError(f"Cannot plot indicator with status={result.status}: {result.message}")

    dates = df.index
    close = df["close"]
    group = result.group
    params_str = _format_params(result.params_used)
    title = f"{result.name} ({params_str})" if params_str else result.name
    subtitle = f"{dates[0].date()} to {dates[-1].date()}"

    if group == PATTERN_GROUP:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, close, label="Close", color="black", linewidth=1)
        out_key = next(iter(result.outputs))
        pattern = result.outputs[out_key]
        hits = pattern != 0
        if hits.any():
            ax.scatter(
                dates[hits],
                close[hits],
                color="red",
                marker="^",
                s=40,
                label="Pattern",
                zorder=5,
            )
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
    elif group == OVERLAP_GROUP:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, close, label="Close", color="black", linewidth=1, alpha=0.8)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, (key, arr) in enumerate(result.outputs.items()):
            ax.plot(dates, arr, label=key, color=colors[i % len(colors)], linewidth=1)
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
    elif group == VOLUME_GROUP and "volume" in df.columns:
        fig, (ax_price, ax_vol, ax_ind) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        ax_price.plot(dates, close, color="black", linewidth=1)
        ax_price.set_ylabel("Close")
        ax_vol.bar(dates, df["volume"], color="gray", alpha=0.5, width=20)
        ax_vol.set_ylabel("Volume")
        for key, arr in result.outputs.items():
            ax_ind.plot(dates, arr, label=key, linewidth=1)
        ax_ind.set_ylabel(result.name)
        ax_ind.legend(loc="upper left")
    else:
        fig, (ax_price, ax_ind) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax_price.plot(dates, close, color="black", linewidth=1)
        ax_price.set_ylabel("Close")
        for key, arr in result.outputs.items():
            ax_ind.plot(dates, arr, label=key, linewidth=1)
        ax_ind.set_ylabel(result.name)
        ax_ind.legend(loc="upper left")
        if group == MOMENTUM_GROUP and result.name == "RSI":
            ax_ind.axhline(70, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            ax_ind.axhline(30, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
            ax_ind.set_ylim(-5, 105)

    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
    fig.tight_layout()

    if output_path is None:
        charts_dir = default_charts_dir() / _safe_filename(group)
        charts_dir.mkdir(parents=True, exist_ok=True)
        output_path = charts_dir / f"{result.name}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_from_csv(
    csv_path: str | Path,
    indicator: str,
    output_path: Path | None = None,
    *,
    show: bool = False,
) -> Path:
    from indicator_testing.data_loader import load_monthly_ohlc

    df = load_monthly_ohlc(csv_path, warn_short=False)
    result = run_indicator(indicator, df)
    return plot_indicator(df, result, output_path, show=show)
