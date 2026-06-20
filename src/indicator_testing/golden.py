from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from indicator_testing.models import IndicatorResult
from indicator_testing.registry import IndicatorRegistry
from indicator_testing.runner import run_indicator

DEFAULT_CHECKPOINT_INDICES = [50, 60, 70, 80, 90, 100, 110]

GOLDEN_INDICATORS = [
    "SMA",
    "EMA",
    "RSI",
    "MACD",
    "BBANDS",
    "ATR",
    "STOCH",
    "ADX",
    "CCI",
    "MOM",
    "OBV",
    "AD",
    "WILLR",
    "ROC",
    "STDDEV",
    "LINEARREG",
    "CDLHAMMER",
    "CDLDOJI",
    "NATR",
    "APO",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_golden_dir() -> Path:
    return _project_root() / "outputs" / "golden"


def _serialize_value(value: Any) -> int | float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def build_snapshot(
    df: pd.DataFrame,
    result: IndicatorResult,
    checkpoint_indices: list[int] | None = None,
) -> dict[str, Any]:
    if result.status != "success":
        raise ValueError(f"Cannot snapshot failed result: {result.message}")

    indices = checkpoint_indices or DEFAULT_CHECKPOINT_INDICES
    warmup = result.warmup_bars or 0
    checkpoints: list[dict[str, Any]] = []

    for idx in indices:
        if idx >= len(df) or idx < warmup:
            continue
        row: dict[str, Any] = {
            "index": idx,
            "date": df.index[idx].strftime("%Y-%m-%d"),
        }
        for out_name, arr in result.outputs.items():
            row[out_name] = _serialize_value(arr[idx])
        checkpoints.append(row)

    return {
        "indicator": result.name,
        "group": result.group,
        "params": result.params_used,
        "warmup_bars": result.warmup_bars,
        "checkpoints": checkpoints,
    }


def write_snapshot(snapshot: dict[str, Any], golden_dir: Path | None = None) -> Path:
    out_dir = golden_dir or default_golden_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{snapshot['indicator']}.json"
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return path


def load_snapshot(name: str, golden_dir: Path | None = None) -> dict[str, Any]:
    path = (golden_dir or default_golden_dir()) / f"{name.upper()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Golden snapshot not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_golden(
    df: pd.DataFrame,
    *,
    indicator: str | None = None,
    run_all: bool = False,
    registry: IndicatorRegistry | None = None,
    golden_dir: Path | None = None,
) -> list[Path]:
    reg = registry or IndicatorRegistry()
    names = GOLDEN_INDICATORS if run_all else [reg.get(indicator or "").name]
    written: list[Path] = []

    for name in names:
        result = run_indicator(name, df, reg, validate=False)
        if result.status != "success":
            continue
        snapshot = build_snapshot(df, result)
        if snapshot["checkpoints"]:
            written.append(write_snapshot(snapshot, golden_dir))

    return written


def compare_with_golden(
    df: pd.DataFrame,
    name: str,
    registry: IndicatorRegistry | None = None,
    golden_dir: Path | None = None,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple[bool, list[str]]:
    reg = registry or IndicatorRegistry()
    snapshot = load_snapshot(name, golden_dir)
    result = run_indicator(name, df, reg, validate=False)

    if result.status != "success":
        return False, [f"Run failed: {result.message}"]

    errors: list[str] = []
    for cp in snapshot["checkpoints"]:
        idx = cp["index"]
        for key, expected in cp.items():
            if key in {"index", "date"}:
                continue
            if key not in result.outputs:
                errors.append(f"Missing output key {key} at index {idx}")
                continue
            actual = result.outputs[key][idx]
            if expected is None:
                if not (np.isnan(actual) if isinstance(actual, float) else False):
                    errors.append(f"{key}[{idx}]: expected NaN, got {actual}")
                continue
            if isinstance(expected, int) or (
                snapshot.get("group") == "Pattern Recognition"
            ):
                if int(actual) != int(expected):
                    errors.append(f"{key}[{idx}]: expected {expected}, got {actual}")
            elif not np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
                errors.append(f"{key}[{idx}]: expected {expected}, got {actual}")

    return len(errors) == 0, errors
