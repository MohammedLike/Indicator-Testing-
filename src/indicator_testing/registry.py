from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import talib
import yaml
from talib import abstract

from indicator_testing.data_loader import has_volume

PATTERN_GROUP = "Pattern Recognition"
VOLUME_GROUP = "Volume Indicators"

VOLUME_INPUT_INDICATORS = {
    "AD",
    "ADOSC",
    "OBV",
}


@dataclass(frozen=True)
class IndicatorInfo:
    name: str
    group: str
    input_names: list[str]
    output_names: list[str]
    default_params: dict[str, Any]
    requires_volume: bool
    is_pattern: bool


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_param_overrides(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = config_path or (_project_root() / "config" / "defaults.yaml")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {k.upper(): v for k, v in data.items() if isinstance(v, dict)}


def _collect_required_inputs(func: abstract.Function) -> list[str]:
    required: list[str] = []
    for key, value in func.input_names.items():
        if key == "price" and isinstance(value, str):
            required.append(value)
        elif key == "prices" and isinstance(value, (list, tuple)):
            required.extend(value)
    return sorted(set(required))


def _coerce_params(params: dict[str, Any], param_specs: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in params.items():
        if key not in param_specs:
            coerced[key] = value
            continue
        target = type(param_specs[key])
        if target is float:
            coerced[key] = float(value)
        elif target is int:
            coerced[key] = int(value)
        else:
            coerced[key] = value
    return coerced


def _requires_volume(name: str, group: str, inputs: list[str]) -> bool:
    if group == VOLUME_GROUP or name in VOLUME_INPUT_INDICATORS:
        return True
    return "volume" in inputs


class IndicatorRegistry:
    def __init__(self, config_path: Path | None = None) -> None:
        self._overrides = load_param_overrides(config_path)
        self._by_name: dict[str, IndicatorInfo] = {}
        self._by_group: dict[str, list[str]] = {}
        self._build()

    def _build(self) -> None:
        groups = talib.get_function_groups()
        for group, names in groups.items():
            sorted_names = sorted(names)
            self._by_group[group] = sorted_names
            for name in sorted_names:
                func = abstract.Function(name)
                inputs = _collect_required_inputs(func)
                info = IndicatorInfo(
                    name=name,
                    group=group,
                    input_names=inputs,
                    output_names=list(func.output_names),
                    default_params=dict(func.parameters),
                    requires_volume=_requires_volume(name, group, inputs),
                    is_pattern=group == PATTERN_GROUP,
                )
                self._by_name[name] = info

    def get(self, name: str) -> IndicatorInfo:
        key = name.upper()
        if key not in self._by_name:
            raise KeyError(f"Unknown indicator: {name}")
        return self._by_name[key]

    def all_names(self) -> list[str]:
        return sorted(self._by_name.keys())

    def names_by_group(self, group: str) -> list[str]:
        for g, names in self._by_group.items():
            if g.lower() == group.lower():
                return list(names)
        raise KeyError(f"Unknown group: {group}")

    def groups(self) -> dict[str, list[str]]:
        return {g: list(names) for g, names in sorted(self._by_group.items())}

    def iter_ordered(self) -> list[IndicatorInfo]:
        return [
            self._by_name[name]
            for group in sorted(self._by_group.keys())
            for name in self._by_group[group]
        ]

    def resolve_params(self, name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        info = self.get(name)
        func = abstract.Function(info.name)
        resolved = dict(info.default_params)
        override = self._overrides.get(name.upper(), {})
        resolved.update(override)
        if params:
            resolved.update(params)
        return _coerce_params(resolved, func.parameters)

    def can_run(self, name: str, df: pd.DataFrame) -> tuple[bool, str]:
        info = self.get(name)
        if info.requires_volume and not has_volume(df):
            return False, "Volume column required but not present in CSV."
        return True, ""

    def build_input_arrays(self, df: pd.DataFrame, *, indicator: str | None = None) -> dict[str, Any]:
        import numpy as np

        arrays: dict[str, Any] = {}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                arrays[col] = df[col].astype("float64").values

        if indicator:
            func = abstract.Function(indicator)
            if "periods" in func.input_names:
                arrays["periods"] = np.full(len(df), 20.0, dtype=np.float64)

        return arrays
