from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Status = Literal["success", "skipped", "error"]


@dataclass
class Check:
    name: str
    passed: bool
    message: str


@dataclass
class ValidationResult:
    passed: bool
    checks: list[Check] = field(default_factory=list)


@dataclass
class IndicatorResult:
    name: str
    group: str
    params_used: dict[str, Any]
    outputs: dict[str, np.ndarray]
    warmup_bars: int | None
    status: Status
    message: str
    validation: ValidationResult | None = None

    def output_names(self) -> list[str]:
        return list(self.outputs.keys())


@dataclass
class BatchReport:
    csv_path: str
    bar_count: int
    has_volume: bool
    results: list[IndicatorResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.status == "success")

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.status == "error")
