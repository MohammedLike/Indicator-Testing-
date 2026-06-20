from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from indicator_testing.models import BatchReport, IndicatorResult


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_reports_dir() -> Path:
    return _project_root() / "outputs" / "reports"


def _result_to_dict(result: IndicatorResult) -> dict:
    return {
        "name": result.name,
        "group": result.group,
        "status": result.status,
        "message": result.message,
        "params_used": result.params_used,
        "warmup_bars": result.warmup_bars,
        "output_names": result.output_names(),
        "validation_passed": result.validation.passed if result.validation else None,
        "validation_checks": [
            {"name": c.name, "passed": c.passed, "message": c.message}
            for c in (result.validation.checks if result.validation else [])
        ],
    }


def build_batch_report(
    csv_path: str | Path,
    df_length: int,
    has_volume: bool,
    results: list[IndicatorResult],
) -> BatchReport:
    return BatchReport(
        csv_path=str(csv_path),
        bar_count=df_length,
        has_volume=has_volume,
        results=results,
    )


def write_json_report(report: BatchReport, output_dir: Path | None = None) -> Path:
    out_dir = output_dir or default_reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"run_{stamp}.json"
    payload = {
        "csv_path": report.csv_path,
        "bar_count": report.bar_count,
        "has_volume": report.has_volume,
        "success_count": report.success_count,
        "skipped_count": report.skipped_count,
        "error_count": report.error_count,
        "results": [_result_to_dict(r) for r in report.results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_csv_report(report: BatchReport, output_dir: Path | None = None) -> Path:
    out_dir = output_dir or default_reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"run_{stamp}.csv"

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "name",
                "group",
                "status",
                "warmup_bars",
                "validation_passed",
                "message",
            ],
        )
        writer.writeheader()
        for r in report.results:
            writer.writerow(
                {
                    "name": r.name,
                    "group": r.group,
                    "status": r.status,
                    "warmup_bars": r.warmup_bars,
                    "validation_passed": r.validation.passed if r.validation else "",
                    "message": r.message,
                }
            )
    return path


def print_summary(report: BatchReport) -> None:
    print(f"CSV: {report.csv_path} ({report.bar_count} bars, volume={report.has_volume})")
    print(
        f"Results: {report.success_count} success, "
        f"{report.skipped_count} skipped, {report.error_count} error"
    )
    print(f"{'NAME':<20} {'GROUP':<22} {'STATUS':<8} {'WARMUP':<8} {'VALID':<6}")
    print("-" * 72)
    for r in report.results:
        valid = ""
        if r.validation is not None:
            valid = "PASS" if r.validation.passed else "FAIL"
        warmup = "" if r.warmup_bars is None else str(r.warmup_bars)
        print(f"{r.name:<20} {r.group:<22} {r.status:<8} {warmup:<8} {valid:<6}")
