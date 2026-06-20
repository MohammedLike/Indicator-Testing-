from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from indicator_testing.data_loader import describe_ohlc, has_volume, load_monthly_ohlc, load_ohlc
from indicator_testing.golden import GOLDEN_INDICATORS, default_golden_dir, update_golden
from indicator_testing.plotter import plot_indicator
from indicator_testing.registry import IndicatorRegistry
from indicator_testing.report import (
    build_batch_report,
    print_summary,
    write_csv_report,
    write_json_report,
)
from indicator_testing.runner import run_batch, run_indicator

app = typer.Typer(
    name="indicator-testing",
    help="Test TA-Lib indicators one-by-one on monthly OHLC CSV data.",
)
golden_app = typer.Typer(help="Manage golden reference snapshots.")
app.add_typer(golden_app, name="golden")


def _default_sample_csv() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "sample_monthly.csv"


def _load_csv(
    csv: Path,
    resample: str | None,
    symbol: str | None,
    *,
    warn_short: bool = True,
):
    if resample == "none":
        return load_ohlc(csv, resample=None, symbol=symbol, warn_short=warn_short)
    if resample is not None or symbol is not None:
        return load_ohlc(csv, resample=resample, symbol=symbol, warn_short=warn_short)
    return load_monthly_ohlc(csv, warn_short=warn_short)


@app.command("inspect")
def inspect_cmd(
    csv: Path = typer.Option(..., "--csv", help="Path to OHLC CSV."),
    resample: Optional[str] = typer.Option(
        None,
        "--resample",
        "-r",
        help="Aggregate bars: monthly, daily, weekly. Auto-monthly for intraday data.",
    ),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Filter by symbol column."),
) -> None:
    """Summarize CSV columns, date range, and bar count."""
    raw = __import__("pandas").read_csv(csv)
    typer.echo(f"File: {csv}")
    typer.echo(f"Raw rows: {len(raw)}")
    typer.echo(f"Columns: {list(raw.columns)}")

    df = _load_csv(csv, resample, symbol, warn_short=False)
    info = describe_ohlc(df)
    typer.echo(f"\nLoaded bars: {info['bars']}")
    typer.echo(f"Range: {info['start']} -> {info['end']}")
    typer.echo(f"Frequency: {info['inferred_frequency']}")
    typer.echo(f"Volume: {info['has_volume']}")
    typer.echo(f"Close range: {info['close_min']:.2f} - {info['close_max']:.2f}")
    if resample:
        typer.echo(f"Resample: {resample}")
    typer.echo("\nLast 5 bars:")
    typer.echo(df.tail().to_string())


@app.command("list")
def list_indicators(
    group: Optional[str] = typer.Option(None, "--group", "-g", help="Filter by group name."),
) -> None:
    """List all TA-Lib indicators by group with required inputs."""
    registry = IndicatorRegistry()
    groups = registry.groups()

    if group:
        try:
            names = registry.names_by_group(group)
        except KeyError:
            typer.echo(f"Unknown group: {group}", err=True)
            raise typer.Exit(1)
        info_list = [registry.get(n) for n in names]
        typer.echo(f"=== {group} ({len(info_list)}) ===")
        for info in info_list:
            vol = " [volume]" if info.requires_volume else ""
            typer.echo(f"  {info.name:<20} inputs={info.input_names} outputs={info.output_names}{vol}")
        return

    for gname, names in groups.items():
        typer.echo(f"\n=== {gname} ({len(names)}) ===")
        for name in names:
            info = registry.get(name)
            vol = " [volume]" if info.requires_volume else ""
            typer.echo(f"  {info.name:<20} inputs={info.input_names}{vol}")


@app.command("run")
def run_cmd(
    csv: Path = typer.Option(..., "--csv", help="Path to monthly OHLC CSV."),
    indicator: Optional[str] = typer.Option(None, "--indicator", "-i", help="Single indicator name."),
    group: Optional[str] = typer.Option(None, "--group", "-g", help="Run all indicators in a group."),
    all_: bool = typer.Option(False, "--all", help="Run all indicators."),
    plot: bool = typer.Option(False, "--plot", help="Generate chart(s) for successful runs."),
    no_report: bool = typer.Option(False, "--no-report", help="Skip writing JSON/CSV reports."),
    resample: Optional[str] = typer.Option(
        None,
        "--resample",
        "-r",
        help="Aggregate bars: monthly, daily, weekly. Auto-monthly for intraday data.",
    ),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Filter by symbol column."),
) -> None:
    """Run indicator(s) with validation."""
    if not all_ and not indicator and not group:
        typer.echo("Specify --indicator, --group, or --all.", err=True)
        raise typer.Exit(1)

    df = _load_csv(csv, resample, symbol)
    registry = IndicatorRegistry()
    results = run_batch(
        df,
        registry,
        indicator=indicator,
        group=group,
        run_all=all_,
        validate=True,
    )

    report = build_batch_report(csv, len(df), has_volume(df), results)
    print_summary(report)

    if not no_report:
        json_path = write_json_report(report)
        csv_path = write_csv_report(report)
        typer.echo(f"\nReports: {json_path}, {csv_path}")

    if plot:
        for r in results:
            if r.status == "success":
                try:
                    path = plot_indicator(df, r)
                    typer.echo(f"Chart: {path}")
                except Exception as exc:
                    typer.echo(f"Plot failed for {r.name}: {exc}", err=True)


@app.command("plot")
def plot_cmd(
    csv: Path = typer.Option(..., "--csv", help="Path to monthly OHLC CSV."),
    indicator: str = typer.Option(..., "--indicator", "-i", help="Indicator name."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output PNG path."),
    show: bool = typer.Option(False, "--show", help="Display chart interactively."),
    resample: Optional[str] = typer.Option(None, "--resample", "-r"),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s"),
) -> None:
    """Plot a single indicator on price data."""
    from indicator_testing.plotter import plot_indicator
    from indicator_testing.runner import run_indicator

    df = _load_csv(csv, resample, symbol, warn_short=False)
    result = run_indicator(indicator, df)
    path = plot_indicator(df, result, output, show=show)
    typer.echo(f"Chart saved: {path}")


@app.command("report")
def report_cmd(
    csv: Path = typer.Option(..., "--csv", help="Path to monthly OHLC CSV."),
    all_: bool = typer.Option(True, "--all", help="Run all indicators (default)."),
    indicator: Optional[str] = typer.Option(None, "--indicator", "-i"),
    group: Optional[str] = typer.Option(None, "--group", "-g"),
    resample: Optional[str] = typer.Option(None, "--resample", "-r"),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s"),
) -> None:
    """Run batch and write JSON + CSV reports only."""
    df = _load_csv(csv, resample, symbol)
    registry = IndicatorRegistry()
    results = run_batch(
        df,
        registry,
        indicator=indicator,
        group=group,
        run_all=all_ if not indicator and not group else bool(all_),
        validate=True,
    )
    report = build_batch_report(csv, len(df), has_volume(df), results)
    json_path = write_json_report(report)
    csv_path = write_csv_report(report)
    print_summary(report)
    typer.echo(f"\nReports: {json_path}, {csv_path}")


@golden_app.command("update")
def golden_update(
    csv: Path = typer.Option(
        _default_sample_csv(),
        "--csv",
        help="CSV used to generate golden snapshots.",
    ),
    indicator: Optional[str] = typer.Option(None, "--indicator", "-i"),
    all_: bool = typer.Option(False, "--all", help="Update all golden indicators."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Golden output directory."),
) -> None:
    """Regenerate golden reference JSON snapshots."""
    if not all_ and not indicator:
        typer.echo("Specify --indicator or --all.", err=True)
        raise typer.Exit(1)

    df = load_monthly_ohlc(csv, warn_short=False)
    paths = update_golden(
        df,
        indicator=indicator,
        run_all=all_,
        golden_dir=output_dir or default_golden_dir(),
    )
    typer.echo(f"Wrote {len(paths)} golden snapshot(s) to {output_dir or default_golden_dir()}")
    for p in paths:
        typer.echo(f"  {p.name}")


@golden_app.command("list")
def golden_list() -> None:
    """List indicators included in golden snapshot set."""
    for name in GOLDEN_INDICATORS:
        typer.echo(name)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
