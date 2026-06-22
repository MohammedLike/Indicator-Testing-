"""Streamlit UI for TA-Lib, pandas, and SMC indicator testing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from indicator_testing.data_loader import DataLoadError, describe_ohlc, load_ohlc
from indicator_testing.models import IndicatorResult
from indicator_testing.pandas_indicators import PandasIndicatorRegistry
from indicator_testing.pandas_runner import run_pandas_batch, run_pandas_indicator
from indicator_testing.plotter import make_indicator_figure
from indicator_testing.registry import IndicatorRegistry
from indicator_testing.runner import run_batch, run_indicator
from indicator_testing.smc_indicators import SmcIndicatorRegistry
from indicator_testing.smc_runner import run_smc_batch, run_smc_indicator

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_ROOT / "questdb-query-1781940224994.csv"
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_monthly.csv"

LIBRARIES = {
    "TA-Lib": "talib",
    "Pandas": "pandas",
    "SMC (Smart Money Concepts)": "smc",
}


def _init_session() -> None:
    if "df" not in st.session_state:
        st.session_state.df = None
    if "data_info" not in st.session_state:
        st.session_state.data_info = None


def _load_data(source: Path | None, uploaded, resample: str, symbol: str | None) -> None:
    resample_arg = None if resample == "none" else resample
    try:
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                path = Path(tmp.name)
        elif source and source.exists():
            path = source
        else:
            st.error("No CSV selected or file not found.")
            return

        df = load_ohlc(path, resample=resample_arg, symbol=symbol or None, warn_short=False)
        st.session_state.df = df
        st.session_state.data_info = describe_ohlc(df)
        st.session_state.csv_label = uploaded.name if uploaded else str(source)
    except DataLoadError as exc:
        st.error(str(exc))


def _registry_for(library: str):
    if library == "talib":
        return IndicatorRegistry()
    if library == "pandas":
        return PandasIndicatorRegistry()
    return SmcIndicatorRegistry()


def _indicator_options(library: str) -> dict[str, list[str]]:
    reg = _registry_for(library)
    if library == "talib":
        return reg.groups()
    return reg.groups()


def _run_one(library: str, name: str, df: pd.DataFrame, params: dict) -> IndicatorResult:
    if library == "talib":
        return run_indicator(name, df, validate=True)
    if library == "pandas":
        return run_pandas_indicator(name, df, params=params or None, validate=True)
    return run_smc_indicator(name, df, params=params or None, validate=True)


def _run_all(library: str, df: pd.DataFrame) -> list[IndicatorResult]:
    if library == "talib":
        return run_batch(df, run_all=True, validate=True)
    if library == "pandas":
        return run_pandas_batch(df, validate=True)
    return run_smc_batch(df, validate=True)


def _result_table(results: list[IndicatorResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "indicator": r.name,
                "group": r.group,
                "status": r.status,
                "warmup": r.warmup_bars,
                "valid": r.validation.passed if r.validation else None,
                "message": r.message if r.status != "success" else "",
            }
        )
    return pd.DataFrame(rows)


def _outputs_dataframe(df: pd.DataFrame, result: IndicatorResult, tail: int = 20) -> pd.DataFrame:
    out = pd.DataFrame({"close": df["close"]})
    for k, arr in result.outputs.items():
        out[k] = arr
    out.index = df.index
    numeric = out.select_dtypes(include="number")
    if len(numeric.columns):
        hits = numeric[(numeric != 0).any(axis=1)]
        if len(hits):
            return hits.tail(tail)
    return out.tail(tail)


def _plot_result(df: pd.DataFrame, result: IndicatorResult, library: str) -> None:
    if result.status != "success" or not result.outputs:
        st.warning("Nothing to plot.")
        return

    try:
        if library == "talib":
            fig = make_indicator_figure(df, result)
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            ax_price, ax_ind = axes[0], axes[1]
            ax_price.plot(df.index, df["close"], color="black", lw=1)
            ax_price.set_ylabel("Close")
            for key, arr in result.outputs.items():
                if getattr(arr, "dtype", None) == object:
                    series = pd.Series(arr, index=df.index)
                    mask = series.notna() & (series.astype(str) != "") & (series.astype(str) != "nan")
                    if mask.any() and library == "smc":
                        ax_price.scatter(df.index[mask], df["close"][mask], s=25, label=key)
                    continue
                series = pd.Series(arr, index=df.index)
                if (series.fillna(0) != 0).any():
                    ax_ind.plot(df.index, arr, label=key, lw=1)
                elif any(x in key for x in ("top", "bottom", "high", "low", "equilibrium", "range")):
                    ax_ind.plot(df.index, arr, label=key, lw=0.8, alpha=0.7)
            if ax_ind.get_legend_handles_labels()[0]:
                ax_ind.legend(loc="upper left", fontsize=8)
            ax_ind.set_ylabel(result.name)
            fig.suptitle(f"{result.name} ({library})")
            fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:
        st.warning(f"Chart error: {exc}")


def _validation_panel(result: IndicatorResult) -> None:
    cols = st.columns(4)
    cols[0].metric("Status", result.status)
    cols[1].metric("Warmup bars", result.warmup_bars if result.warmup_bars is not None else "—")
    valid = "—"
    if result.validation is not None:
        valid = "PASS" if result.validation.passed else "FAIL"
    cols[2].metric("Validation", valid)
    cols[3].metric("Outputs", len(result.outputs))

    if result.message and result.status != "success":
        st.info(result.message)

    if result.validation:
        for check in result.validation.checks:
            icon = "✅" if check.passed else "❌"
            st.write(f"{icon} **{check.name}**: {check.message}")


def main() -> None:
    st.set_page_config(page_title="Indicator Testing", page_icon="📈", layout="wide")
    _init_session()

    st.title("Indicator Testing App")
    st.caption("Test TA-Lib, pandas, and SMC indicators on OHLC CSV data.")

    with st.sidebar:
        st.header("Data")
        use_upload = st.radio("Source", ["Bundled NIFTY CSV", "Sample monthly", "Upload CSV"], index=0)
        uploaded = None
        source_path = DEFAULT_CSV
        if use_upload == "Sample monthly":
            source_path = SAMPLE_CSV
        elif use_upload == "Upload CSV":
            uploaded = st.file_uploader("OHLC CSV", type=["csv"])
            source_path = None

        resample = st.selectbox(
            "Resample",
            ["none", "daily", "monthly", "weekly"],
            index=1,
            help="none = raw bars (best for TA-Lib). daily = good for SMC.",
        )
        symbol = st.text_input("Symbol filter (optional)", value="")

        if st.button("Load data", type="primary", use_container_width=True):
            _load_data(source_path, uploaded, resample, symbol.strip() or None)

        st.divider()
        st.header("Library")
        library_label = st.selectbox("Indicator library", list(LIBRARIES.keys()))
        library = LIBRARIES[library_label]

        swing_length = 2
        if library == "smc":
            swing_length = st.number_input("Swing length", min_value=1, max_value=10, value=2)

        if st.session_state.data_info:
            info = st.session_state.data_info
            st.success(f"Loaded **{info['bars']}** bars")
            st.write(f"Range: {info['start'][:10]} → {info['end'][:10]}")
            st.write(f"Frequency: {info['inferred_frequency']}")

    df: pd.DataFrame | None = st.session_state.df
    if df is None:
        st.info("Load a CSV from the sidebar to begin.")
        return

    tab_single, tab_batch, tab_preview = st.tabs(["Single indicator", "Batch run", "Data preview"])

    groups = _indicator_options(library)
    group_names = list(groups.keys())
    flat_names = [n for names in groups.values() for n in names]

    with tab_preview:
        st.subheader("OHLC preview")
        st.dataframe(df.tail(50), use_container_width=True)

    with tab_single:
        col1, col2 = st.columns([1, 2])
        with col1:
            group_pick = st.selectbox("Group", group_names)
            indicator = st.selectbox("Indicator", groups[group_pick])
            run_btn = st.button("Run indicator", type="primary", key="run_single")

        if run_btn:
            params = {"length": int(swing_length)} if library == "smc" else {}
            with st.spinner(f"Running {indicator}..."):
                result = _run_one(library, indicator, df, params)

            with col2:
                st.subheader(f"{indicator} — {library_label}")
                _validation_panel(result)

            st.subheader("Chart")
            _plot_result(df, result, library)

            st.subheader("Output sample")
            st.dataframe(_outputs_dataframe(df, result), use_container_width=True)

    with tab_batch:
        st.write(f"Run all **{len(flat_names)}** indicators in **{library_label}**.")
        batch_btn = st.button("Run all indicators", type="primary", key="run_batch")

        if batch_btn:
            with st.spinner("Running batch..."):
                results = _run_all(library, df)

            summary = _result_table(results)
            ok = (summary["status"] == "success").sum()
            err = (summary["status"] == "error").sum()
            skip = (summary["status"] == "skipped").sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Success", int(ok))
            m2.metric("Skipped", int(skip))
            m3.metric("Error", int(err))
            m4.metric("Total", len(results))

            status_filter = st.multiselect(
                "Filter status",
                ["success", "skipped", "error"],
                default=["success", "error"],
            )
            st.dataframe(summary[summary["status"].isin(status_filter)], use_container_width=True)

            csv_bytes = summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                csv_bytes,
                file_name=f"indicator_run_{library}.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
