from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

IndicatorFunc = Callable[..., pd.Series | pd.DataFrame]


@dataclass(frozen=True)
class SmcIndicatorInfo:
    name: str
    group: str
    func: IndicatorFunc
    default_params: dict[str, Any]
    output_names: list[str]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["open", "high", "low", "close"]:
        out[col] = out[col].astype(float)
    return out


def _swing_mask(series: pd.Series, length: int, *, mode: str) -> pd.Series:
    """Mark swing points; length = bars on each side."""
    n = len(series)
    flags = np.zeros(n, dtype=bool)
    values = series.to_numpy()
    for i in range(length, n - length):
        window = values[i - length : i + length + 1]
        if mode == "high" and values[i] == window.max() and (window == values[i]).sum() == 1:
            flags[i] = True
        elif mode == "low" and values[i] == window.min() and (window == values[i]).sum() == 1:
            flags[i] = True
    return pd.Series(flags, index=series.index)


def swing_highs(df: pd.DataFrame, length: int = 2) -> pd.DataFrame:
    df = _prepare(df)
    mask = _swing_mask(df["high"], length, mode="high")
    level = df["high"].where(mask)
    return pd.DataFrame({"swing_high": level, "is_swing_high": mask.astype(int)})


def swing_lows(df: pd.DataFrame, length: int = 2) -> pd.DataFrame:
    df = _prepare(df)
    mask = _swing_mask(df["low"], length, mode="low")
    level = df["low"].where(mask)
    return pd.DataFrame({"swing_low": level, "is_swing_low": mask.astype(int)})


def swing_points(df: pd.DataFrame, length: int = 2) -> pd.DataFrame:
    hi = swing_highs(df, length)
    lo = swing_lows(df, length)
    return pd.concat([hi, lo], axis=1)


def fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """3-candle FVG: bull gap when low > high[t-2], bear when high < low[t-2]."""
    df = _prepare(df)
    bull = df["low"] > df["high"].shift(2)
    bear = df["high"] < df["low"].shift(2)
    bull_top = df["low"].where(bull)
    bull_bottom = df["high"].shift(2).where(bull)
    bear_top = df["low"].shift(2).where(bear)
    bear_bottom = df["high"].where(bear)
    return pd.DataFrame(
        {
            "fvg_bull": bull.astype(int),
            "fvg_bear": bear.astype(int),
            "fvg_bull_top": bull_top,
            "fvg_bull_bottom": bull_bottom,
            "fvg_bear_top": bear_top,
            "fvg_bear_bottom": bear_bottom,
        }
    )


def fvg_bullish(df: pd.DataFrame) -> pd.DataFrame:
    out = fair_value_gaps(df)
    return pd.DataFrame(
        {
            "fvg_bull": out["fvg_bull"],
            "fvg_bull_top": out["fvg_bull_top"],
            "fvg_bull_bottom": out["fvg_bull_bottom"],
        }
    )


def fvg_bearish(df: pd.DataFrame) -> pd.DataFrame:
    out = fair_value_gaps(df)
    return pd.DataFrame(
        {
            "fvg_bear": out["fvg_bear"],
            "fvg_bear_top": out["fvg_bear_top"],
            "fvg_bear_bottom": out["fvg_bear_bottom"],
        }
    )


def market_structure(df: pd.DataFrame, length: int = 2) -> pd.DataFrame:
    """Label swing sequence: HH, HL, LH, LL and trend bias."""
    df = _prepare(df)
    sh = swing_highs(df, length)["swing_high"]
    sl = swing_lows(df, length)["swing_low"]

    structure = pd.Series(np.nan, index=df.index, dtype=object)
    trend = pd.Series(0, index=df.index, dtype=int)  # 1 bull, -1 bear

    prev_sh = np.nan
    prev_sl = np.nan
    bias = 0

    for i in range(len(df)):
        if not np.isnan(sh.iloc[i]):
            if not np.isnan(prev_sh):
                structure.iloc[i] = "HH" if sh.iloc[i] > prev_sh else "LH"
                if sh.iloc[i] > prev_sh:
                    bias = 1
                elif sh.iloc[i] < prev_sh and bias == 1:
                    bias = 0
            prev_sh = sh.iloc[i]
        if not np.isnan(sl.iloc[i]):
            if not np.isnan(prev_sl):
                label = "HL" if sl.iloc[i] > prev_sl else "LL"
                if structure.iloc[i] is np.nan or pd.isna(structure.iloc[i]):
                    structure.iloc[i] = label
                if sl.iloc[i] > prev_sl:
                    bias = 1
                elif sl.iloc[i] < prev_sl:
                    bias = -1
            prev_sl = sl.iloc[i]
        trend.iloc[i] = bias

    return pd.DataFrame({"structure_label": structure, "trend_bias": trend})


def bos(df: pd.DataFrame, length: int = 2) -> pd.DataFrame:
    """Break of Structure: continuation break in trend direction."""
    df = _prepare(df)
    sh = swing_highs(df, length)["swing_high"].ffill()
    sl = swing_lows(df, length)["swing_low"].ffill()
    ms = market_structure(df, length)

    bos_bull = (ms["trend_bias"] >= 0) & (df["close"] > sh.shift(1)) & sh.shift(1).notna()
    bos_bear = (ms["trend_bias"] <= 0) & (df["close"] < sl.shift(1)) & sl.shift(1).notna()

    return pd.DataFrame({"bos_bull": bos_bull.astype(int), "bos_bear": bos_bear.astype(int)})


def choch(df: pd.DataFrame, length: int = 2) -> pd.DataFrame:
    """Change of Character: first break against prior bias."""
    df = _prepare(df)
    sh = swing_highs(df, length)["swing_high"].ffill()
    sl = swing_lows(df, length)["swing_low"].ffill()
    ms = market_structure(df, length)

    choch_bear = (ms["trend_bias"] == 1) & (df["close"] < sl.shift(1)) & sl.shift(1).notna()
    choch_bull = (ms["trend_bias"] == -1) & (df["close"] > sh.shift(1)) & sh.shift(1).notna()

    return pd.DataFrame({"choch_bull": choch_bull.astype(int), "choch_bear": choch_bear.astype(int)})


def order_blocks(df: pd.DataFrame, length: int = 2, impulse_pct: float = 0.003) -> pd.DataFrame:
    """Simplified OB: last opposite candle before impulsive break of structure."""
    df = _prepare(df)
    bos_df = bos(df, length)
    bearish = df["close"] < df["open"]
    bullish = df["close"] > df["open"]

    ob_bull = pd.Series(0, index=df.index, dtype=int)
    ob_bear = pd.Series(0, index=df.index, dtype=int)
    ob_bull_low = pd.Series(np.nan, index=df.index)
    ob_bull_high = pd.Series(np.nan, index=df.index)
    ob_bear_low = pd.Series(np.nan, index=df.index)
    ob_bear_high = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):
        move = (df["close"].iloc[i] - df["close"].iloc[i - 1]) / df["close"].iloc[i - 1]
        if bos_df["bos_bull"].iloc[i] and move >= impulse_pct:
            j = i - 1
            while j >= 0 and not bearish.iloc[j]:
                j -= 1
            if j >= 0:
                ob_bull.iloc[i] = 1
                ob_bull_low.iloc[i] = df["low"].iloc[j]
                ob_bull_high.iloc[i] = df["high"].iloc[j]
        if bos_df["bos_bear"].iloc[i] and move <= -impulse_pct:
            j = i - 1
            while j >= 0 and not bullish.iloc[j]:
                j -= 1
            if j >= 0:
                ob_bear.iloc[i] = 1
                ob_bear_low.iloc[i] = df["low"].iloc[j]
                ob_bear_high.iloc[i] = df["high"].iloc[j]

    return pd.DataFrame(
        {
            "ob_bull": ob_bull,
            "ob_bull_low": ob_bull_low,
            "ob_bull_high": ob_bull_high,
            "ob_bear": ob_bear,
            "ob_bear_low": ob_bear_low,
            "ob_bear_high": ob_bear_high,
        }
    )


def order_block_bull(df: pd.DataFrame, length: int = 2, impulse_pct: float = 0.003) -> pd.DataFrame:
    ob = order_blocks(df, length, impulse_pct)
    return ob[["ob_bull", "ob_bull_low", "ob_bull_high"]]


def order_block_bear(df: pd.DataFrame, length: int = 2, impulse_pct: float = 0.003) -> pd.DataFrame:
    ob = order_blocks(df, length, impulse_pct)
    return ob[["ob_bear", "ob_bear_low", "ob_bear_high"]]


def liquidity_sweeps(df: pd.DataFrame, length: int = 2) -> pd.DataFrame:
    """Sweep: wick takes prior swing high/low then close reverts inside."""
    df = _prepare(df)
    sh = swing_highs(df, length)["swing_high"].shift(1)
    sl = swing_lows(df, length)["swing_low"].shift(1)

    sweep_high = (df["high"] > sh) & (df["close"] < sh) & sh.notna()
    sweep_low = (df["low"] < sl) & (df["close"] > sl) & sl.notna()

    return pd.DataFrame(
        {"liquidity_sweep_high": sweep_high.astype(int), "liquidity_sweep_low": sweep_low.astype(int)}
    )


def equal_highs_lows(df: pd.DataFrame, length: int = 2, tolerance_pct: float = 0.0005) -> pd.DataFrame:
    """Equal highs/lows = nearby liquidity pools."""
    df = _prepare(df)
    sh = swing_highs(df, length)
    sl = swing_lows(df, length)

    eq_high = pd.Series(0, index=df.index, dtype=int)
    eq_low = pd.Series(0, index=df.index, dtype=int)
    last_sh = np.nan
    last_sl = np.nan

    for i in range(len(df)):
        if sh["is_swing_high"].iloc[i]:
            if not np.isnan(last_sh) and abs(sh["swing_high"].iloc[i] - last_sh) / last_sh <= tolerance_pct:
                eq_high.iloc[i] = 1
            last_sh = sh["swing_high"].iloc[i]
        if sl["is_swing_low"].iloc[i]:
            if not np.isnan(last_sl) and abs(sl["swing_low"].iloc[i] - last_sl) / last_sl <= tolerance_pct:
                eq_low.iloc[i] = 1
            last_sl = sl["swing_low"].iloc[i]

    return pd.DataFrame({"equal_highs": eq_high, "equal_lows": eq_low})


def premium_discount(df: pd.DataFrame, length: int = 2) -> pd.DataFrame:
    """Position in current dealing range (0=discount, 100=premium)."""
    df = _prepare(df)
    sh = swing_highs(df, length)["swing_high"].ffill()
    sl = swing_lows(df, length)["swing_low"].ffill()
    eq = (sh + sl) / 2
    rng = (sh - sl).replace(0, np.nan)
    pct = 100 * (df["close"] - sl) / rng
    zone = pd.Series("equilibrium", index=df.index)
    zone = zone.mask(pct > 50, "premium").mask(pct < 50, "discount")
    return pd.DataFrame(
        {
            "range_pct": pct,
            "equilibrium": eq,
            "zone": zone,
        }
    )


def dealing_range(df: pd.DataFrame, length: int = 2) -> pd.DataFrame:
    df = _prepare(df)
    sh = swing_highs(df, length)["swing_high"].ffill()
    sl = swing_lows(df, length)["swing_low"].ffill()
    eq = (sh + sl) / 2
    return pd.DataFrame({"range_high": sh, "range_low": sl, "equilibrium": eq})


def internal_range(df: pd.DataFrame, length: int = 1) -> pd.DataFrame:
    """Shorter swing structure (internal liquidity / iBOS context)."""
    return swing_points(df, length)


SMC_INDICATORS: list[SmcIndicatorInfo] = [
    SmcIndicatorInfo("SWING_HIGH", "Market Structure", swing_highs, {"length": 2}, ["swing_high", "is_swing_high"]),
    SmcIndicatorInfo("SWING_LOW", "Market Structure", swing_lows, {"length": 2}, ["swing_low", "is_swing_low"]),
    SmcIndicatorInfo("SWING_POINTS", "Market Structure", swing_points, {"length": 2}, ["swing_high", "is_swing_high", "swing_low", "is_swing_low"]),
    SmcIndicatorInfo("MARKET_STRUCTURE", "Market Structure", market_structure, {"length": 2}, ["structure_label", "trend_bias"]),
    SmcIndicatorInfo("BOS", "Market Structure", bos, {"length": 2}, ["bos_bull", "bos_bear"]),
    SmcIndicatorInfo("CHOCH", "Market Structure", choch, {"length": 2}, ["choch_bull", "choch_bear"]),
    SmcIndicatorInfo("INTERNAL_SWINGS", "Market Structure", internal_range, {"length": 1}, ["swing_high", "is_swing_high", "swing_low", "is_swing_low"]),
    SmcIndicatorInfo("FVG", "Fair Value Gaps", fair_value_gaps, {}, ["fvg_bull", "fvg_bear", "fvg_bull_top", "fvg_bull_bottom", "fvg_bear_top", "fvg_bear_bottom"]),
    SmcIndicatorInfo("FVG_BULL", "Fair Value Gaps", fvg_bullish, {}, ["fvg_bull", "fvg_bull_top", "fvg_bull_bottom"]),
    SmcIndicatorInfo("FVG_BEAR", "Fair Value Gaps", fvg_bearish, {}, ["fvg_bear", "fvg_bear_top", "fvg_bear_bottom"]),
    SmcIndicatorInfo("ORDER_BLOCKS", "Order Blocks", order_blocks, {"length": 2, "impulse_pct": 0.003}, ["ob_bull", "ob_bull_low", "ob_bull_high", "ob_bear", "ob_bear_low", "ob_bear_high"]),
    SmcIndicatorInfo("ORDER_BLOCK_BULL", "Order Blocks", order_block_bull, {"length": 2, "impulse_pct": 0.003}, ["ob_bull", "ob_bull_low", "ob_bull_high"]),
    SmcIndicatorInfo("ORDER_BLOCK_BEAR", "Order Blocks", order_block_bear, {"length": 2, "impulse_pct": 0.003}, ["ob_bear", "ob_bear_low", "ob_bear_high"]),
    SmcIndicatorInfo("LIQUIDITY_SWEEPS", "Liquidity", liquidity_sweeps, {"length": 2}, ["liquidity_sweep_high", "liquidity_sweep_low"]),
    SmcIndicatorInfo("EQUAL_HIGHS_LOWS", "Liquidity", equal_highs_lows, {"length": 2, "tolerance_pct": 0.0005}, ["equal_highs", "equal_lows"]),
    SmcIndicatorInfo("PREMIUM_DISCOUNT", "Premium / Discount", premium_discount, {"length": 2}, ["range_pct", "equilibrium", "zone"]),
    SmcIndicatorInfo("DEALING_RANGE", "Premium / Discount", dealing_range, {"length": 2}, ["range_high", "range_low", "equilibrium"]),
]


class SmcIndicatorRegistry:
    def __init__(self) -> None:
        self._by_name = {i.name: i for i in SMC_INDICATORS}
        self._by_group: dict[str, list[str]] = {}
        for info in SMC_INDICATORS:
            self._by_group.setdefault(info.group, []).append(info.name)

    def get(self, name: str) -> SmcIndicatorInfo:
        key = name.upper()
        if key not in self._by_name:
            raise KeyError(f"Unknown SMC indicator: {name}")
        return self._by_name[key]

    def all_names(self) -> list[str]:
        return [i.name for i in SMC_INDICATORS]

    def groups(self) -> dict[str, list[str]]:
        return dict(self._by_group)

    def iter_ordered(self) -> list[SmcIndicatorInfo]:
        return list(SMC_INDICATORS)
