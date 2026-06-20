from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

IndicatorFunc = Callable[..., pd.Series | pd.DataFrame]


@dataclass(frozen=True)
class PandasIndicatorInfo:
    name: str
    group: str
    func: IndicatorFunc
    default_params: dict[str, Any]
    output_names: list[str]
    requires_volume: bool = False


def _hlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    return df["high"], df["low"], df["close"]


def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = _hlc(df)
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


# --- Overlap studies ---


def sma(df: pd.DataFrame, length: int = 20) -> pd.Series:
    return df["close"].rolling(length).mean().rename("sma")


def ema(df: pd.DataFrame, length: int = 20) -> pd.Series:
    return df["close"].ewm(span=length, adjust=False).mean().rename("ema")


def wma(df: pd.DataFrame, length: int = 20) -> pd.Series:
    weights = np.arange(1, length + 1)

    def _wma(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return df["close"].rolling(length).apply(_wma, raw=True).rename("wma")


def dema(df: pd.DataFrame, length: int = 20) -> pd.Series:
    e1 = ema(df, length)
    e2 = e1.ewm(span=length, adjust=False).mean()
    return (2 * e1 - e2).rename("dema")


def tema(df: pd.DataFrame, length: int = 20) -> pd.Series:
    e1 = ema(df, length)
    e2 = e1.ewm(span=length, adjust=False).mean()
    e3 = e2.ewm(span=length, adjust=False).mean()
    return (3 * e1 - 3 * e2 + e3).rename("tema")


def hma(df: pd.DataFrame, length: int = 20) -> pd.Series:
    half = max(int(length / 2), 1)
    sqrt_len = max(int(np.sqrt(length)), 1)
    wma_half = wma(pd.DataFrame({"close": df["close"]}), half)
    wma_full = wma(pd.DataFrame({"close": df["close"]}), length)
    raw = 2 * wma_half - wma_full
    return wma(pd.DataFrame({"close": raw}), sqrt_len).rename("hma")


def vwma(df: pd.DataFrame, length: int = 20) -> pd.Series:
    if "volume" not in df.columns:
        raise ValueError("volume column required")
    pv = df["close"] * df["volume"]
    return (pv.rolling(length).sum() / df["volume"].rolling(length).sum()).rename("vwma")


# --- Momentum ---


def rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).rename("rsi")


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
    line = fast_ema - slow_ema
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "macdsignal": sig, "macdhist": line - sig},
    )


def roc(df: pd.DataFrame, length: int = 10) -> pd.Series:
    return (df["close"].pct_change(length) * 100).rename("roc")


def mom(df: pd.DataFrame, length: int = 10) -> pd.Series:
    return df["close"].diff(length).rename("mom")


def stoch(
    df: pd.DataFrame,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    high, low, close = _hlc(df)
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    pct_k = 100 * (close - lowest) / (highest - lowest)
    slow_k = pct_k.rolling(smooth_k).mean()
    slow_d = slow_k.rolling(d).mean()
    return pd.DataFrame({"slowk": slow_k, "slowd": slow_d})


def stochrsi(df: pd.DataFrame, length: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    r = rsi(df, length)
    lowest = r.rolling(length).min()
    highest = r.rolling(length).max()
    stoch = 100 * (r - lowest) / (highest - lowest)
    k_line = stoch.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return pd.DataFrame({"stochrsi_k": k_line, "stochrsi_d": d_line})


def cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    high, low, close = _hlc(df)
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return ((tp - sma_tp) / (0.015 * mad)).rename("cci")


def willr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = _hlc(df)
    highest = high.rolling(length).max()
    lowest = low.rolling(length).min()
    return (-100 * (highest - close) / (highest - lowest)).rename("willr")


def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    if "volume" not in df.columns:
        raise ValueError("volume column required")
    high, low, close = _hlc(df)
    tp = (high + low + close) / 3
    rmf = tp * df["volume"]
    direction = tp.diff()
    pos = rmf.where(direction > 0, 0.0)
    neg = rmf.where(direction < 0, 0.0)
    pos_sum = pos.rolling(length).sum()
    neg_sum = neg.rolling(length).sum()
    return (100 - 100 / (1 + pos_sum / neg_sum)).rename("mfi")


def cmo(df: pd.DataFrame, length: int = 14) -> pd.Series:
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(length).sum()
    down = (-delta.clip(upper=0)).rolling(length).sum()
    return (100 * (up - down) / (up + down)).rename("cmo")


def ppo(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
    return (100 * (fast_ema - slow_ema) / slow_ema).rename("ppo")


def trix(df: pd.DataFrame, length: int = 15) -> pd.Series:
    e1 = df["close"].ewm(span=length, adjust=False).mean()
    e2 = e1.ewm(span=length, adjust=False).mean()
    e3 = e2.ewm(span=length, adjust=False).mean()
    return (e3.pct_change() * 100).rename("trix")


# --- Volatility ---


def bbands(df: pd.DataFrame, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = df["close"].rolling(length).mean()
    dev = df["close"].rolling(length).std()
    return pd.DataFrame(
        {
            "upperband": mid + std * dev,
            "middleband": mid,
            "lowerband": mid - std * dev,
        }
    )


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return _true_range(df).rolling(length).mean().rename("atr")


def natr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    a = atr(df, length)
    return (100 * a / df["close"]).rename("natr")


def trange(df: pd.DataFrame) -> pd.Series:
    return _true_range(df).rename("trange")


def donchian(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    high, low, _ = _hlc(df)
    upper = high.rolling(length).max()
    lower = low.rolling(length).min()
    return pd.DataFrame(
        {
            "donchian_upper": upper,
            "donchian_mid": (upper + lower) / 2,
            "donchian_lower": lower,
        }
    )


# --- Volume ---


def obv(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        raise ValueError("volume column required")
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum().rename("obv")


def ad(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        raise ValueError("volume column required")
    high, low, close = _hlc(df)
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
    return (clv * df["volume"]).cumsum().rename("ad")


def adosc(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
    line = ad(df)
    return (line.ewm(span=fast, adjust=False).mean() - line.ewm(span=slow, adjust=False).mean()).rename(
        "adosc"
    )


def cmf(df: pd.DataFrame, length: int = 20) -> pd.Series:
    if "volume" not in df.columns:
        raise ValueError("volume column required")
    high, low, close = _hlc(df)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * df["volume"]
    return (mfv.rolling(length).sum() / df["volume"].rolling(length).sum()).rename("cmf")


def vwap(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns:
        raise ValueError("volume column required")
    high, low, close = _hlc(df)
    tp = (high + low + close) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


# --- Price transform ---


def avgprice(df: pd.DataFrame) -> pd.Series:
    return ((df["open"] + df["high"] + df["low"] + df["close"]) / 4).rename("avgprice")


def medprice(df: pd.DataFrame) -> pd.Series:
    return ((df["high"] + df["low"]) / 2).rename("medprice")


def typprice(df: pd.DataFrame) -> pd.Series:
    return ((df["high"] + df["low"] + df["close"]) / 3).rename("typprice")


def wclprice(df: pd.DataFrame) -> pd.Series:
    high, low, close = _hlc(df)
    return ((high + low + 2 * close) / 4).rename("wclprice")


# --- Statistics ---


def stddev(df: pd.DataFrame, length: int = 5) -> pd.Series:
    return df["close"].rolling(length).std().rename("stddev")


def variance(df: pd.DataFrame, length: int = 5) -> pd.Series:
    return df["close"].rolling(length).var().rename("var")


def zscore(df: pd.DataFrame, length: int = 20) -> pd.Series:
    mean = df["close"].rolling(length).mean()
    std = df["close"].rolling(length).std()
    return ((df["close"] - mean) / std).rename("zscore")


def linearreg(df: pd.DataFrame, length: int = 14) -> pd.Series:
    def _linreg(y: np.ndarray) -> float:
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (len(y) - 1) + intercept

    return df["close"].rolling(length).apply(_linreg, raw=True).rename("linearreg")


PANDAS_INDICATORS: list[PandasIndicatorInfo] = [
    PandasIndicatorInfo("SMA", "Overlap Studies", sma, {"length": 20}, ["sma"]),
    PandasIndicatorInfo("EMA", "Overlap Studies", ema, {"length": 20}, ["ema"]),
    PandasIndicatorInfo("WMA", "Overlap Studies", wma, {"length": 20}, ["wma"]),
    PandasIndicatorInfo("DEMA", "Overlap Studies", dema, {"length": 20}, ["dema"]),
    PandasIndicatorInfo("TEMA", "Overlap Studies", tema, {"length": 20}, ["tema"]),
    PandasIndicatorInfo("HMA", "Overlap Studies", hma, {"length": 20}, ["hma"]),
    PandasIndicatorInfo("VWMA", "Overlap Studies", vwma, {"length": 20}, ["vwma"], True),
    PandasIndicatorInfo("RSI", "Momentum Indicators", rsi, {"length": 14}, ["rsi"]),
    PandasIndicatorInfo("MACD", "Momentum Indicators", macd, {"fast": 12, "slow": 26, "signal": 9}, ["macd", "macdsignal", "macdhist"]),
    PandasIndicatorInfo("ROC", "Momentum Indicators", roc, {"length": 10}, ["roc"]),
    PandasIndicatorInfo("MOM", "Momentum Indicators", mom, {"length": 10}, ["mom"]),
    PandasIndicatorInfo("STOCH", "Momentum Indicators", stoch, {"k": 14, "d": 3, "smooth_k": 3}, ["slowk", "slowd"]),
    PandasIndicatorInfo("STOCHRSI", "Momentum Indicators", stochrsi, {"length": 14, "k": 3, "d": 3}, ["stochrsi_k", "stochrsi_d"]),
    PandasIndicatorInfo("CCI", "Momentum Indicators", cci, {"length": 20}, ["cci"]),
    PandasIndicatorInfo("WILLR", "Momentum Indicators", willr, {"length": 14}, ["willr"]),
    PandasIndicatorInfo("MFI", "Momentum Indicators", mfi, {"length": 14}, ["mfi"], True),
    PandasIndicatorInfo("CMO", "Momentum Indicators", cmo, {"length": 14}, ["cmo"]),
    PandasIndicatorInfo("PPO", "Momentum Indicators", ppo, {"fast": 12, "slow": 26}, ["ppo"]),
    PandasIndicatorInfo("TRIX", "Momentum Indicators", trix, {"length": 15}, ["trix"]),
    PandasIndicatorInfo("BBANDS", "Volatility Indicators", bbands, {"length": 20, "std": 2.0}, ["upperband", "middleband", "lowerband"]),
    PandasIndicatorInfo("ATR", "Volatility Indicators", atr, {"length": 14}, ["atr"]),
    PandasIndicatorInfo("NATR", "Volatility Indicators", natr, {"length": 14}, ["natr"]),
    PandasIndicatorInfo("TRANGE", "Volatility Indicators", trange, {}, ["trange"]),
    PandasIndicatorInfo("DONCHIAN", "Volatility Indicators", donchian, {"length": 20}, ["donchian_upper", "donchian_mid", "donchian_lower"]),
    PandasIndicatorInfo("OBV", "Volume Indicators", obv, {}, ["obv"], True),
    PandasIndicatorInfo("AD", "Volume Indicators", ad, {}, ["ad"], True),
    PandasIndicatorInfo("ADOSC", "Volume Indicators", adosc, {"fast": 3, "slow": 10}, ["adosc"], True),
    PandasIndicatorInfo("CMF", "Volume Indicators", cmf, {"length": 20}, ["cmf"], True),
    PandasIndicatorInfo("VWAP", "Volume Indicators", vwap, {}, ["vwap"], True),
    PandasIndicatorInfo("AVGPRICE", "Price Transform", avgprice, {}, ["avgprice"]),
    PandasIndicatorInfo("MEDPRICE", "Price Transform", medprice, {}, ["medprice"]),
    PandasIndicatorInfo("TYPPRICE", "Price Transform", typprice, {}, ["typprice"]),
    PandasIndicatorInfo("WCLPRICE", "Price Transform", wclprice, {}, ["wclprice"]),
    PandasIndicatorInfo("STDDEV", "Statistic Functions", stddev, {"length": 5}, ["stddev"]),
    PandasIndicatorInfo("VAR", "Statistic Functions", variance, {"length": 5}, ["var"]),
    PandasIndicatorInfo("ZSCORE", "Statistic Functions", zscore, {"length": 20}, ["zscore"]),
    PandasIndicatorInfo("LINEARREG", "Statistic Functions", linearreg, {"length": 14}, ["linearreg"]),
]


class PandasIndicatorRegistry:
    def __init__(self) -> None:
        self._by_name = {i.name: i for i in PANDAS_INDICATORS}
        self._by_group: dict[str, list[str]] = {}
        for info in PANDAS_INDICATORS:
            self._by_group.setdefault(info.group, []).append(info.name)

    def get(self, name: str) -> PandasIndicatorInfo:
        key = name.upper()
        if key not in self._by_name:
            raise KeyError(f"Unknown pandas indicator: {name}")
        return self._by_name[key]

    def all_names(self) -> list[str]:
        return [i.name for i in PANDAS_INDICATORS]

    def groups(self) -> dict[str, list[str]]:
        return dict(self._by_group)

    def iter_ordered(self) -> list[PandasIndicatorInfo]:
        return list(PANDAS_INDICATORS)
