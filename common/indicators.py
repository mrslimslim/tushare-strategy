from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kdj(group: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    """对单个股票分组数据计算 K、D、J。
    需要列：['high','low','close']，按交易日升序。
    """
    high_n = group["high"].rolling(window=n, min_periods=n).max()
    low_n = group["low"].rolling(window=n, min_periods=n).min()

    rsv = (group["close"] - low_n) / (high_n - low_n) * 100.0
    rsv = rsv.replace([np.inf, -np.inf], np.nan).fillna(50.0)

    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3.0 * k - 2.0 * d

    group["K"] = k
    group["D"] = d
    group["J"] = j
    return group


def compute_macd_diff(group: pd.DataFrame) -> pd.DataFrame:
    """计算 MACD 的 DIFF（12/26）。需要列：['close']。"""
    ema12 = group["close"].ewm(span=12, adjust=False).mean()
    ema26 = group["close"].ewm(span=26, adjust=False).mean()
    group["diff"] = ema12 - ema26
    return group


def compute_ma(group: pd.DataFrame, window: int = 30, col: str = "close", out_col: str | None = None) -> pd.DataFrame:
    """计算简单移动平均线。默认为 MA30。"""
    out_col = out_col or f"ma{window}"
    group[out_col] = group[col].rolling(
        window=window, min_periods=window).mean()
    return group


def compute_ema(group: pd.DataFrame, span: int = 13, col: str = "close", out_col: str | None = None) -> pd.DataFrame:
    """计算指数移动平均 EMA。默认 span=13。"""
    out_col = out_col or f"ema{span}"
    group[out_col] = group[col].ewm(span=span, adjust=False).mean()
    return group


def compute_zx_short_trend(group: pd.DataFrame) -> pd.DataFrame:
    """知行短期趋势线：EMA(EMA(close, 10), 10)。输出列：'zx_short_trend'。

    该指标在技术公式中标注为白色细线，这里只计算数值供筛选使用。
    """
    ema10_once = group["close"].ewm(span=10, adjust=False).mean()
    zx = ema10_once.ewm(span=10, adjust=False).mean()
    group["zx_short_trend"] = zx
    # 额外补充：常用参考线 MA1=MA(close,60)，MA2=EMA(close,13)
    if "ma60" not in group.columns:
        group = compute_ma(group, window=60, col="close", out_col="ma60")
    if "ema13" not in group.columns:
        group = compute_ema(group, span=13, col="close", out_col="ema13")
    return group


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """对包含多支股票（ts_code）的日线数据计算 KDJ、MACD DIFF、MA30、VOL_MA10。

    预期列：['ts_code','trade_date','open','high','low','close','vol'] 至少包含
    要求按 ts_code, trade_date 升序。
    """
    if df.empty:
        return df

    df = df.sort_values(["ts_code", "trade_date"]).copy()
    df = df.groupby("ts_code", group_keys=False).apply(compute_kdj)
    df = df.groupby("ts_code", group_keys=False).apply(compute_macd_diff)
    df = df.groupby("ts_code", group_keys=False).apply(
        lambda g: compute_ma(g, window=30, col="close", out_col="ma30"))
    # 成交量10日均量
    df = df.groupby("ts_code", group_keys=False).apply(
        lambda g: compute_ma(g, window=10, col="vol", out_col="vol_ma10"))
    # 知行短期趋势线：EMA(EMA(close, 10), 10)
    df = df.groupby("ts_code", group_keys=False).apply(compute_zx_short_trend)
    return df
