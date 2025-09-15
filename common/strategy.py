from __future__ import annotations

import numpy as np
import pandas as pd


def add_signal_columns(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    为日线数据添加策略信号列。

    条件：
    - J < 30
    - DIFF > 0
    - MA30 上移（当日 ma30 > 前一日 ma30）
    - 成交量为近10个交易日中倒数第1或第2或第3
    - 当日涨跌幅 ∈ [-2.5%, 2.25%]
    - 收盘相对知行短期趋势线偏离绝对值 ≤ 1.5%
    - 当日振幅（(high - low) / 前收盘）< 4%
    - 近14个交易日内：至少1日涨幅 > 6%

    依赖列：'ts_code','trade_date','open','high','low','close','vol','J','diff','ma30','zx_short_trend'
    若缺少 'prev_close' 或 'pct_chg' 将在函数内自动补充。
    """
    if enriched is None or enriched.empty:
        df = enriched.copy() if enriched is not None else pd.DataFrame()
        df["signal"] = False
        return df

    df = enriched.sort_values(["ts_code", "trade_date"]).copy()

    # 前收和涨跌幅（若缺失则计算）
    if "prev_close" not in df.columns:
        df["prev_close"] = df.groupby("ts_code")["close"].shift(1)
    if "pct_chg" not in df.columns:
        df["pct_chg"] = (df["close"] / df["prev_close"] - 1.0) * 100.0

    # MA30 上移
    df["ma30_prev"] = df.groupby("ts_code")["ma30"].shift(1)
    df["ma30_rising"] = (
        df["ma30"].notna() & df["ma30_prev"].notna() & (
            df["ma30"] > df["ma30_prev"])
    )

    # 与知行短期趋势线的偏离
    if "zx_short_trend" in df.columns:
        df["zx_dev_pct"] = (
            (df["close"] / df["zx_short_trend"]) - 1.0).abs() * 100.0
        cond_zx = (
            df["zx_short_trend"].notna() & df["close"].notna() & (
                df["zx_dev_pct"] <= 1.5)
        )
    else:
        df["zx_dev_pct"] = pd.NA
        cond_zx = pd.Series(False, index=df.index)

    # 成交量条件：当日成交量在近10个交易日内处于最小、次小或第三小（按去重排序）
    def _is_bottom_k_unique(a: np.ndarray, k: int = 2) -> float:
        # 要求窗口内10个值均为有限数
        if not np.all(np.isfinite(a)):
            return np.nan
        last = a[-1]
        uniq_sorted = np.unique(a)
        if uniq_sorted.size <= 1:
            # 窗口内全相等，视为满足（处于最小组）
            return 1.0
        idx = min(k - 1, uniq_sorted.size - 1)
        threshold = uniq_sorted[idx]  # 第k小的唯一值（若唯一值不足k个则取最大）
        return 1.0 if last <= threshold else 0.0

    vol_is_bottom3 = (
        df.groupby("ts_code", group_keys=False)["vol"]
          .apply(lambda s: s.rolling(window=10, min_periods=10)
                 .apply(_is_bottom_k_unique, raw=True))
    )
    cond_vol = vol_is_bottom3.notna() & (vol_is_bottom3 > 0.0)

    # 当日振幅：相对前收盘
    df["amp_pct"] = ((df["high"] - df["low"]) / df["prev_close"]).replace(
        [np.inf, -np.inf], np.nan
    ) * 100.0
    cond_amp = df["amp_pct"].notna() & (df["amp_pct"] < 4.0)

    # 近14个交易日的单日涨跌极值
    df["max_pct14"] = df.groupby("ts_code")["pct_chg"].transform(
        lambda s: s.rolling(window=14, min_periods=14).max()
    )
    cond_up8 = df["max_pct14"].notna() & (df["max_pct14"] > 6.0)

    # 其他条件
    cond_j = df["J"] < 30
    cond_diff = df["diff"] > 0
    cond_pct = df["pct_chg"].notna() & (
        df["pct_chg"] >= -2.5) & (df["pct_chg"] <= 2.25)

    signal = (
        cond_j
        & cond_diff
        & cond_vol
        & cond_pct
        & cond_zx
        & cond_amp
        & cond_up8
        & df["ma30_rising"].fillna(False)
    )
    df["signal"] = signal

    # 清理临时列
    # 保留 ma30_rising、zx_dev_pct、amp_pct 以便调试/导出
    df = df.drop(columns=["ma30_prev"])
    return df
