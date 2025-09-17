from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


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
    def _is_bottom_k_unique(a: np.ndarray, k: int = 1) -> float:
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
          .apply(lambda s: s.rolling(window=14, min_periods=14)
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


def add_score_columns(
    df: pd.DataFrame,
    norm: str = "zscore",
    weight_scheme: str = "equal",
    min_turnover: float | None = None,
    prefer_quiet_volume: bool = True,
    smooth_alpha: float = 0.4,
    industry_bonus: Optional[Dict[str, float]] = None,
    area_bonus: Optional[Dict[str, float]] = None,
    streak_bonus: float = 0.0,
) -> pd.DataFrame:
    """
    基于多维因子为每个交易日的截面计算分项得分与综合得分。

    模块与要素：
    - 动量（momentum）：pct_chg（越大越好）、J（越小越好）、zx_dev_pct（越小越好）
    - 趋势（trend）：ma30_rising（布尔）、ema13 相对 ma30/60（越大越好）、close 相对 zx_short_trend（越大越好）
    - 成交量（volume）：vol 相对 vol_ma10（prefer_quiet_volume=True 时越小越好），叠加流动性（turnover=vol*close）
    - 风险（risk）：amp_pct（越小越稳）、max_pct14（越小越稳）

    规范化：对每个交易日做截面标准化（z-score 或排名百分位），并统一方向（越大越好）。
    平滑：对综合得分按 ts_code 做 EWM 平滑（alpha=smooth_alpha）。

    额外：可配置行业/地域加分（industry_bonus/area_bonus），以及流动性门槛（min_turnover）。
    输出列（新增）：
        'score_momentum','score_trend','score_volume','score_risk',
        'score_raw','score_smoothed','score','score_rank_pct',
        'signal_streak3','eligible_liquidity','turnover','vol_rel_ma10',
        'ema13_rel_ma30','ema13_rel_ma60','close_rel_zx'
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.sort_values(["ts_code", "trade_date"]).copy()

    # 基础派生列
    def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        r = a / b
        r = r.replace([np.inf, -np.inf], np.nan)
        return r

    if "vol_ma10" in out.columns and "vol" in out.columns:
        out["vol_rel_ma10"] = _safe_ratio(out["vol"], out["vol_ma10"])
    else:
        out["vol_rel_ma10"] = np.nan

    if "ema13" in out.columns and "ma30" in out.columns:
        out["ema13_rel_ma30"] = _safe_ratio(out["ema13"], out["ma30"]) - 1.0
    else:
        out["ema13_rel_ma30"] = np.nan
    if "ema13" in out.columns and "ma60" in out.columns:
        out["ema13_rel_ma60"] = _safe_ratio(out["ema13"], out["ma60"]) - 1.0
    else:
        out["ema13_rel_ma60"] = np.nan

    if "zx_short_trend" in out.columns and "close" in out.columns:
        out["close_rel_zx"] = _safe_ratio(
            out["close"], out["zx_short_trend"]) - 1.0
    else:
        out["close_rel_zx"] = np.nan

    # 流动性（成交额近似）：单位依赖于数据源，阈值需按实际调整
    if "vol" in out.columns and "close" in out.columns:
        out["turnover"] = (out["vol"].astype(float) *
                           out["close"].astype(float))
    else:
        out["turnover"] = np.nan

    # 规范化方法
    def _norm_cs(series: pd.Series) -> pd.Series:
        # 对每个交易日截面归一
        if norm == "rank":
            return series.groupby(out["trade_date"]).transform(
                lambda x: x.rank(pct=True, method="average") - 0.5
            )
        # z-score

        def _z(x: pd.Series) -> pd.Series:
            m = x.mean(skipna=True)
            s = x.std(ddof=0, skipna=True)
            if not np.isfinite(s) or s == 0:
                return pd.Series(0.0, index=x.index)
            return (x - m) / s
        return series.groupby(out["trade_date"]).transform(_z)

    # 统一方向的子项得分（越大越好）
    s_pct = _norm_cs(out["pct_chg"]) if "pct_chg" in out.columns else pd.Series(
        np.nan, index=out.index)
    s_J = - \
        _norm_cs(out["J"]) if "J" in out.columns else pd.Series(
            np.nan, index=out.index)
    s_zx_dev = -_norm_cs(out["zx_dev_pct"]
                         ) if "zx_dev_pct" in out.columns else pd.Series(np.nan, index=out.index)

    s_ema13_ma30 = _norm_cs(out["ema13_rel_ma30"]) if "ema13_rel_ma30" in out.columns else pd.Series(
        np.nan, index=out.index)
    s_ema13_ma60 = _norm_cs(out["ema13_rel_ma60"]) if "ema13_rel_ma60" in out.columns else pd.Series(
        np.nan, index=out.index)
    s_close_zx = _norm_cs(out["close_rel_zx"]) if "close_rel_zx" in out.columns else pd.Series(
        np.nan, index=out.index)
    s_ma30_rising = out.get("ma30_rising", pd.Series(
        False, index=out.index)).astype(float)

    # 成交量偏好：静默（相对均量更低）更好，或相反
    s_volQuiet = -_norm_cs(out["vol_rel_ma10"]) if prefer_quiet_volume else _norm_cs(out["vol_rel_ma10"])  # noqa: N816
    s_liquidity = _norm_cs(out["turnover"])  # 流动性越高越好（仅用于volume模块的次要项）

    s_amp = - \
        _norm_cs(out["amp_pct"]) if "amp_pct" in out.columns else pd.Series(
            np.nan, index=out.index)
    s_max14 = -_norm_cs(out["max_pct14"]
                        ) if "max_pct14" in out.columns else pd.Series(np.nan, index=out.index)

    # 模块内聚合：忽略 NaN 取均值
    def _nanmean_cols(cols: list[pd.Series]) -> pd.Series:
        arr = np.vstack([c.astype(float).to_numpy(copy=False) for c in cols])
        with np.errstate(invalid="ignore"):
            m = np.nanmean(arr, axis=0)
        # 全 NaN 时返回 0
        m = np.where(np.isfinite(m), m, 0.0)
        return pd.Series(m, index=out.index)

    out["score_momentum"] = _nanmean_cols([s_pct, s_J, s_zx_dev])
    out["score_trend"] = _nanmean_cols(
        [s_ma30_rising, s_ema13_ma30, s_ema13_ma60, s_close_zx])
    # 成交量：以“静默度”为主，叠加少量流动性（确保可执行）
    out["score_volume"] = 0.7 * s_volQuiet + 0.3 * s_liquidity
    out["score_risk"] = _nanmean_cols([s_amp, s_max14])

    # 跨模块权重
    if weight_scheme == "momentum_heavy":
        w_m, w_t, w_v, w_r = 0.40, 0.30, 0.20, 0.10
    elif weight_scheme == "momentum_tilt":
        # 轻度动量倾斜：动量/趋势/量/风险 = 30%/30%/25%/15%
        w_m, w_t, w_v, w_r = 0.30, 0.30, 0.25, 0.15
    else:
        w_m, w_t, w_v, w_r = 0.25, 0.25, 0.25, 0.25

    out["score_raw"] = (
        w_m * out["score_momentum"]
        + w_t * out["score_trend"]
        + w_v * out["score_volume"]
        + w_r * out["score_risk"]
    )

    # 行业与地域加分（若提供映射，建议值在 [-0.2, +0.2] 之间）
    bonus = 0.0
    if industry_bonus is not None and "industry" in out.columns:
        out["_ind_bonus"] = out["industry"].map(industry_bonus).fillna(0.0)
        bonus = bonus + out["_ind_bonus"]
    else:
        out["_ind_bonus"] = 0.0
    if area_bonus is not None and "area" in out.columns:
        out["_area_bonus"] = out["area"].map(area_bonus).fillna(0.0)
        bonus = bonus + out["_area_bonus"]
    else:
        out["_area_bonus"] = 0.0

    out["score_smoothed"] = out.groupby("ts_code", group_keys=False)["score_raw"].apply(
        lambda s: s.ewm(alpha=float(smooth_alpha), adjust=False).mean()
    )
    out["score"] = out["score_smoothed"] + bonus

    # 信号连续性（过去3天 signal=True 的天数，含当日；若无 signal 列则记0）
    if "signal" in out.columns:
        out["signal_streak3"] = (
            out.sort_values(["ts_code", "trade_date"])  # ensure order
            .groupby("ts_code")["signal"]
            .apply(lambda s: s.rolling(window=3, min_periods=1).sum())
            .reset_index(level=0, drop=True)
        )
        # 可选：将连续性弱化为次级加分，不直接写入 score，以便回测阶段试验
    else:
        out["signal_streak3"] = 0.0

    # 若配置了 streak 加分，则在最终分数上叠加（幅度建议 <=0.1）
    if streak_bonus is not None and float(streak_bonus) != 0.0:
        try:
            coef = float(streak_bonus)
        except Exception:
            coef = 0.0
        if coef != 0.0:
            out["score"] = out["score"] + coef * \
                out["signal_streak3"].astype(float)

    # 每日得分百分位（用于阈值或二次排序）
    out["score_rank_pct"] = out.groupby(
        "trade_date")["score"].rank(pct=True, method="average")

    # 流动性资格
    if min_turnover is not None and np.isfinite(min_turnover):
        out["eligible_liquidity"] = out["turnover"].astype(
            float) >= float(min_turnover)
    else:
        out["eligible_liquidity"] = True

    # 清理临时列标记
    return out
