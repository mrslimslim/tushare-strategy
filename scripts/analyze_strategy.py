import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

try:
    from common.indicators import compute_all_indicators
    from common.env import get_tushare_pro
    from common.strategy import add_signal_columns, add_score_columns
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.indicators import compute_all_indicators
    from common.env import get_tushare_pro
    from common.strategy import add_signal_columns, add_score_columns


def get_latest_trade_date_from_cal(days_back: int = 30) -> str | None:
    """从交易日历获取最近一个交易日（返回 yyyymmdd 字符串）。
    若无法访问 Tushare（无 token），返回 None。
    """
    try:
        pro = get_tushare_pro()
    except Exception:
        return None
    today = datetime.today()
    start = (today - timedelta(days=days_back)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    cal = cal[cal["is_open"] == 1]
    if cal.empty:
        return None
    # 取区间内的最大交易日（而非依赖返回顺序）
    try:
        latest = int(cal["cal_date"].astype(int).max())
    except Exception:
        latest = str(cal["cal_date"].astype(str).max())
        return latest
    return str(latest)


def analyze(
    data_dir: Path,
    prefer_trade_cal: bool = True,
    specified_date: str | None = None,
    weight_scheme: str = "equal",
    norm: str = "zscore",
    min_turnover: float | None = None,
    topn: int | None = None,
    streak_bonus: float = 0.0,
) -> pd.DataFrame:
    basics_fp = data_dir / "stock_basic.csv"
    daily_fp = data_dir / "daily.csv"

    basics = pd.read_csv(basics_fp)
    daily = pd.read_csv(daily_fp)

    # 指标计算
    use_cols = [
        "ts_code", "trade_date", "open", "high", "low", "close", "vol"
    ]
    # 日线数据确保字段存在
    for col in use_cols:
        if col not in daily.columns:
            raise RuntimeError(f"日线缺少列: {col}")

    enriched = compute_all_indicators(daily[use_cols].copy())

    # 计算涨跌幅（若原始数据未提供 pct_chg，则用前一日收盘推算）
    enriched = enriched.sort_values(["ts_code", "trade_date"]).copy()
    enriched["prev_close"] = enriched.groupby("ts_code")["close"].shift(1)
    enriched["pct_chg"] = (enriched["close"] /
                           enriched["prev_close"] - 1.0) * 100
    # 统一由共用策略模块生成信号（含：振幅<4%、偏离≤1.5%、MA30上移等）
    df_sig = add_signal_columns(enriched)
    # 打分（全量数据按截面标准化）
    df_scored = add_score_columns(
        df_sig,
        norm=norm,
        weight_scheme=weight_scheme,
        min_turnover=min_turnover,
        streak_bonus=streak_bonus,
    )

    # 目标交易日：优先使用交易日历的最近交易日；若数据中不存在则回退为数据最大日期
    if enriched.empty:
        return pd.DataFrame()

    last_in_data = enriched["trade_date"].max()
    # 确定目标交易日：优先使用指定日期；否则遵循原有逻辑
    if specified_date is not None:
        target_date = specified_date
    else:
        target_date = last_in_data
        if prefer_trade_cal:
            latest_from_cal = get_latest_trade_date_from_cal(days_back=120)
            if latest_from_cal is not None:
                target_date = latest_from_cal
    # 统一 dtype 以匹配数据列
    if pd.api.types.is_numeric_dtype(enriched["trade_date"].dtype):
        try:
            target_date_norm = int(target_date)
        except Exception:
            target_date_norm = last_in_data
    else:
        target_date_norm = str(target_date)
    if target_date_norm not in set(enriched["trade_date"].unique()):
        # 数据中没有目标交易日，回退为数据最大日期
        target_date_norm = last_in_data

    latest = df_scored[df_scored["trade_date"] == target_date_norm].copy()
    # 基础资格：流动性 + 策略信号
    base_mask = latest.get("eligible_liquidity", True)
    if isinstance(base_mask, bool):
        base_mask = pd.Series([base_mask] * len(latest), index=latest.index)
    sig_mask = latest.get("signal", False)
    if isinstance(sig_mask, bool):
        sig_mask = pd.Series([sig_mask] * len(latest), index=latest.index)
    selected = latest[base_mask.fillna(True) & sig_mask.fillna(False)].copy()

    # 附加基本信息
    selected = selected.merge(
        basics[["ts_code", "name", "industry", "area", "list_date"]],
        on="ts_code",
        how="left"
    )

    # 排序：综合得分降序；同分按风险更小、偏离更小、J更小
    sort_cols_all = ["score", "amp_pct", "zx_dev_pct", "J"]
    sort_asc_all = [False, True, True, True]
    exist_cols = [c for c in sort_cols_all if c in selected.columns]
    exist_asc = [sort_asc_all[sort_cols_all.index(c)] for c in exist_cols]
    if exist_cols:
        selected = selected.sort_values(exist_cols, ascending=exist_asc)

    # Top N（可选）
    if topn is not None and int(topn) > 0:
        selected = selected.head(int(topn))

    selected["trade_date"] = selected["trade_date"].astype(str)

    # 优先展示名称与核心得分信息
    preferred_cols = ["name", "score", "ts_code"]
    ordered = [col for col in preferred_cols if col in selected.columns]
    ordered += [col for col in selected.columns if col not in ordered]
    selected = selected[ordered]

    return selected


def main():
    parser = argparse.ArgumentParser(
        description=(
            "按策略筛选并打分：动量/趋势/量/风险分项归一与加权，支持TopN与流动性门槛"
        ))
    parser.add_argument("--data", type=str, default="data",
                        help="数据目录（包含 stock_basic.csv, daily.csv）")
    parser.add_argument("--out", type=str,
                        default="output.csv", help="筛选结果CSV")
    parser.add_argument("--no-prefer-trade-cal",
                        action="store_true", help="不从交易日历获取最新交易日，改用数据中最大日期")
    parser.add_argument("--date", type=str, default=None,
                        help="指定交易日（YYYYMMDD）。未提供则默认使用最新交易日")
    parser.add_argument("--weight-scheme", type=str, default="equal",
                        choices=["equal", "momentum_heavy", "momentum_tilt"],
                        help="权重方案：equal=等权；momentum_heavy=动量40/趋势30/量20/风险10")
    parser.add_argument("--norm", type=str, default="zscore",
                        choices=["zscore", "rank"],
                        help="截面规范化方式：zscore 或 rank")
    parser.add_argument("--min-turnover", type=float, default=None,
                        help="成交额门槛（vol*close），用于资格过滤")
    parser.add_argument("--topn", type=int, default=None,
                        help="可选：仅导出综合得分排名前N名")
    parser.add_argument("--streak-bonus", type=float, default=0.0,
                        help="信号连续性加分系数（每连续1天叠加该系数）")
    args = parser.parse_args()

    data_dir = Path(args.data)
    result = analyze(
        data_dir,
        prefer_trade_cal=(not args.no_prefer_trade_cal),
        specified_date=args.date,
        weight_scheme=args.weight_scheme,
        norm=args.norm,
        min_turnover=args.min_turnover,
        topn=args.topn,
        streak_bonus=args.streak_bonus,
    )

    if result.empty:
        print("无符合条件的标的。")
        return

    out_path = Path(args.out)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已输出 {result.shape[0]} 条结果到 {out_path}")


if __name__ == "__main__":
    main()
